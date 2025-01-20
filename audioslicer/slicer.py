import librosa
import numpy as np
import soundfile as sf
from pathlib import Path
import logging
import random
from typing import List, Optional, Tuple
from dataclasses import dataclass
import argparse

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class SlicerConfig:
    threshold_db: float = -40.0  # Less strict silence detection
    min_length: float = 5000    # Shorter minimum segments
    min_interval: float = 300    # Much shorter silence intervals
    hop_size: float = 10       # Keep this the same
    max_silence: float = 1000   # Less silence padding

def random_word(length: int = 5) -> str:
    vowels = 'aeiou'
    consonants = 'bcdfghjklmnpqrstvwxyz'
    word = ''
    for i in range(length):
        word += random.choice(consonants if i % 2 == 0 else vowels)
    return word

class AudioSlicer:
    def __init__(self, config=None):
        self.config = config if config else SlicerConfig()
        
    def _ms_to_samples(self, ms: float, sr: int) -> int:
        return int(np.floor(sr * ms / 1000))
        
    def _get_rms_frames(self, audio: np.ndarray, sr: int) -> np.ndarray:
        hop_length = self._ms_to_samples(self.config.hop_size, sr)
        frame_length = min(hop_length * 2, len(audio))
        
        rms = librosa.feature.rms(
            y=audio, 
            frame_length=frame_length, 
            hop_length=hop_length,
            center=True
        )[0]
        
        with np.errstate(divide='ignore'):
            rms_db = librosa.amplitude_to_db(rms, ref=np.max)
        return np.nan_to_num(rms_db, neginf=self.config.threshold_db - 10)
        
    def _find_silence_regions(self, rms: np.ndarray, sr: int) -> List[Tuple[int, int]]:
        logger.info(f"RMS range: {np.min(rms):.1f}dB to {np.max(rms):.1f}dB")
        
        # Use a bit more sensitive threshold for initial detection
        detection_threshold = self.config.threshold_db + 5
        silent_frames = rms < detection_threshold
        
        # Merge very short non-silent regions
        min_activity = int(0.03 * sr / self._ms_to_samples(self.config.hop_size, sr))  # 30ms
        silent_frames = np.array([1 if sum(silent_frames[max(0, i-min_activity):min(len(silent_frames), i+min_activity+1)]) > min_activity 
                                else 0 for i in range(len(silent_frames))])
        
        logger.info(f"Found {np.sum(silent_frames)} silent frames after merging")
        
        changes = np.diff(silent_frames.astype(int), prepend=0, append=0)
        silence_starts = np.where(changes == 1)[0]
        silence_ends = np.where(changes == -1)[0]
        
        hop_samples = self._ms_to_samples(self.config.hop_size, sr)
        return [(int(start * hop_samples), int(end * hop_samples)) 
                for start, end in zip(silence_starts, silence_ends)]

    def process_folder(self, input_folder: str) -> None:
        input_path = Path(input_folder)
        if not input_path.exists():
            raise FileNotFoundError(f"Folder not found: {input_path}")

        output_path = input_path / 'sliced_audio'
        output_path.mkdir(exist_ok=True)

        audio_files = list(input_path.glob('*.wav')) + list(input_path.glob('*.mp3'))
        
        for audio_file in audio_files:
            try:
                logger.info(f"Processing: {audio_file.name}")
                y, sr = librosa.load(audio_file, sr=None)
                if len(y) == 0 or np.isnan(y).any():
                    logger.warning(f"Skipping invalid file: {audio_file}")
                    continue

                rms = self._get_rms_frames(y, sr)
                silence_regions = self._find_silence_regions(rms, sr)
                logger.info(f"Found {len(silence_regions)} silence regions")
                
                min_interval = self._ms_to_samples(self.config.min_interval, sr)
                original_regions = silence_regions.copy()
                silence_regions = [
                    (start, end) for start, end in silence_regions
                    if end - start >= min_interval
                ]
                
                # Debug information about silence regions
                for i, (start, end) in enumerate(original_regions):
                    length_ms = (end - start) * 1000 / sr
                    logger.info(f"Silence region {i}: {length_ms:.0f}ms")
                    
                logger.info(f"After filtering minimum interval {self.config.min_interval}ms: {len(silence_regions)} valid silence regions")
                
                segments = []
                last_end = 0
                
                # First identify all potential cut points
                for i, (silence_start, silence_end) in enumerate(silence_regions):
                    segment_length = silence_start - last_end
                    min_samples = self._ms_to_samples(self.config.min_length, sr)
                    segment_length_ms = segment_length * 1000 / sr
                    
                    logger.info(f"Checking segment {i}: {segment_length_ms:.0f}ms (min: {self.config.min_length}ms)")
                    
                    # Either the segment is long enough, or it's the last segment
                    if segment_length >= min_samples or i == len(silence_regions) - 1:
                        # Find optimal cutting point
                        silence = y[silence_start:silence_end]
                        if len(silence) > 1:
                            rms_silence = librosa.feature.rms(y=silence)[0]
                            cut_point = silence_start + np.argmin(rms_silence)
                        else:
                            cut_point = silence_start
                            
                        segments.append((last_end, cut_point))
                        last_end = cut_point
                
                # Add final segment if long enough
                if len(y) - last_end >= min_samples:
                    segments.append((last_end, len(y)))
                
                # Save all segments
                for i, (start, end) in enumerate(segments, 1):
                    segment = y[start:end]
                    if len(segment) > 0:
                        output_file = output_path / f"{audio_file.stem}_{random_word()}_{i:03d}.wav"
                        sf.write(output_file, segment, sr)
                        logger.info(f"Created segment {i}: {(end-start) * 1000 / sr:.0f}ms")
                
                logger.info(f"Created {len(segments)} segments from {audio_file.name}")
                
            except Exception as e:
                logger.error(f"Failed to process {audio_file}: {str(e)}")

def main():
    parser = argparse.ArgumentParser(description="Audio file slicer based on silence detection")
    parser.add_argument("input_folder", help="Folder containing audio files")
    defaults = SlicerConfig()
    parser.add_argument("-t", "--threshold", type=float, default=None, 
                       help=f"Silence threshold in dB (default: {defaults.threshold_db})")
    parser.add_argument("-l", "--min_length", type=float, default=None, 
                       help=f"Minimum segment length in ms (default: {defaults.min_length})")
    parser.add_argument("-i", "--min_interval", type=float, default=None, 
                       help=f"Minimum silence interval in ms (default: {defaults.min_interval})")
    parser.add_argument("-s", "--hop_size", type=float, default=None, 
                       help=f"Analysis window size in ms (default: {defaults.hop_size})")
    parser.add_argument("-m", "--max_silence", type=float, default=None, 
                       help=f"Maximum silence to keep in ms (default: {defaults.max_silence})")

    args = parser.parse_args()

    # Create config with default values
    config = SlicerConfig()
    
    # Override only the specified arguments
    if args.threshold is not None:
        config.threshold_db = args.threshold
    if args.min_length is not None:
        config.min_length = args.min_length
    if args.min_interval is not None:
        config.min_interval = args.min_interval
    if args.hop_size is not None:
        config.hop_size = args.hop_size
    if args.max_silence is not None:
        config.max_silence = args.max_silence

    try:
        slicer = AudioSlicer(config)
        slicer.process_folder(args.input_folder)
    except Exception as e:
        logger.error(f"Processing failed: {str(e)}")
        return 1
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())
