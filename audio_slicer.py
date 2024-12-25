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
    threshold_db: float = -40.0
    min_length: float = 5000
    min_interval: float = 300
    hop_size: float = 10
    max_silence: float = 1000

def random_word(length: int = 5) -> str:
    vowels = 'aeiou'
    consonants = 'bcdfghjklmnpqrstvwxyz'
    word = ''
    for i in range(length):
        word += random.choice(consonants if i % 2 == 0 else vowels)
    return word


class AudioSlicer:
    def __init__(self):
        self.config = SlicerConfig()
        
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
        silent_frames = rms < self.config.threshold_db
        changes = np.diff(silent_frames.astype(int), prepend=0, append=0)
        silence_starts = np.where(changes == 1)[0]
        silence_ends = np.where(changes == -1)[0]
        
        hop_samples = self._ms_to_samples(self.config.hop_size, sr)
        return [(int(start * hop_samples), int(end * hop_samples)) 
                for start, end in zip(silence_starts, silence_ends)]
    
    def _process_segment(self, audio: np.ndarray, sr: int, start: int, end: int, 
                        silence_start: int, silence_end: int) -> Optional[Tuple[int, np.ndarray]]:
        min_samples = self._ms_to_samples(self.config.min_length, sr)
        
        if end - start < min_samples:
            return None
            
        silence = audio[silence_start:silence_end]
        if len(silence) > 1:
            rms_silence = librosa.feature.rms(y=silence)[0]
            cut_point = silence_start + np.argmin(rms_silence)
        else:
            cut_point = silence_start
            
        max_silence = self._ms_to_samples(self.config.max_silence, sr)
        pad_before = min(max_silence // 2, cut_point - silence_start)
        pad_after = min(max_silence // 2, silence_end - cut_point)
        
        segment = audio[max(0, start):min(cut_point + pad_after, len(audio))]
        return cut_point - pad_before, segment

    def process_folder(self, input_folder: str) -> None:
        input_path = Path(input_folder)
        if not input_path.exists():
            raise FileNotFoundError(f"Folder not found: {input_path}")

        # Create output folder next to input folder
        output_path = input_path / 'sliced_audio'
        output_path.mkdir(exist_ok=True)

        audio_files = list(input_path.glob('*.wav')) + list(input_path.glob('*.mp3'))
        
        for audio_file in audio_files:
            try:
                y, sr = librosa.load(audio_file, sr=None)
                if len(y) == 0 or np.isnan(y).any():
                    logger.warning(f"Skipping invalid file: {audio_file}")
                    continue

                rms = self._get_rms_frames(y, sr)
                silence_regions = self._find_silence_regions(rms, sr)
                
                min_interval = self._ms_to_samples(self.config.min_interval, sr)
                silence_regions = [
                    (start, end) for start, end in silence_regions
                    if end - start >= min_interval
                ]
                
                last_end = 0
                
                for i, (silence_start, silence_end) in enumerate(silence_regions, 1):
                    result = self._process_segment(
                        y, sr, last_end, silence_start, 
                        silence_start, silence_end
                    )
                    
                    if result:
                        new_end, segment = result
                        if len(segment) > 0:
                            output_file = output_path / f"{audio_file.stem}_{random_word()}_{i:03d}.wav"
                            sf.write(output_file, segment, sr)
                            last_end = new_end
                
                if len(y) - last_end >= self._ms_to_samples(self.config.min_length, sr):
                    final_segment = y[last_end:]
                    if len(final_segment) > 0:
                        output_file = output_path / f"{audio_file.stem}_{random_word()}_final.wav"
                        sf.write(output_file, final_segment, sr)
                
                logger.info(f"Processed: {audio_file.name}")
                
            except Exception as e:
                logger.error(f"Failed to process {audio_file}: {str(e)}")


def main():
    parser = argparse.ArgumentParser(description="Audio file slicer based on silence detection")
    parser.add_argument("input_folder", help="Folder containing audio files")
    parser.add_argument("-t", "--threshold", type=float, default=-40.0, help="Silence threshold in dB")
    parser.add_argument("-l", "--min_length", type=float, default=5000, help="Minimum segment length (ms)")
    parser.add_argument("-i", "--min_interval", type=float, default=300, help="Minimum silence interval (ms)")
    parser.add_argument("-s", "--hop_size", type=float, default=10, help="Analysis window size (ms)")
    parser.add_argument("-m", "--max_silence", type=float, default=1000, help="Maximum silence to keep (ms)")

    args = parser.parse_args()

    config = SlicerConfig(
        threshold_db=args.threshold,
        min_length=args.min_length,
        min_interval=args.min_interval,
        hop_size=args.hop_size,
        max_silence=args.max_silence
    )

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
