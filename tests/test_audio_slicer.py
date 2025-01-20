# tests/test_audio_slicer.py
import pytest
import numpy as np
import soundfile as sf
import tempfile
from pathlib import Path
from unittest.mock import patch

from audioslicer import AudioSlicer, SlicerConfig

@pytest.fixture
def default_config():
    return SlicerConfig()

@pytest.fixture
def slicer(default_config):
    return AudioSlicer(default_config)

@pytest.fixture
def sample_audio():
    duration = 3
    sr = 22050
    t = np.linspace(0, duration, int(sr * duration))
    audio = np.sin(2 * np.pi * 440 * t)  # 440 Hz tone
    audio[sr:2*sr] = 0  # Add 1 second of silence
    return audio, sr

@pytest.fixture
def temp_audio_folder():
    with tempfile.TemporaryDirectory() as tmpdir:
        temp_path = Path(tmpdir)
        duration = 3
        sr = 22050
        t = np.linspace(0, duration, int(sr * duration))
        audio = np.sin(2 * np.pi * 440 * t)
        
        test_file = temp_path / "test.wav"
        sf.write(str(test_file), audio, sr)
        yield temp_path

def test_slicer_config_defaults():
    config = SlicerConfig()
    assert config.threshold_db == -40.0
    assert config.min_length == 5000
    assert config.min_interval == 300
    assert config.hop_size == 10
    assert config.max_silence == 1000

def test_ms_to_samples(slicer):
    sr = 22050
    assert slicer._ms_to_samples(1000, sr) == 22050
    assert slicer._ms_to_samples(500, sr) == 11025
    assert slicer._ms_to_samples(0, sr) == 0

def test_get_rms_frames(slicer, sample_audio):
    audio, sr = sample_audio
    rms = slicer._get_rms_frames(audio, sr)
    assert isinstance(rms, np.ndarray)
    assert len(rms) > 0
    assert not np.isnan(rms).any()

def test_find_silence_regions(slicer, sample_audio):
    audio, sr = sample_audio
    rms = slicer._get_rms_frames(audio, sr)
    regions = slicer._find_silence_regions(rms, sr)
    
    assert isinstance(regions, list)
    assert all(isinstance(region, tuple) for region in regions)
    assert all(len(region) == 2 for region in regions)
    assert all(region[1] > region[0] for region in regions)

def test_process_folder(slicer, temp_audio_folder):
    with patch('librosa.load') as mock_load:
        mock_load.return_value = (np.zeros(22050), 22050)
        slicer.process_folder(str(temp_audio_folder))
        output_folder = temp_audio_folder / 'sliced_audio'
        assert output_folder.exists()

def test_process_folder_nonexistent():
    slicer = AudioSlicer()
    with pytest.raises(FileNotFoundError):
        slicer.process_folder("nonexistent_folder")

def test_main_with_args():
    with patch('sys.argv', ['script.py', 'test_folder', '-t', '-35', '-l', '4000']):
        with patch('audioslicer.slicer.AudioSlicer') as MockSlicer:
            from audioslicer.slicer import main
            main()
            config = MockSlicer.call_args[0][0]
            assert config.threshold_db == -35
            assert config.min_length == 4000

def test_find_silence_regions_no_silence(slicer):
    """Test silence detection with no silence in audio"""
    duration = 1
    sr = 22050
    t = np.linspace(0, duration, sr)
    audio = np.sin(2 * np.pi * 440 * t)  # Constant 440Hz tone
    
    rms = slicer._get_rms_frames(audio, sr)
    regions = slicer._find_silence_regions(rms, sr)
    assert len(regions) == 0  # Should find no silence regions

def test_process_folder_with_no_audio_files(tmp_path):
    """Test processing folder with no audio files"""
    slicer = AudioSlicer()
    slicer.process_folder(str(tmp_path))
    output_folder = tmp_path / 'sliced_audio'
    assert output_folder.exists()
    assert len(list(output_folder.glob('*.wav'))) == 0
