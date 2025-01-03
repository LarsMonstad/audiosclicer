# Audio Slicer
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

A Python tool for automatically slicing audio files based on silence detection, designed for creating ML datasets and processing large audio collections.

## Features
- Automatic silence detection using RMS analysis
- Batch processing of WAV and MP3 files  
- Random word generation for unique file naming
- Configurable silence thresholds and segment lengths
- Maintains original sample rate

## Installation
```bash
pip install librosa numpy soundfile
```

## Usage
```bash
python audio_slicer.py /path/to/audio/folder [options]
```

### Options
- `-t, --threshold`: Silence threshold in dB (default: -40.0)
- `-l, --min_length`: Minimum segment length in ms (default: 5000)
- `-i, --min_interval`: Minimum silence interval in ms (default: 300)
- `-s, --hop_size`: Analysis window size in ms (default: 10)
- `-m, --max_silence`: Maximum silence to keep in ms (default: 1000)

### Example
```bash
python audio_slicer.py ./my_audio -t -35 -l 3000
```

## How It Works
1. Analyzes audio using RMS (root mean square) to detect silence
2. Identifies silence regions below the threshold
3. Splits audio at optimal points within silence regions  
4. Generates unique filenames using random pronounceable words
5. Saves segments in a new 'sliced_audio' directory

## Output
- Creates 'sliced_audio' folder in the input directory
- Naming format: `originalname_randomword_number.wav`
- Preserves original audio quality and sample rate

## Technical Notes
- Minimum segment length prevents creation of tiny clips
- Hop size affects precision vs processing speed
- Maximum silence parameter trims long silence periods
- Uses librosa for robust audio processing


## Acknowledgments
- Inspired by [GUI Audio Slicer](https://github.com/flutydeer/audio-slicer)

## License
MIT License

---
Made by Lars Monstad
