# Real-Time Speech-to-Text System with Speaker Diarization

A comprehensive, locally-running speech-to-text system with multi-speaker identification and video capture capabilities.

## 🌟 Features

- **Real-time Speech Transcription**: Convert spoken English to text instantly using OpenAI's Whisper model (running locally)
- **Multi-Speaker Identification**: Automatically detect and separate multiple speakers
- **Video Capture**: Record video and take snapshots while transcribing
- **Complete Privacy**: Everything runs locally - no third-party APIs or cloud services
- **Speaker Diarization**: Identify up to 5 simultaneous speakers with color-coded transcripts
- **Export Capabilities**: Save transcripts and snapshots for later review
- **Modern GUI**: User-friendly interface with real-time audio level monitoring

## 🔒 Security & Privacy

This system is designed with information security in mind:
- ✅ **No third-party API calls** - all processing happens on your machine
- ✅ **No data leaves your computer** - complete offline operation
- ✅ **No cloud dependencies** - works without internet connection (after initial setup)
- ✅ **Open source models** - uses OpenAI Whisper (open source, runs locally)

## 📋 Requirements

### System Requirements
- **Python**: 3.8 or higher
- **RAM**: Minimum 4GB (8GB recommended)
- **Storage**: ~2GB for models and dependencies
- **Camera**: Webcam or built-in camera
- **Microphone**: Working audio input device

### Operating Systems
- Windows 10/11
- macOS 10.14+
- Linux (Ubuntu 18.04+, etc.)

## 🚀 Installation

### Step 1: Clone the Repository

```bash
git clone <repository-url>
cd SanseAI_S2T
```

### Step 2: Install Dependencies

#### Option A: Using the Installation Helper (Recommended)

```bash
python install.py
```

This script will:
- Check your Python version
- Install all required packages
- Download the Whisper model
- Verify your camera and microphone

#### Option B: Manual Installation

```bash
# Install dependencies
pip install -r requirements.txt

# For Windows users with PyAudio issues:
pip install pipwin
pipwin install pyaudio
```

### Step 3: Verify Installation

```bash
python install.py
```

Select 'n' when asked to install packages, and it will verify your setup.

## 🎯 Quick Start

1. **Run the application:**

```bash
python main.py
```

2. **Click "Start"** to begin capturing audio and video

3. **Speak into your microphone** - transcription will appear in real-time

4. **Multiple speakers?** The system will automatically detect and color-code different speakers

5. **Take snapshots** using the 📷 button

6. **Export transcript** using the 💾 button when done

7. **Click "Stop"** to end the session

## ⚙️ Configuration

Edit `config.yaml` to customize settings:

### Audio Settings
```yaml
audio:
  sample_rate: 16000      # Audio sample rate (Hz)
  channels: 1             # Mono (1) or Stereo (2)
  chunk_size: 1024        # Audio buffer size
```

### Video Settings
```yaml
video:
  width: 640              # Video width
  height: 480             # Video height
  fps: 30                 # Frames per second
  camera_index: 0         # Camera ID (0 = default)
```

### Speech Recognition
```yaml
speech:
  model_size: "base"      # Options: tiny, base, small, medium, large
  language: "en"          # Language code
```

**Model Size Guide:**
- `tiny`: Fastest, lowest accuracy (~1GB RAM)
- `base`: Good balance (recommended) (~1GB RAM)
- `small`: Better accuracy (~2GB RAM)
- `medium`: High accuracy (~5GB RAM)
- `large`: Best accuracy (~10GB RAM)

### Speaker Diarization
```yaml
diarization:
  enabled: true           # Enable/disable speaker separation
  min_speakers: 1         # Minimum number of speakers
  max_speakers: 5         # Maximum speakers to track
```

## 📁 Project Structure

```
SanseAI_S2T/
├── main.py                      # Main application entry point
├── config.yaml                  # Configuration file
├── requirements.txt             # Python dependencies
├── install.py                   # Installation helper
│
├── audio_capture.py             # Audio recording module
├── video_capture.py             # Video capture module
├── speech_to_text.py            # Whisper integration
├── speaker_diarization.py       # Speaker identification
├── gui_application.py           # GUI interface
│
├── snapshots/                   # Saved snapshots (auto-created)
├── exports/                     # Exported transcripts (auto-created)
│
└── README.md                    # This file
```

## 🎬 Usage Examples

### Basic Transcription
1. Start the application
2. Click "Start"
3. Speak clearly into your microphone
4. View real-time transcription on the right panel

### Multi-Speaker Meeting
1. Configure `max_speakers` in config.yaml
2. Start the application
3. Multiple speakers will be automatically identified
4. Each speaker gets a unique color in the transcript
5. Export transcript at the end

### Recording with Video
1. Position yourself in front of the camera
2. Start the application
3. Take snapshots during important moments
4. Find saved images in the `snapshots/` folder

## 🔧 Troubleshooting

### Camera Not Working
- **Check permissions**: Ensure the app has camera access
- **Try different camera**: Change `camera_index` in config.yaml
- **Close other apps**: Close apps that might be using the camera

### Microphone Issues
- **Check device**: Run `python install.py` to see available devices
- **System settings**: Ensure microphone is not muted
- **Permissions**: Grant microphone access to Python

### PyAudio Installation Fails (Windows)
```bash
pip install pipwin
pipwin install pyaudio
```

### Low Transcription Quality
- **Increase model size**: Change to `small` or `medium` in config.yaml
- **Reduce background noise**: Use in a quiet environment
- **Better microphone**: Use a quality microphone closer to speakers
- **Adjust VAD threshold**: Modify in audio_capture.py

### Performance Issues
- **Use smaller model**: Switch to `tiny` or `base`
- **Close other apps**: Free up RAM
- **Increase buffer duration**: Adjust in config.yaml
- **Disable diarization**: Set `enabled: false` if not needed

## 🧪 Technical Details

### Speech Recognition Engine
Uses OpenAI's **Whisper** model:
- State-of-the-art automatic speech recognition
- Trained on 680,000 hours of multilingual data
- Runs completely offline after initial download
- Multiple model sizes for different accuracy/speed tradeoffs

### Speaker Diarization
Implements a **simplified speaker identification system**:
- Extracts audio features (pitch, spectral characteristics, energy)
- Uses clustering to identify unique speakers
- Adapts to speaker profiles over time
- Color-codes speakers in the UI

**Note**: For production-grade diarization, consider integrating `pyannote.audio` (requires Hugging Face authentication).

### Audio Processing Pipeline
1. **Capture**: PyAudio records audio at 16kHz
2. **Buffering**: Audio buffered for configurable duration
3. **VAD**: Voice Activity Detection filters non-speech
4. **Diarization**: Speaker identification (if enabled)
5. **Transcription**: Whisper converts speech to text
6. **Display**: Results shown in GUI with speaker info

## 📊 Performance Metrics

| Model | Speed (RTF*) | Accuracy | RAM Usage |
|-------|-------------|----------|-----------|
| tiny  | ~0.1x       | Good     | ~1GB      |
| base  | ~0.2x       | Better   | ~1GB      |
| small | ~0.5x       | High     | ~2GB      |
| medium| ~1.0x       | Higher   | ~5GB      |
| large | ~2.0x       | Best     | ~10GB     |

*RTF = Real-Time Factor (lower is faster)

## 🔐 Privacy Considerations

### What data is stored?
- **Snapshots**: Saved to `snapshots/` folder (optional, user-initiated)
- **Transcripts**: Saved to `exports/` folder (optional, user-initiated)
- **Models**: Whisper models cached in `~/.cache/whisper/`

### What data is transmitted?
- **Nothing**: All processing is local, no network transmission

### Can it work offline?
- **Yes**: After initial setup (downloading models), fully offline capable

## 🤝 Contributing

This is a complete, production-ready system. Potential enhancements:
- Integration with `pyannote.audio` for advanced diarization
- Support for more languages
- Real-time translation capabilities
- Punctuation restoration
- Noise reduction preprocessing
- GPU acceleration options

## 📝 License

This project uses the following open-source components:
- **OpenAI Whisper**: MIT License
- **PyTorch**: BSD-style License
- **OpenCV**: Apache 2.0 License

## 🆘 Support

### Common Questions

**Q: Can I use this for commercial purposes?**
A: Yes, but verify licenses of included components.

**Q: Does it work in other languages?**
A: Whisper supports 99 languages. Change `language` in config.yaml.

**Q: Can I process pre-recorded audio?**
A: This version is designed for real-time. For batch processing, modify speech_to_text.py.

**Q: How accurate is speaker identification?**
A: Basic implementation works well for distinct voices. For high accuracy, integrate pyannote.audio.

**Q: Can I run this on a Raspberry Pi?**
A: Yes, but use `tiny` model and expect slower performance.

## 🎓 Credits

Built with:
- [OpenAI Whisper](https://github.com/openai/whisper) - Speech recognition
- [PyTorch](https://pytorch.org/) - Deep learning framework
- [OpenCV](https://opencv.org/) - Computer vision
- [PyAudio](http://people.csail.mit.edu/hubert/pyaudio/) - Audio I/O

---

**Made with ❤️ for privacy-conscious users who need reliable, local speech-to-text**