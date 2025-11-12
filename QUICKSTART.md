# Quick Start Guide

Get up and running in 5 minutes!

## Prerequisites

- Python 3.8 or higher
- Webcam
- Microphone
- 4GB+ RAM
- 2GB free disk space

## Installation (First Time Only)

### Step 1: Install

**Windows:**
```bash
python install.py
```

**Mac/Linux:**
```bash
python3 install.py
```

Answer 'y' when prompted to install packages and download the model.

This will take 5-10 minutes depending on your internet speed.

## Running the Application

### Method 1: Use the launcher script

**Windows:**
Double-click `run.bat` or in terminal:
```bash
run.bat
```

**Mac/Linux:**
```bash
chmod +x run.sh
./run.sh
```

### Method 2: Direct command

**Windows:**
```bash
python main.py
```

**Mac/Linux:**
```bash
python3 main.py
```

## Using the Application

1. **Click "Start"** button
2. **Speak into your microphone**
3. **Watch transcription appear** in real-time on the right
4. **Click "Stop"** when done
5. **Click "Export"** to save transcript (optional)

## That's It!

You're ready to go. For more detailed instructions, see:
- **USAGE_GUIDE.md** - Detailed usage instructions
- **README.md** - Full documentation
- **TROUBLESHOOTING.md** - Problem solving

## Common First-Time Issues

### Camera Permission Denied
- **Windows**: Settings → Privacy → Camera → Allow
- **Mac**: System Preferences → Security → Camera → Grant access to Terminal
- **Linux**: `sudo usermod -a -G video $USER` then logout/login

### Microphone Not Working
- Check system sound settings
- Ensure microphone is not muted
- Grant microphone permission to Python/Terminal

### Application Slow
Edit `config.yaml`:
```yaml
speech:
  model_size: "tiny"  # Faster but less accurate
```

### Model Download Fails
- Check internet connection
- Run install.py again
- May take time on slow connections

## Quick Settings

Edit `config.yaml` for:
- Model size (accuracy vs speed)
- Video resolution
- Speaker diarization on/off
- Processing buffer duration

## Verify Installation

Test all components:
```bash
python test_components.py
```

## Need Help?

1. Check **TROUBLESHOOTING.md**
2. Run `python test_components.py`
3. Check console for error messages

## What's Next?

- Try speaking with multiple people (speaker separation)
- Take snapshots during meetings
- Export transcripts for documentation
- Adjust settings for your use case

---

**Have fun transcribing! 🎤→📝**

