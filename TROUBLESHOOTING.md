# Troubleshooting Guide

This guide covers common issues and their solutions.

## Installation Issues

### PyAudio Won't Install (Windows)

**Problem:** `pip install pyaudio` fails with compilation errors

**Solution:**
```bash
pip install pipwin
pipwin install pyaudio
```

**Alternative:**
Download the appropriate `.whl` file from:
https://www.lfd.uci.edu/~gohlke/pythonlibs/#pyaudio

Then install:
```bash
pip install PyAudio‑0.2.11‑cp39‑cp39‑win_amd64.whl
```
(adjust filename for your Python version)

### PyAudio Won't Install (Mac)

**Problem:** Installation fails with missing `portaudio` error

**Solution:**
```bash
brew install portaudio
pip install pyaudio
```

### PyAudio Won't Install (Linux)

**Problem:** Installation fails

**Solution:**
```bash
# Ubuntu/Debian
sudo apt-get install portaudio19-dev python3-pyaudio
pip install pyaudio

# Fedora
sudo dnf install portaudio-devel
pip install pyaudio
```

### PyTorch Installation Issues

**Problem:** PyTorch download is very slow or fails

**Solution:**
Install CPU version explicitly:
```bash
# Windows/Linux
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu

# Mac
pip install torch torchaudio
```

### Whisper Model Download Fails

**Problem:** Model download times out or fails

**Solution:**
1. Check internet connection
2. Manually download and place in cache:
   - Download from: https://github.com/openai/whisper/discussions
   - Place in: `~/.cache/whisper/` (Linux/Mac) or `%USERPROFILE%\.cache\whisper\` (Windows)

## Runtime Issues

### Application Won't Start

**Problem:** Nothing happens when running `python main.py`

**Checklist:**
1. Check Python version: `python --version` (need 3.8+)
2. Verify installation: `python test_components.py`
3. Check console for error messages
4. Try running with verbose output: `python -v main.py`

**Common causes:**
- Missing dependencies → Run `pip install -r requirements.txt`
- Wrong Python version → Upgrade Python
- Config file missing → Should auto-create, check permissions

### Camera Issues

#### Camera Not Detected

**Problem:** "Failed to start video capture" error

**Solutions:**

**Windows:**
1. Check camera permissions:
   - Settings → Privacy → Camera
   - Enable "Allow apps to access your camera"
2. Close other apps using camera (Zoom, Teams, Skype, etc.)
3. Try different camera_index in config.yaml (0, 1, 2...)

**Mac:**
1. System Preferences → Security & Privacy → Camera
2. Grant permission to Terminal/Python
3. Restart terminal after granting permission

**Linux:**
1. Check camera exists: `ls /dev/video*`
2. Add user to video group: `sudo usermod -a -G video $USER`
3. Logout and login again
4. Check permissions: `ls -l /dev/video0`

#### Black Screen in Video Feed

**Problem:** Camera opens but shows black screen

**Solutions:**
1. Check if camera LED is on
2. Try different USB port
3. Restart application
4. Check camera works in other apps
5. Update camera drivers

#### Wrong Camera Selected

**Problem:** Built-in camera used instead of external

**Solution:**
Find available cameras:
```python
import cv2
for i in range(5):
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        print(f"Camera {i} available")
        cap.release()
```

Then set in config.yaml:
```yaml
video:
  camera_index: 1  # Use the correct index
```

### Microphone Issues

#### No Audio Detected

**Problem:** Audio level bar shows no activity

**Solutions:**

**Windows:**
1. Right-click speaker icon → Sounds → Recording
2. Select microphone → Properties → Levels
3. Ensure not muted and volume is up
4. Set as default device

**Mac:**
1. System Preferences → Sound → Input
2. Select correct microphone
3. Check input level
4. Grant microphone permission to Terminal

**Linux:**
1. Check with: `arecord -l`
2. Test recording: `arecord -d 5 test.wav`
3. Adjust with: `alsamixer`

#### Find Available Audio Devices

Run this script:
```python
import pyaudio
p = pyaudio.PyAudio()
for i in range(p.get_device_count()):
    info = p.get_device_info_by_index(i)
    if info['maxInputChannels'] > 0:
        print(f"[{i}] {info['name']}")
p.terminate()
```

Or run: `python install.py` (shows devices)

#### Audio Clipping or Distortion

**Problem:** Transcription is poor, audio sounds distorted

**Solutions:**
1. Reduce microphone gain/volume
2. Move further from microphone
3. Use pop filter
4. Check for hardware issues

### Transcription Issues

#### No Transcription Appearing

**Problem:** Audio detected but no text appears

**Possible causes and solutions:**

1. **Speaking too quietly**
   - Speak louder or move closer to mic
   - Reduce VAD threshold in audio_capture.py

2. **Wrong language setting**
   - Check config.yaml: `language: "en"`
   - Change to correct language code

3. **Model not loaded**
   - Check console for "Loading Whisper model..." message
   - If missing, reinstall: `pip install --upgrade openai-whisper`

4. **Insufficient audio buffer**
   - Speak for at least 2-3 seconds
   - Increase buffer_duration in config.yaml

#### Poor Transcription Quality

**Problem:** Many errors in transcription

**Solutions:**

1. **Use larger model**
   ```yaml
   speech:
     model_size: "medium"  # or "large"
   ```

2. **Improve audio quality**
   - Reduce background noise
   - Use better microphone
   - Speak clearly and at normal pace
   - Ensure proper microphone positioning

3. **Check language setting**
   - Verify correct language in config.yaml
   - Whisper works best with English

4. **Increase buffer duration**
   ```yaml
   processing:
     buffer_duration: 4.0  # More context
   ```

#### Transcription Lag/Delay

**Problem:** Large delay between speaking and text appearing

**Solutions:**

1. **Use smaller model**
   ```yaml
   speech:
     model_size: "tiny"  # or "base"
   ```

2. **Reduce buffer duration**
   ```yaml
   processing:
     buffer_duration: 2.0
   ```

3. **Use GPU if available**
   - Install CUDA-enabled PyTorch
   - Whisper will automatically use GPU

4. **Close other applications**
   - Free up CPU and RAM
   - Check Task Manager/Activity Monitor

#### Words Cut Off

**Problem:** First/last words of sentences missing

**Solution:**
Increase overlap:
```yaml
processing:
  overlap_duration: 1.0  # More overlap
```

### Speaker Diarization Issues

#### All Speakers Identified as Same Person

**Problem:** Multiple speakers not separated

**Possible causes:**

1. **Diarization disabled**
   ```yaml
   diarization:
     enabled: true
   ```

2. **Similar voices**
   - System may struggle with similar-sounding voices
   - This is a limitation of the simple diarization

3. **Poor audio quality**
   - Improve microphone quality
   - Reduce background noise
   - Ensure speakers are close to mic

4. **Threshold too high**
   - Edit speaker_diarization.py
   - Lower `self.similarity_threshold` (default: 0.7)
   - Try 0.5 for more separation

#### Too Many Speakers Detected

**Problem:** Single speaker split into multiple

**Solutions:**

1. **Increase similarity threshold**
   - Edit speaker_diarization.py
   - Increase `self.similarity_threshold` to 0.85

2. **Reduce max speakers**
   ```yaml
   diarization:
     max_speakers: 2  # Limit to expected number
   ```

3. **Consistent audio conditions**
   - Keep same distance from mic
   - Avoid moving around
   - Maintain consistent volume

#### Reset Speaker Profiles

**Problem:** Want to reset speaker identification

**Solution:**
Stop and restart the application. Speaker profiles are not persisted between sessions.

### Performance Issues

#### Application Running Slow

**Problem:** UI laggy, high CPU usage

**Solutions:**

1. **Optimize configuration**
   ```yaml
   speech:
     model_size: "tiny"
   
   processing:
     buffer_duration: 2.0
   
   diarization:
     enabled: false
   
   video:
     width: 320
     height: 240
     fps: 15
   ```

2. **Close other applications**
   - Close browser tabs
   - Close other resource-intensive apps
   - Check background processes

3. **Check system resources**
   - Open Task Manager (Windows) / Activity Monitor (Mac)
   - Ensure not running out of RAM
   - Check CPU usage

#### Out of Memory Errors

**Problem:** Application crashes with memory error

**Solutions:**

1. **Use smaller model**
   ```yaml
   speech:
     model_size: "tiny"
   ```

2. **Close other applications**
   - Free up RAM
   - Restart computer if needed

3. **Reduce buffer size**
   ```yaml
   processing:
     buffer_duration: 2.0
   ```

4. **Disable diarization**
   ```yaml
   diarization:
     enabled: false
   ```

### GUI Issues

#### Window Not Appearing

**Problem:** Application runs but no window shows

**Solutions:**
1. Check if window is minimized or behind other windows
2. Try on different display (if multi-monitor)
3. Update graphics drivers
4. Reinstall tkinter:
   - Linux: `sudo apt-get install python3-tk`
   - Mac: Included with Python
   - Windows: Included with Python

#### GUI Frozen

**Problem:** Window not responding

**Solutions:**
1. Wait 30 seconds (may be processing)
2. Check console for errors
3. Force close and restart
4. Check if model is being downloaded (first run)

#### Text Not Displaying

**Problem:** Transcript area remains empty despite processing

**Solutions:**
1. Check console for transcription output
2. Verify transcription is actually happening
3. Try clearing and restarting
4. Check if scrolled to wrong position

## Error Messages

### "Failed to initialize audio: Invalid device index"

**Cause:** Audio device not found or wrong index

**Solution:**
Run `python install.py` to see available devices, then update config.yaml

### "CUDA out of memory"

**Cause:** GPU doesn't have enough memory for model

**Solution:**
Use CPU instead:
```yaml
speech:
  compute_type: "float32"
```

Or use smaller model

### "Could not find ffmpeg"

**Cause:** FFmpeg not installed (required by some audio processing)

**Solution:**
- Windows: Download from https://ffmpeg.org/download.html
- Mac: `brew install ffmpeg`
- Linux: `sudo apt-get install ffmpeg`

### "Permission denied" errors

**Cause:** No permission to access camera/microphone/files

**Solution:**
- Windows: Check app permissions in Settings
- Mac: Grant permission in System Preferences
- Linux: Check user groups and file permissions

### "Module not found" errors

**Cause:** Dependency not installed

**Solution:**
```bash
pip install -r requirements.txt
```

## Platform-Specific Issues

### Windows

#### "DLL load failed"
- Install Visual C++ Redistributable
- Download from Microsoft website

#### Antivirus blocking
- Add Python to antivirus exceptions
- Add project folder to exceptions

### macOS

#### "Operation not permitted"
- Grant Full Disk Access to Terminal
- System Preferences → Security & Privacy → Full Disk Access

#### Apple Silicon (M1/M2) Issues
- Use native ARM Python
- Some packages may need Rosetta 2

### Linux

#### "Permission denied: /dev/video0"
```bash
sudo usermod -a -G video $USER
# Logout and login
```

#### "ALSA lib" warnings
These are usually harmless warnings and can be ignored.

#### Missing libraries
```bash
# Ubuntu/Debian
sudo apt-get install python3-tk python3-dev portaudio19-dev

# Fedora
sudo dnf install python3-tkinter python3-devel portaudio-devel
```

## Still Having Issues?

### Diagnostic Steps

1. **Run component tests**
   ```bash
   python test_components.py
   ```

2. **Check versions**
   ```bash
   python --version
   pip list | grep -i whisper
   pip list | grep -i torch
   ```

3. **Reinstall from scratch**
   ```bash
   pip uninstall -y openai-whisper torch torchaudio
   pip install -r requirements.txt
   ```

4. **Try minimal configuration**
   Edit config.yaml:
   ```yaml
   speech:
     model_size: "tiny"
   
   diarization:
     enabled: false
   
   video:
     width: 320
     height: 240
   ```

5. **Check console output**
   - Run from terminal/command prompt
   - Look for error messages
   - Note where it fails

### Getting More Help

If you're still stuck:

1. Check the console output carefully
2. Note your:
   - Operating system and version
   - Python version
   - Installed package versions
   - Error messages (full text)
3. Try searching the error message
4. Check OpenAI Whisper issues: https://github.com/openai/whisper/issues

## Prevention Tips

### Regular Maintenance

1. **Keep dependencies updated**
   ```bash
   pip install --upgrade -r requirements.txt
   ```

2. **Clear cache if issues arise**
   ```bash
   # Windows
   rmdir /s %USERPROFILE%\.cache\whisper
   
   # Linux/Mac
   rm -rf ~/.cache/whisper
   ```

3. **Backup your config**
   - Keep a copy of config.yaml
   - Document your settings

### Best Practices

1. **Always test before important use**
   - Run test_components.py
   - Do a quick test recording
   - Verify camera and microphone

2. **Use appropriate model for your hardware**
   - Lower-end: tiny or base
   - Mid-range: base or small
   - High-end: medium or large

3. **Keep environment consistent**
   - Same Python version
   - Same virtual environment
   - Don't mix different installation methods

---

**Need more help? Check README.md and USAGE_GUIDE.md for additional information.**

