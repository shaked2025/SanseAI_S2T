# Usage Guide - Speech-to-Text System

## Getting Started

### First Time Setup

1. **Install Dependencies**
   ```bash
   python install.py
   ```
   This will install all required packages and download the Whisper model.

2. **Test Your Setup**
   - The install script will check your camera and microphone
   - Make sure both are working before proceeding

3. **Launch the Application**
   ```bash
   python main.py
   ```

## Using the Application

### Main Interface

The application window is divided into several sections:

#### Left Panel - Video Feed
- Shows your camera feed in real-time
- Video preview helps you position yourself properly
- Below the video is an audio level indicator showing microphone input

#### Right Panel - Transcription
- **Active Speakers**: Shows color-coded badges for each detected speaker
- **Live Transcript**: Displays transcriptions in real-time with timestamps
- Each speaker's text is shown in a unique color

#### Bottom Control Bar
- **▶ Start**: Begin capturing audio and video
- **⬛ Stop**: End the capture session
- **📷 Snapshot**: Take a photo (saved to `snapshots/` folder)
- **🗑 Clear**: Clear the transcript display
- **💾 Export**: Save transcript to a text file

### Basic Workflow

#### Single Speaker Transcription

1. Click **Start**
2. Position yourself in front of the camera
3. Speak clearly into your microphone
4. Watch as your words appear in real-time
5. Click **Stop** when done
6. Click **Export** to save the transcript

**Tips:**
- Speak at a normal pace
- Minimize background noise
- Stay within 1-2 feet of the microphone
- Use proper lighting for video

#### Multi-Speaker Transcription

1. Ensure `diarization: enabled: true` in config.yaml
2. Click **Start**
3. Multiple people can speak (one at a time works best)
4. System automatically identifies different speakers
5. Each speaker gets assigned a color
6. Transcripts show speaker ID and their text

**Tips:**
- Speakers should have distinct voice characteristics
- Works best when speakers take turns
- System adapts over time - first few utterances may not separate perfectly
- Maximum 5 speakers by default (configurable)

### Advanced Features

#### Taking Snapshots

Use snapshots to:
- Capture important moments during meetings
- Document who is speaking
- Create visual records alongside transcripts

**How to use:**
1. While running, click **📷 Snapshot**
2. Image saved to `snapshots/snapshot_YYYYMMDD_HHMMSS.jpg`
3. Check the status bar for confirmation

#### Exporting Transcripts

Export creates a text file with:
- Timestamps for each entry
- Speaker identification
- Full transcription text

**Format:**
```
[2025-11-12 14:30:15] Speaker 1: Hello, this is a test.
[2025-11-12 14:30:20] Speaker 2: Yes, I can hear you clearly.
```

**Location:** `exports/transcript_YYYYMMDD_HHMMSS.txt`

## Configuration Options

### Adjusting Model Size

Edit `config.yaml`:

```yaml
speech:
  model_size: "base"  # Change to: tiny, small, medium, or large
```

**When to use each model:**
- **tiny**: Testing, low-end hardware, speed priority
- **base**: Default, good balance (recommended)
- **small**: Better accuracy, slightly slower
- **medium**: High accuracy, needs more RAM
- **large**: Best accuracy, high-end hardware only

### Changing Camera

If you have multiple cameras:

```yaml
video:
  camera_index: 0  # Try 0, 1, 2, etc.
```

To find your camera index, run:
```bash
python install.py
```

### Audio Settings

For better quality:

```yaml
audio:
  sample_rate: 16000  # Don't change (Whisper requirement)
  channels: 1         # Use 1 for mono, 2 for stereo
```

### Processing Settings

Adjust real-time processing:

```yaml
processing:
  buffer_duration: 3.0   # Seconds of audio per transcription
  overlap_duration: 0.5  # Overlap between chunks
```

**Buffer Duration:**
- Lower (1-2s): Faster response, may cut off words
- Higher (3-5s): More accurate, slight delay
- Recommended: 3.0s

## Troubleshooting

### No Transcription Appearing

**Check:**
1. Is audio level bar showing activity?
   - If no: Microphone not working or muted
   - If yes: Speak louder or reduce noise threshold

2. Is status showing "Processing"?
   - If no: Audio may not be loud enough
   - Try speaking louder or closer to mic

3. Model loaded successfully?
   - Check console for error messages
   - May need to re-download model

### Wrong Speaker Assignment

**Solutions:**
1. Reset speaker profiles:
   - Stop and restart the application
   
2. Adjust similarity threshold:
   - Edit `speaker_diarization.py`
   - Change `self.similarity_threshold` (default: 0.7)
   - Lower = more likely to create new speakers
   - Higher = more likely to merge into existing speakers

3. For better accuracy:
   - Consider speakers with distinct voices
   - Ensure good audio quality
   - Let each speaker talk for a few seconds initially

### Video Feed Frozen

**Solutions:**
1. Stop and restart the application
2. Close other applications using the camera
3. Check camera connection
4. Try different camera_index

### Application Slow or Freezing

**Solutions:**
1. Use smaller model (`tiny` or `base`)
2. Close other applications
3. Reduce buffer duration
4. Disable diarization if not needed
5. Check CPU usage in Task Manager/Activity Monitor

### Audio Echo or Feedback

**Solutions:**
1. Use headphones
2. Reduce speaker volume
3. Position microphone away from speakers
4. Use unidirectional microphone

## Best Practices

### For Meetings

1. **Pre-meeting:**
   - Test camera and microphone
   - Position camera to see all participants
   - Ensure good lighting
   - Close unnecessary applications

2. **During meeting:**
   - Have speakers identify themselves initially
   - Encourage clear speech
   - Take snapshots of important moments
   - Use mute when not speaking (if using speakers)

3. **Post-meeting:**
   - Export transcript immediately
   - Review and edit transcript if needed
   - Rename speakers in transcript manually if desired

### For Interviews

1. Position camera to see interviewer
2. Use quality microphone
3. Quiet environment
4. Take snapshots of key moments
5. Export transcript for notes

### For Lectures/Presentations

1. Position near speaker
2. Use medium or large model for best accuracy
3. Take snapshots of slides
4. Export for study notes

## Keyboard Shortcuts

Currently the application uses button controls. Future versions may include:
- `Ctrl+S`: Start/Stop
- `Ctrl+T`: Take snapshot
- `Ctrl+E`: Export
- `Ctrl+K`: Clear transcript

## Performance Tips

### Optimize for Speed
```yaml
speech:
  model_size: "tiny"
  
processing:
  buffer_duration: 2.0
  
diarization:
  enabled: false
```

### Optimize for Accuracy
```yaml
speech:
  model_size: "medium"
  
processing:
  buffer_duration: 4.0
  
diarization:
  enabled: true
  max_speakers: 5
```

### Optimize for Multiple Speakers
```yaml
diarization:
  enabled: true
  min_speakers: 2
  max_speakers: 5
  
processing:
  buffer_duration: 3.0
```

## Common Use Cases

### 1. Solo Content Creation
- Model: base or small
- Diarization: disabled
- Buffer: 2-3 seconds
- Focus: Fast, accurate transcription

### 2. Team Meeting
- Model: base or medium
- Diarization: enabled (2-5 speakers)
- Buffer: 3-4 seconds
- Focus: Speaker separation

### 3. Interview/Podcast
- Model: medium or large
- Diarization: enabled (2 speakers)
- Buffer: 3-4 seconds
- Focus: High accuracy

### 4. Classroom/Lecture
- Model: medium
- Diarization: enabled (1-2 speakers)
- Buffer: 4-5 seconds
- Focus: Accuracy over speed

## Privacy & Security

### Data Storage
- All audio processing happens in RAM
- No audio files saved (unless you explicitly record)
- Snapshots only saved when you click the button
- Transcripts only saved when you export

### Data Transmission
- **Zero network activity** for transcription
- All models run locally
- No cloud services used
- Complete offline operation

### Cleanup
To remove all generated data:
```bash
# Remove snapshots
rm -rf snapshots/

# Remove exports
rm -rf exports/

# Remove model cache (will re-download on next run)
rm -rf ~/.cache/whisper/
```

## FAQ

**Q: How do I change the language?**
A: Edit config.yaml, change `language: "en"` to your language code (es, fr, de, etc.)

**Q: Can multiple people speak simultaneously?**
A: The system works best with one speaker at a time. Simultaneous speech will be transcribed together.

**Q: How accurate is it?**
A: With the base model, typically 90-95% accurate in quiet environments. Use larger models for better accuracy.

**Q: Does it need internet?**
A: Only for initial setup (downloading models). After that, fully offline.

**Q: Can I edit transcripts in the app?**
A: Not currently. Export and edit in a text editor.

**Q: How much delay is there?**
A: Typically 2-4 seconds depending on buffer_duration and model size.

**Q: Can I use it for transcribing videos?**
A: Not directly. This is for real-time microphone input. For video files, use Whisper directly.

## Getting Help

1. Check the console/terminal for error messages
2. Run `python install.py` to verify setup
3. Review the troubleshooting section
4. Check the main README.md for technical details

## Next Steps

- Experiment with different model sizes
- Try different buffer durations
- Test with multiple speakers
- Explore configuration options
- Consider integrating with your workflow

---

**Happy Transcribing! 🎤**

