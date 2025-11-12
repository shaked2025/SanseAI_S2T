"""
Component Test Script
Tests individual components of the speech-to-text system
"""

import sys
import numpy as np


def test_audio_capture():
    """Test audio capture module"""
    print("\n" + "="*60)
    print("Testing Audio Capture...")
    print("="*60)
    
    try:
        from audio_capture import AudioCapture, VoiceActivityDetector
        
        # Test initialization
        audio = AudioCapture(sample_rate=16000, channels=1, chunk_size=1024)
        print("✓ AudioCapture initialized")
        
        # Test VAD
        vad = VoiceActivityDetector(sample_rate=16000)
        print("✓ VoiceActivityDetector initialized")
        
        # Test with dummy data
        dummy_audio = np.random.randint(-1000, 1000, 16000, dtype=np.int16)
        is_speech = vad.is_speech(dummy_audio)
        print(f"✓ VAD test: is_speech = {is_speech}")
        
        audio.cleanup()
        return True
        
    except Exception as e:
        print(f"❌ Audio capture test failed: {e}")
        return False


def test_video_capture():
    """Test video capture module"""
    print("\n" + "="*60)
    print("Testing Video Capture...")
    print("="*60)
    
    try:
        from video_capture import VideoCapture
        
        # Test initialization
        video = VideoCapture(camera_index=0, width=640, height=480)
        print("✓ VideoCapture initialized")
        
        # Try to start (may fail if no camera)
        if video.start():
            print("✓ Camera opened successfully")
            
            # Try to get a frame
            import time
            time.sleep(0.5)
            frame = video.get_frame()
            
            if frame is not None:
                print(f"✓ Frame captured: {frame.shape}")
            else:
                print("⚠ Camera opened but no frame captured")
                
            video.stop()
        else:
            print("⚠ No camera available (this is OK if you don't have one)")
            
        video.cleanup()
        return True
        
    except Exception as e:
        print(f"❌ Video capture test failed: {e}")
        return False


def test_speech_to_text():
    """Test speech-to-text module"""
    print("\n" + "="*60)
    print("Testing Speech-to-Text...")
    print("="*60)
    
    try:
        from speech_to_text import SpeechToText, TranscriptManager
        
        # Test transcript manager
        manager = TranscriptManager(max_entries=10)
        manager.add_transcript("Test transcript", speaker_id=0)
        print("✓ TranscriptManager working")
        
        # Test Whisper loading (this will take time on first run)
        print("\nLoading Whisper model (this may take a moment)...")
        stt = SpeechToText(model_size="tiny", language="en")
        print("✓ Whisper model loaded")
        
        # Test with dummy audio
        dummy_audio = np.zeros(16000, dtype=np.int16)  # 1 second of silence
        result = stt.transcribe(dummy_audio, sample_rate=16000)
        print(f"✓ Transcription test completed (empty audio): '{result['text']}'")
        
        return True
        
    except Exception as e:
        print(f"❌ Speech-to-text test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_speaker_diarization():
    """Test speaker diarization module"""
    print("\n" + "="*60)
    print("Testing Speaker Diarization...")
    print("="*60)
    
    try:
        from speaker_diarization import SimpleSpeakerDiarization, SpeakerManager
        
        # Test speaker manager
        manager = SpeakerManager()
        info = manager.get_speaker_info(0)
        print(f"✓ SpeakerManager working: {info['name']}")
        
        # Test diarization
        diarization = SimpleSpeakerDiarization(min_speakers=1, max_speakers=5)
        print("✓ SimpleSpeakerDiarization initialized")
        
        # Test with dummy audio
        dummy_audio = np.random.randint(-5000, 5000, 16000, dtype=np.int16)
        features = diarization.extract_features(dummy_audio, sample_rate=16000)
        print(f"✓ Feature extraction: {len(features)} features")
        
        speaker_id = diarization.identify_speaker(dummy_audio, sample_rate=16000)
        print(f"✓ Speaker identification: Speaker {speaker_id}")
        
        return True
        
    except Exception as e:
        print(f"❌ Speaker diarization test failed: {e}")
        return False


def test_gui():
    """Test GUI components (import only, don't start)"""
    print("\n" + "="*60)
    print("Testing GUI Components...")
    print("="*60)
    
    try:
        import tkinter as tk
        print("✓ Tkinter available")
        
        from gui_application import SpeechToTextGUI
        print("✓ GUI module imported successfully")
        
        # Don't actually create the GUI, just test import
        return True
        
    except Exception as e:
        print(f"❌ GUI test failed: {e}")
        return False


def test_config():
    """Test configuration loading"""
    print("\n" + "="*60)
    print("Testing Configuration...")
    print("="*60)
    
    try:
        import yaml
        
        with open('config.yaml', 'r') as f:
            config = yaml.safe_load(f)
            
        print("✓ config.yaml loaded")
        print(f"  Audio sample rate: {config['audio']['sample_rate']}")
        print(f"  Video resolution: {config['video']['width']}x{config['video']['height']}")
        print(f"  Speech model: {config['speech']['model_size']}")
        print(f"  Diarization: {'enabled' if config['diarization']['enabled'] else 'disabled'}")
        
        return True
        
    except Exception as e:
        print(f"❌ Config test failed: {e}")
        return False


def main():
    """Run all tests"""
    print("="*60)
    print(" "*15 + "Component Test Suite")
    print(" "*10 + "Speech-to-Text System")
    print("="*60)
    
    tests = [
        ("Configuration", test_config),
        ("Audio Capture", test_audio_capture),
        ("Video Capture", test_video_capture),
        ("Speaker Diarization", test_speaker_diarization),
        ("GUI Components", test_gui),
        ("Speech-to-Text", test_speech_to_text),  # This one last as it downloads model
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"\n❌ {test_name} test crashed: {e}")
            results[test_name] = False
    
    # Summary
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for test_name, result in results.items():
        status = "✓ PASS" if result else "❌ FAIL"
        print(f"{test_name:30} {status}")
    
    print("="*60)
    print(f"Results: {passed}/{total} tests passed")
    print("="*60)
    
    if passed == total:
        print("\n✓ All tests passed! The system is ready to use.")
        print("Run 'python main.py' to start the application.")
        return 0
    else:
        print("\n⚠ Some tests failed. Please check the errors above.")
        print("You may need to:")
        print("  1. Install missing dependencies: pip install -r requirements.txt")
        print("  2. Check camera/microphone permissions")
        print("  3. Ensure compatible Python version (3.8+)")
        return 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\nTests interrupted by user.")
        sys.exit(1)

