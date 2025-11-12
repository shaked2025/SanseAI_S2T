"""
Main Application
Integrates all components for real-time speech-to-text with speaker diarization
"""

import tkinter as tk
import yaml
import threading
import time
import numpy as np
from datetime import datetime
import os
import sys

# Import our modules
from audio_capture import AudioCapture, VoiceActivityDetector
from video_capture import VideoCapture
from speech_to_text import SpeechToText, TranscriptManager
from speaker_diarization import SimpleSpeakerDiarization, SpeakerManager
from gui_application import SpeechToTextGUI


class SpeechToTextApplication:
    """Main application class that integrates all components"""
    
    def __init__(self, config_file="config.yaml"):
        # Load configuration
        self.config = self.load_config(config_file)
        
        # Initialize components
        print("Initializing components...")
        
        # Audio
        audio_config = self.config.get('audio', {})
        self.audio_capture = AudioCapture(
            sample_rate=audio_config.get('sample_rate', 16000),
            channels=audio_config.get('channels', 1),
            chunk_size=audio_config.get('chunk_size', 1024)
        )
        
        # Voice activity detection
        processing_config = self.config.get('processing', {})
        self.vad = VoiceActivityDetector(
            sample_rate=audio_config.get('sample_rate', 16000),
            threshold=500,
            min_duration=0.3
        )
        
        # Video
        video_config = self.config.get('video', {})
        self.video_capture = VideoCapture(
            camera_index=video_config.get('camera_index', 0),
            width=video_config.get('width', 640),
            height=video_config.get('height', 480),
            fps=video_config.get('fps', 30)
        )
        
        # Speech recognition
        speech_config = self.config.get('speech', {})
        print(f"Loading speech recognition model (this may take a moment)...")
        self.speech_to_text = SpeechToText(
            model_size=speech_config.get('model_size', 'base'),
            language=speech_config.get('language', 'en')
        )
        
        # Speaker diarization
        diarization_config = self.config.get('diarization', {})
        self.diarization_enabled = diarization_config.get('enabled', True)
        if self.diarization_enabled:
            self.speaker_diarization = SimpleSpeakerDiarization(
                min_speakers=diarization_config.get('min_speakers', 1),
                max_speakers=diarization_config.get('max_speakers', 5)
            )
        else:
            self.speaker_diarization = None
            
        # Speaker manager
        self.speaker_manager = SpeakerManager()
        
        # Transcript manager
        self.transcript_manager = TranscriptManager(max_entries=100)
        
        # GUI
        self.root = tk.Tk()
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.gui = SpeechToTextGUI(self.root, self.config)
        self.gui.set_callbacks(
            start_cb=self.start_capture,
            stop_cb=self.stop_capture,
            snapshot_cb=self.take_snapshot,
            export_cb=self.export_transcript
        )
        
        # Force window to front
        self.root.lift()
        self.root.attributes('-topmost', True)
        self.root.after_idle(self.root.attributes, '-topmost', False)
        
        # State
        self.is_running = False
        self.processing_thread = None
        self.sample_rate = audio_config.get('sample_rate', 16000)
        self.buffer_duration = processing_config.get('buffer_duration', 3.0)
        
        print("Initialization complete!")
        
    def load_config(self, config_file):
        """Load configuration from YAML file"""
        try:
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)
            print(f"Configuration loaded from {config_file}")
            return config
        except FileNotFoundError:
            print(f"Warning: Config file {config_file} not found, using defaults")
            return {}
        except Exception as e:
            print(f"Error loading config: {e}")
            return {}
            
    def start_capture(self):
        """Start audio/video capture and processing"""
        if self.is_running:
            return
            
        try:
            print("\n" + "="*50)
            print("Starting capture...")
            print("="*50)
            
            # Start video capture
            print("Starting video capture...")
            if not self.video_capture.start():
                print("Warning: Video capture failed")
                self.gui.queue_update({'type': 'status', 'message': 'Warning: Camera not available'})
                # Continue anyway - audio still works
                
            # Start audio capture
            print("Starting audio capture...")
            self.audio_capture.start()
            
            # Update GUI state
            self.is_running = True
            self.gui.queue_update({'type': 'state', 'is_running': True})
            self.gui.queue_update({'type': 'status', 'message': 'Running - Listening for speech...'})
            
            # Start processing thread
            print("Starting processing thread...")
            self.processing_thread = threading.Thread(target=self.processing_loop, daemon=True)
            self.processing_thread.start()
            
            # Start video update thread
            print("Starting video thread...")
            self.video_thread = threading.Thread(target=self.video_update_loop, daemon=True)
            self.video_thread.start()
            
            # Start audio level update thread
            print("Starting audio level thread...")
            self.audio_level_thread = threading.Thread(target=self.audio_level_loop, daemon=True)
            self.audio_level_thread.start()
            
            print("✅ Capture started successfully!")
            print("🔴 LIVE - Continuous streaming mode activated")
            print("💬 Speak naturally - transcription will appear in real-time")
            print("Speak into your microphone to see transcription...")
            
        except Exception as e:
            print(f"Error starting capture: {e}")
            import traceback
            traceback.print_exc()
            self.gui.queue_update({'type': 'status', 'message': f'Error: {str(e)}'})
            self.is_running = False
        
    def stop_capture(self):
        """Stop audio/video capture"""
        if not self.is_running:
            return
            
        print("\nStopping capture...")
        
        self.is_running = False
        
        # Wait for threads to finish
        if self.processing_thread:
            self.processing_thread.join(timeout=3.0)
            
        # Stop captures
        self.audio_capture.stop()
        self.video_capture.stop()
        
        # Update GUI
        self.gui.queue_update({'type': 'state', 'is_running': False})
        self.gui.queue_update({'type': 'status', 'message': 'Stopped'})
        
        print("Capture stopped")
        
    def video_update_loop(self):
        """Update video display - optimized for smooth streaming"""
        try:
            frame_count = 0
            while self.is_running:
                try:
                    frame = self.video_capture.get_frame()
                    if frame is not None:
                        # Only update every 2nd frame to reduce GUI load (15fps instead of 30fps)
                        if frame_count % 2 == 0:
                            self.gui.queue_update({'type': 'video', 'frame': frame})
                        frame_count += 1
                except Exception as e:
                    print(f"Video update error: {e}")
                time.sleep(0.033)  # ~30 fps capture, but display at 15fps
        except Exception as e:
            print(f"Video loop error: {e}")
            
    def audio_level_loop(self):
        """Update audio level display - faster for live feel"""
        try:
            while self.is_running:
                try:
                    level = self.audio_capture.get_audio_level()
                    self.gui.queue_update({'type': 'audio_level', 'level': level})
                except Exception as e:
                    print(f"Audio level error: {e}")
                time.sleep(0.05)  # Update every 50ms for smoother visual feedback
        except Exception as e:
            print(f"Audio loop error: {e}")
            
    def processing_loop(self):
        """Main processing loop for speech recognition"""
        print("Processing loop started - Continuous streaming mode")
        
        last_process_time = time.time()
        process_interval = self.config.get('processing', {}).get('process_interval', 0.5)
        
        print(f"Processing interval: {process_interval}s for continuous streaming")
        
        while self.is_running:
            try:
                current_time = time.time()
                
                # Check if it's time to process (more frequent for streaming)
                if current_time - last_process_time < process_interval:
                    time.sleep(0.05)  # Shorter sleep for faster response
                    continue
                    
                # Get audio buffer
                audio_data = self.audio_capture.get_buffer(duration=self.buffer_duration)
                
                if len(audio_data) < self.sample_rate * 0.3:  # At least 0.3 seconds
                    time.sleep(0.05)
                    continue
                    
                # Check if speech is present
                if not self.vad.is_speech(audio_data):
                    time.sleep(0.05)
                    continue
                    
                print(f"🎤 Processing {len(audio_data)/self.sample_rate:.2f}s of audio...")
                
                # Process in a separate thread to avoid blocking
                # This prevents UI freezing
                def process_audio():
                    try:
                        # Identify speaker if diarization is enabled
                        speaker_id = 0
                        if self.diarization_enabled and self.speaker_diarization:
                            speaker_id = self.speaker_diarization.identify_speaker(audio_data, self.sample_rate)
                            print(f"👤 Speaker identified: Speaker {speaker_id + 1}")
                            
                        # Get speaker info
                        speaker_info = self.speaker_manager.get_speaker_info(speaker_id)
                        
                        # Update speakers display immediately (live)
                        all_speakers = self.speaker_manager.get_all_speakers()
                        self.gui.queue_update({'type': 'speakers', 'speakers': all_speakers})
                        
                        # Transcribe
                        result = self.speech_to_text.transcribe(audio_data, self.sample_rate)
                        
                        if result['text']:
                            print(f"📝 Transcript: [{speaker_info['name']}] {result['text']}")
                            
                            # Add to transcript manager
                            self.transcript_manager.add_transcript(
                                result['text'],
                                speaker_id=speaker_id,
                                timestamp=result['timestamp']
                            )
                            
                            # Update GUI immediately
                            self.gui.queue_update({
                                'type': 'transcript',
                                'text': result['text'],
                                'speaker_id': speaker_id,
                                'speaker_name': speaker_info['name'],
                                'color': speaker_info['color']
                            })
                            
                            # Update status to show it's live
                            self.gui.queue_update({
                                'type': 'status',
                                'message': f'🔴 LIVE - {speaker_info["name"]} speaking...'
                            })
                        else:
                            print("⚠️ No speech detected in audio segment")
                    except Exception as e:
                        print(f"❌ Error in audio processing: {e}")
                        import traceback
                        traceback.print_exc()
                
                # Run transcription in background to keep UI responsive
                transcribe_thread = threading.Thread(target=process_audio, daemon=True)
                transcribe_thread.start()
                    
                last_process_time = current_time
                
            except Exception as e:
                print(f"❌ Error in processing loop: {e}")
                import traceback
                traceback.print_exc()
                time.sleep(0.5)  # Shorter sleep to recover faster
                
        print("Processing loop ended")
        
    def take_snapshot(self):
        """Take a snapshot from video"""
        filename = self.video_capture.take_snapshot()
        if filename:
            self.gui.show_info("Snapshot", f"Snapshot saved to:\n{filename}")
            self.gui.queue_update({'type': 'status', 'message': f'Snapshot saved: {os.path.basename(filename)}'})
        else:
            self.gui.show_error("Error", "Failed to take snapshot")
            
    def export_transcript(self):
        """Export transcript to file"""
        try:
            # Create exports directory
            os.makedirs("exports", exist_ok=True)
            
            # Generate filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"exports/transcript_{timestamp}.txt"
            
            # Export
            self.transcript_manager.export_to_text(filename)
            
            self.gui.show_info("Export", f"Transcript exported to:\n{filename}")
            self.gui.queue_update({'type': 'status', 'message': f'Transcript exported: {os.path.basename(filename)}'})
            
        except Exception as e:
            self.gui.show_error("Error", f"Failed to export transcript:\n{str(e)}")
            
    def on_closing(self):
        """Handle window close event"""
        print("\nClosing application...")
        self.cleanup()
        self.root.destroy()
        
    def run(self):
        """Run the application"""
        print("\n" + "="*50)
        print("Speech-to-Text Application Ready")
        print("="*50)
        print("\nClick 'Start' to begin capturing and transcribing speech.")
        print("The system will automatically detect and separate multiple speakers.")
        print("\nPress 'Stop' to end the session.")
        print("="*50 + "\n")
        
        # Run GUI main loop
        try:
            print("Starting GUI main loop...")
            self.root.mainloop()
            print("GUI main loop ended")
        except KeyboardInterrupt:
            print("\nShutting down...")
        except Exception as e:
            print(f"Error in main loop: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.cleanup()
            
    def cleanup(self):
        """Cleanup resources"""
        print("\nCleaning up...")
        
        if self.is_running:
            self.stop_capture()
            
        self.audio_capture.cleanup()
        self.video_capture.cleanup()
        
        print("Cleanup complete")


def main():
    """Main entry point"""
    print("="*60)
    print(" "*10 + "Real-Time Speech-to-Text System")
    print(" "*15 + "with Speaker Diarization")
    print("="*60)
    print()
    print("This system runs completely locally - no third-party APIs!")
    print("Features:")
    print("  ✓ Real-time speech transcription")
    print("  ✓ Multiple speaker identification")
    print("  ✓ Video capture and snapshots")
    print("  ✓ Export transcripts")
    print()
    print("Loading components...")
    print()
    
    try:
        app = SpeechToTextApplication()
        app.run()
    except Exception as e:
        print(f"\nError starting application: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

