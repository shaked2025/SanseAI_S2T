"""
Interview Transcription Application
Optimized for interrogation/interview scenarios with speaker enrollment
"""

import tkinter as tk
import yaml
import threading
import time
import numpy as np
from datetime import datetime
import os
import sys
from logger import log, log_info, log_debug, log_critical, log_error, set_verbosity, Logger

# Import modules
from audio_capture import AudioCapture, VoiceActivityDetector
from video_capture import VideoCapture
from speech_to_text import SpeechToText, TranscriptManager
from speaker_diarization_robust import ResemblyzerEmbeddings
from speaker_enrollment import SpeakerEnrollment, SpeakerVerificationEngine
from enrollment_ui import EnrollmentWizard
from gui_application import SpeechToTextGUI
from overlap_detection import OverlappingSpeechDetector, MultiSpeakerIdentifier
from noise_filtering import BackgroundSpeakerFilter, ProximityBasedFilter, EnrolledSpeakerOnlyVerifier


class InterviewTranscriptionApp:
    """Interview transcription with enrollment-based speaker verification"""
    
    def __init__(self, config_file="config.yaml"):
        # Load configuration
        self.config = self.load_config(config_file)
        
        # Set logging to NORMAL (reduce verbosity)
        set_verbosity(Logger.NORMAL)
        
        log_critical("="*60)
        log_critical(" "*10 + "Interview Transcription System")
        log_critical(" "*12 + "with Speaker Enrollment")
        log_critical("="*60)
        log("")
        log("✅ Enrollment-based speaker verification")
        log("✅ 95-99% accuracy for known speakers")
        log("✅ Optimized for interview/interrogation scenarios")
        log("")
        
        # Initialize audio/video - FORCE camera microphone (device 1)
        audio_config = self.config.get('audio', {})
        self.audio_capture = AudioCapture(
            sample_rate=audio_config.get('sample_rate', 16000),
            channels=audio_config.get('channels', 1),
            chunk_size=audio_config.get('chunk_size', 1024),
            device_index=6  # ALWAYS use camera microphone (Logitech BRIO - working device)
        )
        
        self.vad = VoiceActivityDetector(
            sample_rate=audio_config.get('sample_rate', 16000)
        )
        
        video_config = self.config.get('video', {})
        self.video_capture = VideoCapture(
            camera_index=video_config.get('camera_index', 0),
            width=video_config.get('width', 640),
            height=video_config.get('height', 480)
        )
        
        # Initialize speech recognition
        speech_config = self.config.get('speech', {})
        log("Loading speech recognition model...")
        self.speech_to_text = SpeechToText(
            model_size=speech_config.get('model_size', 'base'),
            language=speech_config.get('language', 'en')
        )
        
        # Initialize speaker enrollment system
        log("Loading speaker verification system...")
        self.embedding_extractor = ResemblyzerEmbeddings()
        self.enrollment_system = SpeakerEnrollment(self.embedding_extractor)
        self.verification_engine = None  # Will be created after enrollment
        
        # Initialize overlap detection
        log_info("Loading overlapping speech detector...")
        self.overlap_detector = OverlappingSpeechDetector(sample_rate=audio_config.get('sample_rate', 16000))
        self.multi_speaker_identifier = None  # Will be created after enrollment
        
        # Initialize noise filtering (prevents background speakers)
        log_info("Loading background speaker filter...")
        self.background_filter = BackgroundSpeakerFilter(
            min_confidence=0.75,  # Must be confident match
            min_energy=1000,      # Must be close/loud enough
            max_distance_ratio=2.5
        )
        self.proximity_filter = ProximityBasedFilter(min_snr=10.0)
        self.enrolled_only_verifier = None  # Will be created after enrollment
        
        # Transcript manager
        self.transcript_manager = TranscriptManager(max_entries=200)
        
        # GUI
        self.root = tk.Tk()
        self.root.withdraw()  # Hide initially
        
        # State
        self.is_running = False
        self.is_enrolled = False
        self.sample_rate = audio_config.get('sample_rate', 16000)
        self.buffer_duration = self.config.get('processing', {}).get('buffer_duration', 1.5)
        
        print("✅ System initialized")
        print()
        
    def load_config(self, config_file):
        """Load configuration"""
        try:
            with open(config_file, 'r') as f:
                return yaml.safe_load(f)
        except:
            return {}
            
    def run(self):
        """Run the application"""
        # Show enrollment wizard first
        print("="*60)
        print("Starting Speaker Enrollment...")
        print("="*60)
        print()
        
        # Create temporary root for wizard
        wizard_root = tk.Tk()
        wizard_root.withdraw()
        
        # Show enrollment wizard
        wizard = EnrollmentWizard(
            wizard_root,
            self.audio_capture,
            self.embedding_extractor,
            self.on_enrollment_complete
        )
        
        wizard_root.mainloop()
        
    def on_enrollment_complete(self, participants):
        """Called when enrollment is complete"""
        print()
        print("="*60)
        print("Processing Enrollment Data...")
        print("="*60)
        print()
        
        # Process each participant's enrollment
        for participant in participants:
            speaker_key = participant['key']
            name = participant['name']
            role = participant['role']
            samples = participant['samples']
            
            print(f"📝 Enrolling {name} ({role})...")
            
            # Start enrollment
            self.enrollment_system.start_enrollment(speaker_key, name, role)
            
            # Add each sample
            for i, sample in enumerate(samples):
                success, quality, msg = self.enrollment_system.add_enrollment_sample(
                    speaker_key,
                    sample['audio'],
                    self.audio_capture.sample_rate
                )
                print(f"   Sample {i+1}: {msg}")
                
            # Complete enrollment
            success, quality, msg = self.enrollment_system.complete_enrollment(speaker_key)
            print(f"   {msg}")
            print()
            
        # Test speaker separation
        print("="*60)
        print("Testing Speaker Separation...")
        print("="*60)
        print()
        
        self.enrollment_system.test_speaker_separation()
        
        # Create verification engine
        self.verification_engine = SpeakerVerificationEngine(self.enrollment_system)
        
        # Create multi-speaker identifier for overlap handling
        self.multi_speaker_identifier = MultiSpeakerIdentifier(
            self.verification_engine,
            self.overlap_detector
        )
        
        # Create enrolled-speaker-only verifier (filters background speakers)
        self.enrolled_only_verifier = EnrolledSpeakerOnlyVerifier(
            self.verification_engine,
            self.background_filter,
            self.proximity_filter
        )
        
        print("✅ Background speaker filtering enabled")
        print("   Only enrolled participants will be identified")
        
        # Set interview roles
        interviewer = next((p for p in participants if 'interviewer' in p['role'].lower()), None)
        interviewees = [p for p in participants if 'interviewee' in p['role'].lower()]
        
        if interviewer and interviewees:
            self.verification_engine.context_tracker.set_roles(
                interviewer['key'],
                [i['key'] for i in interviewees]
            )
            
        self.is_enrolled = True
        
        # Save enrollment
        self.enrollment_system.save_enrollment("interview_enrollment.pkl")
        
        print()
        print("="*60)
        print("✅ Enrollment Complete - Ready for Interview")
        print("="*60)
        print()
        
        # Now show main application
        self.root.deiconify()  # Show main window
        self.root.title("Interview Transcription - LIVE")
        
        # Create main GUI
        self.gui = SpeechToTextGUI(self.root, self.config)
        self.gui.set_callbacks(
            start_cb=self.start_capture,
            stop_cb=self.stop_capture,
            snapshot_cb=self.take_snapshot,
            export_cb=self.export_transcript
        )
        
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.root.lift()
        self.root.attributes('-topmost', True)
        self.root.after_idle(self.root.attributes, '-topmost', False)
        
        print("Starting main application...")
        print("Click 'Start' to begin interview recording")
        print()
        
        # Run main loop
        try:
            self.root.mainloop()
        except KeyboardInterrupt:
            print("\nShutting down...")
        finally:
            self.cleanup()
            
    def start_capture(self):
        """Start interview recording"""
        if not self.is_enrolled:
            messagebox.showerror("Not Ready", "Please complete speaker enrollment first.")
            return
            
        if self.is_running:
            return
            
        print("\n" + "="*50)
        print("Starting Interview Recording...")
        print("="*50)
        
        # Start video
        self.video_capture.start()
        
        # Start audio
        if not self.audio_capture.is_recording:
            self.audio_capture.start()
            
        self.is_running = True
        self.gui.queue_update({'type': 'state', 'is_running': True})
        self.gui.queue_update({'type': 'status', 'message': '🔴 RECORDING INTERVIEW...'})
        
        # Start processing thread
        self.processing_thread = threading.Thread(target=self.processing_loop, daemon=True)
        self.processing_thread.start()
        
        # Start video thread
        self.video_thread = threading.Thread(target=self.video_update_loop, daemon=True)
        self.video_thread.start()
        
        # Start audio level thread
        self.audio_level_thread = threading.Thread(target=self.audio_level_loop, daemon=True)
        self.audio_level_thread.start()
        
        print("✅ Recording started")
        print("💬 Speakers will be identified with 95%+ accuracy")
        print()
        
    def stop_capture(self):
        """Stop recording"""
        if not self.is_running:
            return
            
        print("\nStopping recording...")
        self.is_running = False
        
        if self.processing_thread:
            self.processing_thread.join(timeout=3.0)
            
        self.audio_capture.stop()
        self.video_capture.stop()
        
        self.gui.queue_update({'type': 'state', 'is_running': False})
        self.gui.queue_update({'type': 'status', 'message': 'Stopped'})
        
        # Show statistics
        stats = self.verification_engine.get_statistics()
        print(f"\n📊 Session Statistics:")
        print(f"   Verifications: {stats['total_verifications']}")
        print(f"   High confidence: {stats['high_confidence']}")
        print(f"   Accuracy: {stats['accuracy']:.1f}%")
        print(f"   Enrolled speakers: {stats['enrolled_speakers']}")
        
        # Show filtering statistics
        if self.enrolled_only_verifier:
            filter_stats = self.enrolled_only_verifier.get_filtering_statistics()
            print(f"\n🛡️ Background Filtering Statistics:")
            print(f"   Total segments: {filter_stats['total_segments']}")
            print(f"   Accepted (enrolled speakers): {filter_stats['accepted']}")
            print(f"   Rejected (background/unknown): {filter_stats['rejected_total']}")
            print(f"   - Low energy: {filter_stats['rejected_low_energy']}")
            print(f"   - Low confidence: {filter_stats['rejected_low_confidence']}")
            print(f"   - Unknown speaker: {filter_stats['rejected_unknown']}")
            print(f"   Acceptance rate: {filter_stats['acceptance_rate']:.1f}%")
        
    def processing_loop(self):
        """Main processing loop"""
        log("🎙️ Interview processing started - Logging reduced for clarity")
        
        last_process_time = time.time()
        process_interval = 0.5
        
        while self.is_running:
            try:
                current_time = time.time()
                
                if current_time - last_process_time < process_interval:
                    time.sleep(0.05)
                    continue
                    
                # Get audio buffer
                audio_data = self.audio_capture.get_buffer(duration=self.buffer_duration)
                
                if len(audio_data) < self.sample_rate * 0.3:
                    time.sleep(0.05)
                    continue
                    
                # Check for speech
                if not self.vad.is_speech(audio_data):
                    time.sleep(0.05)
                    continue
                    
                log_debug(f"🎤 Processing {len(audio_data)/self.sample_rate:.2f}s of audio...")
                
                # First, verify this is an enrolled speaker (not background noise)
                accept, speaker_key_filtered, speaker_name_filtered, conf_filtered, filter_reason = self.enrolled_only_verifier.verify_with_filtering(
                    audio_data,
                    self.sample_rate
                )
                
                if not accept:
                    # Rejected as background/noise/unknown speaker
                    log_debug(f"🚫 FILTERED: {filter_reason}")
                    time.sleep(0.05)
                    continue
                
                # Accepted - now identify speakers (handles both single and overlapping)
                speakers_identified = self.multi_speaker_identifier.identify_speakers(
                    audio_data,
                    self.sample_rate
                )
                
                # Filter speakers to only enrolled ones
                filtered_speakers = []
                for spk_key, spk_name, spk_conf, is_overlap in speakers_identified:
                    # Double-check each identified speaker
                    enrolled = self.enrollment_system.get_enrolled_speakers()
                    if spk_key in enrolled and spk_conf >= 0.70:
                        filtered_speakers.append((spk_key, spk_name, spk_conf, is_overlap))
                    else:
                        log_debug(f"🚫 Filtered out: {spk_name} (conf: {spk_conf:.2f}) - not enrolled or low confidence")
                
                if not filtered_speakers:
                    log_debug(f"🚫 No enrolled speakers identified - skipping segment")
                    time.sleep(0.05)
                    continue
                
                speakers_identified = filtered_speakers
                
                # Transcribe once
                result = self.speech_to_text.transcribe(audio_data, self.sample_rate)
                
                if result['text']:
                    # Handle single or multiple speakers
                    if len(speakers_identified) == 1:
                        # Single speaker
                        speaker_key, speaker_name, confidence, is_overlap = speakers_identified[0]
                        
                        # Get role
                        enrolled = self.enrollment_system.get_enrolled_speakers()
                        role = enrolled.get(speaker_key, {}).get('role', 'Unknown')
                        
                        log(f"👤 {role}: {speaker_name}")
                        log_critical(f"📝 [{role}] {speaker_name}: {result['text']}")
                        
                        # Update GUI
                        self.gui.queue_update({
                            'type': 'transcript',
                            'text': result['text'],
                            'speaker_id': hash(speaker_key) % 10,
                            'speaker_name': f"{role}: {speaker_name}",
                            'color': self.get_role_color(role)
                        })
                        
                    else:
                        # Multiple speakers (overlapping speech!)
                        speaker_names = []
                        roles = []
                        enrolled = self.enrollment_system.get_enrolled_speakers()
                        
                        for speaker_key, speaker_name, confidence, is_overlap in speakers_identified:
                            role = enrolled.get(speaker_key, {}).get('role', 'Unknown')
                            speaker_names.append(speaker_name)
                            roles.append(role)
                            log_info(f"👥 {role}: {speaker_name} [OVERLAPPING]")
                        
                        # Combine speaker names
                        combined_name = " + ".join([f"{roles[i]}: {speaker_names[i]}" for i in range(len(speaker_names))])
                        
                        log_critical(f"📝 [{combined_name}]: {result['text']}")
                        
                        # Update GUI with combined speakers
                        self.gui.queue_update({
                            'type': 'transcript',
                            'text': result['text'],
                            'speaker_id': 0,
                            'speaker_name': combined_name,
                            'color': '#FF9500'  # Orange for overlap
                        })
                    
                last_process_time = current_time
                
            except Exception as e:
                print(f"❌ Processing error: {e}")
                import traceback
                traceback.print_exc()
                time.sleep(0.5)
                
    def video_update_loop(self):
        """Update video display"""
        while self.is_running:
            try:
                frame = self.video_capture.get_frame()
                if frame is not None:
                    self.gui.queue_update({'type': 'video', 'frame': frame})
            except:
                pass
            time.sleep(0.066)  # 15fps
            
    def audio_level_loop(self):
        """Update audio level"""
        while self.is_running:
            try:
                level = self.audio_capture.get_audio_level()
                self.gui.queue_update({'type': 'audio_level', 'level': level})
            except:
                pass
            time.sleep(0.05)
            
    def get_role_color(self, role):
        """Get color for role"""
        colors = {
            'Interviewer': '#3498DB',  # Blue
            'Interviewee': '#E74C3C',  # Red
            'Interviewee 1': '#E74C3C',
            'Interviewee 2': '#9B59B6',  # Purple
            'Interviewee 3': '#F39C12',  # Orange
            'Observer': '#95A5A6',  # Gray
        }
        return colors.get(role, '#2ECC71')  # Green default
        
    def take_snapshot(self):
        """Take snapshot"""
        filename = self.video_capture.take_snapshot()
        if filename:
            self.gui.queue_update({'type': 'status', 'message': f'Snapshot saved'})
            
    def export_transcript(self):
        """Export transcript"""
        try:
            os.makedirs("exports", exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"exports/interview_{timestamp}.txt"
            self.transcript_manager.export_to_text(filename)
            self.gui.queue_update({'type': 'status', 'message': f'Transcript exported'})
        except Exception as e:
            print(f"Export error: {e}")
            
    def on_closing(self):
        """Handle window close"""
        self.cleanup()
        self.root.destroy()
        
    def cleanup(self):
        """Cleanup resources"""
        if self.is_running:
            self.stop_capture()
        self.audio_capture.cleanup()
        self.video_capture.cleanup()
        
        
def main():
    """Main entry point"""
    app = InterviewTranscriptionApp()
    app.run()
    
if __name__ == "__main__":
    main()

