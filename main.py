"""
CLEAN Interview Transcription System
Requirements:
- Each speaker records 6 recordings of 5 seconds each
- Then live transcription with speaker identification
- Microphone only (device 5 - external mic)
- NO CAMERA
"""

import tkinter as tk
from tkinter import scrolledtext
import whisper
import numpy as np
from datetime import datetime
import threading
import time
from audio_capture import AudioCapture
from speaker_diarization_robust import ResemblyzerEmbeddings
from speaker_enrollment import SpeakerEnrollment, SpeakerVerificationEngine
from unknown_speaker_rejection import AdvancedSpeakerRejection, calculate_audio_quality, MultiMetricVerifier

# NO VIDEO/CAMERA IMPORTS - AUDIO ONLY!
# NO GUI_APPLICATION - Custom lightweight GUI only


class InterviewSystem:
    """Clean interview transcription system"""
    
    def __init__(self):
        print("Loading Interview System...")
        
        # Whisper model
        self.model = whisper.load_model("base")
        
        # Audio (device 5 = external microphone)
        self.audio = AudioCapture(sample_rate=16000, channels=1, device_index=5)
        
        # Speaker enrollment
        self.embedder = ResemblyzerEmbeddings()
        self.enrollment = SpeakerEnrollment(self.embedder)
        self.verifier = None
        
        # ADVANCED unknown speaker rejection
        print("Loading advanced unknown speaker rejection system...")
        self.rejector = AdvancedSpeakerRejection(nu=0.10)  # Moderate: 10% outliers expected (less strict for small sets)
        self.multi_metric = MultiMetricVerifier()
        print("✅ Unknown speaker rejection ready")
        
        self.is_running = False
        self.is_recording_enrollment = False
        self.current_speaker_enrolling = None
        self.enrollment_count = 0
        
        print("✅ Ready")
        
    def create_gui(self):
        """Create GUI"""
        self.root = tk.Tk()
        self.root.title("Interview Transcription - Microphone Only")
        self.root.geometry("900x700")
        
        # === ENROLLMENT SECTION ===
        enroll_frame = tk.LabelFrame(self.root, text="STEP 1: Enroll Speakers (6 recordings each)", 
                                     font=('Arial', 13, 'bold'), padx=15, pady=15)
        enroll_frame.pack(fill=tk.X, padx=20, pady=15)
        
        # Speaker 1
        s1_frame = tk.Frame(enroll_frame)
        s1_frame.pack(fill=tk.X, pady=8)
        
        tk.Label(s1_frame, text="Speaker 1:", font=('Arial', 12, 'bold'), width=12, anchor='e').pack(side=tk.LEFT, padx=5)
        self.name1_entry = tk.Entry(s1_frame, width=20, font=('Arial', 11))
        self.name1_entry.insert(0, "Interviewer")
        self.name1_entry.pack(side=tk.LEFT, padx=5)
        
        self.record1_btn = tk.Button(
            s1_frame,
            text="🔴 RECORD 6 SAMPLES (5s each)",
            command=lambda: self.enroll_speaker(0, self.name1_entry.get()),
            bg='#E74C3C',
            fg='white',
            font=('Arial', 12, 'bold'),
            padx=25,
            pady=12
        )
        self.record1_btn.pack(side=tk.LEFT, padx=10)
        
        self.status1_label = tk.Label(s1_frame, text="Not enrolled", font=('Arial', 10), fg='gray')
        self.status1_label.pack(side=tk.LEFT, padx=10)
        
        # Speaker 2
        s2_frame = tk.Frame(enroll_frame)
        s2_frame.pack(fill=tk.X, pady=8)
        
        tk.Label(s2_frame, text="Speaker 2:", font=('Arial', 12, 'bold'), width=12, anchor='e').pack(side=tk.LEFT, padx=5)
        self.name2_entry = tk.Entry(s2_frame, width=20, font=('Arial', 11))
        self.name2_entry.insert(0, "Interviewee")
        self.name2_entry.pack(side=tk.LEFT, padx=5)
        
        self.record2_btn = tk.Button(
            s2_frame,
            text="🔴 RECORD 6 SAMPLES (5s each)",
            command=lambda: self.enroll_speaker(1, self.name2_entry.get()),
            bg='#E74C3C',
            fg='white',
            font=('Arial', 12, 'bold'),
            padx=25,
            pady=12
        )
        self.record2_btn.pack(side=tk.LEFT, padx=10)
        
        self.status2_label = tk.Label(s2_frame, text="Not enrolled", font=('Arial', 10), fg='gray')
        self.status2_label.pack(side=tk.LEFT, padx=10)
        
        # Recording status
        self.enroll_status = tk.Label(
            enroll_frame,
            text="",
            font=('Arial', 14, 'bold'),
            fg='red'
        )
        self.enroll_status.pack(pady=15)
        
        # === TRANSCRIPT SECTION ===
        trans_frame = tk.LabelFrame(self.root, text="STEP 2: Live Interview Transcript", 
                                    font=('Arial', 13, 'bold'))
        trans_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=15)
        
        self.transcript = scrolledtext.ScrolledText(
            trans_frame,
            wrap=tk.WORD,
            font=('Consolas', 11),
            bg='#2C3E50',
            fg='white'
        )
        self.transcript.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # === CONTROL BUTTONS ===
        btn_frame = tk.Frame(self.root)
        btn_frame.pack(pady=15)
        
        self.start_btn = tk.Button(
            btn_frame,
            text="▶ START INTERVIEW",
            command=self.start_interview,
            bg='#27AE60',
            fg='white',
            font=('Arial', 16, 'bold'),
            padx=40,
            pady=18,
            state=tk.DISABLED
        )
        self.start_btn.pack(side=tk.LEFT, padx=8)
        
        self.stop_btn = tk.Button(
            btn_frame,
            text="⬛ STOP",
            command=self.stop_interview,
            bg='#E74C3C',
            fg='white',
            font=('Arial', 16, 'bold'),
            padx=40,
            pady=18,
            state=tk.DISABLED
        )
        self.stop_btn.pack(side=tk.LEFT, padx=8)
        
    def enroll_speaker(self, speaker_num, name):
        """Enroll a speaker with 6 recordings of 5 seconds each"""
        if not name:
            return
            
        if self.is_recording_enrollment:
            return
            
        print(f"\n{'='*60}")
        print(f"ENROLLING: {name}")
        print(f"Recording 6 samples × 5 seconds each")
        print(f"{'='*60}")
        
        # Disable all buttons
        self.record1_btn.config(state=tk.DISABLED)
        self.record2_btn.config(state=tk.DISABLED)
        self.name1_entry.config(state=tk.DISABLED)
        self.name2_entry.config(state=tk.DISABLED)
        
        self.is_recording_enrollment = True
        self.current_speaker_enrolling = speaker_num
        self.enrollment_count = 0
        
        # Start audio
        if not self.audio.is_recording:
            self.audio.start()
            time.sleep(1.0)
            
        # Start enrollment thread
        threading.Thread(target=self.enrollment_loop, args=(speaker_num, name), daemon=True).start()
        
    def enrollment_loop(self, speaker_num, name):
        """Record 6 samples sequentially"""
        speaker_key = f"speaker_{speaker_num}"
        samples = []
        
        # Start enrollment
        self.enrollment.start_enrollment(speaker_key, name, "Interviewer" if speaker_num == 0 else "Interviewee")
        
        # Record 6 samples
        for i in range(6):
            print(f"\nSample {i+1}/6 - SPEAK NOW!")
            
            # Update UI
            self.enroll_status.config(
                text=f"🔴 Sample {i+1}/6 for {name} - SPEAK NOW!",
                fg='red'
            )
            self.root.update()
            
            # Clear buffer
            self.audio.clear_queue()
            time.sleep(0.3)
            
            # Record for 5 seconds with countdown
            start = time.time()
            while (time.time() - start) < 5.0:
                remaining = 5.0 - (time.time() - start)
                self.enroll_status.config(
                    text=f"🔴 Sample {i+1}/6 - {remaining:.1f}s - SPEAK!"
                )
                self.root.update()
                time.sleep(0.1)
                
            # Get audio
            audio_data = self.audio.get_buffer(duration=5.5)
            
            if len(audio_data) > 16000:  # Got audio
                # Add to enrollment
                success, quality, msg = self.enrollment.add_enrollment_sample(speaker_key, audio_data, 16000)
                samples.append(audio_data)
                print(f"  ✅ Sample {i+1} recorded - {msg}")
                
                # Brief pause
                self.enroll_status.config(text=f"✅ Sample {i+1}/6 saved", fg='green')
                self.root.update()
                time.sleep(0.8)
            else:
                print(f"  ❌ Sample {i+1} failed - no audio")
                
        # Complete enrollment
        success, quality, msg = self.enrollment.complete_enrollment(speaker_key)
        
        print(f"\n✅ {name} enrollment complete!")
        print(f"   Quality: {quality:.1%}")
        
        # Update UI
        status_label = self.status1_label if speaker_num == 0 else self.status2_label
        status_label.config(text=f"✅ Enrolled ({quality:.0%} quality)", fg='green', font=('Arial', 10, 'bold'))
        
        self.enroll_status.config(
            text=f"✅ {name} enrolled! Ready for interview or enroll more speakers.",
            fg='green'
        )
        
        # Re-enable buttons
        self.record1_btn.config(state=tk.NORMAL)
        self.record2_btn.config(state=tk.NORMAL)
        self.name1_entry.config(state=tk.NORMAL)
        self.name2_entry.config(state=tk.NORMAL)
        
        # Enable start interview if we have enrolled speakers
        enrolled = self.enrollment.get_enrolled_speakers()
        if len(enrolled) >= 1:
            self.start_btn.config(state=tk.NORMAL)
            
        self.is_recording_enrollment = False
        
    def start_interview(self):
        """Start live interview transcription"""
        if self.is_running:
            return
            
        # Create verifier
        self.verifier = SpeakerVerificationEngine(self.enrollment)
        
        # TRAIN unknown speaker rejection model
        print("\n" + "="*60)
        print("TRAINING UNKNOWN SPEAKER REJECTION")
        print("="*60)
        
        success = self.rejector.fit_enrolled_speakers(self.enrollment)
        
        if not success:
            print("⚠️ Warning: Rejection model not trained - will use basic filtering")
        
        print("\n" + "="*60)
        print("STARTING LIVE INTERVIEW")
        print("="*60)
        print("🛡️ Advanced unknown speaker rejection: ACTIVE")
        print("   Only enrolled speakers will be transcribed")
        print("="*60)
        
        self.is_running = True
        self.start_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        self.record1_btn.config(state=tk.DISABLED)
        self.record2_btn.config(state=tk.DISABLED)
        
        # Start audio
        if not self.audio.is_recording:
            self.audio.start()
            
        # Start transcription thread
        threading.Thread(target=self.transcribe_loop, daemon=True).start()
        
    def stop_interview(self):
        """Stop interview"""
        self.is_running = False
        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        print("Interview stopped")
        
    def transcribe_loop(self):
        """Real-time transcription with speaker ID"""
        print("🎙️ Transcription loop started")
        print("   Listening for speech...")
        
        last_time = time.time()
        
        while self.is_running:
            try:
                if time.time() - last_time < 1.5:  # Faster processing
                    time.sleep(0.1)
                    continue
                    
                # Get audio
                audio_data = self.audio.get_buffer(duration=2.5)
                
                if len(audio_data) < 16000:
                    time.sleep(0.1)
                    continue
                    
                # Check if speech
                rms = np.sqrt(np.mean(audio_data.astype(np.float32) ** 2))
                
                print(f"   Audio level: {int(rms)}")  # DEBUG
                
                if rms < 800:  # Lower threshold
                    time.sleep(0.1)
                    continue
                    
                print(f"🎤 Processing speech (level: {int(rms)})...")
                
                # Calculate audio quality
                audio_quality = calculate_audio_quality(audio_data, 16000)
                print(f"   Audio quality: {audio_quality:.2f}")
                
                # Identify speaker with basic verifier
                speaker_key, speaker_name, raw_confidence, metadata = self.verifier.verify_speaker(audio_data, 16000, use_context=False)
                
                # Get enrolled profile
                enrolled = self.enrollment.get_enrolled_speakers()
                if speaker_key not in enrolled:
                    print(f"🚫 REJECTED: Speaker not in enrolled set")
                    last_time = time.time()
                    continue
                    
                enrolled_profile = enrolled[speaker_key]
                
                # Extract test embedding
                test_embedding = self.embedder.extract_embedding(audio_data, 16000)
                
                # DEBUG: Check if embedding is valid
                if np.allclose(test_embedding, 0):
                    print("⚠️ WARNING: Zero embedding extracted - skipping")
                    last_time = time.time()
                    continue
                    
                # DEBUG: Check similarity with ALL enrolled speakers directly
                print(f"\n   Direct similarity check:")
                for check_key, check_profile in enrolled.items():
                    direct_sim = np.dot(test_embedding, check_profile['mean_embedding'])
                    print(f"      vs {check_profile['name']}: {direct_sim:.3f}")
                
                # ADVANCED REJECTION CHECK
                accept, final_score, reason, details = self.rejector.verify_and_reject(
                    test_embedding,
                    speaker_key,
                    raw_confidence,
                    enrolled_profile,
                    audio_quality
                )
                
                if not accept:
                    # REJECTED as unknown/impostor speaker
                    print(f"🚫 REJECTED: {reason}")
                    print(f"   Scores: cosine={details.get('cosine', 0):.3f}, "
                          f"fused={details.get('fused_score', 0):.3f}, "
                          f"z-score={details.get('z_score', 0):.2f}")
                    print(f"   Votes: {details.get('votes', {})}")
                    last_time = time.time()
                    continue
                    
                # ACCEPTED - proceed with transcription
                print(f"✅ ACCEPTED: {speaker_name} (score: {final_score:.3f}, quality: {audio_quality:.2f})")
                print(f"   Z-score: {details.get('z_score', 0):.2f}, SVM: {details.get('ocsvm_inlier', False)}")
                
                # Transcribe
                audio_float = audio_data.astype(np.float32) / 32768.0
                result = self.model.transcribe(audio_float, language='en', fp16=False, verbose=False)
                
                if result['text'].strip():
                    timestamp = datetime.now().strftime("%H:%M:%S")
                    text = f"[{timestamp}] {speaker_name}: {result['text'].strip()}\n\n"
                    
                    self.transcript.insert(tk.END, text)
                    self.transcript.see(tk.END)
                    self.root.update()  # Force GUI update
                    
                    print(f"📝 [{timestamp}] {speaker_name}: {result['text'].strip()}")
                else:
                    print("   (No speech in transcript)")
                    
                last_time = time.time()
                
            except Exception as e:
                print(f"❌ Error: {e}")
                import traceback
                traceback.print_exc()
                time.sleep(1)
                
    def run(self):
        """Run application"""
        self.create_gui()
        self.root.mainloop()
        self.audio.cleanup()


if __name__ == "__main__":
    print("="*60)
    print("  INTERVIEW TRANSCRIPTION SYSTEM")
    print("  Microphone Only - No Camera")
    print("="*60)
    print()
    
    app = InterviewSystem()
    app.run()

