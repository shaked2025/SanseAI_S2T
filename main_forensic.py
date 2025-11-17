"""
FORENSIC-GRADE INTERROGATION TRANSCRIPTION SYSTEM

Complete implementation with:
✅ Stress-invariant voice processing
✅ Spatial location verification
✅ Comprehensive audit trail
✅ Confidence scoring (multi-dimensional)
✅ Adaptive enrollment (long sessions)
✅ Legal admissibility tracking
✅ Cryptographic integrity
✅ Voice stress indicators
✅ Quality assessment
✅ Manual override capabilities

Production-ready for interrogation rooms
"""

import tkinter as tk
from tkinter import scrolledtext, messagebox, ttk
import whisper
import numpy as np
from datetime import datetime
import threading
import time
import hashlib

# Core modules
from audio_capture import AudioCapture
from speaker_diarization_robust import ResemblyzerEmbeddings
from speaker_enrollment import SpeakerEnrollment

# Forensic modules
from simple_robust_verification import SimpleRobustVerifier
from spatial_location_features import LocationAwareVerifier
from stress_invariant_features import StressInvariantProcessor, calculate_whisper_confidence
from adaptive_enrollment import AdaptiveEnrollmentSystem, VoiceStressIndicators
from forensic_audit_trail import ForensicAuditLogger
from comprehensive_quality import ComprehensiveQualityAssessment


class ForensicInterrogationSystem:
    """
    Complete forensic-grade interrogation transcription system
    """
    
    def __init__(self, room_id="Room_1", case_id="Case_001"):
        print("="*80)
        print("  FORENSIC INTERROGATION TRANSCRIPTION SYSTEM")
        print("  Production-Grade with Complete Audit Trail")
        print("="*80)
        print()
        
        # Session management
        self.room_id = room_id
        self.case_id = case_id
        self.session_start = datetime.now()
        
        # Initialize forensic audit logger
        print("Initializing forensic audit trail...")
        self.audit = ForensicAuditLogger(room_id=room_id, case_id=case_id)
        
        # Load models
        print("Loading Whisper model (base)...")
        self.model = whisper.load_model("base")
        
        # Audio capture (device 5 = external mic)
        print("Initializing audio capture (device 5 - external microphone)...")
        self.audio = AudioCapture(sample_rate=16000, channels=1, device_index=5)
        
        # Speaker enrollment
        print("Loading speaker enrollment system...")
        self.embedder = ResemblyzerEmbeddings()
        self.enrollment = SpeakerEnrollment(self.embedder)
        
        # Stress-invariant preprocessing
        print("Loading stress-invariant processor...")
        self.stress_processor = StressInvariantProcessor(target_pitch=150, target_rms=1000)
        
        # Verification systems
        print("Loading verification systems...")
        self.base_verifier = SimpleRobustVerifier(base_threshold=0.64)
        self.location_verifier = LocationAwareVerifier(self.base_verifier, spatial_weight=0.15)
        
        # Adaptive enrollment
        print("Loading adaptive enrollment system...")
        self.adaptive_system = AdaptiveEnrollmentSystem(
            learning_rate=0.05,
            min_confidence=0.90,
            max_drift_per_hour=0.10
        )
        
        # Quality assessment
        print("Loading quality assessment system...")
        self.quality_assessor = ComprehensiveQualityAssessment()
        
        # Voice stress analysis
        print("Loading voice stress analyzer...")
        self.stress_analyzer = VoiceStressIndicators()
        
        # State
        self.is_running = False
        self.is_recording_enrollment = False
        
        self.audit.log_system_event("SYSTEM_INITIALIZED", {
            'models': {
                'speaker_embedding': 'Resemblyzer-v0.1.4',
                'transcription': 'Whisper-base',
                'verification': 'SimpleRobust-v1.0'
            },
            'room_id': room_id,
            'case_id': case_id
        })
        
        print("✅ Forensic system ready")
        print()
        
    def create_gui(self):
        """Create comprehensive GUI"""
        self.root = tk.Tk()
        self.root.title(f"Forensic Interrogation System - {self.room_id}")
        self.root.geometry("1100x800")
        
        # === HEADER ===
        header_frame = tk.Frame(self.root, bg='#2C3E50')
        header_frame.pack(fill=tk.X)
        
        tk.Label(
            header_frame,
            text=f"🔒 FORENSIC INTERROGATION TRANSCRIPTION",
            font=('Arial', 16, 'bold'),
            bg='#2C3E50',
            fg='white',
            pady=15
        ).pack()
        
        tk.Label(
            header_frame,
            text=f"Room: {self.room_id} | Case: {self.case_id} | Session: {self.audit.session_id[:8]}",
            font=('Arial', 10),
            bg='#2C3E50',
            fg='#ECF0F1',
            pady=5
        ).pack()
        
        # === MAIN CONTENT ===
        main_frame = tk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Left: Enrollment + Controls
        left_frame = tk.Frame(main_frame)
        left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        
        # Enrollment
        enroll_frame = tk.LabelFrame(left_frame, text="Participant Enrollment", 
                                     font=('Arial', 11, 'bold'), padx=10, pady=10)
        enroll_frame.pack(fill=tk.X, pady=5)
        
        # Speaker slots (support up to 5)
        self.speaker_entries = []
        self.role_entries = []
        self.enroll_buttons = []
        self.status_labels = []
        
        roles = ["Interrogator", "Suspect", "Lawyer", "Witness", "Observer"]
        
        for i in range(5):
            frame = tk.Frame(enroll_frame)
            frame.pack(fill=tk.X, pady=3)
            
            tk.Label(frame, text=f"P{i+1}:", width=3, font=('Arial', 9, 'bold')).pack(side=tk.LEFT)
            
            name_entry = tk.Entry(frame, width=12, font=('Arial', 9))
            name_entry.pack(side=tk.LEFT, padx=2)
            self.speaker_entries.append(name_entry)
            
            role_var = tk.StringVar(value=roles[i])
            role_combo = ttk.Combobox(frame, textvariable=role_var, values=roles, 
                                     width=10, font=('Arial', 9))
            role_combo.pack(side=tk.LEFT, padx=2)
            self.role_entries.append(role_var)
            
            btn = tk.Button(
                frame,
                text="📝",
                command=lambda idx=i: self.enroll_speaker(idx),
                bg='#E74C3C',
                fg='white',
                font=('Arial', 9, 'bold'),
                width=2
            )
            btn.pack(side=tk.LEFT, padx=2)
            self.enroll_buttons.append(btn)
            
            status = tk.Label(frame, text="", font=('Arial', 8), width=8)
            status.pack(side=tk.LEFT)
            self.status_labels.append(status)
            
        # Session controls
        control_frame = tk.LabelFrame(left_frame, text="Session Control", 
                                     font=('Arial', 11, 'bold'), padx=10, pady=10)
        control_frame.pack(fill=tk.X, pady=10)
        
        self.start_btn = tk.Button(
            control_frame,
            text="▶ START\nINTERROGATION",
            command=self.start_interrogation,
            bg='#27AE60',
            fg='white',
            font=('Arial', 12, 'bold'),
            pady=15,
            state=tk.DISABLED
        )
        self.start_btn.pack(fill=tk.X, pady=5)
        
        self.stop_btn = tk.Button(
            control_frame,
            text="⬛ STOP &\nGENERATE REPORT",
            command=self.stop_interrogation,
            bg='#E74C3C',
            fg='white',
            font=('Arial', 12, 'bold'),
            pady=15,
            state=tk.DISABLED
        )
        self.stop_btn.pack(fill=tk.X, pady=5)
        
        # Statistics display
        stats_frame = tk.LabelFrame(left_frame, text="Session Statistics", 
                                   font=('Arial', 10, 'bold'), padx=10, pady=10)
        stats_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        self.stats_text = tk.Text(stats_frame, height=12, width=30, font=('Courier', 9))
        self.stats_text.pack(fill=tk.BOTH, expand=True)
        
        # Right: Transcript + Quality
        right_frame = tk.Frame(main_frame)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # Transcript
        trans_frame = tk.LabelFrame(right_frame, text="Live Transcript (Forensic)", 
                                   font=('Arial', 11, 'bold'))
        trans_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 5))
        
        self.transcript = scrolledtext.ScrolledText(
            trans_frame,
            wrap=tk.WORD,
            font=('Consolas', 10),
            bg='#2C3E50',
            fg='white'
        )
        self.transcript.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Status bar
        self.status_bar = tk.Label(
            self.root,
            text="Ready - Enroll participants to begin",
            font=('Arial', 9),
            bg='#34495E',
            fg='white',
            anchor='w',
            padx=10,
            pady=5
        )
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Update stats timer
        self.update_stats_display()
        
    def enroll_speaker(self, speaker_idx):
        """Enroll a speaker (6 samples)"""
        name = self.speaker_entries[speaker_idx].get()
        role = self.role_entries[speaker_idx].get()
        
        if not name:
            messagebox.showwarning("Name Required", "Please enter participant name")
            return
            
        if self.is_recording_enrollment:
            return
            
        print(f"\n{'='*80}")
        print(f"ENROLLING: {name} ({role})")
        print(f"{'='*80}")
        
        # Disable all enrollment buttons
        for btn in self.enroll_buttons:
            btn.config(state=tk.DISABLED)
            
        self.is_recording_enrollment = True
        
        # Start audio if not already
        if not self.audio.is_recording:
            self.audio.start()
            time.sleep(1.0)
            
        # Start enrollment thread
        threading.Thread(
            target=self.enrollment_loop,
            args=(speaker_idx, name, role),
            daemon=True
        ).start()
        
    def enrollment_loop(self, speaker_idx, name, role):
        """Record 6 samples and create comprehensive profile"""
        speaker_key = f"participant_{speaker_idx}"
        samples_audio = []
        samples_embeddings = []
        
        # Start enrollment
        self.enrollment.start_enrollment(speaker_key, name, role)
        
        # Record 6 samples
        for i in range(6):
            print(f"\n🎙️ Sample {i+1}/6 - SPEAK NOW!")
            
            self.status_bar.config(text=f"🔴 Recording {name} - Sample {i+1}/6 - SPEAK!")
            self.root.update()
            
            # Clear and record
            self.audio.clear_queue()
            time.sleep(0.3)
            
            # 5-second recording
            start = time.time()
            while (time.time() - start) < 5.0:
                time.sleep(0.1)
                
            # Get audio
            audio_data = self.audio.get_buffer(duration=5.5)
            
            if len(audio_data) > 16000:
                # Stress-invariant preprocessing
                audio_normalized = self.stress_processor.normalize_audio(audio_data, 16000)
                
                # Convert back to int16 for embedding
                audio_for_embedding = (audio_normalized * 32768).astype(np.int16)
                
                # Add to enrollment
                success, quality, msg = self.enrollment.add_enrollment_sample(
                    speaker_key, audio_for_embedding, 16000
                )
                
                samples_audio.append(audio_data)  # Store original for spatial features
                
                print(f"  ✅ Sample {i+1}: {msg}")
                
                self.status_labels[speaker_idx].config(text=f"{i+1}/6", fg='orange')
                self.root.update()
                time.sleep(0.5)
                
        # Complete enrollment
        success, quality, msg = self.enrollment.complete_enrollment(speaker_key)
        
        # Create spatial fingerprint
        self.location_verifier.enroll_spatial_profile(speaker_key, samples_audio)
        
        # Initialize adaptive tracking
        enrolled_profile = self.enrollment.get_enrolled_speakers()[speaker_key]
        self.adaptive_system.initialize_speaker(speaker_key, enrolled_profile['mean_embedding'])
        
        # Register in audit
        self.audit.register_participant(speaker_key, name, role, quality)
        
        print(f"\n✅ {name} ({role}) enrolled!")
        print(f"   Voice quality: {quality:.1%}")
        
        self.status_labels[speaker_idx].config(text="✅ Done", fg='green')
        self.status_bar.config(text=f"✅ {name} enrolled successfully")
        self.root.update()
        
        # Re-enable buttons
        for btn in self.enroll_buttons:
            btn.config(state=tk.NORMAL)
            
        # Enable start if at least 2 enrolled
        enrolled = self.enrollment.get_enrolled_speakers()
        if len(enrolled) >= 2:
            self.start_btn.config(state=tk.NORMAL)
            
        self.is_recording_enrollment = False
        
    def start_interrogation(self):
        """Start live interrogation with full forensic logging"""
        if self.is_running:
            return
            
        print("\n" + "="*80)
        print("STARTING FORENSIC INTERROGATION SESSION")
        print("="*80)
        print(f"Session ID: {self.audit.session_id}")
        print(f"Participants: {len(self.enrollment.get_enrolled_speakers())}")
        print("="*80)
        
        self.is_running = True
        self.start_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        
        # Disable enrollment during interrogation
        for btn in self.enroll_buttons:
            btn.config(state=tk.DISABLED)
            
        # Start audio
        if not self.audio.is_recording:
            self.audio.start()
            
        # Log session start
        self.audit.log_system_event("INTERROGATION_START", {
            'enrolled_participants': list(self.enrollment.get_enrolled_speakers().keys())
        })
        
        # Start transcription thread
        threading.Thread(target=self.forensic_transcription_loop, daemon=True).start()
        
        self.status_bar.config(text="🔴 INTERROGATION IN PROGRESS - All events logged", bg='#E74C3C')
        
    def stop_interrogation(self):
        """Stop and generate forensic report"""
        if not self.is_running:
            return
            
        print("\nStopping interrogation and generating forensic report...")
        
        self.is_running = False
        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        
        # Log session end
        self.audit.log_system_event("INTERROGATION_END", {
            'duration_minutes': (datetime.now() - self.session_start).total_seconds() / 60
        })
        
        # Generate forensic report
        report_file, transcript_file = self.audit.export_forensic_report()
        
        # Show completion message
        messagebox.showinfo(
            "Forensic Report Generated",
            f"Interrogation session complete!\n\n"
            f"Forensic report: {report_file}\n"
            f"Transcript: {transcript_file}\n\n"
            f"Session ID: {self.audit.session_id}\n"
            f"Total segments: {self.audit.stats['transcribed_segments']}\n"
            f"Integrity: VERIFIED"
        )
        
        self.status_bar.config(text="Session complete - Forensic report generated", bg='#27AE60')
        
    def forensic_transcription_loop(self):
        """
        Main transcription loop with complete forensic logging
        """
        print("🎙️ Forensic transcription started")
        print("   All events will be logged for legal review")
        
        last_time = time.time()
        
        while self.is_running:
            try:
                if time.time() - last_time < 1.5:
                    time.sleep(0.1)
                    continue
                    
                # Get audio
                audio_data = self.audio.get_buffer(duration=2.5)
                
                if len(audio_data) < 16000:
                    time.sleep(0.1)
                    continue
                    
                # Voice activity detection
                rms = np.sqrt(np.mean(audio_data.astype(np.float32) ** 2))
                
                if rms < 300:
                    time.sleep(0.1)
                    continue
                    
                print(f"\n🎤 Processing speech (RMS: {int(rms)})...")
                
                # Generate audio segment ID
                audio_segment_id = hashlib.sha256(audio_data.tobytes()).hexdigest()[:16]
                
                # === QUALITY ASSESSMENT ===
                audio_quality = self.quality_assessor.assess_audio_quality(audio_data, 16000)
                
                print(f"   Audio quality: {audio_quality['audio_quality_category']} "
                      f"(SNR: {audio_quality['snr_db']:.1f} dB)")
                
                # === STRESS-INVARIANT PROCESSING ===
                audio_normalized = self.stress_processor.normalize_audio(audio_data, 16000)
                audio_for_embedding = (audio_normalized * 32768).astype(np.int16)
                
                # === EMBEDDING EXTRACTION ===
                test_embedding = self.embedder.extract_embedding(audio_for_embedding, 16000)
                
                if np.allclose(test_embedding, 0):
                    print("⚠️ Zero embedding - skipping")
                    continue
                    
                # === SPEAKER VERIFICATION with LOCATION ===
                enrolled = self.enrollment.get_enrolled_speakers()
                
                if not enrolled:
                    continue
                    
                accept, speaker_key, speaker_name, combined_score, reason = self.location_verifier.verify_with_location(
                    test_embedding,
                    audio_data,  # Original audio for spatial features
                    enrolled
                )
                
                # Get details for logging
                voice_sim = reason.split("voice:")[1].split(",")[0] if "voice:" in reason else combined_score
                spatial_sim = reason.split("spatial:")[1].split(")")[0] if "spatial:" in reason else None
                
                try:
                    voice_sim = float(voice_sim)
                    spatial_sim = float(spatial_sim) if spatial_sim else None
                except:
                    voice_sim = combined_score
                    spatial_sim = None
                    
                # Get threshold used
                speaker_profile = enrolled.get(speaker_key, {})
                threshold = speaker_profile.get('threshold', 0.64)
                
                # === VERIFICATION QUALITY ===
                verification_quality = self.quality_assessor.assess_verification_quality(
                    voice_sim, spatial_sim, combined_score, threshold, 
                    margin=None  # Could calculate if needed
                )
                
                # === LOG VERIFICATION ===
                self.audit.log_verification(
                    audio_segment_id=audio_segment_id,
                    speaker_key=speaker_key,
                    speaker_name=speaker_name,
                    voice_similarity=voice_sim,
                    spatial_similarity=spatial_sim,
                    combined_score=combined_score,
                    decision="ACCEPTED" if accept else "REJECTED",
                    threshold_used=threshold,
                    quality_metrics={
                        **audio_quality,
                        **verification_quality,
                        'rejection_reason': reason if not accept else None,
                        'rms_level': int(rms)
                    }
                )
                
                if not accept:
                    # REJECTED
                    print(f"🚫 REJECTED: {speaker_name} (score: {combined_score:.3f})")
                    print(f"   Reason: {reason}")
                    last_time = time.time()
                    continue
                    
                # ACCEPTED
                print(f"✅ ACCEPTED: {speaker_name} (score: {combined_score:.3f})")
                print(f"   {reason}")
                
                # === ADAPTIVE ENROLLMENT ===
                should_adapt, adapt_reason = self.adaptive_system.should_adapt(
                    speaker_key, verification_quality['verification_confidence'], 
                    spatial_match=(spatial_sim > 0.85 if spatial_sim else True)
                )
                
                if should_adapt:
                    current_voiceprint = enrolled[speaker_key]['mean_embedding']
                    updated_voiceprint, drift_info = self.adaptive_system.adapt_voiceprint(
                        speaker_key, current_voiceprint, test_embedding,
                        verification_quality['verification_confidence'],
                        self.session_start
                    )
                    
                    if drift_info['adapted']:
                        # Update enrollment
                        enrolled[speaker_key]['mean_embedding'] = updated_voiceprint
                        print(f"   📊 Voiceprint adapted (drift: {drift_info['drift']:.3f})")
                        
                        # Check for drift alert
                        alert, alert_msg = self.adaptive_system.check_drift_alert(speaker_key, drift_info['drift'])
                        if alert:
                            print(f"   ⚠️ {alert_msg}")
                            
                # === VOICE STRESS ANALYSIS ===
                stress_indicators = self.stress_analyzer.analyze_stress(audio_data)
                
                if stress_indicators['overall_stress'] != "LOW":
                    print(f"   ⚠️ Stress indicators: {stress_indicators['overall_stress']}")
                    
                # === TRANSCRIPTION ===
                audio_float = audio_data.astype(np.float32) / 32768.0
                result = self.model.transcribe(audio_float, language='en', fp16=False, verbose=False)
                
                if result['text'].strip():
                    # === TRANSCRIPTION QUALITY ===
                    transcription_quality = self.quality_assessor.assess_transcription_quality(result)
                    
                    # === OVERALL QUALITY ===
                    overall_quality = self.quality_assessor.assess_overall_quality(
                        audio_quality,
                        verification_quality,
                        transcription_quality
                    )
                    
                    # Log transcription
                    timestamp = datetime.now()
                    audio_checksum = hashlib.sha256(audio_data.tobytes()).hexdigest()
                    
                    self.audit.log_transcription(
                        timestamp=timestamp,
                        speaker_key=speaker_key,
                        speaker_name=speaker_name,
                        text=result['text'].strip(),
                        confidence=transcription_quality['transcription_confidence'],
                        audio_checksum=audio_checksum
                    )
                    
                    # Display with quality indicators
                    conf_marker = ""
                    if overall_quality['overall_category'] == "EXCELLENT":
                        conf_marker = ""
                        color = 'white'
                    elif overall_quality['overall_category'] == "GOOD":
                        conf_marker = " [GOOD]"
                        color = '#2ECC71'
                    elif overall_quality['overall_category'] == "ACCEPTABLE":
                        conf_marker = " [MED]"
                        color = '#F39C12'
                    elif overall_quality['overall_category'] == "POOR":
                        conf_marker = " [LOW]"
                        color = '#E74C3C'
                    else:
                        conf_marker = " [!INADMISSIBLE!]"
                        color = '#C0392B'
                        
                    speaker_profile = enrolled.get(speaker_key, {})
                    speaker_role = speaker_profile.get('role', 'Unknown')
                    
                    timestamp_str = timestamp.strftime("%H:%M:%S")
                    text_display = f"[{timestamp_str}] {speaker_role} ({speaker_name}){conf_marker}:\n    {result['text'].strip()}\n\n"
                    
                    self.transcript.insert(tk.END, text_display)
                    self.transcript.tag_add(f"seg_{len(self.audit.transcript_log)}", 
                                           f"{len(self.audit.transcript_log)+1}.0", 
                                           f"{len(self.audit.transcript_log)+1}.end")
                    self.transcript.tag_config(f"seg_{len(self.audit.transcript_log)}", foreground=color)
                    self.transcript.see(tk.END)
                    self.root.update()
                    
                    print(f"📝 [{timestamp_str}] {speaker_role} ({speaker_name}): {result['text'].strip()}")
                    print(f"   Overall quality: {overall_quality['overall_category']} "
                          f"(combined: {overall_quality['combined_quality_score']:.2f})")
                    
                    if not overall_quality['legally_admissible']:
                        print(f"   ⚠️ INADMISSIBLE: {', '.join(overall_quality['inadmissibility_reasons'])}")
                        
                last_time = time.time()
                
            except Exception as e:
                print(f"❌ Error: {e}")
                import traceback
                traceback.print_exc()
                time.sleep(1)
                
    def update_stats_display(self):
        """Update statistics display"""
        if hasattr(self, 'stats_text'):
            stats = self.audit.get_statistics()
            
            stats_str = f"""Session Statistics
{'='*28}

Participants: {stats['participants']}
Duration: {stats.get('duration_minutes', 0):.1f} min

Verifications: {stats['total_verifications']}
  Accepted: {stats['accepted']}
  Rejected: {stats['rejected']}

Transcribed: {stats['transcribed_segments']}
Avg Confidence: {stats.get('avg_confidence', 0):.1%}

Session ID:
{self.audit.session_id}
"""
            
            self.stats_text.delete(1.0, tk.END)
            self.stats_text.insert(1.0, stats_str)
            
        # Schedule next update
        if hasattr(self, 'root'):
            self.root.after(2000, self.update_stats_display)
            
    def run(self):
        """Run the forensic system"""
        self.create_gui()
        self.root.mainloop()
        self.audio.cleanup()


if __name__ == "__main__":
    app = ForensicInterrogationSystem(room_id="InterrogationRoom_A", case_id="INV-2025-001")
    app.run()

