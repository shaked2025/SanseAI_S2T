"""
COMPREHENSIVE FORENSIC INTERROGATION SYSTEM
Addresses ALL identified weaknesses

Implements:
✅ Enhanced acoustic features (50+ features: shimmer, formants, energy dynamics, pauses)
✅ Linguistic/semantic analysis (sentiment, deception markers, cognitive load)
✅ Topic modeling (automatic segmentation and per-topic stress)
✅ Temporal pattern analysis (baselines, trends, change points)
✅ Conversation dynamics (turn-taking, response latency, coherence)
✅ Improved unknown rejection (95%+ TRR target)
✅ Complete forensic compliance

Production-grade for interrogation rooms.
"""

import tkinter as tk
from tkinter import scrolledtext, ttk, messagebox
import whisper
import numpy as np
from datetime import datetime
import threading
import time
import hashlib
import json

# Core
from audio_capture import AudioCapture
from speaker_diarization_robust import ResemblyzerEmbeddings
from speaker_enrollment import SpeakerEnrollment

# Verification
from simple_robust_verification import SimpleRobustVerifier
from spatial_location_features import LocationAwareVerifier
from improved_unknown_rejection import ImprovedUnknownRejection

# Analysis
from enhanced_acoustic_features import ComprehensiveAcousticAnalyzer
from linguistic_stress_analysis import LinguisticStressAnalyzer, ConversationDynamicsAnalyzer
from topic_modeling_analysis import TopicSegmentationSystem, TemporalStressAnalyzer
from stress_invariant_features import StressInvariantProcessor, calculate_whisper_confidence

# Forensic
from forensic_audit_trail import ForensicAuditLogger
from comprehensive_quality import ComprehensiveQualityAssessment


class ComprehensiveInterrogationSystem:
    """
    Complete system with ALL enhancements
    """
    
    def __init__(self, room_id="Room_1", case_id="Case_001"):
        print("="*90)
        print(" "*20 + "COMPREHENSIVE FORENSIC INTERROGATION SYSTEM")
        print(" "*25 + "Production-Grade with Full Analysis")
        print("="*90)
        print()
        
        self.room_id = room_id
        self.case_id = case_id
        self.session_start = datetime.now()
        
        # === CORE COMPONENTS ===
        print("Loading core components...")
        
        self.model = whisper.load_model("base")
        self.audio = AudioCapture(sample_rate=16000, channels=1, device_index=5)
        self.embedder = ResemblyzerEmbeddings()
        self.enrollment = SpeakerEnrollment(self.embedder)
        
        # === VERIFICATION COMPONENTS ===
        print("Loading verification systems...")
        
        self.base_verifier = SimpleRobustVerifier(base_threshold=0.64)
        self.location_verifier = LocationAwareVerifier(self.base_verifier, spatial_weight=0.15)
        self.improved_rejector = ImprovedUnknownRejection(base_threshold=0.64, strict_mode=True)
        self.stress_processor = StressInvariantProcessor()
        
        # === ANALYSIS COMPONENTS ===
        print("Loading comprehensive analysis systems...")
        
        self.acoustic_analyzer = ComprehensiveAcousticAnalyzer()
        self.linguistic_analyzer = LinguisticStressAnalyzer()
        self.conversation_analyzer = ConversationDynamicsAnalyzer()
        self.topic_system = TopicSegmentationSystem(similarity_threshold=0.65)
        self.temporal_analyzer = TemporalStressAnalyzer(baseline_duration_minutes=5)
        
        # === FORENSIC COMPONENTS ===
        print("Loading forensic compliance systems...")
        
        self.audit = ForensicAuditLogger(room_id=room_id, case_id=case_id)
        self.quality_assessor = ComprehensiveQualityAssessment()
        
        # State
        self.is_running = False
        self.is_recording_enrollment = False
        
        print("✅ All systems loaded")
        print(f"   Session ID: {self.audit.session_id}")
        print()
        
    def create_gui(self):
        """Create comprehensive GUI with analysis displays"""
        self.root = tk.Tk()
        self.root.title(f"Comprehensive Forensic System - {self.room_id}")
        self.root.geometry("1400x900")
        
        # Header
        header = tk.Frame(self.root, bg='#1ABC9C')
        header.pack(fill=tk.X)
        
        tk.Label(
            header,
            text="🔒 COMPREHENSIVE FORENSIC INTERROGATION ANALYSIS",
            font=('Arial', 18, 'bold'),
            bg='#1ABC9C',
            fg='white',
            pady=15
        ).pack()
        
        info_text = f"Room: {self.room_id} | Case: {self.case_id} | Session: {self.audit.session_id[:12]}"
        tk.Label(
            header,
            text=info_text,
            font=('Arial', 10),
            bg='#1ABC9C',
            fg='white',
            pady=5
        ).pack()
        
        # Main layout: 3 columns
        main_frame = tk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # === LEFT COLUMN: Enrollment & Control ===
        left_frame = tk.Frame(main_frame, width=300)
        left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 5))
        left_frame.pack_propagate(False)
        
        # Enrollment (compact for 5 participants)
        enroll_frame = tk.LabelFrame(left_frame, text="Participants", font=('Arial', 10, 'bold'))
        enroll_frame.pack(fill=tk.X, pady=5)
        
        self.participant_widgets = []
        
        for i in range(5):
            pframe = tk.Frame(enroll_frame)
            pframe.pack(fill=tk.X, pady=2, padx=5)
            
            name_entry = tk.Entry(pframe, width=12, font=('Arial', 9))
            name_entry.pack(side=tk.LEFT, padx=1)
            
            role_var = tk.StringVar(value=["Int", "Sus", "Law", "Wit", "Obs"][i])
            role_combo = ttk.Combobox(pframe, textvariable=role_var, 
                                     values=["Int", "Sus", "Law", "Wit", "Obs"],
                                     width=4, font=('Arial', 9))
            role_combo.pack(side=tk.LEFT, padx=1)
            
            enroll_btn = tk.Button(pframe, text="📝", command=lambda idx=i: self.enroll(idx),
                                  font=('Arial', 8), width=2, bg='#E74C3C', fg='white')
            enroll_btn.pack(side=tk.LEFT, padx=1)
            
            status = tk.Label(pframe, text="", font=('Arial', 8), width=6)
            status.pack(side=tk.LEFT)
            
            self.participant_widgets.append({
                'name': name_entry,
                'role': role_var,
                'button': enroll_btn,
                'status': status
            })
            
        # Controls
        control_frame = tk.LabelFrame(left_frame, text="Session", font=('Arial', 10, 'bold'))
        control_frame.pack(fill=tk.X, pady=10)
        
        self.start_btn = tk.Button(control_frame, text="▶ START", command=self.start,
                                   bg='#27AE60', fg='white', font=('Arial', 14, 'bold'),
                                   pady=12, state=tk.DISABLED)
        self.start_btn.pack(fill=tk.X, padx=5, pady=3)
        
        self.stop_btn = tk.Button(control_frame, text="⬛ STOP", command=self.stop,
                                 bg='#E74C3C', fg='white', font=('Arial', 14, 'bold'),
                                 pady=12, state=tk.DISABLED)
        self.stop_btn.pack(fill=tk.X, padx=5, pady=3)
        
        # Real-time stats
        stats_frame = tk.LabelFrame(left_frame, text="Real-Time Analysis", 
                                   font=('Arial', 10, 'bold'))
        stats_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        self.stats_display = tk.Text(stats_frame, font=('Courier', 8), height=25, width=35)
        self.stats_display.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # === CENTER COLUMN: Transcript ===
        center_frame = tk.Frame(main_frame)
        center_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        
        trans_frame = tk.LabelFrame(center_frame, text="Live Transcript with Quality Indicators",
                                   font=('Arial', 11, 'bold'))
        trans_frame.pack(fill=tk.BOTH, expand=True)
        
        self.transcript = scrolledtext.ScrolledText(trans_frame, wrap=tk.WORD,
                                                    font=('Consolas', 10),
                                                    bg='#2C3E50', fg='white')
        self.transcript.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # === RIGHT COLUMN: Topic & Stress Analysis ===
        right_frame = tk.Frame(main_frame, width=350)
        right_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=(5, 0))
        right_frame.pack_propagate(False)
        
        # Topic summary
        topic_frame = tk.LabelFrame(right_frame, text="Topic Analysis",
                                   font=('Arial', 10, 'bold'))
        topic_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 5))
        
        self.topic_display = tk.Text(topic_frame, font=('Courier', 9), width=40)
        self.topic_display.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Stress timeline
        stress_frame = tk.LabelFrame(right_frame, text="Stress Timeline",
                                    font=('Arial', 10, 'bold'))
        stress_frame.pack(fill=tk.BOTH, expand=True)
        
        self.stress_display = tk.Text(stress_frame, font=('Courier', 9), width=40)
        self.stress_display.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Status bar
        self.status_bar = tk.Label(self.root, text="Ready", font=('Arial', 9),
                                  bg='#34495E', fg='white', anchor='w', padx=10, pady=5)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Update displays periodically
        self.update_displays()
        
    def enroll(self, idx):
        """Enroll a participant"""
        name = self.participant_widgets[idx]['name'].get()
        role = self.participant_widgets[idx]['role'].get()
        
        if not name:
            return
            
        # Enrollment logic (same as before, just condensed)
        print(f"\nEnrolling: {name} ({role})")
        
        # Disable buttons
        for w in self.participant_widgets:
            w['button'].config(state=tk.DISABLED)
            
        self.is_recording_enrollment = True
        
        if not self.audio.is_recording:
            self.audio.start()
            time.sleep(1.0)
            
        threading.Thread(target=self.enrollment_thread, 
                        args=(idx, name, role), daemon=True).start()
        
    def enrollment_thread(self, idx, name, role):
        """Enrollment thread - 6 samples"""
        speaker_key = f"participant_{idx}"
        samples = []
        
        self.enrollment.start_enrollment(speaker_key, name, role)
        
        for i in range(6):
            self.status_bar.config(text=f"🔴 Recording {name} - Sample {i+1}/6")
            self.root.update()
            
            self.audio.clear_queue()
            time.sleep(0.3)
            
            start = time.time()
            while (time.time() - start) < 5.0:
                time.sleep(0.1)
                
            audio_data = self.audio.get_buffer(duration=5.5)
            
            if len(audio_data) > 16000:
                # Stress-invariant preprocessing
                audio_norm = self.stress_processor.normalize_audio(audio_data, 16000)
                audio_for_emb = (audio_norm * 32768).astype(np.int16)
                
                self.enrollment.add_enrollment_sample(speaker_key, audio_for_emb, 16000)
                samples.append(audio_data)
                
                self.participant_widgets[idx]['status'].config(text=f"{i+1}/6", fg='orange')
                self.root.update()
                time.sleep(0.5)
                
        # Complete
        success, quality, msg = self.enrollment.complete_enrollment(speaker_key)
        
        # Spatial fingerprint
        self.location_verifier.enroll_spatial_profile(speaker_key, samples)
        
        # Register
        self.audit.register_participant(speaker_key, name, role, quality)
        
        print(f"✅ {name} enrolled (quality: {quality:.1%})")
        
        self.participant_widgets[idx]['status'].config(text="✅", fg='green')
        
        # Re-enable
        for w in self.participant_widgets:
            w['button'].config(state=tk.NORMAL)
            
        # Enable start if 2+ enrolled
        if len(self.enrollment.get_enrolled_speakers()) >= 2:
            self.start_btn.config(state=tk.NORMAL)
            
        self.is_recording_enrollment = False
        
    def start(self):
        """Start comprehensive interrogation"""
        if self.is_running:
            return
            
        print("\n" + "="*90)
        print("STARTING COMPREHENSIVE INTERROGATION ANALYSIS")
        print("="*90)
        
        # Fit improved rejection model
        self.improved_rejector.fit_on_enrolled(self.enrollment)
        
        self.is_running = True
        self.start_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        
        if not self.audio.is_recording:
            self.audio.start()
            
        self.audit.log_system_event("INTERROGATION_START", {})
        
        threading.Thread(target=self.comprehensive_analysis_loop, daemon=True).start()
        
        self.status_bar.config(text="🔴 COMPREHENSIVE ANALYSIS IN PROGRESS", bg='#E74C3C')
        
    def stop(self):
        """Stop and generate comprehensive report"""
        if not self.is_running:
            return
            
        print("\nGenerating comprehensive forensic report...")
        
        self.is_running = False
        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        
        # Export forensic report
        report_file, transcript_file = self.audit.export_forensic_report()
        
        # Export topic analysis
        topic_report = self.export_topic_analysis()
        
        # Export stress timeline
        stress_report = self.export_stress_timeline()
        
        messagebox.showinfo(
            "Comprehensive Report Generated",
            f"Reports generated:\n\n"
            f"1. Forensic audit: {report_file}\n"
            f"2. Transcript: {transcript_file}\n"
            f"3. Topic analysis: {topic_report}\n"
            f"4. Stress timeline: {stress_report}\n\n"
            f"All reports include complete stress and linguistic analysis."
        )
        
        self.status_bar.config(text="Complete - All reports generated", bg='#27AE60')
        
    def comprehensive_analysis_loop(self):
        """
        Main loop with COMPLETE analysis
        """
        print("🎙️ Comprehensive analysis started")
        
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
                    
                # Check speech
                rms = np.sqrt(np.mean(audio_data.astype(np.float32) ** 2))
                
                if rms < 300:
                    time.sleep(0.1)
                    continue
                    
                print(f"\n{'='*90}")
                print(f"🎤 Processing utterance (RMS: {int(rms)})")
                
                # === COMPREHENSIVE ACOUSTIC ANALYSIS ===
                print("   Extracting 50+ acoustic features...")
                acoustic_features = self.acoustic_analyzer.extract_all_features(audio_data)
                acoustic_stress = self.acoustic_analyzer.assess_stress_from_acoustics(acoustic_features)
                
                print(f"   Acoustic stress: {acoustic_stress['acoustic_stress_category']} "
                      f"({acoustic_stress['acoustic_stress_probability']:.2f})")
                
                # === STRESS-INVARIANT EMBEDDING ===
                audio_norm = self.stress_processor.normalize_audio(audio_data, 16000)
                audio_for_emb = (audio_norm * 32768).astype(np.int16)
                test_embedding = self.embedder.extract_embedding(audio_for_emb, 16000)
                
                if np.allclose(test_embedding, 0):
                    continue
                    
                enrolled = self.enrollment.get_enrolled_speakers()
                
                if not enrolled:
                    continue
                    
                # === IMPROVED UNKNOWN SPEAKER REJECTION ===
                print("   Multi-layer speaker verification...")
                
                # Get voice similarity
                similarities = {}
                for key, profile in enrolled.items():
                    similarities[key] = np.dot(test_embedding, profile['mean_embedding'])
                    
                best_key = max(similarities, key=similarities.get)
                voice_sim = similarities[best_key]
                
                # Get spatial similarity
                spatial_accept, _, _, combined_score, reason = self.location_verifier.verify_with_location(
                    test_embedding, audio_data, enrolled
                )
                
                # Extract spatial sim from reason
                spatial_sim = None
                if "spatial:" in reason:
                    try:
                        spatial_sim = float(reason.split("spatial:")[1].split(")")[0])
                    except:
                        pass
                        
                # ENHANCED REJECTION
                accept, confidence, method_results = self.improved_rejector.verify_with_enhanced_rejection(
                    test_embedding, best_key, voice_sim, spatial_sim, enrolled
                )
                
                speaker_name = enrolled[best_key]['name']
                speaker_role = enrolled[best_key]['role']
                
                print(f"   Speaker: {speaker_name} ({speaker_role})")
                print(f"   Voice: {voice_sim:.3f}, Spatial: {spatial_sim:.3f if spatial_sim else 'N/A'}")
                print(f"   Methods passed: {method_results['decision']['methods_passed']}/{method_results['decision']['methods_total']}")
                
                if not accept:
                    # REJECTED
                    print(f"   🚫 REJECTED")
                    print(f"   Failed methods: {[k for k, v in method_results.items() if isinstance(v, dict) and not v.get('pass', True)]}")
                    
                    # Add to impostor cohort
                    self.improved_rejector.add_impostor_sample(test_embedding)
                    
                    # Log rejection
                    self.audit.log_verification(
                        hashlib.md5(audio_data.tobytes()).hexdigest()[:16],
                        best_key, speaker_name, voice_sim, spatial_sim, combined_score,
                        "REJECTED", 0.64, {'method_results': str(method_results)}
                    )
                    
                    last_time = time.time()
                    continue
                    
                # ACCEPTED
                print(f"   ✅ ACCEPTED (confidence: {confidence:.3f})")
                
                # === TRANSCRIPTION ===
                print("   Transcribing...")
                audio_float = audio_data.astype(np.float32) / 32768.0
                result = self.model.transcribe(audio_float, language='en', fp16=False, verbose=False)
                
                if not result['text'].strip():
                    continue
                    
                text = result['text'].strip()
                print(f"   Text: {text[:60]}...")
                
                # === LINGUISTIC ANALYSIS ===
                print("   Analyzing linguistic features...")
                linguistic_features = self.linguistic_analyzer.analyze_text(text, speaker_role)
                
                print(f"   Linguistic stress: {linguistic_features['linguistic_stress_category']} "
                      f"({linguistic_features['linguistic_stress_probability']:.2f})")
                
                # === TOPIC ASSIGNMENT ===
                print("   Assigning to topic...")
                topic_assignment = self.topic_system.add_utterance(
                    datetime.now(), best_key, speaker_role, text,
                    acoustic_stress, linguistic_features
                )
                
                print(f"   Topic: {topic_assignment['topic_label']} (ID: {topic_assignment['topic_id']})")
                if topic_assignment['is_topic_return']:
                    print(f"   ⚡ Topic return! (last mentioned {topic_assignment['time_since_last_mention']/60:.1f} min ago)")
                    
                # === TEMPORAL STRESS TRACKING ===
                combined_stress = 0.6 * acoustic_stress['acoustic_stress_probability'] + \
                                 0.4 * linguistic_features['linguistic_stress_probability']
                                 
                self.temporal_analyzer.add_measurement(
                    datetime.now(), 
                    acoustic_stress['acoustic_stress_probability'],
                    linguistic_features['linguistic_stress_probability']
                )
                
                # === CONVERSATION DYNAMICS ===
                self.conversation_analyzer.add_utterance(
                    datetime.now(), best_key, speaker_role, text,
                    acoustic_features, linguistic_features
                )
                
                # === QUALITY ASSESSMENT ===
                transcription_quality = self.quality_assessor.assess_transcription_quality(result)
                audio_quality = self.quality_assessor.assess_audio_quality(audio_data, 16000)
                
                # === DISPLAY ===
                timestamp_str = datetime.now().strftime("%H:%M:%S")
                
                # Quality marker
                if combined_stress >= 0.60:
                    stress_marker = " [HIGH STRESS]"
                    color = '#E74C3C'
                elif combined_stress >= 0.35:
                    stress_marker = " [MOD STRESS]"
                    color = '#F39C12'
                else:
                    stress_marker = ""
                    color = 'white'
                    
                display_text = f"[{timestamp_str}] {speaker_role} ({speaker_name}){stress_marker}:\n"
                display_text += f"    {text}\n"
                display_text += f"    Topic: {topic_assignment['topic_label']} | "
                display_text += f"A-Stress: {acoustic_stress['acoustic_stress_probability']:.2f} | "
                display_text += f"L-Stress: {linguistic_features['linguistic_stress_probability']:.2f}\n\n"
                
                self.transcript.insert(tk.END, display_text)
                self.transcript.see(tk.END)
                
                # Log
                self.audit.log_transcription(
                    datetime.now(), best_key, speaker_name, text,
                    transcription_quality['transcription_confidence'],
                    hashlib.sha256(audio_data.tobytes()).hexdigest()
                )
                
                print(f"   ✅ Transcribed and analyzed")
                
                last_time = time.time()
                
            except Exception as e:
                print(f"❌ Error: {e}")
                import traceback
                traceback.print_exc()
                time.sleep(1)
                
    def update_displays(self):
        """Update real-time analysis displays"""
        if hasattr(self, 'stats_display'):
            # Session stats
            stats = self.audit.get_statistics()
            
            stats_text = f"""SESSION STATS
{'='*32}
Duration: {stats.get('duration_minutes', 0):.1f} min
Participants: {stats['participants']}

Verifications: {stats['total_verifications']}
Accepted: {stats['accepted']}
Rejected: {stats['rejected']}

Transcribed: {stats['transcribed_segments']}

Session ID:
{self.audit.session_id[:24]}
"""
            
            self.stats_display.delete(1.0, tk.END)
            self.stats_display.insert(1.0, stats_text)
            
        if hasattr(self, 'topic_display') and self.topic_system.topics:
            # Topic summary
            topic_text = f"TOPICS ({len(self.topic_system.topics)})\n{'='*32}\n\n"
            
            for topic_id in sorted(self.topic_system.topics.keys()):
                analysis = self.topic_system.analyze_topic_stress_patterns(topic_id)
                if analysis:
                    topic_text += f"{analysis['topic_label']}: {analysis['utterance_count']} mentions\n"
                    topic_text += f"  Stress: {analysis.get('topic_stress_category', 'N/A')}\n"
                    topic_text += f"  Duration: {analysis['total_duration_seconds']/60:.1f} min\n\n"
                    
            self.topic_display.delete(1.0, tk.END)
            self.topic_display.insert(1.0, topic_text)
            
        # Schedule next update
        if hasattr(self, 'root'):
            self.root.after(2000, self.update_displays)
            
    def export_topic_analysis(self):
        """Export per-topic stress analysis"""
        topic_summaries = self.topic_system.get_all_topics_summary()
        
        filename = f"forensic_reports/topic_analysis_{self.audit.session_id}.json"
        
        with open(filename, 'w') as f:
            json.dump({
                'session_id': self.audit.session_id,
                'total_topics': len(topic_summaries),
                'topics': topic_summaries,
                'avoidance_patterns': self.topic_system.detect_topic_avoidance()
            }, f, indent=2)
            
        print(f"   Topic analysis: {filename}")
        return filename
        
    def export_stress_timeline(self):
        """Export stress timeline and change points"""
        change_points = self.temporal_analyzer.detect_change_points()
        trend = self.temporal_analyzer.calculate_stress_trend()
        
        filename = f"forensic_reports/stress_timeline_{self.audit.session_id}.json"
        
        with open(filename, 'w') as f:
            json.dump({
                'session_id': self.audit.session_id,
                'baseline_acoustic': float(self.temporal_analyzer.baseline_acoustic) if self.temporal_analyzer.baseline_acoustic else None,
                'baseline_linguistic': float(self.temporal_analyzer.baseline_linguistic) if self.temporal_analyzer.baseline_linguistic else None,
                'overall_trend': float(trend),
                'change_points': change_points,
                'timeline': self.temporal_analyzer.stress_timeline
            }, f, indent=2, default=str)
            
        print(f"   Stress timeline: {filename}")
        return filename
        
    def run(self):
        """Run the system"""
        self.create_gui()
        self.root.mainloop()
        self.audio.cleanup()


if __name__ == "__main__":
    app = ComprehensiveInterrogationSystem(
        room_id="InterrogationRoom_A",
        case_id="INV-2025-001"
    )
    app.run()

