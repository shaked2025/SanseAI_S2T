"""
AUDIO-ONLY Interview Transcription
NO CAMERA - Microphone only with speaker enrollment
"""

import tkinter as tk
from tkinter import ttk, scrolledtext
import whisper
import numpy as np
from datetime import datetime
import threading
import time
from audio_capture import AudioCapture
from speaker_diarization_robust import ResemblyzerEmbeddings
from speaker_enrollment import SpeakerEnrollment, SpeakerVerificationEngine

class AudioOnlyTranscriber:
    """Simple audio-only transcription with speaker enrollment"""
    
    def __init__(self):
        print("="*60)
        print("  AUDIO-ONLY INTERVIEW TRANSCRIPTION")
        print("="*60)
        print()
        
        # Load Whisper
        print("Loading Whisper (base model)...")
        self.model = whisper.load_model("base")
        
        # Audio ONLY - device 5 (your external microphone)
        print("Loading EXTERNAL microphone (device 5)...")
        self.audio = AudioCapture(sample_rate=16000, channels=1, device_index=5)
        
        # Speaker enrollment
        print("Loading speaker enrollment...")
        self.embedder = ResemblyzerEmbeddings()
        self.enrollment = SpeakerEnrollment(self.embedder)
        self.verifier = None
        
        self.is_running = False
        self.enrolled = False
        
        print("✅ Ready!")
        print()
        
    def create_gui(self):
        """Create simple GUI - NO VIDEO"""
        self.root = tk.Tk()
        self.root.title("Audio-Only Interview Transcription")
        self.root.geometry("800x600")
        
        # Title
        title = tk.Label(
            self.root,
            text="🎤 Audio-Only Interview Transcription",
            font=('Arial', 18, 'bold'),
            bg='#2C3E50',
            fg='white',
            pady=15
        )
        title.pack(fill=tk.X)
        
        # Enrollment section
        enroll_frame = tk.LabelFrame(self.root, text="Step 1: Enroll Speakers", font=('Arial', 12, 'bold'), padx=20, pady=15)
        enroll_frame.pack(fill=tk.X, padx=20, pady=10)
        
        tk.Label(
            enroll_frame,
            text="Enter number of speakers and their names, then record 5-second sample for each.",
            font=('Arial', 10)
        ).pack()
        
        # Speaker inputs
        input_frame = tk.Frame(enroll_frame)
        input_frame.pack(pady=10)
        
        tk.Label(input_frame, text="Speaker 1:", font=('Arial', 11)).grid(row=0, column=0, padx=5, sticky='e')
        self.name1 = tk.Entry(input_frame, width=20, font=('Arial', 11))
        self.name1.insert(0, "Interviewer")
        self.name1.grid(row=0, column=1, padx=5)
        
        self.enroll1_btn = tk.Button(
            input_frame,
            text="🔴 Record 5s",
            command=lambda: self.record_speaker(0),
            bg='#E74C3C',
            fg='white',
            font=('Arial', 10, 'bold'),
            padx=15,
            pady=5
        )
        self.enroll1_btn.grid(row=0, column=2, padx=5)
        
        self.status1 = tk.Label(input_frame, text="", font=('Arial', 9))
        self.status1.grid(row=0, column=3, padx=5)
        
        tk.Label(input_frame, text="Speaker 2:", font=('Arial', 11)).grid(row=1, column=0, padx=5, sticky='e')
        self.name2 = tk.Entry(input_frame, width=20, font=('Arial', 11))
        self.name2.insert(0, "Interviewee")
        self.name2.grid(row=1, column=1, padx=5, pady=5)
        
        self.enroll2_btn = tk.Button(
            input_frame,
            text="🔴 Record 5s",
            command=lambda: self.record_speaker(1),
            bg='#E74C3C',
            fg='white',
            font=('Arial', 10, 'bold'),
            padx=15,
            pady=5
        )
        self.enroll2_btn.grid(row=1, column=2, padx=5)
        
        self.status2 = tk.Label(input_frame, text="", font=('Arial', 9))
        self.status2.grid(row=1, column=3, padx=5)
        
        # Recording status
        self.enroll_status = tk.Label(
            enroll_frame,
            text="Click 'Record 5s' for each speaker. They will speak for 5 seconds.",
            font=('Arial', 10),
            fg='#7F8C8D'
        )
        self.enroll_status.pack(pady=10)
        
        # Transcript section
        transcript_frame = tk.LabelFrame(self.root, text="Step 2: Live Transcription", font=('Arial', 12, 'bold'))
        transcript_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        self.transcript = scrolledtext.ScrolledText(
            transcript_frame,
            wrap=tk.WORD,
            font=('Arial', 11),
            bg='#34495E',
            fg='white',
            height=15
        )
        self.transcript.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Control buttons
        btn_frame = tk.Frame(self.root)
        btn_frame.pack(pady=10)
        
        self.start_btn = tk.Button(
            btn_frame,
            text="▶ START INTERVIEW",
            command=self.start_interview,
            bg='#27AE60',
            fg='white',
            font=('Arial', 14, 'bold'),
            padx=30,
            pady=15,
            state=tk.DISABLED
        )
        self.start_btn.pack(side=tk.LEFT, padx=5)
        
        self.stop_btn = tk.Button(
            btn_frame,
            text="⬛ STOP",
            command=self.stop_interview,
            bg='#E74C3C',
            fg='white',
            font=('Arial', 14, 'bold'),
            padx=30,
            pady=15,
            state=tk.DISABLED
        )
        self.stop_btn.pack(side=tk.LEFT, padx=5)
        
    def record_speaker(self, speaker_num):
        """Record 5-second sample for a speaker"""
        name = self.name1.get() if speaker_num == 0 else self.name2.get()
        if not name:
            return
            
        print(f"\n🎙️ Recording speaker: {name}")
        
        # Disable buttons
        self.enroll1_btn.config(state=tk.DISABLED)
        self.enroll2_btn.config(state=tk.DISABLED)
        
        # Update status
        status_label = self.status1 if speaker_num == 0 else self.status2
        status_label.config(text="🔴 Recording 5s...", fg='red')
        self.enroll_status.config(text=f"🔴 RECORDING {name} - SPEAK NOW!", fg='red', font=('Arial', 12, 'bold'))
        self.root.update()
        
        # Start audio
        if not self.audio.is_recording:
            self.audio.start()
            time.sleep(0.8)
            
        # Clear and record
        self.audio.clear_queue()
        time.sleep(0.3)
        
        # Record for 5 seconds
        start = time.time()
        while (time.time() - start) < 5.0:
            elapsed = time.time() - start
            remaining = 5.0 - elapsed
            self.enroll_status.config(text=f"🔴 RECORDING {name} - {remaining:.1f}s remaining")
            self.root.update()
            time.sleep(0.1)
            
        # Get audio
        audio_data = self.audio.get_buffer(duration=5.5)
        
        print(f"   Got {len(audio_data)/16000:.1f}s of audio")
        
        # Process enrollment
        self.enroll_status.config(text=f"Processing {name}...", fg='orange')
        self.root.update()
        
        # Extract embedding
        embedding = self.embedder.extract_embedding(audio_data, 16000)
        
        # Create or update speaker
        speaker_key = f"speaker_{speaker_num}"
        
        # Start enrollment
        self.enrollment.start_enrollment(speaker_key, name, "Interviewer" if speaker_num == 0 else "Interviewee")
        self.enrollment.add_enrollment_sample(speaker_key, audio_data, 16000)
        self.enrollment.complete_enrollment(speaker_key)
        
        # Update UI
        status_label.config(text="✅ Enrolled", fg='green')
        self.enroll_status.config(text=f"✅ {name} enrolled! Record others or start interview.", fg='green')
        
        # Re-enable buttons
        self.enroll1_btn.config(state=tk.NORMAL)
        self.enroll2_btn.config(state=tk.NORMAL)
        
        # Enable start if at least one enrolled
        enrolled_speakers = self.enrollment.get_enrolled_speakers()
        if len(enrolled_speakers) >= 1:
            self.start_btn.config(state=tk.NORMAL)
            self.enrolled = True
            
        print(f"✅ {name} enrolled!")
        
    def start_interview(self):
        """Start live interview"""
        if self.is_running or not self.enrolled:
            return
            
        # Create verifier
        self.verifier = SpeakerVerificationEngine(self.enrollment)
        
        self.is_running = True
        self.start_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        
        # Disable enrollment
        self.enroll1_btn.config(state=tk.DISABLED)
        self.enroll2_btn.config(state=tk.DISABLED)
        self.name1.config(state=tk.DISABLED)
        self.name2.config(state=tk.DISABLED)
        
        # Start audio if not already
        if not self.audio.is_recording:
            self.audio.start()
            
        # Start transcription thread
        threading.Thread(target=self.transcribe_loop, daemon=True).start()
        
        print("\n🎙️ Interview started!")
        
    def stop_interview(self):
        """Stop interview"""
        self.is_running = False
        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        
        print("Stopped")
        
    def transcribe_loop(self):
        """Transcribe audio in real-time"""
        last_time = time.time()
        
        while self.is_running:
            try:
                # Every 2 seconds
                if time.time() - last_time < 2.0:
                    time.sleep(0.1)
                    continue
                    
                # Get audio
                audio_data = self.audio.get_buffer(duration=3.0)
                
                if len(audio_data) < 16000:
                    time.sleep(0.1)
                    continue
                    
                # Check if speech
                rms = np.sqrt(np.mean(audio_data.astype(np.float32) ** 2))
                if rms < 1000:
                    time.sleep(0.1)
                    continue
                    
                # Verify speaker
                speaker_key, speaker_name, confidence, metadata = self.verifier.verify_speaker(
                    audio_data, 16000
                )
                
                # Transcribe
                audio_float = audio_data.astype(np.float32) / 32768.0
                result = self.model.transcribe(audio_float, language='en', fp16=False)
                
                if result['text'].strip():
                    timestamp = datetime.now().strftime("%H:%M:%S")
                    text = f"[{timestamp}] {speaker_name}: {result['text'].strip()}\n\n"
                    
                    self.transcript.insert(tk.END, text)
                    self.transcript.see(tk.END)
                    
                    print(f"📝 {speaker_name}: {result['text'].strip()}")
                    
                last_time = time.time()
                
            except Exception as e:
                print(f"Error: {e}")
                time.sleep(1)
                
    def run(self):
        """Run application"""
        self.create_gui()
        self.root.mainloop()
        
        # Cleanup
        if self.is_running:
            self.stop_interview()
        self.audio.cleanup()


if __name__ == "__main__":
    app = AudioOnlyTranscriber()
    app.run()

