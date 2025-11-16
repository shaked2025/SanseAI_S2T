"""
Enrollment UI Wizard for Interview Transcription
Guides users through speaker enrollment process
"""

import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
import threading
import time
import numpy as np
from auto_enrollment import ContinuousEnrollmentRecorder


class EnrollmentWizard:
    """Wizard for enrolling speakers before interview"""
    
    def __init__(self, root, audio_capture, embedding_extractor, on_complete_callback):
        """
        Initialize enrollment wizard
        
        Args:
            root: Tkinter root window
            audio_capture: AudioCapture instance
            embedding_extractor: Embedding extractor (Resemblyzer)
            on_complete_callback: Function to call when enrollment complete
        """
        self.root = root
        self.audio_capture = audio_capture
        self.embedding_extractor = embedding_extractor
        self.on_complete = on_complete_callback
        
        # Create new window
        self.window = tk.Toplevel(root)
        self.window.title("Speaker Enrollment - Interview Setup")
        self.window.geometry("800x600")
        self.window.protocol("WM_DELETE_WINDOW", self.on_cancel)
        
        # Force to front
        self.window.lift()
        self.window.attributes('-topmost', True)
        self.window.after_idle(self.window.attributes, '-topmost', False)
        
        # Data
        self.num_participants = 0
        self.participants = []  # [{key, name, role, samples}]
        self.current_step = 0
        self.current_participant_idx = 0
        self.current_sample_idx = 0
        self.is_recording = False
        
        # Create UI
        self.create_ui()
        self.show_welcome_screen()
        
    def create_ui(self):
        """Create wizard UI"""
        # Main container
        self.main_frame = ttk.Frame(self.window, padding=20)
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Title
        self.title_label = tk.Label(
            self.main_frame,
            text="🎙️ Speaker Enrollment Wizard",
            font=('Arial', 18, 'bold'),
            fg='#2C3E50'
        )
        self.title_label.pack(pady=(0, 20))
        
        # Content area (will be replaced for each step)
        self.content_frame = ttk.Frame(self.main_frame)
        self.content_frame.pack(fill=tk.BOTH, expand=True)
        
        # Navigation buttons
        self.nav_frame = ttk.Frame(self.main_frame)
        self.nav_frame.pack(fill=tk.X, pady=(20, 0))
        
        self.back_button = tk.Button(
            self.nav_frame,
            text="← Back",
            command=self.go_back,
            state=tk.DISABLED,
            font=('Arial', 11),
            padx=20,
            pady=8
        )
        self.back_button.pack(side=tk.LEFT)
        
        self.next_button = tk.Button(
            self.nav_frame,
            text="Next →",
            command=self.go_next,
            bg='#3498DB',
            fg='white',
            font=('Arial', 11, 'bold'),
            padx=20,
            pady=8
        )
        self.next_button.pack(side=tk.RIGHT)
        
        self.cancel_button = tk.Button(
            self.nav_frame,
            text="Cancel",
            command=self.on_cancel,
            font=('Arial', 11),
            padx=20,
            pady=8
        )
        self.cancel_button.pack(side=tk.RIGHT, padx=10)
        
    def clear_content(self):
        """Clear content area"""
        for widget in self.content_frame.winfo_children():
            widget.destroy()
            
    def show_welcome_screen(self):
        """Show welcome screen"""
        self.clear_content()
        
        # Welcome message
        welcome_text = tk.Label(
            self.content_frame,
            text="Welcome to Interview Transcription Setup",
            font=('Arial', 16, 'bold'),
            fg='#2C3E50'
        )
        welcome_text.pack(pady=20)
        
        info_text = tk.Label(
            self.content_frame,
            text="This wizard will help you enroll all participants.\n\n"
                 "For accurate speaker identification, each participant will:\n"
                 "• Provide their name and role\n"
                 "• Record 5 voice samples\n"
                 "• Build a unique voiceprint\n\n"
                 "This ensures 95%+ accuracy during the interview.",
            font=('Arial', 11),
            justify=tk.LEFT
        )
        info_text.pack(pady=20)
        
        # Number of participants
        num_frame = ttk.Frame(self.content_frame)
        num_frame.pack(pady=30)
        
        tk.Label(
            num_frame,
            text="How many participants (including interviewer)?",
            font=('Arial', 12, 'bold')
        ).pack()
        
        num_buttons_frame = ttk.Frame(num_frame)
        num_buttons_frame.pack(pady=15)
        
        for i in range(2, 7):
            btn = tk.Button(
                num_buttons_frame,
                text=str(i),
                command=lambda n=i: self.set_num_participants(n),
                font=('Arial', 14, 'bold'),
                width=3,
                height=1,
                bg='#ECF0F1',
                activebackground='#3498DB'
            )
            btn.pack(side=tk.LEFT, padx=5)
            
    def set_num_participants(self, num):
        """Set number of participants and move to next step"""
        self.num_participants = num
        print(f"📋 Setting up enrollment for {num} participants")
        
        # Initialize participant list
        self.participants = []
        roles = ['Interviewer'] + [f'Interviewee {i}' for i in range(1, num)]
        
        for i in range(num):
            self.participants.append({
                'key': f'person_{i}',
                'name': '',
                'role': roles[i] if i < len(roles) else 'Participant',
                'samples': [],
                'enrolled': False
            })
            
        self.show_participant_details()
        
    def show_participant_details(self):
        """Show participant name/role entry screen"""
        self.clear_content()
        self.back_button.config(state=tk.NORMAL)
        
        tk.Label(
            self.content_frame,
            text=f"Participant Details ({self.num_participants} people)",
            font=('Arial', 14, 'bold')
        ).pack(pady=10)
        
        # Scrollable frame for participants
        canvas = tk.Canvas(self.content_frame, height=400)
        scrollbar = ttk.Scrollbar(self.content_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Create entry fields
        self.name_entries = []
        self.role_vars = []
        
        for i, participant in enumerate(self.participants):
            frame = ttk.LabelFrame(scrollable_frame, text=f"Person {i + 1}", padding=10)
            frame.pack(fill=tk.X, pady=5, padx=10)
            
            # Name
            name_frame = ttk.Frame(frame)
            name_frame.pack(fill=tk.X, pady=5)
            tk.Label(name_frame, text="Name:", width=10, anchor='w').pack(side=tk.LEFT)
            name_entry = ttk.Entry(name_frame, width=30)
            name_entry.pack(side=tk.LEFT, padx=5)
            name_entry.insert(0, f"Person {i + 1}")
            self.name_entries.append(name_entry)
            
            # Role
            role_frame = ttk.Frame(frame)
            role_frame.pack(fill=tk.X, pady=5)
            tk.Label(role_frame, text="Role:", width=10, anchor='w').pack(side=tk.LEFT)
            
            role_var = tk.StringVar(value=participant['role'])
            role_combo = ttk.Combobox(
                role_frame,
                textvariable=role_var,
                values=['Interviewer', 'Interviewee 1', 'Interviewee 2', 'Interviewee 3', 'Observer', 'Other'],
                width=28
            )
            role_combo.pack(side=tk.LEFT, padx=5)
            self.role_vars.append(role_var)
            
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Update next button
        self.next_button.config(command=self.start_enrollment_process)
        
    def start_enrollment_process(self):
        """Save participant details and start voice enrollment"""
        # Save names and roles
        for i, participant in enumerate(self.participants):
            participant['name'] = self.name_entries[i].get() or f"Person {i + 1}"
            participant['role'] = self.role_vars[i].get()
            participant['key'] = participant['role'].lower().replace(' ', '_')
            
        print(f"✅ Participant details saved")
        for p in self.participants:
            print(f"   {p['role']}: {p['name']}")
            
        # Start enrollment
        self.current_participant_idx = 0
        self.current_sample_idx = 0
        self.show_enrollment_screen()
        
    def show_enrollment_screen(self):
        """Show voice sample recording screen - 5 SEPARATE recordings"""
        self.clear_content()
        
        participant = self.participants[self.current_participant_idx]
        
        # Progress
        progress_text = f"Enrolling {self.current_participant_idx + 1} of {len(self.participants)}"
        tk.Label(
            self.content_frame,
            text=progress_text,
            font=('Arial', 11),
            fg='#7F8C8D'
        ).pack(pady=5)
        
        # Current participant
        tk.Label(
            self.content_frame,
            text=f"🎙️ {participant['name']}",
            font=('Arial', 16, 'bold'),
            fg='#2C3E50'
        ).pack(pady=10)
        
        tk.Label(
            self.content_frame,
            text=f"Role: {participant['role']}",
            font=('Arial', 12),
            fg='#7F8C8D'
        ).pack()
        
        # Instructions
        instructions_frame = ttk.LabelFrame(self.content_frame, text="📝 Instructions", padding=15)
        instructions_frame.pack(fill=tk.X, pady=20, padx=20)
        
        tk.Label(
            instructions_frame,
            text=f"Record 5 voice samples (5 seconds each)\n"
                 f"Read the sentence shown clearly\n"
                 f"Recording will AUTO-STOP after 5 seconds",
            font=('Arial', 11),
            justify=tk.LEFT,
            fg='#2C3E50'
        ).pack()
        
        # Sample prompts
        prompts = [
            f"My name is {participant['name']}, and I am the {participant['role']}.",
            "I am participating in this interview session.",
            "This is my voice sample for speaker identification.",
            "The quick brown fox jumps over the lazy dog.",
            "Thank you for your patience during this enrollment."
        ]
        
        # Current sample
        sample_frame = ttk.LabelFrame(self.content_frame, text=f"Sample {self.current_sample_idx + 1} of 5", padding=15)
        sample_frame.pack(fill=tk.BOTH, expand=True, pady=10, padx=20)
        
        prompt = prompts[self.current_sample_idx]
        
        self.prompt_label = tk.Label(
            sample_frame,
            text=f'"{prompt}"',
            font=('Arial', 13, 'italic'),
            wraplength=600,
            fg='#34495E'
        )
        self.prompt_label.pack(pady=20)
        
        # Recording controls
        controls_frame = ttk.Frame(sample_frame)
        controls_frame.pack(pady=20)
        
        self.record_button = tk.Button(
            controls_frame,
            text="🔴 Start Recording (5 seconds)",
            command=self.start_recording_sample,
            bg='#E74C3C',
            fg='white',
            font=('Arial', 14, 'bold'),
            padx=30,
            pady=15
        )
        self.record_button.pack()
        
        self.recording_label = tk.Label(
            sample_frame,
            text="",
            font=('Arial', 11, 'bold'),
            fg='#E74C3C'
        )
        self.recording_label.pack(pady=10)
        
    def start_continuous_recording(self):
        """Start continuous recording for auto-chunking"""
        if self.is_recording:
            return
        
        print("🎙️ Starting continuous enrollment recording...")
            
        # Start audio capture if not already
        if not self.audio_capture.is_recording:
            print("   Starting audio capture...")
            self.audio_capture.start()
            time.sleep(0.5)  # Wait for audio to initialize
        
        # Clear buffer to start fresh
        print("   Clearing audio buffer...")
        self.audio_capture.clear_queue()
        time.sleep(0.3)
        
        # NOW start recording
        self.is_recording = True
        self.recording_start_time = time.time()
        
        print(f"✅ Recording started at {self.recording_start_time}")
        
        # Update UI
        self.record_button.config(
            text="⬛ Stop Recording (or wait for auto-stop at 30s)",
            bg='#95A5A6',
            command=self.stop_continuous_recording,
            font=('Arial', 11, 'bold')
        )
        
        # Visual feedback
        self.recording_label.config(text="🔴 RECORDING... 0.0s", font=('Arial', 14, 'bold'))
        self.progress_label.config(text="Speak naturally - will auto-stop at 30 seconds")
        
        # Update timer
        self.update_recording_timer()
        
    def update_recording_timer(self):
        """Update recording timer"""
        if self.is_recording:
            elapsed = time.time() - self.recording_start_time
            self.recording_label.config(text=f"🔴 RECORDING... {elapsed:.1f}s")
            
            if elapsed >= 30:
                # Auto-stop after 30 seconds
                self.progress_label.config(text=f"✅ 30 seconds reached - auto-stopping!", fg='#27AE60')
                self.window.update()
                time.sleep(0.5)
                self.stop_continuous_recording()
                return
            elif elapsed >= 20:
                self.progress_label.config(text=f"✅ Good! Will auto-stop at 30s (or click Stop now)", fg='#27AE60')
            elif elapsed >= 10:
                self.progress_label.config(text=f"Keep going... (will auto-stop at 30s)")
            else:
                self.progress_label.config(text=f"Keep speaking... (target: 20-30s)")
            
            self.window.after(100, self.update_recording_timer)
            
    def update_5s_timer(self):
        """Update timer for 5-second recording with auto-stop"""
        if self.is_recording:
            elapsed = time.time() - self.recording_start_time
            self.recording_label.config(text=f"🔴 RECORDING... {elapsed:.1f}s / 5.0s")
            
            if elapsed >= 5.0:
                # Auto-stop at 5 seconds
                self.recording_label.config(text=f"✅ 5 seconds complete!")
                self.window.update()
                time.sleep(0.3)
                self.stop_recording_sample()
                return
            
            self.window.after(100, self.update_5s_timer)
            
    def stop_continuous_recording(self):
        """Stop continuous recording and auto-chunk into samples"""
        if not self.is_recording:
            return
            
        self.is_recording = False
        self.record_button.config(
            text="🔴 Start Continuous Recording",
            bg='#E74C3C',
            command=self.start_continuous_recording,
            state=tk.DISABLED
        )
        
        # Get total recording duration
        duration = time.time() - self.recording_start_time
        
        if duration < 10:
            messagebox.showwarning("Recording Too Short", "Please record for at least 20 seconds.\nTry again.")
            self.record_button.config(state=tk.NORMAL)
            self.recording_label.config(text="")
            self.progress_label.config(text="")
            return
            
        # Get full recording
        self.recording_label.config(text="⏳ Processing recording and extracting samples...")
        self.progress_label.config(text="This may take a moment...")
        self.window.update()
        
        audio_data = self.audio_capture.get_buffer(duration=min(duration, 60))
        
        print(f"Processing {len(audio_data)/self.audio_capture.sample_rate:.1f}s of audio...")
        
        # Auto-chunk into 5 samples
        try:
            from auto_enrollment import AutoEnrollmentChunker
            chunker = AutoEnrollmentChunker(
                sample_rate=self.audio_capture.sample_rate,
                target_samples=5,
                min_duration=3.0,
                max_duration=7.0
            )
            
            chunks = chunker.chunk_recording(audio_data, self.audio_capture.sample_rate)
            
            if len(chunks) < 5:
                messagebox.showerror(
                    "Insufficient Speech",
                    f"Only detected {len(chunks)} speech segments.\n"
                    f"Please speak more (20-30 seconds) and try again."
                )
                self.record_button.config(state=tk.NORMAL)
                self.recording_label.config(text="")
                self.progress_label.config(text="")
                return
                
            # Extract embeddings from each chunk
            participant = self.participants[self.current_participant_idx]
            
            for i, chunk in enumerate(chunks):
                embedding = self.embedding_extractor.extract_embedding(chunk, self.audio_capture.sample_rate)
                
                participant['samples'].append({
                    'audio': chunk,
                    'embedding': embedding,
                    'duration': len(chunk) / self.audio_capture.sample_rate
                })
                
                self.recording_label.config(
                    text=f"✅ Extracted sample {i+1}/5...",
                    fg='#27AE60'
                )
                self.window.update()
                
            # All samples extracted!
            self.recording_label.config(
                text=f"✅ ALL 5 SAMPLES AUTO-EXTRACTED FOR {participant['name']}!",
                fg='#27AE60',
                font=('Arial', 12, 'bold')
            )
            self.progress_label.config(
                text=f"Moving to next participant in 2 seconds...",
                fg='#27AE60'
            )
            
            # Auto-advance to next participant
            self.window.after(2000, self.auto_advance_after_enrollment)
                
        except Exception as e:
            messagebox.showerror("Error", f"Failed to process recording: {str(e)}")
            import traceback
            traceback.print_exc()
            self.record_button.config(state=tk.NORMAL)
            self.recording_label.config(text="")
            self.progress_label.config(text="")
            
    def auto_advance_after_enrollment(self):
        """Auto-advance after completing samples for a participant"""
        self.current_participant_idx += 1
        
        if self.current_participant_idx < len(self.participants):
            # Next participant
            self.current_sample_idx = 0
            messagebox.showinfo(
                "Next Participant",
                f"Moving to next participant:\n{self.participants[self.current_participant_idx]['name']}"
            )
            self.show_enrollment_screen()
        else:
            # All participants enrolled
            messagebox.showinfo(
                "All Enrolled!",
                f"All {len(self.participants)} participants have been enrolled!\n\n"
                f"The system will now validate and start the interview."
            )
            self.show_completion_screen()
    
    def go_next(self):
        """Go to next participant or finish"""
        self.auto_advance_after_enrollment()
            
    def go_back(self):
        """Go back to previous step"""
        # Implementation for going back
        pass
        
    def show_completion_screen(self):
        """Show enrollment completion screen"""
        self.clear_content()
        
        tk.Label(
            self.content_frame,
            text="✅ Enrollment Complete!",
            font=('Arial', 18, 'bold'),
            fg='#27AE60'
        ).pack(pady=20)
        
        tk.Label(
            self.content_frame,
            text=f"Successfully enrolled {len(self.participants)} participants:",
            font=('Arial', 12)
        ).pack(pady=10)
        
        # Show enrolled participants
        for p in self.participants:
            frame = ttk.Frame(self.content_frame)
            frame.pack(fill=tk.X, padx=50, pady=5)
            
            tk.Label(
                frame,
                text=f"✓ {p['name']} ({p['role']}) - {len(p['samples'])} samples",
                font=('Arial', 11)
            ).pack(anchor='w')
            
        tk.Label(
            self.content_frame,
            text="\nThe system is now ready to accurately identify speakers\n"
                 "during the interview with 95%+ accuracy.\n\n"
                 "Starting interview interface in 3 seconds...",
            font=('Arial', 11),
            fg='#7F8C8D'
        ).pack(pady=20)
        
        # Auto-start interview after 3 seconds
        self.window.after(3000, self.finish_enrollment)
        
    def finish_enrollment(self):
        """Complete enrollment and start main app"""
        # Stop audio capture temporarily
        if self.audio_capture.is_recording:
            self.audio_capture.stop()
            
        # Call completion callback with enrollment data
        self.window.destroy()
        if self.on_complete:
            self.on_complete(self.participants)
            
    def on_cancel(self):
        """Cancel enrollment"""
        if messagebox.askyesno("Cancel Enrollment", "Are you sure you want to cancel?\nNo speakers will be enrolled."):
            self.window.destroy()

