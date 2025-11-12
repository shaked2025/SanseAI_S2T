"""
GUI Application
Main interface for the Speech-to-Text system
"""

import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
from PIL import Image, ImageTk
import cv2
import numpy as np
import threading
import queue
from datetime import datetime
import os


class SpeechToTextGUI:
    """Main GUI application for speech-to-text system"""
    
    def __init__(self, root, config):
        self.root = root
        self.config = config
        
        # Set up window
        self.root.title("Real-Time Speech-to-Text with Speaker Diarization")
        window_width = config.get('display', {}).get('window_width', 1200)
        window_height = config.get('display', {}).get('window_height', 800)
        self.root.geometry(f"{window_width}x{window_height}")
        
        # Colors and styles
        self.bg_color = "#2C3E50"
        self.fg_color = "#ECF0F1"
        self.accent_color = "#3498DB"
        
        # Configure styles
        self.setup_styles()
        
        # Initialize components
        self.video_label = None
        self.transcript_text = None
        self.status_label = None
        self.audio_level_bar = None
        self.speaker_frame = None
        
        # State
        self.is_running = False
        self.current_frame = None
        self.frame_lock = threading.Lock()
        
        # Initialize callbacks
        self.start_callback = None
        self.stop_callback = None
        self.snapshot_callback = None
        self.export_callback = None
        
        # Create GUI
        self.create_gui()
        
        # Update queue
        self.update_queue = queue.Queue()
        self.schedule_updates()
        
    def setup_styles(self):
        """Configure GUI styles"""
        style = ttk.Style()
        style.theme_use('clam')
        
        # Configure colors
        style.configure('TFrame', background=self.bg_color)
        style.configure('TLabel', background=self.bg_color, foreground=self.fg_color, font=('Arial', 10))
        style.configure('Title.TLabel', font=('Arial', 16, 'bold'))
        style.configure('Status.TLabel', font=('Arial', 9))
        style.configure('TButton', font=('Arial', 10), padding=10)
        
    def create_gui(self):
        """Create the main GUI layout"""
        
        # Main container
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Title with live indicator
        title_frame = ttk.Frame(main_frame)
        title_frame.pack(pady=(0, 10))
        
        title_label = ttk.Label(
            title_frame,
            text="🎤 Real-Time Speech-to-Text System",
            style='Title.TLabel'
        )
        title_label.pack(side=tk.LEFT)
        
        # Live indicator (will show when running)
        self.live_indicator = tk.Label(
            title_frame,
            text="",
            font=('Arial', 12, 'bold'),
            fg='#E74C3C',
            bg=self.bg_color
        )
        self.live_indicator.pack(side=tk.LEFT, padx=10)
        
        # Content area (split into left and right)
        content_frame = ttk.Frame(main_frame)
        content_frame.pack(fill=tk.BOTH, expand=True)
        
        # Left side - Video
        left_frame = ttk.Frame(content_frame)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        
        # Video display
        video_frame = ttk.LabelFrame(left_frame, text="Video Feed", padding=10)
        video_frame.pack(fill=tk.BOTH, expand=True)
        
        self.video_label = tk.Label(video_frame, bg='black')
        self.video_label.pack(fill=tk.BOTH, expand=True)
        
        # Audio level indicator
        audio_frame = ttk.Frame(left_frame)
        audio_frame.pack(fill=tk.X, pady=10)
        
        ttk.Label(audio_frame, text="Audio Level:").pack(side=tk.LEFT, padx=5)
        
        self.audio_level_canvas = tk.Canvas(audio_frame, height=20, bg='#34495E', highlightthickness=0)
        self.audio_level_canvas.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
        # Right side - Transcripts and controls
        right_frame = ttk.Frame(content_frame)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 0))
        
        # Speakers panel
        speaker_frame = ttk.LabelFrame(right_frame, text="Active Speakers", padding=10)
        speaker_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.speaker_container = ttk.Frame(speaker_frame)
        self.speaker_container.pack(fill=tk.X)
        
        # Transcript display
        transcript_frame = ttk.LabelFrame(right_frame, text="Live Transcript", padding=10)
        transcript_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        # Create text widget with scrollbar
        text_container = ttk.Frame(transcript_frame)
        text_container.pack(fill=tk.BOTH, expand=True)
        
        self.transcript_text = scrolledtext.ScrolledText(
            text_container,
            wrap=tk.WORD,
            font=('Arial', 11),
            bg='#34495E',
            fg='#ECF0F1',
            insertbackground='white',
            relief=tk.FLAT,
            padx=10,
            pady=10
        )
        self.transcript_text.pack(fill=tk.BOTH, expand=True)
        
        # Control buttons
        control_frame = ttk.Frame(right_frame)
        control_frame.pack(fill=tk.X)
        
        self.start_button = tk.Button(
            control_frame,
            text="▶ Start",
            command=self.on_start,
            bg='#27AE60',
            fg='white',
            font=('Arial', 12, 'bold'),
            relief=tk.FLAT,
            padx=20,
            pady=10,
            cursor='hand2'
        )
        self.start_button.pack(side=tk.LEFT, padx=5)
        
        self.stop_button = tk.Button(
            control_frame,
            text="⬛ Stop",
            command=self.on_stop,
            bg='#E74C3C',
            fg='white',
            font=('Arial', 12, 'bold'),
            relief=tk.FLAT,
            padx=20,
            pady=10,
            cursor='hand2',
            state=tk.DISABLED
        )
        self.stop_button.pack(side=tk.LEFT, padx=5)
        
        self.snapshot_button = tk.Button(
            control_frame,
            text="📷 Snapshot",
            command=self.on_snapshot,
            bg='#3498DB',
            fg='white',
            font=('Arial', 12, 'bold'),
            relief=tk.FLAT,
            padx=20,
            pady=10,
            cursor='hand2',
            state=tk.DISABLED
        )
        self.snapshot_button.pack(side=tk.LEFT, padx=5)
        
        self.clear_button = tk.Button(
            control_frame,
            text="🗑 Clear",
            command=self.on_clear,
            bg='#95A5A6',
            fg='white',
            font=('Arial', 12, 'bold'),
            relief=tk.FLAT,
            padx=20,
            pady=10,
            cursor='hand2'
        )
        self.clear_button.pack(side=tk.LEFT, padx=5)
        
        self.export_button = tk.Button(
            control_frame,
            text="💾 Export",
            command=self.on_export,
            bg='#9B59B6',
            fg='white',
            font=('Arial', 12, 'bold'),
            relief=tk.FLAT,
            padx=20,
            pady=10,
            cursor='hand2'
        )
        self.export_button.pack(side=tk.LEFT, padx=5)
        
        # Status bar
        status_frame = ttk.Frame(main_frame)
        status_frame.pack(fill=tk.X, pady=(10, 0))
        
        self.status_label = ttk.Label(
            status_frame,
            text="Ready to start",
            style='Status.TLabel'
        )
        self.status_label.pack(side=tk.LEFT)
        
    def on_start(self):
        """Handle start button click"""
        if self.start_callback:
            self.start_callback()
            
    def on_stop(self):
        """Handle stop button click"""
        if self.stop_callback:
            self.stop_callback()
            
    def on_snapshot(self):
        """Handle snapshot button click"""
        if self.snapshot_callback:
            self.snapshot_callback()
            
    def on_clear(self):
        """Handle clear button click"""
        self.transcript_text.delete(1.0, tk.END)
        self.update_status("Transcript cleared")
        
    def on_export(self):
        """Handle export button click"""
        if self.export_callback:
            self.export_callback()
            
    def set_callbacks(self, start_cb=None, stop_cb=None, snapshot_cb=None, export_cb=None):
        """Set callback functions for buttons"""
        self.start_callback = start_cb
        self.stop_callback = stop_cb
        self.snapshot_callback = snapshot_cb
        self.export_callback = export_cb
        
    def update_video_frame(self, frame):
        """Update video display"""
        if frame is None:
            return
            
        # Resize frame to fit display
        display_height = 400
        aspect_ratio = frame.shape[1] / frame.shape[0]
        display_width = int(display_height * aspect_ratio)
        
        frame_resized = cv2.resize(frame, (display_width, display_height))
        
        # Convert to ImageTk format
        frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)
        imgtk = ImageTk.PhotoImage(image=img)
        
        # Update label
        self.video_label.imgtk = imgtk
        self.video_label.configure(image=imgtk)
        
    def update_transcript(self, text, speaker_id=0, speaker_name=None, color=None):
        """Add new transcript entry"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        if speaker_name is None:
            speaker_name = f"Speaker {speaker_id + 1}"
            
        if color is None:
            color = "#FFFFFF"
            
        # Insert with color
        self.transcript_text.insert(tk.END, f"[{timestamp}] ", "timestamp")
        self.transcript_text.insert(tk.END, f"{speaker_name}: ", f"speaker_{speaker_id}")
        self.transcript_text.insert(tk.END, f"{text}\n\n", "text")
        
        # Configure tags for colors
        self.transcript_text.tag_config("timestamp", foreground="#95A5A6")
        self.transcript_text.tag_config(f"speaker_{speaker_id}", foreground=color, font=('Arial', 11, 'bold'))
        self.transcript_text.tag_config("text", foreground="#ECF0F1")
        
        # Auto-scroll to bottom
        self.transcript_text.see(tk.END)
        
    def update_speakers(self, speakers):
        """Update active speakers display"""
        # Clear existing
        for widget in self.speaker_container.winfo_children():
            widget.destroy()
            
        if not speakers:
            no_speaker_label = ttk.Label(
                self.speaker_container,
                text="No speakers detected",
                foreground="#95A5A6"
            )
            no_speaker_label.pack()
            return
            
        # Add speaker badges
        for speaker in speakers:
            speaker_badge = tk.Frame(
                self.speaker_container,
                bg=speaker.get('color', '#3498DB'),
                padx=10,
                pady=5
            )
            speaker_badge.pack(side=tk.LEFT, padx=5, pady=2)
            
            name_label = tk.Label(
                speaker_badge,
                text=speaker.get('name', f"Speaker {speaker['id'] + 1}"),
                bg=speaker.get('color', '#3498DB'),
                fg='white',
                font=('Arial', 10, 'bold')
            )
            name_label.pack()
            
    def update_audio_level(self, level):
        """Update audio level indicator"""
        # Clear canvas
        self.audio_level_canvas.delete("all")
        
        # Calculate bar width (normalize level to 0-1 range)
        max_level = 5000  # Adjust based on typical audio levels
        normalized_level = min(level / max_level, 1.0)
        
        canvas_width = self.audio_level_canvas.winfo_width()
        bar_width = int(canvas_width * normalized_level)
        
        # Color based on level
        if normalized_level < 0.3:
            color = '#27AE60'  # Green
        elif normalized_level < 0.7:
            color = '#F39C12'  # Orange
        else:
            color = '#E74C3C'  # Red
            
        # Draw bar
        self.audio_level_canvas.create_rectangle(
            0, 0, bar_width, 20,
            fill=color,
            outline=''
        )
        
    def update_status(self, message):
        """Update status message"""
        self.status_label.configure(text=message)
        
    def set_running_state(self, is_running):
        """Update button states based on running state"""
        self.is_running = is_running
        
        if is_running:
            self.start_button.configure(state=tk.DISABLED)
            self.stop_button.configure(state=tk.NORMAL)
            self.snapshot_button.configure(state=tk.NORMAL)
            # Show LIVE indicator
            self.live_indicator.configure(text="🔴 LIVE")
            self.blink_live_indicator()
        else:
            self.start_button.configure(state=tk.NORMAL)
            self.stop_button.configure(state=tk.DISABLED)
            self.snapshot_button.configure(state=tk.DISABLED)
            # Hide LIVE indicator
            self.live_indicator.configure(text="")
            
    def blink_live_indicator(self):
        """Make the live indicator blink for attention"""
        if self.is_running:
            current_text = self.live_indicator.cget("text")
            if current_text == "🔴 LIVE":
                self.live_indicator.configure(text="⚪ LIVE")
            else:
                self.live_indicator.configure(text="🔴 LIVE")
            # Blink every 800ms
            self.root.after(800, self.blink_live_indicator)
            
    def schedule_updates(self):
        """Process queued updates - optimized for continuous streaming"""
        try:
            # Process multiple updates per cycle for smoother streaming
            updates_processed = 0
            max_updates_per_cycle = 5  # Process up to 5 updates at once
            
            while updates_processed < max_updates_per_cycle:
                try:
                    update = self.update_queue.get_nowait()
                    update_type = update.get('type')
                    
                    if update_type == 'video':
                        self.update_video_frame(update['frame'])
                    elif update_type == 'transcript':
                        self.update_transcript(
                            update['text'],
                            update.get('speaker_id', 0),
                            update.get('speaker_name'),
                            update.get('color')
                        )
                    elif update_type == 'speakers':
                        self.update_speakers(update['speakers'])
                    elif update_type == 'audio_level':
                        self.update_audio_level(update['level'])
                    elif update_type == 'status':
                        self.update_status(update['message'])
                    elif update_type == 'state':
                        self.set_running_state(update['is_running'])
                    
                    updates_processed += 1
                    
                except queue.Empty:
                    break
                    
        except Exception as e:
            print(f"Error in schedule_updates: {e}")
            
        # Schedule next update - faster for streaming (20ms instead of 50ms)
        self.root.after(20, self.schedule_updates)
        
    def queue_update(self, update):
        """Queue a GUI update"""
        self.update_queue.put(update)
        
    def show_error(self, title, message):
        """Show error dialog"""
        messagebox.showerror(title, message)
        
    def show_info(self, title, message):
        """Show info dialog"""
        messagebox.showinfo(title, message)

