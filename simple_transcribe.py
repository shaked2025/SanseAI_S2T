"""
SIMPLE Speech-to-Text - No complexity, just works
Uses camera microphone, fast and easy
"""

import tkinter as tk
from tkinter import ttk, scrolledtext
import whisper
import numpy as np
from audio_capture import AudioCapture
from video_capture import VideoCapture
from datetime import datetime
import threading
import time
from PIL import Image, ImageTk
import cv2


class SimpleTranscriber:
    """Simple real-time transcription - no speaker ID complications"""
    
    def __init__(self):
        print("="*50)
        print("  SIMPLE SPEECH-TO-TEXT")
        print("="*50)
        print()
        print("Loading Whisper model (tiny for speed)...")
        
        # Use tiny model for speed
        self.model = whisper.load_model("tiny")
        
        # Audio capture with camera mic
        self.audio = AudioCapture(sample_rate=16000, channels=1, device_index=5)
        
        # Video capture
        self.video = VideoCapture(camera_index=0, width=640, height=480)
        
        self.is_running = False
        
        print("✅ Ready!")
        print()
        
    def create_gui(self):
        """Create simple GUI"""
        self.root = tk.Tk()
        self.root.title("Simple Speech-to-Text")
        self.root.geometry("900x700")
        
        # Title
        title = tk.Label(self.root, text="🎤 Simple Speech-to-Text", font=('Arial', 16, 'bold'))
        title.pack(pady=10)
        
        # Video
        self.video_label = tk.Label(self.root, bg='black')
        self.video_label.pack(pady=10)
        
        # Transcript
        transcript_frame = ttk.LabelFrame(self.root, text="Transcript", padding=10)
        transcript_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        self.transcript = scrolledtext.ScrolledText(
            transcript_frame,
            wrap=tk.WORD,
            font=('Arial', 11),
            bg='#2C3E50',
            fg='white',
            height=15
        )
        self.transcript.pack(fill=tk.BOTH, expand=True)
        
        # Buttons
        btn_frame = tk.Frame(self.root)
        btn_frame.pack(pady=10)
        
        self.start_btn = tk.Button(
            btn_frame,
            text="▶ START",
            command=self.start,
            bg='#27AE60',
            fg='white',
            font=('Arial', 14, 'bold'),
            padx=30,
            pady=10
        )
        self.start_btn.pack(side=tk.LEFT, padx=5)
        
        self.stop_btn = tk.Button(
            btn_frame,
            text="⬛ STOP",
            command=self.stop,
            bg='#E74C3C',
            fg='white',
            font=('Arial', 14, 'bold'),
            padx=30,
            pady=10,
            state=tk.DISABLED
        )
        self.stop_btn.pack(side=tk.LEFT, padx=5)
        
        # Status
        self.status = tk.Label(self.root, text="Ready - Click Start", font=('Arial', 10))
        self.status.pack(pady=5)
        
    def start(self):
        """Start transcription"""
        if self.is_running:
            return
            
        self.is_running = True
        self.start_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        self.status.config(text="🔴 RECORDING...")
        
        # Start audio and video
        self.audio.start()
        self.video.start()
        
        # Start threads
        threading.Thread(target=self.video_loop, daemon=True).start()
        threading.Thread(target=self.transcribe_loop, daemon=True).start()
        
        print("Started!")
        
    def stop(self):
        """Stop transcription"""
        self.is_running = False
        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        self.status.config(text="Stopped")
        
        self.audio.stop()
        self.video.stop()
        
        print("Stopped")
        
    def video_loop(self):
        """Update video display"""
        from PIL import Image, ImageTk
        import cv2
        
        while self.is_running:
            frame = self.video.get_frame()
            if frame is not None:
                # Resize
                frame = cv2.resize(frame, (640, 360))
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame)
                imgtk = ImageTk.PhotoImage(image=img)
                self.video_label.imgtk = imgtk
                self.video_label.configure(image=imgtk)
            time.sleep(0.03)
            
    def transcribe_loop(self):
        """Transcribe audio"""
        last_time = time.time()
        
        while self.is_running:
            try:
                # Every 2 seconds
                if time.time() - last_time < 2.0:
                    time.sleep(0.1)
                    continue
                    
                # Get audio
                audio_data = self.audio.get_buffer(duration=3.0)
                
                if len(audio_data) < 16000:  # At least 1 second
                    time.sleep(0.1)
                    continue
                    
                # Check if speech
                rms = np.sqrt(np.mean(audio_data.astype(np.float32) ** 2))
                if rms < 500:
                    time.sleep(0.1)
                    continue
                    
                # Transcribe
                audio_float = audio_data.astype(np.float32) / 32768.0
                result = self.model.transcribe(audio_float, language='en', fp16=False)
                
                if result['text'].strip():
                    timestamp = datetime.now().strftime("%H:%M:%S")
                    text = f"[{timestamp}] {result['text'].strip()}\n\n"
                    
                    self.transcript.insert(tk.END, text)
                    self.transcript.see(tk.END)
                    
                    print(f"📝 {result['text'].strip()}")
                    
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
            self.stop()
        self.audio.cleanup()
        self.video.cleanup()


if __name__ == "__main__":
    app = SimpleTranscriber()
    app.run()

