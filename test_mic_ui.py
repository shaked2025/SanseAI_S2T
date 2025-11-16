"""
Simple Microphone Test UI
Just to verify microphone is working before full system
NO CAMERA - MICROPHONE ONLY
"""

import tkinter as tk
from tkinter import ttk
import pyaudio
import numpy as np
import threading
import time


class MicrophoneTestUI:
    """Simple UI to test if microphone is working"""
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Microphone Test")
        self.root.geometry("600x400")
        
        # Audio setup
        self.p = pyaudio.PyAudio()
        self.stream = None
        self.is_recording = False
        self.device_index = 6  # Logitech BRIO
        
        # Create UI
        self.create_ui()
        
    def create_ui(self):
        """Create simple test UI"""
        
        # Title
        title = tk.Label(
            self.root,
            text="🎤 Microphone Test",
            font=('Arial', 20, 'bold'),
            fg='#2C3E50'
        )
        title.pack(pady=20)
        
        # Device info
        try:
            device_info = self.p.get_device_info_by_index(self.device_index)
            info_text = f"Using: {device_info['name']}"
        except:
            info_text = "Device not found"
            
        self.device_label = tk.Label(
            self.root,
            text=info_text,
            font=('Arial', 12),
            fg='#7F8C8D'
        )
        self.device_label.pack(pady=10)
        
        # Instructions
        instructions = tk.Label(
            self.root,
            text="Click START and speak into your microphone.\n"
                 "The audio level bar should move if it's working.",
            font=('Arial', 11),
            justify=tk.CENTER
        )
        instructions.pack(pady=20)
        
        # Audio level bar
        level_frame = tk.Frame(self.root)
        level_frame.pack(pady=20, padx=40, fill=tk.X)
        
        tk.Label(level_frame, text="Audio Level:", font=('Arial', 11)).pack()
        
        self.level_canvas = tk.Canvas(level_frame, height=40, bg='#34495E', highlightthickness=2)
        self.level_canvas.pack(fill=tk.X, pady=10)
        
        self.level_text = tk.Label(
            level_frame,
            text="0",
            font=('Arial', 14, 'bold'),
            fg='#E74C3C'
        )
        self.level_text.pack()
        
        # Status
        self.status_label = tk.Label(
            self.root,
            text="Click START to test",
            font=('Arial', 12),
            fg='#7F8C8D'
        )
        self.status_label.pack(pady=20)
        
        # Start button
        self.start_button = tk.Button(
            self.root,
            text="▶ START TEST",
            command=self.start_test,
            bg='#27AE60',
            fg='white',
            font=('Arial', 16, 'bold'),
            padx=40,
            pady=20,
            cursor='hand2'
        )
        self.start_button.pack(pady=10)
        
        # Stop button
        self.stop_button = tk.Button(
            self.root,
            text="⬛ STOP",
            command=self.stop_test,
            bg='#E74C3C',
            fg='white',
            font=('Arial', 16, 'bold'),
            padx=40,
            pady=20,
            state=tk.DISABLED
        )
        self.stop_button.pack(pady=5)
        
    def start_test(self):
        """Start microphone test"""
        if self.is_recording:
            return
            
        print(f"\nStarting microphone test (device {self.device_index})...")
        
        try:
            self.stream = self.p.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=16000,
                input=True,
                input_device_index=self.device_index,
                frames_per_buffer=1024
            )
            
            self.is_recording = True
            self.start_button.config(state=tk.DISABLED)
            self.stop_button.config(state=tk.NORMAL)
            self.status_label.config(text="🔴 RECORDING - SPEAK NOW!", fg='#E74C3C', font=('Arial', 14, 'bold'))
            
            # Start level monitoring thread
            threading.Thread(target=self.monitor_level, daemon=True).start()
            
            print("✅ Microphone test started - SPEAK NOW!")
            
        except Exception as e:
            print(f"❌ Error starting microphone: {e}")
            self.status_label.config(text=f"ERROR: {str(e)}", fg='#E74C3C')
            
    def stop_test(self):
        """Stop microphone test"""
        self.is_recording = False
        
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
            
        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        self.status_label.config(text="Stopped", fg='#7F8C8D', font=('Arial', 12))
        
        print("Stopped")
        
    def monitor_level(self):
        """Monitor audio level"""
        max_level = 0
        
        while self.is_recording and self.stream:
            try:
                data = self.stream.read(1024, exception_on_overflow=False)
                audio = np.frombuffer(data, dtype=np.int16)
                
                # Calculate RMS level
                rms = int(np.sqrt(np.mean(audio.astype(np.float32) ** 2)))
                
                if rms > max_level:
                    max_level = rms
                    
                # Update UI
                self.update_level_bar(rms, max_level)
                
            except Exception as e:
                print(f"Error: {e}")
                break
                
    def update_level_bar(self, current_level, max_level):
        """Update level bar display"""
        # Clear canvas
        self.level_canvas.delete("all")
        
        # Calculate bar width
        canvas_width = self.level_canvas.winfo_width()
        if canvas_width < 10:
            canvas_width = 500
            
        max_display = 5000
        bar_width = int((current_level / max_display) * canvas_width)
        bar_width = min(bar_width, canvas_width)
        
        # Color based on level
        if current_level > 2000:
            color = '#27AE60'  # Green - good
        elif current_level > 500:
            color = '#F39C12'  # Orange - ok
        else:
            color = '#E74C3C'  # Red - too quiet
            
        # Draw bar
        self.level_canvas.create_rectangle(
            0, 0, bar_width, 40,
            fill=color,
            outline=''
        )
        
        # Update text
        self.level_text.config(text=str(current_level))
        
        # Status
        if current_level > 2000:
            self.status_label.config(text="✅ EXCELLENT - Microphone working great!", fg='#27AE60')
        elif current_level > 1000:
            self.status_label.config(text="✅ GOOD - Microphone working", fg='#27AE60')
        elif current_level > 500:
            self.status_label.config(text="⚠️ QUIET - Speak louder or move closer", fg='#F39C12')
        elif current_level > 100:
            self.status_label.config(text="⚠️ VERY QUIET - Microphone barely working", fg='#E74C3C')
        else:
            self.status_label.config(text="❌ NOT WORKING - No audio detected", fg='#E74C3C')
            
        if max_level > 1000:
            print(f"Audio level: {current_level} (max so far: {max_level}) ✅ WORKING")
            
    def run(self):
        """Run the test UI"""
        self.root.mainloop()
        
        # Cleanup
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        self.p.terminate()


if __name__ == "__main__":
    print("="*60)
    print("  MICROPHONE TEST UI")
    print("="*60)
    print()
    print("This will test if your microphone is working")
    print("NO CAMERA - Just microphone")
    print()
    
    app = MicrophoneTestUI()
    app.run()

