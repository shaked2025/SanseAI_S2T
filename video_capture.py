"""
Video Capture Module
Handles real-time video capture from camera
"""

import cv2
import numpy as np
import threading
import queue
from datetime import datetime
import os


class VideoCapture:
    """Real-time video capture from camera"""
    
    def __init__(self, camera_index=0, width=640, height=480, fps=30):
        self.camera_index = camera_index
        self.width = width
        self.height = height
        self.fps = fps
        
        self.cap = None
        self.is_capturing = False
        
        # Current frame
        self.current_frame = None
        self.frame_lock = threading.Lock()
        
        # Capture thread
        self.capture_thread = None
        
    def start(self):
        """Start video capture"""
        if self.is_capturing:
            return
            
        try:
            self.cap = cv2.VideoCapture(self.camera_index)
            
            if not self.cap.isOpened():
                print(f"Error: Could not open camera {self.camera_index}")
                return False
                
            # Set resolution
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self.cap.set(cv2.CAP_PROP_FPS, self.fps)
            
            self.is_capturing = True
            
            # Start capture thread
            self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
            self.capture_thread.start()
            
            print(f"Video capture started: {self.width}x{self.height} @ {self.fps}fps")
            return True
            
        except Exception as e:
            print(f"Error starting video capture: {e}")
            return False
            
    def stop(self):
        """Stop video capture"""
        self.is_capturing = False
        
        if self.capture_thread:
            self.capture_thread.join(timeout=2.0)
            
        if self.cap:
            self.cap.release()
            
        print("Video capture stopped")
        
    def _capture_loop(self):
        """Main capture loop running in separate thread"""
        while self.is_capturing:
            ret, frame = self.cap.read()
            
            if ret:
                with self.frame_lock:
                    self.current_frame = frame.copy()
            else:
                print("Warning: Failed to read frame from camera")
                
    def get_frame(self):
        """Get current frame"""
        with self.frame_lock:
            if self.current_frame is not None:
                return self.current_frame.copy()
            return None
            
    def take_snapshot(self, output_dir="snapshots"):
        """Take a snapshot and save it"""
        frame = self.get_frame()
        
        if frame is None:
            print("No frame available for snapshot")
            return None
            
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(output_dir, f"snapshot_{timestamp}.jpg")
        
        # Save frame
        cv2.imwrite(filename, frame)
        print(f"Snapshot saved: {filename}")
        
        return filename
        
    def get_frame_bgr(self):
        """Get frame in BGR format (OpenCV default)"""
        return self.get_frame()
        
    def get_frame_rgb(self):
        """Get frame in RGB format (for display)"""
        frame = self.get_frame()
        if frame is not None:
            return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return None
        
    def cleanup(self):
        """Cleanup resources"""
        self.stop()

