"""
Audio Capture Module
Handles real-time audio recording from microphone
"""

import pyaudio
import numpy as np
import threading
import queue
import wave
from collections import deque
import time


class AudioCapture:
    """Real-time audio capture with buffering"""
    
    def __init__(self, sample_rate=16000, channels=1, chunk_size=1024, device_index=None):
        self.sample_rate = sample_rate
        self.channels = channels
        self.chunk_size = chunk_size
        self.format = pyaudio.paInt16
        self.device_index = device_index  # None = use default, or specify device number
        
        self.audio = pyaudio.PyAudio()
        self.stream = None
        self.is_recording = False
        
        # Queue for audio chunks
        self.audio_queue = queue.Queue()
        
        # Buffer for accumulating audio
        self.buffer = deque(maxlen=int(sample_rate * 10))  # 10 seconds max
        
        # Thread for recording
        self.record_thread = None
        
    def start(self):
        """Start audio capture"""
        if self.is_recording:
            return
            
        try:
            # Show which device we're using
            if self.device_index is not None:
                device_info = self.audio.get_device_info_by_index(self.device_index)
                print(f"🎤 Using microphone: {device_info['name']} (index {self.device_index})")
            else:
                default_device = self.audio.get_default_input_device_info()
                print(f"🎤 Using default microphone: {default_device['name']}")
                self.device_index = default_device['index']
            
            self.stream = self.audio.open(
                format=self.format,
                channels=self.channels,
                rate=self.sample_rate,
                input=True,
                input_device_index=self.device_index,
                frames_per_buffer=self.chunk_size,
                stream_callback=self._audio_callback
            )
            
            self.is_recording = True
            self.stream.start_stream()
            print(f"✅ Audio capture started: {self.sample_rate}Hz, {self.channels} channel(s)")
            
        except Exception as e:
            print(f"Error starting audio capture: {e}")
            self.is_recording = False
            
    def stop(self):
        """Stop audio capture"""
        self.is_recording = False
        
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
            
        print("Audio capture stopped")
        
    def _audio_callback(self, in_data, frame_count, time_info, status):
        """Callback for audio stream"""
        if status:
            print(f"Audio callback status: {status}")
            
        # Convert bytes to numpy array
        audio_data = np.frombuffer(in_data, dtype=np.int16)
        
        # Add to buffer
        self.buffer.extend(audio_data)
        
        # Put in queue for processing
        self.audio_queue.put(audio_data.copy())
        
        return (in_data, pyaudio.paContinue)
    
    def get_audio_chunk(self, timeout=0.1):
        """Get next audio chunk from queue"""
        try:
            return self.audio_queue.get(timeout=timeout)
        except queue.Empty:
            return None
            
    def get_buffer(self, duration=3.0):
        """Get audio buffer of specified duration"""
        num_samples = int(self.sample_rate * duration)
        
        if len(self.buffer) < num_samples:
            num_samples = len(self.buffer)
            
        if num_samples == 0:
            return np.array([], dtype=np.int16)
            
        # Get last N samples
        buffer_array = np.array(list(self.buffer)[-num_samples:], dtype=np.int16)
        return buffer_array
    
    def clear_queue(self):
        """Clear the audio queue"""
        while not self.audio_queue.empty():
            try:
                self.audio_queue.get_nowait()
            except queue.Empty:
                break
                
    def get_audio_level(self):
        """Get current audio level (RMS)"""
        if len(self.buffer) == 0:
            return 0
            
        # Get last 0.1 seconds
        recent = list(self.buffer)[-int(self.sample_rate * 0.1):]
        if not recent:
            return 0
            
        rms = np.sqrt(np.mean(np.square(recent)))
        return rms
        
    def cleanup(self):
        """Cleanup resources"""
        self.stop()
        self.audio.terminate()


class VoiceActivityDetector:
    """Simple voice activity detection"""
    
    def __init__(self, sample_rate=16000, threshold=500, min_duration=0.3):
        self.sample_rate = sample_rate
        self.threshold = threshold
        self.min_duration = min_duration
        self.min_samples = int(sample_rate * min_duration)
        
    def is_speech(self, audio_data):
        """Detect if audio contains speech"""
        if len(audio_data) < self.min_samples:
            return False
            
        # Calculate RMS energy
        rms = np.sqrt(np.mean(np.square(audio_data.astype(float))))
        
        # Simple threshold-based detection
        return rms > self.threshold
    
    def get_speech_segments(self, audio_data, window_size=0.03):
        """Get segments of audio that contain speech"""
        window_samples = int(self.sample_rate * window_size)
        speech_mask = []
        
        for i in range(0, len(audio_data), window_samples):
            chunk = audio_data[i:i + window_samples]
            if len(chunk) < window_samples:
                break
                
            rms = np.sqrt(np.mean(np.square(chunk.astype(float))))
            speech_mask.append(rms > self.threshold)
            
        return speech_mask

