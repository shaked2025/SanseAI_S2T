"""
Speech-to-Text Module
Handles transcription using OpenAI Whisper (running locally)
"""

import whisper
import numpy as np
import threading
import queue
import time
from datetime import datetime
import torch


class SpeechToText:
    """Speech-to-text engine using Whisper"""
    
    def __init__(self, model_size="base", language="en", device=None):
        """
        Initialize Whisper model
        
        Args:
            model_size: Model size (tiny, base, small, medium, large)
            language: Language code (en for English)
            device: Device to run on (cuda, cpu, or None for auto)
        """
        self.model_size = model_size
        self.language = language
        
        # Determine device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        print(f"Loading Whisper model '{model_size}' on {self.device}...")
        
        try:
            self.model = whisper.load_model(model_size, device=self.device)
            print(f"Whisper model loaded successfully")
        except Exception as e:
            print(f"Error loading Whisper model: {e}")
            raise
            
        # Processing queue
        self.transcription_queue = queue.Queue()
        self.is_processing = False
        self.process_thread = None
        
    def transcribe(self, audio_data, sample_rate=16000):
        """
        Transcribe audio data
        
        Args:
            audio_data: Audio as numpy array (int16)
            sample_rate: Sample rate of audio
            
        Returns:
            Dictionary with transcription results
        """
        try:
            # Convert to float32 and normalize
            audio_float = audio_data.astype(np.float32) / 32768.0
            
            # Resample if needed (Whisper expects 16kHz)
            if sample_rate != 16000:
                print(f"Warning: Audio sample rate is {sample_rate}, but Whisper expects 16000")
                # Simple resampling (for production, use proper resampling)
                
            # Transcribe
            result = self.model.transcribe(
                audio_float,
                language=self.language,
                task="transcribe",
                fp16=(self.device == "cuda"),
                verbose=False
            )
            
            return {
                'text': result['text'].strip(),
                'segments': result.get('segments', []),
                'language': result.get('language', self.language),
                'timestamp': datetime.now()
            }
            
        except Exception as e:
            print(f"Error during transcription: {e}")
            return {
                'text': '',
                'segments': [],
                'error': str(e),
                'timestamp': datetime.now()
            }
    
    def transcribe_with_timestamps(self, audio_data, sample_rate=16000):
        """
        Transcribe audio with word-level timestamps
        
        Args:
            audio_data: Audio as numpy array (int16)
            sample_rate: Sample rate of audio
            
        Returns:
            List of segments with timestamps
        """
        result = self.transcribe(audio_data, sample_rate)
        
        segments = []
        for seg in result.get('segments', []):
            segments.append({
                'start': seg['start'],
                'end': seg['end'],
                'text': seg['text'].strip(),
                'confidence': seg.get('no_speech_prob', 0.0)
            })
            
        return segments
    
    def start_async_processing(self, audio_queue, callback):
        """
        Start async processing of audio chunks
        
        Args:
            audio_queue: Queue to pull audio from
            callback: Function to call with results
        """
        if self.is_processing:
            return
            
        self.is_processing = True
        self.process_thread = threading.Thread(
            target=self._process_loop,
            args=(audio_queue, callback),
            daemon=True
        )
        self.process_thread.start()
        
    def stop_async_processing(self):
        """Stop async processing"""
        self.is_processing = False
        if self.process_thread:
            self.process_thread.join(timeout=5.0)
            
    def _process_loop(self, audio_queue, callback):
        """Main processing loop for async transcription"""
        print("Async transcription processing started")
        
        while self.is_processing:
            try:
                # Get audio from queue (with timeout)
                audio_data = audio_queue.get(timeout=0.5)
                
                if audio_data is None or len(audio_data) == 0:
                    continue
                    
                # Transcribe
                result = self.transcribe(audio_data)
                
                # Call callback with result
                if callback and result['text']:
                    callback(result)
                    
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error in async processing loop: {e}")
                
        print("Async transcription processing stopped")


class TranscriptManager:
    """Manages transcription history and display"""
    
    def __init__(self, max_entries=100):
        self.max_entries = max_entries
        self.transcripts = []
        self.lock = threading.Lock()
        
    def add_transcript(self, text, speaker_id=None, timestamp=None):
        """Add a new transcript entry"""
        if timestamp is None:
            timestamp = datetime.now()
            
        entry = {
            'text': text,
            'speaker': speaker_id if speaker_id is not None else 0,
            'timestamp': timestamp,
            'id': len(self.transcripts)
        }
        
        with self.lock:
            self.transcripts.append(entry)
            
            # Keep only last N entries
            if len(self.transcripts) > self.max_entries:
                self.transcripts = self.transcripts[-self.max_entries:]
                
    def get_recent(self, count=20):
        """Get recent transcript entries"""
        with self.lock:
            return self.transcripts[-count:] if self.transcripts else []
            
    def get_all(self):
        """Get all transcript entries"""
        with self.lock:
            return self.transcripts.copy()
            
    def clear(self):
        """Clear all transcripts"""
        with self.lock:
            self.transcripts = []
            
    def export_to_text(self, filename):
        """Export transcripts to text file"""
        with self.lock:
            with open(filename, 'w', encoding='utf-8') as f:
                for entry in self.transcripts:
                    timestamp_str = entry['timestamp'].strftime("%Y-%m-%d %H:%M:%S")
                    speaker = f"Speaker {entry['speaker']}"
                    f.write(f"[{timestamp_str}] {speaker}: {entry['text']}\n")
                    
        print(f"Transcripts exported to {filename}")

