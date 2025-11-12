"""
Speaker Diarization Module
Handles identification and separation of multiple speakers
"""

import numpy as np
import torch
from collections import defaultdict
import threading
import time
from datetime import datetime


class SimpleSpeakerDiarization:
    """
    Simplified speaker diarization using audio features
    
    Note: For production use, consider using pyannote.audio, but this
    implementation provides a basic solution without additional model downloads.
    """
    
    def __init__(self, min_speakers=1, max_speakers=5):
        self.min_speakers = min_speakers
        self.max_speakers = max_speakers
        
        # Speaker profiles (simplified clustering)
        self.speaker_profiles = []
        self.speaker_history = []
        
        # Similarity threshold for speaker matching  
        self.similarity_threshold = 0.82  # Raised for better consistency
        
    def extract_features(self, audio_data, sample_rate=16000):
        """
        Extract simple audio features for speaker identification
        
        Args:
            audio_data: Audio as numpy array (int16 or float32)
            sample_rate: Sample rate
            
        Returns:
            Feature vector
        """
        # Convert to float if needed
        if audio_data.dtype == np.int16:
            audio_float = audio_data.astype(np.float32) / 32768.0
        else:
            audio_float = audio_data.astype(np.float32)
            
        # Extract basic features
        features = []
        
        # 1. Zero crossing rate
        zcr = np.sum(np.abs(np.diff(np.sign(audio_float)))) / (2 * len(audio_float))
        features.append(zcr)
        
        # 2. Energy
        energy = np.sum(audio_float ** 2) / len(audio_float)
        features.append(energy)
        
        # 3. Spectral features (simplified)
        fft = np.fft.rfft(audio_float)
        magnitude = np.abs(fft)
        
        # Spectral centroid
        freqs = np.fft.rfftfreq(len(audio_float), 1/sample_rate)
        spectral_centroid = np.sum(freqs * magnitude) / (np.sum(magnitude) + 1e-10)
        features.append(spectral_centroid)
        
        # Spectral spread
        spectral_spread = np.sqrt(np.sum(((freqs - spectral_centroid) ** 2) * magnitude) / (np.sum(magnitude) + 1e-10))
        features.append(spectral_spread)
        
        # 4. Pitch estimate (simplified)
        autocorr = np.correlate(audio_float, audio_float, mode='full')
        autocorr = autocorr[len(autocorr)//2:]
        
        # Find first peak after zero
        peaks = []
        for i in range(1, min(len(autocorr)-1, sample_rate//50)):  # Search up to 50 Hz
            if autocorr[i] > autocorr[i-1] and autocorr[i] > autocorr[i+1]:
                peaks.append(i)
                
        if peaks:
            pitch_lag = peaks[0]
            pitch_freq = sample_rate / pitch_lag
        else:
            pitch_freq = 0
            
        features.append(pitch_freq)
        
        return np.array(features)
    
    def identify_speaker(self, audio_data, sample_rate=16000):
        """
        Identify which speaker is speaking
        
        Args:
            audio_data: Audio segment
            sample_rate: Sample rate
            
        Returns:
            Speaker ID (int)
        """
        # Extract features
        features = self.extract_features(audio_data, sample_rate)
        
        # Normalize features
        features = features / (np.linalg.norm(features) + 1e-10)
        
        # If no speakers yet, create first speaker
        if len(self.speaker_profiles) == 0:
            self.speaker_profiles.append({
                'id': 0,
                'features': features,
                'count': 1,
                'last_seen': datetime.now()
            })
            return 0
            
        # Find most similar speaker
        max_similarity = -1
        best_speaker = -1
        
        for i, profile in enumerate(self.speaker_profiles):
            # Cosine similarity
            similarity = np.dot(features, profile['features'])
            
            if similarity > max_similarity:
                max_similarity = similarity
                best_speaker = i
                
        # If similarity is high enough, assign to existing speaker
        if max_similarity > self.similarity_threshold:
            # Update speaker profile (moving average)
            alpha = 0.3  # Learning rate
            self.speaker_profiles[best_speaker]['features'] = (
                alpha * features + (1 - alpha) * self.speaker_profiles[best_speaker]['features']
            )
            self.speaker_profiles[best_speaker]['count'] += 1
            self.speaker_profiles[best_speaker]['last_seen'] = datetime.now()
            return best_speaker
            
        # Otherwise, create new speaker if under max
        if len(self.speaker_profiles) < self.max_speakers:
            new_id = len(self.speaker_profiles)
            self.speaker_profiles.append({
                'id': new_id,
                'features': features,
                'count': 1,
                'last_seen': datetime.now()
            })
            return new_id
            
        # Otherwise, assign to closest speaker
        return best_speaker
    
    def get_active_speakers(self):
        """Get list of currently active speakers"""
        return [p['id'] for p in self.speaker_profiles]
    
    def get_speaker_count(self):
        """Get number of identified speakers"""
        return len(self.speaker_profiles)
    
    def reset(self):
        """Reset speaker profiles"""
        self.speaker_profiles = []
        self.speaker_history = []
        print("Speaker profiles reset")


class AdvancedSpeakerDiarization:
    """
    Advanced speaker diarization using pyannote.audio
    
    This requires downloading pretrained models and accepts their terms of use.
    """
    
    def __init__(self, min_speakers=1, max_speakers=5):
        self.min_speakers = min_speakers
        self.max_speakers = max_speakers
        self.pipeline = None
        self.available = False
        
        try:
            from pyannote.audio import Pipeline
            
            # Note: This requires authentication token from Hugging Face
            # Users need to accept model terms and get token from:
            # https://huggingface.co/pyannote/speaker-diarization
            
            print("Advanced diarization requires Hugging Face authentication")
            print("Please visit: https://huggingface.co/pyannote/speaker-diarization")
            print("Falling back to simple diarization...")
            
            self.available = False
            
        except ImportError:
            print("pyannote.audio not available, using simple diarization")
            self.available = False
    
    def diarize(self, audio_file_path):
        """
        Perform speaker diarization on audio file
        
        Args:
            audio_file_path: Path to audio file
            
        Returns:
            Diarization results with speaker segments
        """
        if not self.available:
            return None
            
        try:
            diarization = self.pipeline(audio_file_path)
            
            segments = []
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                segments.append({
                    'start': turn.start,
                    'end': turn.end,
                    'speaker': speaker
                })
                
            return segments
            
        except Exception as e:
            print(f"Error during diarization: {e}")
            return None


class SpeakerManager:
    """Manages speaker information and colors for display"""
    
    def __init__(self):
        self.speakers = {}
        self.colors = [
            "#FF6B6B",  # Red
            "#4ECDC4",  # Teal
            "#45B7D1",  # Blue
            "#FFA07A",  # Light Salmon
            "#98D8C8",  # Mint
            "#FFD93D",  # Yellow
            "#B19CD9",  # Purple
            "#FF8B94",  # Pink
        ]
        
    def get_speaker_info(self, speaker_id):
        """Get or create speaker info"""
        if speaker_id not in self.speakers:
            self.speakers[speaker_id] = {
                'id': speaker_id,
                'name': f"Speaker {speaker_id + 1}",
                'color': self.colors[speaker_id % len(self.colors)],
                'count': 0,
                'first_seen': datetime.now(),
                'last_seen': datetime.now()
            }
            
        self.speakers[speaker_id]['count'] += 1
        self.speakers[speaker_id]['last_seen'] = datetime.now()
        
        return self.speakers[speaker_id]
    
    def get_all_speakers(self):
        """Get all speaker info"""
        return list(self.speakers.values())
    
    def rename_speaker(self, speaker_id, new_name):
        """Rename a speaker"""
        if speaker_id in self.speakers:
            self.speakers[speaker_id]['name'] = new_name
            
    def clear(self):
        """Clear all speaker data"""
        self.speakers = {}

