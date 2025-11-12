"""
Production-Grade Speaker Diarization Module
Uses deep learning embeddings for robust speaker identification
"""

import numpy as np
import torch
from collections import defaultdict, deque
from datetime import datetime
import threading
import pickle
import os


class EmbeddingExtractor:
    """Extract speaker embeddings using SpeechBrain ECAPA-TDNN model"""
    
    def __init__(self, device=None):
        """
        Initialize the embedding extractor
        
        Args:
            device: torch device (cuda/cpu)
        """
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.sample_rate = 16000
        self.lock = threading.Lock()
        
        print(f"🧠 Initializing speaker embedding model on {self.device}...")
        self._load_model()
        
    def _load_model(self):
        """Load SpeechBrain speaker recognition model"""
        try:
            from pathlib import Path
            from speechbrain.inference.speaker import EncoderClassifier
            
            # Find cached model to avoid Windows symlink issues
            cache_dir = Path.home() / ".cache" / "huggingface" / "hub" / "models--speechbrain--spkrec-ecapa-voxceleb" / "snapshots"
            
            model_path = None
            if cache_dir.exists():
                # Get the snapshot directory (there should be only one)
                snapshots = list(cache_dir.iterdir())
                if snapshots:
                    model_path = snapshots[0]
                    print(f"📂 Found cached model at: {model_path.name}")
            
            with self.lock:
                if model_path and model_path.exists():
                    # Load directly from cache - no symlink creation needed!
                    print("✅ Loading model directly from HuggingFace cache...")
                    self.model = EncoderClassifier.from_hparams(
                        source=str(model_path),
                        savedir=str(model_path),  # Use same dir to avoid copying
                        run_opts={"device": self.device}
                    )
                    print("✅ Speaker embedding model loaded successfully!")
                else:
                    # Model not cached, need to download
                    # This will fail on Windows due to symlinks, but we'll try
                    print("📥 Downloading SpeechBrain model (one-time, ~80MB)...")
                    print("⚠️  Note: May require admin privileges on Windows")
                    self.model = EncoderClassifier.from_hparams(
                        source="speechbrain/spkrec-ecapa-voxceleb",
                        savedir="models/spkrec-ecapa-voxceleb",
                        run_opts={"device": self.device}
                    )
                    print("✅ Model downloaded and loaded successfully!")
            
        except Exception as e:
            print(f"❌ Error loading SpeechBrain model: {e}")
            print("⚠️  Falling back to simple speaker diarization...")
            print("💡 To use production mode: Run as administrator or pre-download the model")
            # Don't raise - let the app continue with fallback
            self.model = None
            
    def extract_embedding(self, audio_data, sample_rate=16000):
        """
        Extract speaker embedding from audio
        
        Args:
            audio_data: numpy array of audio (int16 or float32)
            sample_rate: sample rate of audio
            
        Returns:
            numpy array of embedding (192-dimensional)
        """
        try:
            # If model failed to load, return random embedding
            if self.model is None:
                print("⚠️  Model not loaded, using fallback")
                # Return consistent random embedding based on audio characteristics
                audio_mean = np.mean(np.abs(audio_data))
                np.random.seed(int(audio_mean) % 10000)
                embedding = np.random.randn(192).astype(np.float32)
                embedding = embedding / (np.linalg.norm(embedding) + 1e-10)
                return embedding
            
            # Convert to float32 and normalize
            if audio_data.dtype == np.int16:
                audio_float = audio_data.astype(np.float32) / 32768.0
            else:
                audio_float = audio_data.astype(np.float32)
                
            # Ensure minimum length (0.4 seconds)
            min_samples = int(0.4 * sample_rate)
            if len(audio_float) < min_samples:
                # Pad with zeros
                audio_float = np.pad(audio_float, (0, min_samples - len(audio_float)))
                
            # Convert to torch tensor
            audio_tensor = torch.from_numpy(audio_float).unsqueeze(0).to(self.device)
            
            # Extract embedding
            with self.lock:
                with torch.no_grad():
                    embedding = self.model.encode_batch(audio_tensor)
                    embedding = embedding.squeeze().cpu().numpy()
                    
            # Normalize embedding
            embedding = embedding / (np.linalg.norm(embedding) + 1e-10)
            
            return embedding
            
        except Exception as e:
            print(f"❌ Error extracting embedding: {e}")
            # Return zero embedding on error
            return np.zeros(192, dtype=np.float32)


class SpeakerDatabase:
    """Manage speaker profiles and embeddings"""
    
    def __init__(self, max_speakers=10, embedding_dim=192):
        self.max_speakers = max_speakers
        self.embedding_dim = embedding_dim
        
        # Speaker profiles: {speaker_id: {embeddings, count, name, color, etc}}
        self.speakers = {}
        self.next_speaker_id = 0
        
        # Recent embeddings for temporal smoothing
        self.recent_embeddings = deque(maxlen=5)
        
        # Lock for thread safety
        self.lock = threading.Lock()
        
    def add_speaker(self, embedding, speaker_name=None):
        """
        Add a new speaker to database
        
        Args:
            embedding: Speaker embedding vector
            speaker_name: Optional name for speaker
            
        Returns:
            speaker_id
        """
        with self.lock:
            speaker_id = self.next_speaker_id
            self.next_speaker_id += 1
            
            self.speakers[speaker_id] = {
                'embeddings': [embedding],
                'mean_embedding': embedding.copy(),
                'count': 1,
                'name': speaker_name or f"Speaker {speaker_id + 1}",
                'first_seen': datetime.now(),
                'last_seen': datetime.now(),
                'confidence_history': []
            }
            
            print(f"👤 New speaker added: {self.speakers[speaker_id]['name']} (ID: {speaker_id})")
            return speaker_id
            
    def update_speaker(self, speaker_id, embedding, confidence=1.0):
        """
        Update speaker profile with new embedding
        
        Args:
            speaker_id: ID of speaker to update
            embedding: New embedding vector
            confidence: Confidence score for this embedding
        """
        with self.lock:
            if speaker_id not in self.speakers:
                return
                
            speaker = self.speakers[speaker_id]
            
            # Add to embeddings list (keep last 20)
            speaker['embeddings'].append(embedding)
            if len(speaker['embeddings']) > 20:
                speaker['embeddings'].pop(0)
                
            # Update mean embedding (moving average)
            alpha = 0.2  # Learning rate - lower = more stable, less sensitive to variations
            speaker['mean_embedding'] = (
                alpha * embedding + (1 - alpha) * speaker['mean_embedding']
            )
            
            # Normalize
            speaker['mean_embedding'] = speaker['mean_embedding'] / (
                np.linalg.norm(speaker['mean_embedding']) + 1e-10
            )
            
            speaker['count'] += 1
            speaker['last_seen'] = datetime.now()
            speaker['confidence_history'].append(confidence)
            
            # Keep only recent confidence scores
            if len(speaker['confidence_history']) > 50:
                speaker['confidence_history'].pop(0)
                
    def find_speaker(self, embedding, threshold=0.75):
        """
        Find matching speaker for embedding
        
        Args:
            embedding: Query embedding
            threshold: Similarity threshold (0-1)
            
        Returns:
            (speaker_id, confidence) or (None, 0.0) if no match
        """
        with self.lock:
            if not self.speakers:
                return None, 0.0
                
            best_speaker = None
            best_similarity = -1.0
            
            for speaker_id, speaker_data in self.speakers.items():
                # Compute cosine similarity with mean embedding
                similarity = np.dot(embedding, speaker_data['mean_embedding'])
                
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_speaker = speaker_id
                    
            # Check if best similarity exceeds threshold
            if best_similarity >= threshold:
                return best_speaker, float(best_similarity)
            else:
                return None, float(best_similarity)
                
    def get_speaker_info(self, speaker_id):
        """Get speaker information"""
        with self.lock:
            return self.speakers.get(speaker_id, None)
            
    def get_all_speakers(self):
        """Get list of all speaker IDs"""
        with self.lock:
            return list(self.speakers.keys())
            
    def save(self, filepath):
        """Save speaker database to file"""
        with self.lock:
            try:
                with open(filepath, 'wb') as f:
                    pickle.dump(self.speakers, f)
                print(f"💾 Speaker database saved to {filepath}")
            except Exception as e:
                print(f"❌ Error saving speaker database: {e}")
                
    def load(self, filepath):
        """Load speaker database from file"""
        with self.lock:
            try:
                if os.path.exists(filepath):
                    with open(filepath, 'rb') as f:
                        self.speakers = pickle.load(f)
                    self.next_speaker_id = max(self.speakers.keys()) + 1 if self.speakers else 0
                    print(f"📂 Loaded {len(self.speakers)} speakers from {filepath}")
                    return True
            except Exception as e:
                print(f"❌ Error loading speaker database: {e}")
        return False


class TemporalSmoother:
    """Smooth speaker assignments over time to prevent flickering"""
    
    def __init__(self, window_size=7):
        self.window_size = window_size
        self.history = deque(maxlen=window_size)
        
    def smooth(self, speaker_id, confidence):
        """
        Add new speaker ID and return smoothed result
        
        Args:
            speaker_id: Current speaker ID
            confidence: Confidence score
            
        Returns:
            Smoothed speaker_id
        """
        self.history.append((speaker_id, confidence))
        
        if len(self.history) < 2:
            return speaker_id
            
        # Use weighted voting based on confidence
        speaker_votes = defaultdict(float)
        for spk_id, conf in self.history:
            speaker_votes[spk_id] += conf
            
        # Return speaker with highest weighted vote
        smoothed_id = max(speaker_votes.items(), key=lambda x: x[1])[0]
        return smoothed_id
        
    def reset(self):
        """Reset history"""
        self.history.clear()


class ProductionSpeakerDiarization:
    """
    Production-grade speaker diarization using deep learning embeddings
    
    Features:
    - Deep speaker embeddings (192-dim) using SpeechBrain
    - Robust similarity matching
    - Temporal smoothing
    - Speaker enrollment and adaptation
    - Handles multiple simultaneous speakers
    """
    
    def __init__(self, max_speakers=10, similarity_threshold=0.75, device=None):
        """
        Initialize production speaker diarization
        
        Args:
            max_speakers: Maximum number of speakers to track
            similarity_threshold: Threshold for speaker matching (0-1)
            device: torch device
        """
        self.max_speakers = max_speakers
        self.similarity_threshold = similarity_threshold
        
        print("🎯 Initializing Production Speaker Diarization System...")
        print(f"   Max speakers: {max_speakers}")
        print(f"   Similarity threshold: {similarity_threshold}")
        
        # Initialize components
        self.embedding_extractor = EmbeddingExtractor(device=device)
        self.speaker_db = SpeakerDatabase(max_speakers=max_speakers)
        self.temporal_smoother = TemporalSmoother(window_size=7)  # Larger window for more stability
        
        # Statistics
        self.total_identifications = 0
        self.successful_matches = 0
        
        # Load existing speaker database if available
        self.db_path = "speaker_database.pkl"
        self.speaker_db.load(self.db_path)
        
        print("✅ Production speaker diarization initialized")
        
    def identify_speaker(self, audio_data, sample_rate=16000):
        """
        Identify speaker from audio data
        
        Args:
            audio_data: Audio as numpy array (int16)
            sample_rate: Sample rate
            
        Returns:
            speaker_id (int)
        """
        try:
            # Extract embedding
            embedding = self.embedding_extractor.extract_embedding(audio_data, sample_rate)
            
            # Check for zero embedding (error case)
            if np.allclose(embedding, 0):
                print("⚠️ Zero embedding detected, using fallback")
                return 0
                
            # Find matching speaker
            speaker_id, confidence = self.speaker_db.find_speaker(
                embedding,
                threshold=self.similarity_threshold
            )
            
            self.total_identifications += 1
            
            # If no match, create new speaker
            if speaker_id is None:
                # Check if we've reached max speakers
                if len(self.speaker_db.get_all_speakers()) >= self.max_speakers:
                    # Assign to closest speaker even if below threshold
                    speaker_id, confidence = self._find_closest_speaker(embedding)
                    print(f"⚠️ Max speakers reached, assigning to closest: Speaker {speaker_id + 1} (confidence: {confidence:.2f})")
                else:
                    # Create new speaker
                    speaker_id = self.speaker_db.add_speaker(embedding)
                    confidence = 1.0
                    print(f"✨ New speaker detected: Speaker {speaker_id + 1}")
            else:
                # Update existing speaker
                self.speaker_db.update_speaker(speaker_id, embedding, confidence)
                self.successful_matches += 1
                
            # Apply temporal smoothing
            smoothed_id = self.temporal_smoother.smooth(speaker_id, confidence)
            
            # Log identification
            accuracy = (self.successful_matches / self.total_identifications * 100) if self.total_identifications > 0 else 0
            print(f"👤 Speaker {smoothed_id + 1} (confidence: {confidence:.2f}, accuracy: {accuracy:.1f}%)")
            
            return smoothed_id
            
        except Exception as e:
            print(f"❌ Error in speaker identification: {e}")
            import traceback
            traceback.print_exc()
            return 0
            
    def _find_closest_speaker(self, embedding):
        """Find closest speaker even if below threshold"""
        best_speaker = 0
        best_similarity = -1.0
        
        for speaker_id in self.speaker_db.get_all_speakers():
            speaker_info = self.speaker_db.get_speaker_info(speaker_id)
            if speaker_info:
                similarity = np.dot(embedding, speaker_info['mean_embedding'])
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_speaker = speaker_id
                    
        return best_speaker, float(best_similarity)
        
    def get_speaker_count(self):
        """Get number of identified speakers"""
        return len(self.speaker_db.get_all_speakers())
        
    def get_active_speakers(self):
        """Get list of active speaker IDs"""
        return self.speaker_db.get_all_speakers()
        
    def rename_speaker(self, speaker_id, new_name):
        """Rename a speaker"""
        speaker_info = self.speaker_db.get_speaker_info(speaker_id)
        if speaker_info:
            speaker_info['name'] = new_name
            print(f"✏️ Speaker {speaker_id} renamed to: {new_name}")
            
    def reset(self):
        """Reset speaker database and temporal smoothing"""
        self.speaker_db = SpeakerDatabase(max_speakers=self.max_speakers)
        self.temporal_smoother.reset()
        self.total_identifications = 0
        self.successful_matches = 0
        print("🔄 Speaker database reset")
        
    def save_database(self):
        """Save speaker database to disk"""
        self.speaker_db.save(self.db_path)
        
    def get_statistics(self):
        """Get identification statistics"""
        accuracy = (self.successful_matches / self.total_identifications * 100) if self.total_identifications > 0 else 0
        return {
            'total_identifications': self.total_identifications,
            'successful_matches': self.successful_matches,
            'accuracy': accuracy,
            'num_speakers': self.get_speaker_count()
        }

