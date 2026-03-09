"""
Real-time stress prediction module for processing audio chunks and predicting stress levels.
Uses machine learning models, feature extraction, and audio filtering for real-time predictions.
"""

import torch
import numpy as np
import joblib
import os
import collections
import time
import warnings
from typing import List, Dict, Optional, Any
from .config import CONFIG
from .stresslstm import EnhancedStressLSTM
from .features import AudioFeatures
from .utils import passes_filter, ensure_audio_contiguous, validate_audio_chunk
from .embedding import AudioEmbedding


class RealTimeStressPredictor:
    """
    Real-time stress predictor that processes audio chunks and predicts stress levels.
    
    Combines embedding features and audio features using a trained LSTM model
    to provide real-time stress detection with smoothing and filtering.
    """
    
    def __init__(self, 
                 model_path: str = None,
                 scaler_path: str = None,
                 embedding_model_instance: Optional[AudioEmbedding] = None,
                 feature_set: str = None,
                 chunk_duration_s: float = None,
                 sample_rate: int = None,
                 smoothing_window_size: int = None,
                 buffer_duration_s: float = None,
                 hop_duration_s: float = None):
        """
        Initialize the Real-Time Stress Predictor.
        
        Args:
            model_path: Path to the trained LSTM model
            scaler_path: Path to the feature scaler
            embedding_model_instance: AudioEmbedding instance (optional, will create if None)
            feature_set: Feature set type for audio feature extraction
            chunk_duration_s: Duration of audio chunks in seconds
            sample_rate: Audio sampling rate
            smoothing_window_size: Size of smoothing window for probability averaging
            buffer_duration_s: Duration of audio buffer for real-time processing
            hop_duration_s: Hop duration for sliding window processing
        """
        # Use configuration defaults
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Predictor using device: {self.device}")
        
        self.sample_rate = sample_rate or CONFIG.audio.sample_rate
        self.feature_set = feature_set or CONFIG.features.feature_set
        self.chunk_duration_s = chunk_duration_s or CONFIG.audio.chunk_duration
        self.hop_duration_s = hop_duration_s or CONFIG.audio.hop_duration
        self.smoothing_window_size = smoothing_window_size or CONFIG.predictor.smoothing_window_size
        self.buffer_duration_s = buffer_duration_s or CONFIG.audio.buffer_duration
        
        # Initialize paths
        self.model_path = model_path or CONFIG.predictor.model_path
        self.scaler_path = scaler_path or CONFIG.predictor.scaler_path
        
        # Calculate sample-based parameters
        self.chunk_samples = int(self.chunk_duration_s * self.sample_rate)
        self.hop_samples = int(self.hop_duration_s * self.sample_rate)
        self.buffer_max_samples = int(max(self.buffer_duration_s, self.chunk_duration_s) * self.sample_rate)
        
        # Initialize buffers and state
        self.audio_buffer = collections.deque(maxlen=self.buffer_max_samples)
        self.samples_since_last_pred = 0
        self.prob_history = collections.deque(maxlen=self.smoothing_window_size)
        
        # Initialize components
        self.feature_extractor = AudioFeatures(
            sample_rate=self.sample_rate,
            feature_set=self.feature_set
        )
        
        # Initialize or use provided embedding model
        if embedding_model_instance is not None:
            self.embedding_model = embedding_model_instance
        else:
            self.embedding_model = AudioEmbedding(device=self.device)
            if not self.embedding_model.load_model():
                raise RuntimeError("Failed to load embedding model")
        
        # Determine dimensions
        self.embedding_dim = CONFIG.embedding.embedding_dim
        self.audio_features_dim = self._determine_audio_features_dim()
        
        if self.audio_features_dim == 0:
            raise ValueError("Could not determine audio feature dimension. Required for prediction.")
        
        print(f"Using config: EmbedDim={self.embedding_dim}, AudioFeatDim={self.audio_features_dim}")
        
        # Load models and scaler
        self.model = self._load_model()
        self.scaler = self._load_scaler()
        
        # Essential checks
        if self.model is None:
            raise RuntimeError(f"Failed to load LSTM model from {self.model_path}")
        if self.scaler is None:
            raise RuntimeError(f"Scaler failed to load from {self.scaler_path}, but is required.")
        
        # Validate scaler consistency
        if hasattr(self.scaler, 'n_features_in_') and self.scaler.n_features_in_ != self.audio_features_dim:
            warnings.warn(f"Scaler expected {self.scaler.n_features_in_} features, "
                         f"but found {self.audio_features_dim} in names file. Check consistency!", RuntimeWarning)
        
        print("RealTimeStressPredictor initialized successfully.")
    
    def _determine_audio_features_dim(self) -> int:
        """
        Determines the audio features dimension from the saved feature names file.
        
        Returns:
            Number of audio features
        """
        return self.feature_extractor.get_features_number()
    
    def _load_model(self) -> Optional[EnhancedStressLSTM]:
        """
        Loads the trained LSTM model.
        
        Returns:
            Loaded model or None if loading fails
        """
        if not self.model_path or not os.path.exists(self.model_path):
            print(f"Predictor: Model path '{self.model_path}' not found.")
            return None
        
        try:
            saved_object = torch.load(self.model_path, map_location=self.device)
            state_dict = None
            
            # Handle different saving formats
            if isinstance(saved_object, dict) and 'model_state_dict' in saved_object:
                state_dict = saved_object['model_state_dict']
                print("Predictor: Loaded model format: Dictionary with 'model_state_dict'.")
            elif isinstance(saved_object, dict) and all(isinstance(k, str) for k in saved_object.keys()):
                state_dict = saved_object
                print("Predictor: Loaded model format: Dictionary (assumed state_dict).")
            else:
                state_dict = saved_object
                print("Predictor: Loaded model format: Object (assumed state_dict directly).")
            
            if state_dict is None:
                print("Predictor: Could not extract model state_dict from file.")
                return None
            
            # Instantiate model with determined dimensions
            model = EnhancedStressLSTM(
                embedding_dim=self.embedding_dim,
                audio_features_dim=self.audio_features_dim
            )
            model.load_state_dict(state_dict)
            model.to(self.device)
            model.eval()
            
            print("Predictor: Model weights loaded successfully.")
            return model
            
        except Exception as e:
            print(f"Predictor: Error loading model from {self.model_path}: {e}")
            return None
    
    def _load_scaler(self) -> Optional[Any]:
        """
        Loads the feature scaler.
        
        Returns:
            Loaded scaler or None if loading fails
        """
        if self.scaler_path and os.path.exists(self.scaler_path):
            try:
                scaler = joblib.load(self.scaler_path)
                print(f"Predictor: Scaler loaded from {self.scaler_path}")
                return scaler
            except Exception as e:
                print(f"Predictor: Error loading scaler from {self.scaler_path}: {e}")
        else:
            print(f"Predictor: Scaler path '{self.scaler_path}' not provided or not found.")
        return None
    
    def _get_smoothed_prob(self) -> float:
        """
        Calculates moving average over probability history.
        
        Returns:
            Smoothed probability or NaN if no valid probabilities
        """
        valid_probs = [p for p in self.prob_history if p is not None and not np.isnan(p)]
        return np.mean(valid_probs) if valid_probs else np.nan
    
    def _extract_features_and_embedding(self, audio_chunk: np.ndarray) -> tuple:
        """
        Extracts both embedding and audio features from an audio chunk.
        
        Args:
            audio_chunk: Audio chunk as numpy array
            
        Returns:
            Tuple of (embedding_tensor, audio_features_tensor) or (None, None) if extraction fails
        """
        try:
            # Extract embedding
            embedding = self.embedding_model.extract_embedding_from_numpy(audio_chunk, self.sample_rate)
            if embedding is None:
                raise RuntimeError("Embedding extraction failed")
            
            emb_tensor = embedding.unsqueeze(0).to(self.device)  # Add sequence dimension
            
            # Extract audio features
            if self.scaler is None:
                raise RuntimeError("Scaler not loaded")
            
            raw_audio_features, _ = self.feature_extractor.extract_audio_features(
                audio_chunk, self.sample_rate, self.feature_set
            )
            
            if raw_audio_features is None:
                raise RuntimeError("Feature extraction failed")
            
            if raw_audio_features.shape[0] != self.audio_features_dim:
                raise RuntimeError(f"Feature dim mismatch ({raw_audio_features.shape[0]} vs expected {self.audio_features_dim})")
            
            # Scale features
            scaled_audio_features = self.scaler.transform(raw_audio_features.reshape(1, -1))[0]
            audio_feat_tensor = torch.from_numpy(scaled_audio_features).unsqueeze(0).unsqueeze(0).to(self.device)
            
            return emb_tensor, audio_feat_tensor
            
        except Exception as e:
            print(f"RT Warn: Error during feature/embedding extraction: {e}")
            return None, None
    
    def _predict_stress_probability(self, embedding_tensor: torch.Tensor, audio_features_tensor: torch.Tensor) -> float:
        """
        Predicts stress probability from embedding and audio features.
        
        Args:
            embedding_tensor: Embedding features tensor
            audio_features_tensor: Audio features tensor
            
        Returns:
            Stress probability (0.0 to 1.0) or NaN if prediction fails
        """
        try:
            with torch.no_grad():
                outputs = self.model(x_embedding=embedding_tensor, x_audio=audio_features_tensor)
                probability = torch.sigmoid(outputs).item()
                return probability
        except Exception as e:
            print(f"RT Warn: Error during prediction: {e}")
            return np.nan
    
    def process_audio_chunk(self, new_audio_samples_int16: np.ndarray) -> List[Dict[str, Any]]:
        """
        Processes a new chunk of raw audio samples and returns prediction results.
        
        Args:
            new_audio_samples_int16: Int16 numpy array containing audio data
            
        Returns:
            List of prediction results, each containing:
            - raw_prob: Raw prediction probability
            - smoothed_prob: Smoothed prediction probability  
            - filter_passed: Whether the chunk passed filtering
        """
        # Validate input
        if not isinstance(new_audio_samples_int16, np.ndarray):
            try:
                new_audio_samples_int16 = np.array(new_audio_samples_int16, dtype=np.int16)
            except:
                print("RT Error: Invalid audio input type.")
                return []
        
        # Append new samples and track progress
        self.audio_buffer.extend(new_audio_samples_int16)
        self.samples_since_last_pred += len(new_audio_samples_int16)
        
        results = []
        
        # Process chunks in a loop for multiple hops
        while (self.samples_since_last_pred >= self.hop_samples and 
               len(self.audio_buffer) >= self.chunk_samples):
            
            # Extract chunk and convert to float
            current_buffer_np = np.array(self.audio_buffer)
            chunk_int16 = current_buffer_np[-self.chunk_samples:]
            chunk_float32 = chunk_int16.astype(np.float32) / 32768.0
            
            # Ensure chunk is contiguous
            chunk_float32 = ensure_audio_contiguous(chunk_float32)
            
            # Validate chunk
            if not validate_audio_chunk(chunk_float32):
                print("RT Warn: Invalid audio chunk, skipping")
                self.samples_since_last_pred -= self.hop_samples
                continue
            
            # Apply filtering
            passes = passes_filter(chunk_float32, self.sample_rate)
            
            # Initialize prediction result
            raw_prob = np.nan
            
            if passes:
                # Extract features and embedding
                emb_tensor, audio_feat_tensor = self._extract_features_and_embedding(chunk_float32)
                
                if emb_tensor is not None and audio_feat_tensor is not None:
                    # Make prediction
                    raw_prob = self._predict_stress_probability(emb_tensor, audio_feat_tensor)
                else:
                    passes = False  # Mark as filtered if feature extraction failed
            
            # Update history and calculate smoothed probability
            self.prob_history.append(raw_prob)
            smoothed_prob = self._get_smoothed_prob()
            
            # Create result
            result = {
                'raw_prob': raw_prob,
                'smoothed_prob': smoothed_prob,
                'filter_passed': passes,
                'timestamp': time.time()
            }
            results.append(result)
            
            # Update hop counter
            self.samples_since_last_pred -= self.hop_samples
        
        return results
    
    def predict_single_chunk(self, audio_chunk: np.ndarray) -> Dict[str, Any]:
        """
        Predicts stress for a single audio chunk (convenience method).
        
        Args:
            audio_chunk: Audio chunk as numpy array (float32, range -1 to 1)
            
        Returns:
            Dictionary with prediction results
        """
        # Ensure chunk is the right size
        if len(audio_chunk) != self.chunk_samples:
            print(f"Warning: Expected chunk size {self.chunk_samples}, got {len(audio_chunk)}")
            # Truncate or pad as needed
            if len(audio_chunk) > self.chunk_samples:
                audio_chunk = audio_chunk[:self.chunk_samples]
            else:
                audio_chunk = np.pad(audio_chunk, (0, self.chunk_samples - len(audio_chunk)))
        
        # Apply filtering
        passes = passes_filter(audio_chunk, self.sample_rate)
        
        raw_prob = np.nan
        if passes:
            # Extract features and embedding
            emb_tensor, audio_feat_tensor = self._extract_features_and_embedding(audio_chunk)
            
            if emb_tensor is not None and audio_feat_tensor is not None:
                # Make prediction
                raw_prob = self._predict_stress_probability(emb_tensor, audio_feat_tensor)
        
        return {
            'raw_prob': raw_prob,
            'filter_passed': passes,
            'timestamp': time.time()
        }
    
    def reset(self):
        """
        Resets the internal state of the predictor for a new session.
        """
        print("Resetting RealTimeStressPredictor state...")
        self.audio_buffer.clear()
        self.prob_history.clear()
        self.samples_since_last_pred = 0
        
        # Reset embedding model if it has a reset method
        if hasattr(self.embedding_model, 'reset'):
            self.embedding_model.reset()
        
        print("Predictor state reset.")
    
    def get_predictor_info(self) -> Dict[str, Any]:
        """
        Returns information about the predictor configuration.
        
        Returns:
            Dictionary containing predictor information
        """
        return {
            "sample_rate": self.sample_rate,
            "chunk_duration_s": self.chunk_duration_s,
            "hop_duration_s": self.hop_duration_s,
            "buffer_duration_s": self.buffer_duration_s,
            "chunk_samples": self.chunk_samples,
            "hop_samples": self.hop_samples,
            "embedding_dim": self.embedding_dim,
            "audio_features_dim": self.audio_features_dim,
            "feature_set": self.feature_set,
            "smoothing_window_size": self.smoothing_window_size,
            "model_loaded": self.model is not None,
            "scaler_loaded": self.scaler is not None,
            "embedding_model_loaded": self.embedding_model.is_model_loaded(),
            "device": str(self.device)
        }
    
    def __repr__(self) -> str:
        """
        String representation of the RealTimeStressPredictor.
        
        Returns:
            String representation
        """
        return (f"RealTimeStressPredictor("
                f"sr={self.sample_rate}, "
                f"chunk_dur={self.chunk_duration_s}s, "
                f"device={self.device})") 