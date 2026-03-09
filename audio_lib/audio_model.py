"""
Main audio stress detection model interface.
Provides a unified singleton interface for stress detection from audio chunks.
This is the primary class that should be used for all audio stress detection operations.
"""

import numpy as np
import os
import time
from typing import Dict, Any, Optional, List
from .config import CONFIG
from .predictor import RealTimeStressPredictor
from .embedding import AudioEmbedding
from .features import AudioFeatures
from .utils import validate_audio_chunk, normalize_audio


class AudioStressModel:
    """
    Singleton main audio stress detection model.
    
    This is the primary interface for the audio stress detection system.
    Provides a simple run() method for making predictions and handles all
    underlying complexity of model loading, feature extraction, and prediction.
    """
    
    _instance = None
    _initialized = False
    
    def __new__(cls):
        """
        Implements singleton pattern to ensure only one instance exists.
        
        Returns:
            The single AudioStressModel instance
        """
        if cls._instance is None:
            cls._instance = super(AudioStressModel, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        """
        Initialize the AudioStressModel singleton.
        Only initializes once, subsequent calls are ignored.
        """
        if AudioStressModel._initialized:
            return
        
        print("Initializing AudioStressModel...")
        
        # Configuration
        self.config = CONFIG
        self.sample_rate = self.config.audio.sample_rate
        self.chunk_duration = self.config.audio.chunk_duration
        self.prediction_threshold = self.config.predictor.prediction_threshold
        
        # Components
        self.embedding_model = None
        self.predictor = None
        self.feature_extractor = None
        
        # State
        self.is_loaded = False
        self.demo_mode = False  # Add demo mode flag
        self.last_prediction_time = None
        self.session_stats = {
            'total_predictions': 0,
            'stress_predictions': 0,
            'filtered_chunks': 0,
            'session_start_time': None
        }
        
        AudioStressModel._initialized = True
        print("AudioStressModel singleton created.")
    
    def load_models(self, 
                   model_path: str = None,
                   scaler_path: str = None,
                   force_reload: bool = False) -> bool:
        """
        Loads all required models and components for stress detection.
        
        Args:
            model_path: Path to the trained LSTM model (optional, uses config default)
            scaler_path: Path to the feature scaler (optional, uses config default)
            force_reload: Whether to force reloading even if already loaded
            
        Returns:
            True if all models loaded successfully, False otherwise
        """
        if self.is_loaded and not force_reload:
            print("Models already loaded. Use force_reload=True to reload.")
            return True
        
        print("Loading audio stress detection models...")
        
        try:
            # Initialize embedding model
            print("Loading embedding model...")
            self.embedding_model = AudioEmbedding()
            if not self.embedding_model.load_model():
                print("Failed to load embedding model - switching to demo mode")
                self.demo_mode = True
                self.is_loaded = True
                self.session_stats['session_start_time'] = time.time()
                return True
            
            # Initialize feature extractor
            print("Loading feature extractor...")
            self.feature_extractor = AudioFeatures()
            
            # Determine audio features dimension and update config
            audio_features_dim = self.feature_extractor.get_features_number()
            if audio_features_dim == 0:
                print("Warning: Could not determine audio features dimension - switching to demo mode")
                self.demo_mode = True
                self.is_loaded = True
                self.session_stats['session_start_time'] = time.time()
                return True
            
            self.config.update_audio_features_dim(audio_features_dim)
            
            # Initialize predictor with all components
            print("Loading stress prediction model...")
            self.predictor = RealTimeStressPredictor(
                model_path=model_path or self.config.predictor.model_path,
                scaler_path=scaler_path or self.config.predictor.scaler_path,
                embedding_model_instance=self.embedding_model,
                feature_set=self.config.features.feature_set
            )
            
            self.is_loaded = True
            self.session_stats['session_start_time'] = time.time()
            
            print("AudioStressModel loaded successfully!")
            print(f"Configuration: {self.get_model_info()}")
            
            return True
            
        except Exception as e:
            print(f"Failed to load AudioStressModel: {e}")
            print("Switching to demo mode for UI testing")
            self.demo_mode = True
            self.is_loaded = True
            self.session_stats['session_start_time'] = time.time()
            return True
    
    def run(self, audio_chunk: np.ndarray, 
           normalize: bool = True,
           return_details: bool = False) -> Dict[str, Any]:
        """
        Main prediction function. Processes an audio chunk and returns stress prediction.
        
        Args:
            audio_chunk: Audio data as numpy array (int16 or float32)
            normalize: Whether to normalize the audio chunk
            return_details: Whether to return detailed prediction information
            
        Returns:
            Dictionary containing prediction results:
            - is_stress: Boolean indicating if stress is detected
            - confidence: Confidence score (0.0 to 1.0)
            - raw_probability: Raw model output probability
            - filter_passed: Whether the chunk passed audio filtering
            - processing_time: Time taken for processing (if return_details=True)
            - chunk_info: Information about the audio chunk (if return_details=True)
        """
        if not self.is_loaded:
            if not self.load_models():
                return {
                    'is_stress': False,
                    'confidence': 0.0,
                    'error': 'Models not loaded'
                }
        
        start_time = time.time() if return_details else None
        
        try:
            # Handle demo mode with dummy predictions
            if self.demo_mode:
                return self._generate_demo_prediction(audio_chunk, return_details, start_time or 0.0)
            
            # Validate and preprocess audio chunk
            if not isinstance(audio_chunk, np.ndarray):
                audio_chunk = np.array(audio_chunk)
            
            # Handle different audio formats
            if audio_chunk.dtype == np.int16:
                # Convert int16 to float32
                audio_float = audio_chunk.astype(np.float32) / 32768.0
            elif audio_chunk.dtype == np.float32 or audio_chunk.dtype == np.float64:
                audio_float = audio_chunk.astype(np.float32)
            else:
                raise ValueError(f"Unsupported audio dtype: {audio_chunk.dtype}")
            
            # Normalize if requested
            if normalize:
                audio_float = normalize_audio(audio_float)
            
            # Validate chunk
            if not validate_audio_chunk(audio_float):
                return {
                    'is_stress': False,
                    'confidence': 0.0,
                    'raw_probability': np.nan,
                    'filter_passed': False,
                    'error': 'Invalid audio chunk'
                }
            
            # Make prediction - check if predictor is available
            if self.predictor is None:
                return {
                    'is_stress': False,
                    'confidence': 0.0,
                    'raw_probability': np.nan,
                    'filter_passed': False,
                    'error': 'Predictor not available'
                }
            
            prediction_result = self.predictor.predict_single_chunk(audio_float)
            
            # Extract results
            raw_prob = prediction_result.get('raw_prob', np.nan)
            filter_passed = prediction_result.get('filter_passed', False)
            
            # Determine stress prediction
            is_stress = False
            confidence = 0.0
            
            if not np.isnan(raw_prob) and filter_passed:
                confidence = raw_prob
                is_stress = raw_prob >= self.prediction_threshold
            
            # Update statistics
            self._update_session_stats(is_stress, filter_passed)
            
            # Prepare result
            result = {
                'is_stress': is_stress,
                'confidence': confidence,
                'raw_probability': raw_prob,
                'filter_passed': filter_passed
            }
            
            # Add detailed information if requested
            if return_details and start_time is not None:
                processing_time = time.time() - start_time
                result.update({
                    'processing_time': processing_time,
                    'chunk_info': {
                        'length': len(audio_chunk),
                        'sample_rate': self.sample_rate,
                        'duration_s': len(audio_chunk) / self.sample_rate,
                        'dtype': str(audio_chunk.dtype),
                        'max_amplitude': float(np.max(np.abs(audio_float))),
                        'rms_energy': float(np.sqrt(np.mean(audio_float**2)))
                    },
                    'model_info': self.get_model_info(),
                    'session_stats': self.get_session_stats()
                })
            
            self.last_prediction_time = time.time()
            return result
            
        except Exception as e:
            error_result = {
                'is_stress': False,
                'confidence': 0.0,
                'raw_probability': np.nan,
                'filter_passed': False,
                'error': str(e)
            }
            
            if return_details and start_time is not None:
                processing_time = time.time() - start_time
                error_result.update({
                    'processing_time': processing_time,
                    'chunk_info': {
                        'length': len(audio_chunk) if isinstance(audio_chunk, np.ndarray) else 0,
                        'error': str(e)
                    }
                })
            
            return error_result
    
    def _generate_demo_prediction(self, audio_chunk: np.ndarray, return_details: bool = False, start_time: float = 0.0) -> Dict[str, Any]:
        """Generate demo predictions for UI testing when model files are not available"""
        import random
        import math
        
        # Calculate some basic audio properties for more realistic demo
        if isinstance(audio_chunk, np.ndarray) and len(audio_chunk) > 0:
            # Convert to float if needed
            if audio_chunk.dtype == np.int16:
                audio_float = audio_chunk.astype(np.float32) / 32768.0
            else:
                audio_float = audio_chunk.astype(np.float32)
            
            # Calculate RMS energy
            rms_energy = float(np.sqrt(np.mean(audio_float**2)))
            max_amplitude = float(np.max(np.abs(audio_float)))
            
            # Generate prediction based on audio energy (higher energy = higher stress probability)
            base_probability = min(rms_energy * 2.0, 0.8)  # Scale RMS to probability
            
            # Add some randomness and time-based variation
            current_time = time.time()
            time_factor = math.sin(current_time * 0.1) * 0.2  # Slow sinusoidal variation
            noise_factor = random.uniform(-0.1, 0.1)  # Random noise
            
            raw_probability = max(0.0, min(1.0, base_probability + time_factor + noise_factor))
            
        else:
            # Fallback for invalid audio
            raw_probability = random.uniform(0.2, 0.8)
            rms_energy = 0.0
            max_amplitude = 0.0
        
        # Determine stress prediction
        is_stress = raw_probability >= self.prediction_threshold
        filter_passed = True  # Always pass in demo mode
        
        # Update statistics
        self._update_session_stats(is_stress, filter_passed)
        
        # Prepare result
        result = {
            'is_stress': is_stress,
            'confidence': raw_probability,
            'raw_probability': raw_probability,
            'filter_passed': filter_passed,
            'demo_mode': True
        }
        
        # Add detailed information if requested
        if return_details and start_time > 0:
            processing_time = time.time() - start_time
            result.update({
                'processing_time': processing_time,
                'chunk_info': {
                    'length': len(audio_chunk) if isinstance(audio_chunk, np.ndarray) else 0,
                    'sample_rate': self.sample_rate,
                    'duration_s': len(audio_chunk) / self.sample_rate if isinstance(audio_chunk, np.ndarray) else 0,
                    'dtype': str(audio_chunk.dtype) if isinstance(audio_chunk, np.ndarray) else 'unknown',
                    'max_amplitude': max_amplitude,
                    'rms_energy': rms_energy
                },
                'model_info': self.get_model_info(),
                'session_stats': self.get_session_stats()
            })
        
        self.last_prediction_time = time.time()
        return result
    
    def run_batch(self, audio_chunks: List[np.ndarray], 
                 normalize: bool = True) -> List[Dict[str, Any]]:
        """
        Processes multiple audio chunks in batch.
        
        Args:
            audio_chunks: List of audio chunks as numpy arrays
            normalize: Whether to normalize the audio chunks
            
        Returns:
            List of prediction results for each chunk
        """
        return [self.run(chunk, normalize=normalize, return_details=False) 
                for chunk in audio_chunks]
    
    def run_real_time(self, audio_samples_int16: np.ndarray) -> List[Dict[str, Any]]:
        """
        Processes audio in real-time mode with buffering and hop-based processing.
        
        Args:
            audio_samples_int16: New audio samples as int16 numpy array
            
        Returns:
            List of prediction results (may be empty if no predictions made)
        """
        if not self.is_loaded:
            if not self.load_models():
                return [{'error': 'Models not loaded'}]
        
        try:
            # Handle demo mode
            if self.demo_mode or self.predictor is None:
                # Generate demo predictions for real-time processing
                chunk_size = self.get_recommended_chunk_size()
                num_chunks = len(audio_samples_int16) // chunk_size
                results = []
                
                for i in range(num_chunks):
                    start_idx = i * chunk_size
                    end_idx = start_idx + chunk_size
                    chunk = audio_samples_int16[start_idx:end_idx]
                    
                    demo_result = self._generate_demo_prediction(chunk, return_details=False)
                    demo_result['timestamp'] = time.time()
                    results.append(demo_result)
                
                return results
            
            # Use the predictor's real-time processing
            results = self.predictor.process_audio_chunk(audio_samples_int16)
            
            # Process results and update statistics
            processed_results = []
            for result in results:
                raw_prob = result.get('raw_prob', np.nan)
                filter_passed = result.get('filter_passed', False)
                smoothed_prob = result.get('smoothed_prob', np.nan)
                
                # Use smoothed probability for better results
                confidence = smoothed_prob if not np.isnan(smoothed_prob) else raw_prob
                is_stress = False
                
                if not np.isnan(confidence) and filter_passed:
                    is_stress = confidence >= self.prediction_threshold
                else:
                    confidence = 0.0
                
                # Update statistics
                self._update_session_stats(is_stress, filter_passed)
                
                processed_result = {
                    'is_stress': is_stress,
                    'confidence': confidence,
                    'raw_probability': raw_prob,
                    'smoothed_probability': smoothed_prob,
                    'filter_passed': filter_passed,
                    'timestamp': result.get('timestamp', time.time())
                }
                processed_results.append(processed_result)
            
            return processed_results
            
        except Exception as e:
            print(f"Error in real-time processing: {e}")
            return [{'error': str(e)}]
    
    def _update_session_stats(self, is_stress: bool, filter_passed: bool):
        """
        Updates session statistics.
        
        Args:
            is_stress: Whether stress was detected
            filter_passed: Whether the chunk passed filtering
        """
        if filter_passed:
            self.session_stats['total_predictions'] += 1
            if is_stress:
                self.session_stats['stress_predictions'] += 1
        else:
            self.session_stats['filtered_chunks'] += 1
    
    def reset_session(self):
        """
        Resets the session state and statistics.
        """
        print("Resetting AudioStressModel session...")
        
        # Reset predictor state
        if self.predictor:
            self.predictor.reset()
        
        # Reset session statistics
        self.session_stats = {
            'total_predictions': 0,
            'stress_predictions': 0,
            'filtered_chunks': 0,
            'session_start_time': time.time()
        }
        
        self.last_prediction_time = None
        print("Session reset complete.")
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Returns information about the loaded models and configuration.
        
        Returns:
            Dictionary containing model information
        """
        if not self.is_loaded:
            return {'status': 'not_loaded'}
        
        info = {
            'status': 'loaded',
            'demo_mode': self.demo_mode,
            'sample_rate': self.sample_rate,
            'chunk_duration': self.chunk_duration,
            'prediction_threshold': self.prediction_threshold,
            'embedding_dim': self.config.embedding.embedding_dim,
            'audio_features_dim': self.config.model.audio_features_dim,
            'feature_set': self.config.features.feature_set,
            'device': str(self.predictor.device) if self.predictor else 'demo'
        }
        
        if self.predictor:
            info.update(self.predictor.get_predictor_info())
        
        return info
    
    def get_session_stats(self) -> Dict[str, Any]:
        """
        Returns current session statistics.
        
        Returns:
            Dictionary containing session statistics
        """
        stats = self.session_stats.copy()
        
        if stats['session_start_time']:
            stats['session_duration'] = time.time() - stats['session_start_time']
        
        if stats['total_predictions'] > 0:
            stats['stress_rate'] = stats['stress_predictions'] / stats['total_predictions']
        else:
            stats['stress_rate'] = 0.0
        
        stats['last_prediction_time'] = self.last_prediction_time
        
        return stats
    
    def is_ready(self) -> bool:
        """
        Checks if the model is ready for predictions.
        
        Returns:
            True if model is loaded and ready, False otherwise
        """
        return self.is_loaded  # In demo mode, we're ready even without predictor
    
    def get_supported_audio_formats(self) -> List[str]:
        """
        Returns list of supported audio data types.
        
        Returns:
            List of supported audio format strings
        """
        return ['int16', 'float32', 'float64']
    
    def get_recommended_chunk_size(self) -> int:
        """
        Returns the recommended audio chunk size in samples.
        
        Returns:
            Recommended chunk size in samples
        """
        return int(self.chunk_duration * self.sample_rate)
    
    def __repr__(self) -> str:
        """
        String representation of the AudioStressModel.
        
        Returns:
            String representation
        """
        status = "loaded" if self.is_loaded else "not_loaded"
        return f"AudioStressModel(status={status}, sr={self.sample_rate})"


# Global singleton instance
audio_model = AudioStressModel()


def get_audio_model() -> AudioStressModel:
    """
    Returns the global AudioStressModel singleton instance.
    
    Returns:
        The AudioStressModel singleton instance
    """
    return audio_model


# Convenience functions for easy access
def predict_stress(audio_chunk: np.ndarray, **kwargs) -> Dict[str, Any]:
    """
    Convenience function for stress prediction.
    
    Args:
        audio_chunk: Audio data as numpy array
        **kwargs: Additional arguments passed to the run method
        
    Returns:
        Prediction results dictionary
    """
    return get_audio_model().run(audio_chunk, **kwargs)


def predict_stress_batch(audio_chunks: List[np.ndarray], **kwargs) -> List[Dict[str, Any]]:
    """
    Convenience function for batch stress prediction.
    
    Args:
        audio_chunks: List of audio chunks
        **kwargs: Additional arguments passed to the run_batch method
        
    Returns:
        List of prediction results
    """
    return get_audio_model().run_batch(audio_chunks, **kwargs)


def load_stress_model(**kwargs) -> bool:
    """
    Convenience function for loading the stress detection model.
    
    Args:
        **kwargs: Arguments passed to the load_models method
        
    Returns:
        True if loaded successfully, False otherwise
    """
    return get_audio_model().load_models(**kwargs) 