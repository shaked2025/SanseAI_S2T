"""
Audio stress detection module.

This package provides a complete audio stress detection system with the following components:
- Configuration management
- Audio utilities and filtering
- Audio feature extraction
- Audio embedding extraction
- Data augmentation
- LSTM model for stress detection
- Real-time stress prediction
- Main audio model interface
- Training functionality

Main interface:
    from ai.audio import AudioStressModel
    model = AudioStressModel()
    model.load_models()
    result = model.run(audio_chunk)
"""

from .config import CONFIG, SystemConfig
from .utils import passes_filter, is_chunk_speech, validate_audio_chunk, normalize_audio
from .embedding import AudioEmbedding
from .augmentation import AudioAugmenter
from .stresslstm import EnhancedStressLSTM, create_stress_lstm_model
from .predictor import RealTimeStressPredictor
from .audio_model import AudioStressModel, get_audio_model, predict_stress, predict_stress_batch, load_stress_model
from .train import create_dataset, train_model

# Version info
__version__ = "1.0.0"
__author__ = "Audio VSA Team"

# Main exports
__all__ = [
    # Main interface
    'AudioStressModel',
    'get_audio_model',
    'predict_stress',
    'predict_stress_batch', 
    'load_stress_model',
    
    # Core components
    'AudioEmbedding',
    'AudioAugmenter',
    'EnhancedStressLSTM',
    'RealTimeStressPredictor',
    
    # Utilities
    'passes_filter',
    'is_chunk_speech',
    'validate_audio_chunk',
    'normalize_audio',
    'create_stress_lstm_model',
    
    # Training
    'create_dataset',
    'train_model',
    
    # Configuration
    'CONFIG',
    'SystemConfig'
] 