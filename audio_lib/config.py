"""
Configuration module containing all hyperparameters for the audio stress detection system.
Uses dataclasses to organize parameters by functionality for better maintainability.
"""

from dataclasses import dataclass
from typing import Optional, List, Dict, Any
import torch


@dataclass
class AudioConfig:
    """
    Configuration for audio processing parameters.
    
    Attributes:
        sample_rate: Audio sampling rate in Hz
        chunk_duration: Duration of audio chunks in seconds
        hop_duration: Hop duration for sliding window in seconds
        buffer_duration: Buffer duration for real-time processing in seconds
    """
    sample_rate: int = 16000
    chunk_duration: float = 2.0
    hop_duration: float = 0.5
    buffer_duration: float = 3.0
    rms_frame_length: int = 2048
    rms_hop_length: int = 512


@dataclass
class VADConfig:
    """
    Configuration for Voice Activity Detection (VAD) parameters.
    
    Attributes:
        aggressiveness: VAD aggressiveness level (0-3, higher is more aggressive)
        frame_duration_ms: Frame duration for VAD analysis in milliseconds
        speech_threshold: Proportion of frames needed to be speech for chunk to be considered speech
        max_rms_threshold: Maximum RMS threshold for filtering
    """
    aggressiveness: int = 2
    frame_duration_ms: int = 30
    speech_threshold: float = 0.2
    max_rms_threshold: Optional[float] = 0.5


@dataclass
class EmbeddingConfig:
    """
    Configuration for audio embedding model.
    
    Attributes:
        model_source: Source identifier for the embedding model
        embedding_dim: Dimension of the embedding vectors
        device: Computing device (cpu/cuda)
    """
    model_source: str = "speechbrain/spkrec-ecapa-voxceleb"
    embedding_dim: int = 192
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class FeatureConfig:
    """
    Configuration for audio feature extraction.
    
    Attributes:
        feature_set: Type of feature set to extract ('basic' or 'comprehensive')
        n_mfcc: Number of MFCC coefficients to extract
        pitch_min: Minimum pitch frequency in Hz
        pitch_max: Maximum pitch frequency in Hz
        chroma_bins: Number of chroma bins for comprehensive features
    """
    feature_set: str = 'basic'
    n_mfcc: int = 13
    pitch_min: float = 75.0
    pitch_max: float = 600.0
    chroma_bins: int = 12


@dataclass
class ModelConfig:
    """
    Configuration for the stress detection LSTM model.
    
    Attributes:
        hidden_dim: Hidden dimension size for LSTM
        num_layers: Number of LSTM layers
        dropout_rate: Dropout rate for regularization
        embedding_dim: Dimension of embedding features
        audio_features_dim: Dimension of audio features
    """
    hidden_dim: int = 128
    num_layers: int = 2
    dropout_rate: float = 0.3
    embedding_dim: int = 192
    audio_features_dim: int = 0  # Will be determined dynamically


@dataclass
class TrainingConfig:
    """
    Configuration for model training parameters.
    
    Attributes:
        epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Learning rate for optimizer
        n_splits: Number of folds for cross-validation
        patience_limit: Early stopping patience
        max_grad_norm: Maximum gradient norm for clipping
    """
    epochs: int = 15
    batch_size: int = 32
    learning_rate: float = 0.0005
    n_splits: int = 5
    patience_limit: int = 6
    max_grad_norm: float = 1.0


@dataclass
class AugmentationConfig:
    """
    Configuration for data augmentation parameters.
    
    Attributes:
        augment_data: Whether to enable data augmentation
        augmentation_factor: How many augmented samples to create per original
        time_shift_factor: Maximum time shift as fraction of audio length
        noise_level: Level of noise to add for noise augmentation
        pitch_shift_range: Range of pitch shift in semitones
        speed_change_factor: Maximum speed change factor
        mask_factor: Maximum portion of audio to mask for time masking
    """
    augment_data: bool = False
    augmentation_factor: int = 1
    time_shift_factor: float = 0.2
    noise_level: float = 0.005
    pitch_shift_range: int = 2
    speed_change_factor: float = 0.2
    mask_factor: float = 0.1


@dataclass
class PredictorConfig:
    """
    Configuration for real-time stress prediction.
    
    Attributes:
        smoothing_window_size: Size of smoothing window for probability averaging
        prediction_threshold: Threshold for binary stress classification
        model_path: Path to trained model file
        scaler_path: Path to feature scaler file
        feature_names_path: Path to feature names file
    """
    smoothing_window_size: int = 15
    prediction_threshold: float = 0.5
    model_path: str = "model/enhanced_stress_lstm_best.pth"
    scaler_path: str = "model/audio_features_scaler_final.joblib"
    feature_names_path: str = "model/feature_names.txt"


@dataclass
class PathConfig:
    """
    Configuration for file paths and directories.
    
    Attributes:
        model_dir: Directory containing model files
        output_dir: Directory for prediction outputs
        temp_dir: Directory for temporary files
    """
    model_dir: str = "model"
    output_dir: str = "predict_output"
    temp_dir: str = "temp"


@dataclass
class SystemConfig:
    """
    Main configuration container that holds all sub-configurations.
    
    This is the primary configuration class that should be used throughout
    the application to access all hyperparameters.
    """
    def __post_init__(self):
        if not hasattr(self, 'audio'):
            self.audio = AudioConfig()
            self.vad = VADConfig()
            self.embedding = EmbeddingConfig()
            self.features = FeatureConfig()
            self.model = ModelConfig()
            self.training = TrainingConfig()
            self.augmentation = AugmentationConfig()
            self.predictor = PredictorConfig()
            self.paths = PathConfig()

    def update_audio_features_dim(self, dim: int):
        """
        Updates the audio features dimension in model config.
        
        Args:
            dim: The determined audio features dimension
        """
        self.model.audio_features_dim = dim

# Global configuration instance
CONFIG = SystemConfig() 