"""
Audio embedding module for extracting speaker embeddings from audio signals.
Uses SpeechBrain ECAPA-TDNN model for generating audio embeddings.
"""

import torch
import warnings
import logging
from typing import Optional
from .config import CONFIG

# Suppress SpeechBrain warnings
warnings.filterwarnings("ignore", category=FutureWarning)
logging.getLogger("speechbrain").setLevel(logging.WARNING)

try:
    from speechbrain.inference.speaker import EncoderClassifier
except ImportError:
    warnings.warn("SpeechBrain library not found. Audio embedding functionality will be disabled. "
                  "Install with 'pip install speechbrain'.")
    EncoderClassifier = None


class AudioEmbedding:
    """
    Handles audio embedding extraction using SpeechBrain ECAPA-TDNN model.
    
    This class provides functionality to load the embedding model and extract
    speaker embeddings from audio signals for stress detection.
    """
    
    def __init__(self, 
                 model_source: str = None, 
                 device: str = None,
                 embedding_dim: int = None):
        """
        Initialize the AudioEmbedding class.
        
        Args:
            model_source: Source identifier for the embedding model
            device: Computing device ('cpu' or 'cuda')
            embedding_dim: Expected dimension of embeddings
        """
        self.model_source = model_source or CONFIG.embedding.model_source
        self.device = device or CONFIG.embedding.device
        self.embedding_dim = embedding_dim or CONFIG.embedding.embedding_dim
        self.model = None
        self._is_loaded = False
        
        # Auto-detect device if not specified
        if self.device == "cuda" and not torch.cuda.is_available():
            print("CUDA not available, falling back to CPU")
            self.device = "cpu"
        
        print(f"AudioEmbedding initialized for device: {self.device}")
    
    def load_model(self) -> bool:
        """
        Loads the SpeechBrain ECAPA-TDNN embedding model.
        
        Returns:
            True if model loaded successfully, False otherwise
        """
        if self._is_loaded:
            print("Embedding model already loaded")
            return True
            
        if EncoderClassifier is None:
            print("Error: SpeechBrain library not available")
            return False
        
        try:
            print(f"Loading SpeechBrain model from: {self.model_source}")
            self.model = EncoderClassifier.from_hparams(
                source=self.model_source,
                run_opts={"device": self.device}
            )
            
            # Verify model device
            self.model = self.model.to(self.device)
            self.model.eval()
            self._is_loaded = True
            
            print("SpeechBrain ECAPA-TDNN model loaded successfully")
            return True
            
        except Exception as e:
            print(f"Error loading SpeechBrain model: {e}")
            return False
    
    def extract_embedding(self, audio_chunk: torch.Tensor) -> Optional[torch.Tensor]:
        """
        Extracts embedding from an audio chunk.
        
        Args:
            audio_chunk: Audio tensor, shape should be (1, sequence_length) or (sequence_length,)
            
        Returns:
            Embedding tensor of shape (embedding_dim,) or None if extraction fails
        """
        if not self._is_loaded:
            if not self.load_model():
                print("Error: Cannot extract embedding, model not loaded")
                return None
        
        try:
            # Ensure correct tensor format
            if audio_chunk.dim() == 1:
                audio_chunk = audio_chunk.unsqueeze(0)  # Add batch dimension
            
            # Move to correct device
            audio_chunk = audio_chunk.to(self.device)
            
            # Extract embedding
            with torch.no_grad():
                embedding = self.model.encode_batch(audio_chunk)
                
                # Remove batch dimension and return
                if embedding.dim() > 1:
                    embedding = embedding.squeeze(0)
                
                return embedding.detach()
                
        except Exception as e:
            print(f"Error extracting embedding: {e}")
            return None
    
    def extract_embedding_from_numpy(self, audio_chunk_np, sample_rate: int = None) -> Optional[torch.Tensor]:
        """
        Extracts embedding from a numpy audio chunk.
        
        Args:
            audio_chunk_np: Numpy array containing audio data
            sample_rate: Sample rate of the audio (for validation)
            
        Returns:
            Embedding tensor of shape (embedding_dim,) or None if extraction fails
        """
        if sample_rate is None:
            sample_rate = CONFIG.audio.sample_rate
            
        try:
            # Convert numpy to tensor
            audio_tensor = torch.from_numpy(audio_chunk_np).float()
            return self.extract_embedding(audio_tensor)
            
        except Exception as e:
            print(f"Error converting numpy to tensor for embedding: {e}")
            return None
    
    def get_embedding_dimension(self) -> int:
        """
        Returns the dimension of the embeddings produced by this model.
        
        Returns:
            Embedding dimension
        """
        return self.embedding_dim
    
    def is_model_loaded(self) -> bool:
        """
        Checks if the embedding model is loaded and ready.
        
        Returns:
            True if model is loaded, False otherwise
        """
        return self._is_loaded and self.model is not None
    
    def get_model_info(self) -> dict:
        """
        Returns information about the loaded model.
        
        Returns:
            Dictionary containing model information
        """
        return {
            "model_source": self.model_source,
            "device": self.device,
            "embedding_dim": self.embedding_dim,
            "is_loaded": self._is_loaded,
            "model_available": self.model is not None
        }
    
    def reset(self):
        """
        Resets the embedding model, clearing any cached states.
        Useful for starting fresh processing sessions.
        """
        if self.model is not None and hasattr(self.model, 'reset'):
            try:
                self.model.reset()
                print("Embedding model state reset")
            except Exception as e:
                print(f"Warning: Could not reset embedding model state: {e}")
    
    def __del__(self):
        """
        Cleanup when the object is destroyed.
        """
        if hasattr(self, 'model') and self.model is not None:
            del self.model
            
    def __repr__(self) -> str:
        """
        String representation of the AudioEmbedding object.
        
        Returns:
            String representation
        """
        return f"AudioEmbedding(source='{self.model_source}', device='{self.device}', loaded={self._is_loaded})" 