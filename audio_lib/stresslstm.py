"""
Enhanced Stress LSTM model for stress detection from audio features.
Combines embedding features and audio features using an early fusion approach.
"""

import torch
import torch.nn as nn
from typing import Tuple
from .config import CONFIG


class EnhancedStressLSTM(nn.Module):
    """
    Enhanced LSTM model for stress detection that combines embedding and audio features.
    
    Uses an early fusion approach where embedding features and audio features are
    concatenated before being processed through the LSTM network.
    """
    
    def __init__(self, 
                 embedding_dim: int = None, 
                 audio_features_dim: int = None, 
                 hidden_dim: int = None, 
                 num_layers: int = None, 
                 dropout_rate: float = None):
        """
        Initialize the Enhanced Stress LSTM model.
        
        Args:
            embedding_dim: Dimension of the embedding features
            audio_features_dim: Dimension of the audio features
            hidden_dim: Hidden dimension size for LSTM layers
            num_layers: Number of LSTM layers
            dropout_rate: Dropout rate for regularization
        """
        super(EnhancedStressLSTM, self).__init__()
        
        # Use config defaults if parameters not provided
        self.embedding_dim = embedding_dim or CONFIG.model.embedding_dim
        self.audio_features_dim = audio_features_dim or CONFIG.model.audio_features_dim
        self.hidden_dim = hidden_dim or CONFIG.model.hidden_dim
        self.num_layers = num_layers or CONFIG.model.num_layers
        self.dropout_rate = dropout_rate or CONFIG.model.dropout_rate
        
        # Early fusion approach - concatenate features before LSTM
        self.combined_input_dim = self.embedding_dim + self.audio_features_dim
        
        # Main LSTM that processes combined features
        self.lstm_combined = nn.LSTM(
            self.combined_input_dim, 
            self.hidden_dim, 
            num_layers=self.num_layers, 
            batch_first=True, 
            dropout=self.dropout_rate if self.num_layers > 1 else 0
        )
        
        # Output layers
        self.fc1 = nn.Linear(self.hidden_dim, self.hidden_dim // 2)
        self.dropout = nn.Dropout(self.dropout_rate)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(self.hidden_dim // 2, 1)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """
        Initialize the weights of the model using appropriate initialization schemes.
        """
        for name, param in self.named_parameters():
            if 'weight' in name:
                if 'lstm' in name:
                    # Xavier initialization for LSTM weights
                    nn.init.xavier_uniform_(param.data)
                elif 'fc' in name:
                    # Xavier initialization for linear layers
                    nn.init.xavier_uniform_(param.data)
            elif 'bias' in name:
                # Initialize biases to zero
                nn.init.constant_(param.data, 0)
    
    def forward(self, x_embedding: torch.Tensor, x_audio: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the Enhanced Stress LSTM model.
        
        Args:
            x_embedding: Embedding features tensor, shape (batch_size, seq_len, embedding_dim)
            x_audio: Audio features tensor, shape (batch_size, seq_len, audio_features_dim)
            
        Returns:
            Output tensor with stress predictions, shape (batch_size, 1)
        """
        # Get batch size and sequence lengths
        batch_size = x_embedding.size(0)
        emb_seq_len = x_embedding.size(1)
        audio_seq_len = x_audio.size(1)
        
        # Ensure sequences are the same length
        # For simplicity, this example assumes they're the same length
        # In practice, you might need preprocessing to align them
        if emb_seq_len != audio_seq_len:
            # Take the minimum length for safety
            min_len = min(emb_seq_len, audio_seq_len)
            x_embedding = x_embedding[:, :min_len, :]
            x_audio = x_audio[:, :min_len, :]
        
        # Concatenate features along the feature dimension (early fusion)
        combined_input = torch.cat((x_embedding, x_audio), dim=2)
        
        # Process through combined LSTM
        lstm_out, _ = self.lstm_combined(combined_input)
        
        # Get final hidden state (last time step)
        final_hidden = lstm_out[:, -1, :]
        
        # Process through output layers
        x = self.fc1(final_hidden)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x
    
    def get_model_info(self) -> dict:
        """
        Returns information about the model architecture.
        
        Returns:
            Dictionary containing model architecture information
        """
        return {
            "model_type": "EnhancedStressLSTM",
            "embedding_dim": self.embedding_dim,
            "audio_features_dim": self.audio_features_dim,
            "combined_input_dim": self.combined_input_dim,
            "hidden_dim": self.hidden_dim,
            "num_layers": self.num_layers,
            "dropout_rate": self.dropout_rate,
            "total_parameters": sum(p.numel() for p in self.parameters()),
            "trainable_parameters": sum(p.numel() for p in self.parameters() if p.requires_grad)
        }
    
    def freeze_lstm_layers(self):
        """
        Freezes the LSTM layers to prevent them from being updated during training.
        Useful for transfer learning scenarios.
        """
        for param in self.lstm_combined.parameters():
            param.requires_grad = False
        print("LSTM layers frozen")
    
    def unfreeze_lstm_layers(self):
        """
        Unfreezes the LSTM layers to allow them to be updated during training.
        """
        for param in self.lstm_combined.parameters():
            param.requires_grad = True
        print("LSTM layers unfrozen")
    
    def freeze_output_layers(self):
        """
        Freezes the output layers (fully connected layers).
        """
        for param in self.fc1.parameters():
            param.requires_grad = False
        for param in self.fc2.parameters():
            param.requires_grad = False
        print("Output layers frozen")
    
    def unfreeze_output_layers(self):
        """
        Unfreezes the output layers (fully connected layers).
        """
        for param in self.fc1.parameters():
            param.requires_grad = True
        for param in self.fc2.parameters():
            param.requires_grad = True
        print("Output layers unfrozen")
    
    def get_lstm_hidden_states(self, x_embedding: torch.Tensor, x_audio: torch.Tensor) -> torch.Tensor:
        """
        Returns the hidden states from the LSTM for analysis purposes.
        
        Args:
            x_embedding: Embedding features tensor
            x_audio: Audio features tensor
            
        Returns:
            LSTM hidden states tensor, shape (batch_size, seq_len, hidden_dim)
        """
        # Ensure sequences are the same length
        emb_seq_len = x_embedding.size(1)
        audio_seq_len = x_audio.size(1)
        
        if emb_seq_len != audio_seq_len:
            min_len = min(emb_seq_len, audio_seq_len)
            x_embedding = x_embedding[:, :min_len, :]
            x_audio = x_audio[:, :min_len, :]
        
        # Concatenate features
        combined_input = torch.cat((x_embedding, x_audio), dim=2)
        
        # Get LSTM hidden states
        lstm_out, _ = self.lstm_combined(combined_input)
        
        return lstm_out
    
    def __repr__(self) -> str:
        """
        String representation of the model.
        
        Returns:
            String representation
        """
        return (f"EnhancedStressLSTM("
                f"embedding_dim={self.embedding_dim}, "
                f"audio_features_dim={self.audio_features_dim}, "
                f"hidden_dim={self.hidden_dim}, "
                f"num_layers={self.num_layers}, "
                f"dropout_rate={self.dropout_rate})")


def create_stress_lstm_model(embedding_dim: int = None, 
                           audio_features_dim: int = None, 
                           **kwargs) -> EnhancedStressLSTM:
    """
    Factory function to create an Enhanced Stress LSTM model with default configurations.
    
    Args:
        embedding_dim: Dimension of embedding features
        audio_features_dim: Dimension of audio features
        **kwargs: Additional arguments to pass to the model constructor
        
    Returns:
        Initialized EnhancedStressLSTM model
    """
    model = EnhancedStressLSTM(
        embedding_dim=embedding_dim,
        audio_features_dim=audio_features_dim,
        **kwargs
    )
    
    print(f"Created Enhanced Stress LSTM model with {model.get_model_info()['total_parameters']} parameters")
    return model 