"""
Standalone MagVIT Feature Extractor
No external dependencies - self-contained implementation
"""
import torch
import torch.nn as nn
from typing import Tuple


class MagVITExtractor(nn.Module):
    """
    MagVIT-inspired video feature extractor (standalone version).
    
    Architecture:
    - Spatial encoder: CNN for per-frame features
    - Temporal encoder: 1D conv for temporal modeling
    - Tokenizer: Multi-head attention for aggregation
    """
    
    def __init__(
        self,
        feature_dim: int = 512,
        spatial_dim: int = 512,
        dropout: float = 0.1
    ):
        """Initialize MagVIT extractor."""
        super().__init__()
        
        self.feature_dim = feature_dim
        self.spatial_dim = spatial_dim
        
        # Spatial encoder: Process each frame independently
        self.encoder = nn.Sequential(
            # Input: (B*T, 3, 64, 64)
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # (B*T, 64, 16, 16)
            
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # (B*T, 128, 4, 4)
            
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            # (B*T, 256, 4, 4)
            
            nn.Flatten(),  # (B*T, 256 * 4 * 4) = (B*T, 4096)
            nn.Linear(256 * 4 * 4, spatial_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Temporal encoder: Process sequence with 1D convolutions
        self.temporal_encoder = nn.Sequential(
            # Input: (B, D, T) for Conv1d
            nn.Conv1d(spatial_dim, spatial_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(spatial_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Conv1d(spatial_dim, feature_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU()
        )
        
        # Multi-head attention for temporal aggregation
        self.tokenizer = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=4,
            dropout=dropout,
            batch_first=True
        )
        
        self.layer_norm = nn.LayerNorm(feature_dim)
    
    def extract(self, video: torch.Tensor) -> torch.Tensor:
        """
        Extract features from video.
        
        Args:
            video: (B, T, C, H, W) video tensor
        
        Returns:
            features: (B, T, feature_dim) temporal features
        """
        B, T, C, H, W = video.shape
        
        # Step 1: Spatial encoding (per frame)
        frames = video.view(B * T, C, H, W)
        spatial_features = self.encoder(frames)  # (B*T, spatial_dim)
        spatial_features = spatial_features.view(B, T, self.spatial_dim)  # (B, T, spatial_dim)
        
        # Step 2: Temporal encoding
        # Conv1d expects (B, C, T), so transpose
        spatial_features_transposed = spatial_features.transpose(1, 2)  # (B, spatial_dim, T)
        temporal_features = self.temporal_encoder(spatial_features_transposed)  # (B, feature_dim, T)
        temporal_features = temporal_features.transpose(1, 2)  # (B, T, feature_dim)
        
        # Step 3: Self-attention for temporal context
        attended_features, _ = self.tokenizer(
            temporal_features,
            temporal_features,
            temporal_features
        )
        
        # Residual connection + layer norm
        features = self.layer_norm(attended_features + temporal_features)
        
        return features
