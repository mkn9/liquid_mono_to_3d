"""
MagVIT-Based Video Feature Extractor
=====================================

MagVIT (Masked Generative Video Transformer) uses tokenization to convert
videos into discrete tokens. This implementation uses a simplified encoder
architecture inspired by MagVIT for feature extraction.

Architecture:
    Video (B, T, C, H, W)
        ↓
    Spatial Encoder (Conv2d per frame) → (B, T, D_spatial)
        ↓
    Temporal Encoder (1D Conv + Attention) → (B, T, D)
        ↓
    Features (B, T, D)

TDD Evidence: See artifacts/tdd_magvit_*.txt
"""

import torch
import torch.nn as nn
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'shared'))
from base_extractor import FeatureExtractor


class MagVITExtractor(FeatureExtractor, nn.Module):
    """
    MagVIT-inspired video feature extractor.
    
    Args:
        feature_dim (int): Output feature dimension (default: 256)
        spatial_dim (int): Intermediate spatial feature dimension (default: 512)
        dropout (float): Dropout probability (default: 0.1)
    """
    
    def __init__(
        self,
        feature_dim: int = 256,
        spatial_dim: int = 512,
        dropout: float = 0.1
    ):
        """Initialize MagVIT extractor."""
        nn.Module.__init__(self)
        
        self._feature_dim = feature_dim
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
        
        # Optional: Simple attention for temporal aggregation
        self.tokenizer = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=4,
            dropout=dropout,
            batch_first=True
        )
        
        self.layer_norm = nn.LayerNorm(feature_dim)
    
    @property
    def feature_dim(self) -> int:
        """Return feature dimension."""
        return self._feature_dim
    
    def extract(self, video: torch.Tensor) -> torch.Tensor:
        """
        Extract features from video using MagVIT-inspired architecture.
        
        Args:
            video (torch.Tensor): Input video (B, T, C, H, W)
        
        Returns:
            torch.Tensor: Extracted features (B, T, D)
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
        attn_output, _ = self.tokenizer(
            temporal_features,
            temporal_features,
            temporal_features
        )
        
        # Step 4: Residual connection and layer norm
        features = self.layer_norm(temporal_features + attn_output)
        
        return features
    
    def forward(self, video: torch.Tensor) -> torch.Tensor:
        """
        Forward pass (calls extract).
        
        Args:
            video (torch.Tensor): Input video (B, T, C, H, W)
        
        Returns:
            torch.Tensor: Extracted features (B, T, D)
        """
        return self.extract(video)
    
    def __repr__(self):
        """String representation."""
        return (
            f"MagVITExtractor(\n"
            f"  feature_dim={self._feature_dim},\n"
            f"  spatial_dim={self.spatial_dim}\n"
            f")"
        )


if __name__ == "__main__":
    print("=" * 70)
    print("MagVIT Video Feature Extractor - Demo")
    print("=" * 70)
    
    # Create extractor
    print("\n1. Creating MagVIT extractor...")
    extractor = MagVITExtractor(feature_dim=256)
    print(f"✅ Created: {extractor}")
    
    # Count parameters
    total_params = sum(p.numel() for p in extractor.parameters())
    print(f"\nTotal parameters: {total_params:,}")
    
    # Test extraction
    print("\n2. Testing feature extraction...")
    video = torch.randn(2, 16, 3, 64, 64)
    print(f"Input shape: {video.shape}")
    
    features = extractor.extract(video)
    print(f"Output shape: {features.shape}")
    print(f"Expected shape: (2, 16, {extractor.feature_dim})")
    
    if features.shape == (2, 16, extractor.feature_dim):
        print("✅ Shape correct!")
    
    if torch.all(torch.isfinite(features)):
        print("✅ All values finite!")
    
    # Test with UnifiedModel
    print("\n3. Testing integration with UnifiedModel...")
    from unified_model import UnifiedModel
    
    model = UnifiedModel(extractor, num_classes=4)
    output = model(video)
    
    print(f"Classification output: {output['classification'].shape}")
    print(f"Prediction output: {output['prediction'].shape}")
    print("✅ Integration successful!")
    
    print("\n" + "=" * 70)
    print("Demo complete. Ready for TDD GREEN phase verification.")
    print("=" * 70)

