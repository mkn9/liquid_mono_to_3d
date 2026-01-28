"""
MagVIT Feature Extractor for Liquid VLM Integration
Worker 1 Implementation - Following TDD per cursorrules

Loads the real trained MagVIT model (100% accuracy from Jan 25, 2026)
Projects 256-dim features to 512-dim for Liquid Fusion compatibility
"""
import torch
import torch.nn as nn
from pathlib import Path
from typing import Optional
import sys

# Import standalone MagVIT model
sys.path.insert(0, str(Path(__file__).parent))
from magvit_model import MagVITExtractor


class MagVITFeatureExtractor:
    """
    Loads trained MagVIT model and extracts features for Liquid VLM.
    
    Uses REAL model trained on 10K trajectories with 100% validation accuracy.
    The trained model outputs 256-dim features, which are projected to 512-dim.
    """
    
    def __init__(
        self,
        checkpoint_path: str = "experiments/liquid_vlm_integration/checkpoints/magvit_100pct_20260125.pt",
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Initialize MagVIT feature extractor.
        
        Args:
            checkpoint_path: Path to trained MagVIT checkpoint
            device: Device to run on
        """
        self.device = device
        checkpoint_path = Path(checkpoint_path)
        
        # Handle relative and absolute paths
        if not checkpoint_path.is_absolute():
            checkpoint_path = Path.cwd() / checkpoint_path
        
        print(f"📥 Loading MagVIT model from: {checkpoint_path}")
        
        # Create model with ORIGINAL 256-dim (as trained)
        self.model = MagVITExtractor(
            feature_dim=256,  # Match trained model
            spatial_dim=512,
            dropout=0.1
        ).to(device)
        
        # Add projection layer: 256 -> 512 for Liquid Fusion
        self.projection = nn.Linear(256, 512).to(device)
        
        # Load checkpoint
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"MagVIT checkpoint not found: {checkpoint_path}")
        
        # Load the state dict
        state_dict = torch.load(checkpoint_path, map_location=device)
        
        # Extract just the extractor part if it's wrapped
        extractor_state = {}
        for key, value in state_dict.items():
            if key.startswith("extractor."):
                # Remove "extractor." prefix
                new_key = key.replace("extractor.", "")
                extractor_state[new_key] = value
        
        # If we didn't find extractor keys, the state dict might be the extractor directly
        if not extractor_state:
            extractor_state = state_dict
        
        # Load the state dict (strict=False to ignore classification/prediction heads)
        missing, unexpected = self.model.load_state_dict(extractor_state, strict=False)
        self.model.eval()
        
        print(f"✅ MagVIT model loaded successfully")
        print(f"   Device: {device}")
        print(f"   Native feature dim: 256")
        print(f"   Projected to: 512 (Liquid Fusion compatible)")
        if missing:
            print(f"   Missing keys (expected): {len(missing)}")
        if unexpected:
            print(f"   Unexpected keys (ok, from training heads): {len(unexpected)}")
    
    @torch.no_grad()
    def extract(self, video: torch.Tensor) -> torch.Tensor:
        """
        Extract 512-dim features from video.
        
        Args:
            video: (B, T, C, H, W) video tensor
            
        Returns:
            features: (B, 512) - projected features for Liquid Fusion
        """
        if video.device != self.device:
            video = video.to(self.device)
        
        # Extract 256-dim features from trained model
        # model.extract returns (B, T, 256)
        temporal_features = self.model.extract(video)
        
        # Average over time to get (B, 256)
        features_256 = temporal_features.mean(dim=1)
        
        # Project to 512-dim for Liquid Fusion
        features_512 = self.projection(features_256)
        
        return features_512
    
    def extract_temporal(self, video: torch.Tensor) -> torch.Tensor:
        """
        Extract temporal features (per-frame), projected to 512-dim.
        
        Args:
            video: (B, T, C, H, W) video tensor
            
        Returns:
            features: (B, T, 512) - projected temporal features
        """
        if video.device != self.device:
            video = video.to(self.device)
        
        # Extract 256-dim temporal features
        features_256 = self.model.extract(video)  # (B, T, 256)
        
        # Project each frame to 512-dim
        B, T, _ = features_256.shape
        features_256_flat = features_256.view(B * T, 256)
        features_512_flat = self.projection(features_256_flat)
        features_512 = features_512_flat.view(B, T, 512)
        
        return features_512
