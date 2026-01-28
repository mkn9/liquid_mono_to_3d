"""
TDD GREEN Phase: MagVIT Model Loader Tests
Worker 1: Following cursorrules TDD requirements
"""
import torch
import pytest
from pathlib import Path
import sys
import os

sys.path.insert(0, str(Path(__file__).parent.parent))


def test_magvit_checkpoint_exists():
    """Test that MagVIT checkpoint file exists."""
    checkpoint_path = Path("experiments/liquid_vlm_integration/checkpoints/magvit_100pct_20260125.pt")
    
    assert checkpoint_path.exists(), f"MagVIT checkpoint not found at {checkpoint_path}"
    assert checkpoint_path.stat().st_size > 1e7, f"Checkpoint file should be >10MB, got {checkpoint_path.stat().st_size}"
    
    print(f"\\n✅ MagVIT checkpoint found: {checkpoint_path}")
    print(f"   Size: {checkpoint_path.stat().st_size / 1e6:.2f} MB")


def test_magvit_loader_can_import():
    """Test that MagVIT loader module can be imported."""
    try:
        from magvit_loader import MagVITFeatureExtractor
        assert True
    except ImportError as e:
        pytest.fail(f"Cannot import MagVITFeatureExtractor: {e}")


def test_magvit_model_loads():
    """Test that MagVIT model loads successfully."""
    from magvit_loader import MagVITFeatureExtractor
    
    extractor = MagVITFeatureExtractor(
        checkpoint_path="experiments/liquid_vlm_integration/checkpoints/magvit_100pct_20260125.pt",
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    assert extractor.model is not None
    print("\\n✅ MagVIT model loaded successfully")


def test_magvit_extracts_features():
    """Test that MagVIT can extract features from video tensor."""
    from magvit_loader import MagVITFeatureExtractor
    
    extractor = MagVITFeatureExtractor(
        checkpoint_path="experiments/liquid_vlm_integration/checkpoints/magvit_100pct_20260125.pt",
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    # Create dummy video: (B, T, C, H, W) = (1, 16, 3, 64, 64)
    video = torch.randn(1, 16, 3, 64, 64)
    
    features = extractor.extract(video)
    
    # MagVIT should output 512-dim embeddings (compatible with Liquid Fusion)
    assert features.shape == (1, 512), f"Expected (1, 512), got {features.shape}"
    assert torch.isfinite(features).all(), "Features should be finite"
    
    print(f"\\n✅ Features extracted: {features.shape}")


def test_magvit_feature_consistency():
    """Test that same video produces same features (deterministic)."""
    from magvit_loader import MagVITFeatureExtractor
    
    extractor = MagVITFeatureExtractor(
        checkpoint_path="experiments/liquid_vlm_integration/checkpoints/magvit_100pct_20260125.pt",
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    video = torch.randn(1, 16, 3, 64, 64)
    
    features1 = extractor.extract(video)
    features2 = extractor.extract(video)
    
    assert torch.allclose(features1, features2, rtol=1e-4), "Features should be deterministic"
    print("\\n✅ Feature extraction is deterministic")
