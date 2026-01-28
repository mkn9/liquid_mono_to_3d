"""
Worker 2: Real 2D Feature Extraction Tests (TDD GREEN Phase)
"""
import torch
import pytest
import numpy as np
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

def test_can_load_real_video_sample():
    """Test loading real trajectory video from project."""
    from extract_2d_features import load_trajectory_video
    
    # Use real video from simple_3d_tracker
    video = load_trajectory_video(sample_id=0)
    
    assert video is not None
    assert video.shape[0] == 1, "Batch size should be 1"
    assert video.shape[1] >= 5, "Should have at least 5 frames"
    assert video.shape[2] == 3, "Should have RGB channels"
    print(f"\n✅ Real video loaded: {video.shape}")

def test_extract_2d_features_from_real_video():
    """Test extracting 512-dim features from real video."""
    from extract_2d_features import extract_2d_features_batch
    
    features = extract_2d_features_batch(num_samples=3)
    
    assert features.shape[0] == 3, "Should extract 3 samples"
    assert features.shape[1] == 512, "Should be 512-dim"
    assert torch.isfinite(features).all()
    print(f"\n✅ Real 2D features extracted: {features.shape}")

def test_features_have_reasonable_distribution():
    """Test that features have reasonable statistical properties."""
    from extract_2d_features import extract_2d_features_batch
    
    features = extract_2d_features_batch(num_samples=5)
    
    # Check reasonable statistics
    mean = features.mean().item()
    std = features.std().item()
    
    assert -10 < mean < 10, "Mean should be reasonable"
    assert 0 < std < 10, "Std should be reasonable"
    print(f"\n✅ Feature stats: mean={mean:.3f}, std={std:.3f}")
