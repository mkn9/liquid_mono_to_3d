"""
TDD Tests for MagVIT Feature Extractor
=======================================

RED Phase: Tests should FAIL initially
GREEN Phase: Tests should PASS after implementation
REFACTOR Phase: Tests should still PASS after cleanup

MagVIT (Masked Generative Video Transformer) is a tokenizer-based approach
that quantizes video into discrete tokens for generation tasks.

Test Requirements:
1. MagVITExtractor inherits from FeatureExtractor
2. Uses video tokenization/quantization
3. Returns (B, T, D) features
4. Handles different batch sizes and temporal lengths
5. Integrates with UnifiedModel
"""

import pytest
import torch
import sys
from pathlib import Path

# Add shared module to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'shared'))
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_magvit_extractor_inherits_from_base():
    """Test 1: MagVITExtractor inherits from FeatureExtractor."""
    from base_extractor import FeatureExtractor
    from feature_extractor import MagVITExtractor
    
    extractor = MagVITExtractor(feature_dim=256)
    assert isinstance(extractor, FeatureExtractor)


def test_magvit_extractor_feature_dim_property():
    """Test 2: feature_dim property returns correct dimension."""
    from feature_extractor import MagVITExtractor
    
    for dim in [128, 256, 512]:
        extractor = MagVITExtractor(feature_dim=dim)
        assert extractor.feature_dim == dim


def test_magvit_extractor_extract_output_shape():
    """Test 3: extract() returns correct shape (B, T, D)."""
    from feature_extractor import MagVITExtractor
    
    extractor = MagVITExtractor(feature_dim=256)
    
    video = torch.randn(2, 16, 3, 64, 64)
    features = extractor.extract(video)
    
    assert features.shape == (2, 16, 256)
    assert torch.all(torch.isfinite(features))


def test_magvit_extractor_batch_size_flexibility():
    """Test 4: Works with different batch sizes."""
    from feature_extractor import MagVITExtractor
    
    extractor = MagVITExtractor(feature_dim=512)
    
    for batch_size in [1, 4, 8]:
        video = torch.randn(batch_size, 16, 3, 64, 64)
        features = extractor.extract(video)
        assert features.shape == (batch_size, 16, 512)


def test_magvit_extractor_temporal_flexibility():
    """Test 5: Works with different temporal lengths."""
    from feature_extractor import MagVITExtractor
    
    extractor = MagVITExtractor(feature_dim=256)
    
    for num_frames in [8, 16, 32]:
        video = torch.randn(2, num_frames, 3, 64, 64)
        features = extractor.extract(video)
        assert features.shape == (2, num_frames, 256)


def test_magvit_extractor_has_encoder():
    """Test 6: MagVIT uses an encoder architecture."""
    from feature_extractor import MagVITExtractor
    
    extractor = MagVITExtractor(feature_dim=256)
    
    # Check if encoder exists
    assert hasattr(extractor, 'encoder') or hasattr(extractor, 'vqvae') or hasattr(extractor, 'tokenizer')


def test_magvit_extractor_integration_with_unified_model():
    """Test 7: Can be integrated with UnifiedModel."""
    from unified_model import UnifiedModel, compute_loss
    from feature_extractor import MagVITExtractor
    
    extractor = MagVITExtractor(feature_dim=256)
    model = UnifiedModel(extractor, num_classes=4)
    
    video = torch.randn(2, 16, 3, 64, 64)
    output = model(video)
    
    assert 'classification' in output
    assert 'prediction' in output
    assert output['classification'].shape == (2, 4)
    assert output['prediction'].shape == (2, 3)


def test_magvit_extractor_training_step():
    """Test 8: Can perform training step (forward + backward)."""
    from unified_model import UnifiedModel, compute_loss
    from feature_extractor import MagVITExtractor
    
    extractor = MagVITExtractor(feature_dim=128)
    model = UnifiedModel(extractor, num_classes=4)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Forward
    video = torch.randn(2, 8, 3, 64, 64)
    output = model(video)
    
    targets = {
        'class_label': torch.tensor([0, 2]),
        'future_position': torch.randn(2, 3)
    }
    
    total_loss, _, _ = compute_loss(output, targets)
    
    # Backward
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()
    
    # Check gradients
    for param in model.parameters():
        if param.requires_grad:
            assert param.grad is not None
            assert torch.all(torch.isfinite(param.grad))


def test_magvit_extractor_deterministic():
    """Test 9: Same input produces same output (with eval mode)."""
    from feature_extractor import MagVITExtractor
    
    extractor = MagVITExtractor(feature_dim=256)
    extractor.eval()
    
    video = torch.randn(2, 16, 3, 64, 64)
    
    with torch.no_grad():
        features_1 = extractor.extract(video)
        features_2 = extractor.extract(video)
    
    assert torch.allclose(features_1, features_2)


def test_magvit_extractor_efficient():
    """Test 10: MagVIT is computationally efficient (fewer params than I3D)."""
    from feature_extractor import MagVITExtractor
    
    extractor = MagVITExtractor(feature_dim=256)
    
    # Count parameters
    total_params = sum(p.numel() for p in extractor.parameters())
    
    # MagVIT should be relatively lightweight (<10M parameters)
    assert total_params < 10_000_000, f"Too many parameters: {total_params:,}"


# Mark this file as TDD evidence
if __name__ == "__main__":
    print("=" * 70)
    print("TDD RED PHASE: Running tests for MagVITExtractor")
    print("Expected: ALL TESTS SHOULD FAIL (module doesn't exist yet)")
    print("=" * 70)
    pytest.main([__file__, "-v", "--tb=short"])

