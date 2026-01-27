"""
TDD Tests for Pre-trained ResNet Tokenizer
Worker 2: Use frozen ResNet-18 features instead of simple CNN
"""

import pytest
import torch
import torch.nn as nn
from pathlib import Path
import sys
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pretrained_tokenizer import (
    PretrainedTokenizer,
    ObjectToken
)


class TestPretrainedTokenizer:
    """Test ResNet-based object tokenizer."""
    
    def test_tokenizer_initialization(self):
        """Test tokenizer can be initialized with ResNet."""
        tokenizer = PretrainedTokenizer(feature_dim=256)
        
        assert tokenizer.feature_dim == 256
        assert hasattr(tokenizer, 'feature_extractor')
        assert hasattr(tokenizer, 'projection')
    
    def test_resnet_is_frozen(self):
        """Test that ResNet parameters are frozen."""
        tokenizer = PretrainedTokenizer(feature_dim=256)
        
        # Check that feature extractor parameters don't require grad
        for param in tokenizer.feature_extractor.parameters():
            assert param.requires_grad is False, "ResNet should be frozen"
    
    def test_projection_is_trainable(self):
        """Test that projection layer is trainable."""
        tokenizer = PretrainedTokenizer(feature_dim=256)
        
        # Check that projection parameters DO require grad
        for param in tokenizer.projection.parameters():
            assert param.requires_grad is True, "Projection should be trainable"
    
    def test_extract_patch(self):
        """Test extracting object patch from frame."""
        tokenizer = PretrainedTokenizer(feature_dim=256)
        
        frame = np.random.rand(64, 64, 3).astype(np.float32)
        bbox = (10, 10, 20, 20)  # x, y, width, height
        
        patch = tokenizer.extract_patch(frame, bbox)
        
        # Should be resized to 224x224 for ResNet
        assert patch.shape == (224, 224, 3)
        assert patch.min() >= 0
        assert patch.max() <= 1
    
    def test_encode_patch_to_features(self):
        """Test encoding patch with ResNet."""
        tokenizer = PretrainedTokenizer(feature_dim=256)
        
        patch = np.random.rand(224, 224, 3).astype(np.float32)
        
        features = tokenizer.encode_patch_to_features(patch)
        
        # Should output 256-dim features (after projection)
        assert features.shape == (256,)
        assert isinstance(features, torch.Tensor)
    
    def test_resnet_output_is_512dim_before_projection(self):
        """Test ResNet outputs 512-dim before projection."""
        tokenizer = PretrainedTokenizer(feature_dim=256)
        
        # Create dummy input
        batch = torch.randn(1, 3, 224, 224)
        
        with torch.no_grad():
            resnet_output = tokenizer.feature_extractor(batch)
        
        # ResNet-18 outputs 512-dim features
        assert resnet_output.shape == (1, 512, 1, 1) or resnet_output.shape == (1, 512)
    
    def test_tokenize_single_object(self):
        """Test tokenizing a single detected object."""
        tokenizer = PretrainedTokenizer(feature_dim=256)
        
        frame = np.random.rand(64, 64, 3).astype(np.float32)
        
        # Mock detected object
        class MockDetection:
            def __init__(self):
                self.bbox = (10, 10, 20, 20)
                self.confidence = 0.95
        
        detection = MockDetection()
        track_id = 1
        frame_idx = 5
        
        token = tokenizer.create_token(
            frame=frame,
            detection=detection,
            track_id=track_id,
            frame_idx=frame_idx
        )
        
        # Should return ObjectToken
        assert isinstance(token, ObjectToken)
        assert token.features.shape == (256,)
        assert token.track_id == 1
        assert token.frame_idx == 5
        assert token.confidence == 0.95
    
    def test_tokenize_frame_with_multiple_objects(self):
        """Test tokenizing multiple objects in a frame."""
        tokenizer = PretrainedTokenizer(feature_dim=256)
        
        frame = np.random.rand(64, 64, 3).astype(np.float32)
        
        # Mock multiple detections
        class MockDetection:
            def __init__(self, bbox):
                self.bbox = bbox
                self.confidence = 0.9
        
        detections = [
            MockDetection((10, 10, 20, 20)),
            MockDetection((30, 30, 15, 15)),
            MockDetection((5, 50, 10, 10))
        ]
        track_ids = [1, 2, 3]
        frame_idx = 0
        
        tokens = tokenizer.tokenize_frame(
            frame=frame,
            detections=detections,
            track_ids=track_ids,
            frame_idx=frame_idx
        )
        
        assert len(tokens) == 3
        assert all(isinstance(t, ObjectToken) for t in tokens)
        assert all(t.features.shape == (256,) for t in tokens)
    
    def test_batch_tokenization(self):
        """Test tokenizing multiple frames efficiently."""
        tokenizer = PretrainedTokenizer(feature_dim=256)
        
        # Mock video with multiple frames
        video = np.random.rand(10, 64, 64, 3).astype(np.float32)
        
        # Each frame has 2 objects
        detections_per_frame = []
        for _ in range(10):
            class MockDetection:
                def __init__(self, bbox):
                    self.bbox = bbox
                    self.confidence = 0.9
            
            detections_per_frame.append([
                MockDetection((10, 10, 20, 20)),
                MockDetection((30, 30, 15, 15))
            ])
        
        all_tokens = []
        for frame_idx, (frame, detections) in enumerate(zip(video, detections_per_frame)):
            tokens = tokenizer.tokenize_frame(
                frame=frame,
                detections=detections,
                track_ids=[1, 2],
                frame_idx=frame_idx
            )
            all_tokens.extend(tokens)
        
        # Should have 10 frames * 2 objects = 20 tokens
        assert len(all_tokens) == 20


class TestResNetIntegration:
    """Test ResNet integration and feature quality."""
    
    def test_resnet_produces_different_features_for_different_patches(self):
        """Test ResNet produces different features for different inputs."""
        tokenizer = PretrainedTokenizer(feature_dim=256)
        
        # Create two different patches
        patch1 = np.ones((224, 224, 3), dtype=np.float32) * 0.2  # Dark
        patch2 = np.ones((224, 224, 3), dtype=np.float32) * 0.8  # Bright
        
        features1 = tokenizer.encode_patch_to_features(patch1)
        features2 = tokenizer.encode_patch_to_features(patch2)
        
        # Features should be different
        assert not torch.allclose(features1, features2, atol=0.01)
    
    def test_resnet_features_are_normalized(self):
        """Test that ResNet features are reasonably normalized."""
        tokenizer = PretrainedTokenizer(feature_dim=256)
        
        patch = np.random.rand(224, 224, 3).astype(np.float32)
        features = tokenizer.encode_patch_to_features(patch)
        
        # Features should not explode
        assert features.abs().max() < 100
        assert features.abs().mean() < 10
    
    def test_feature_consistency(self):
        """Test that same input produces same features (deterministic)."""
        tokenizer = PretrainedTokenizer(feature_dim=256)
        tokenizer.eval()  # Ensure eval mode
        
        patch = np.random.rand(224, 224, 3).astype(np.float32)
        
        features1 = tokenizer.encode_patch_to_features(patch)
        features2 = tokenizer.encode_patch_to_features(patch)
        
        # Should be identical
        assert torch.allclose(features1, features2, atol=1e-6)


# RED PHASE: These tests should FAIL initially
if __name__ == "__main__":
    pytest.main([__file__, '-v'])

