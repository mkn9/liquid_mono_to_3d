"""
Test suite for object tokenizer (TDD RED Phase).

Tests conversion of detected objects to transformer tokens.
"""

import pytest
import torch
import numpy as np
from pathlib import Path
import sys

# Add parent dir to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.object_tokenizer import ObjectTokenizer, ObjectToken
from src.object_detector import DetectedObject


class TestObjectTokenizer:
    """Test object tokenizer functionality."""
    
    @pytest.fixture
    def tokenizer(self):
        """Create tokenizer instance."""
        return ObjectTokenizer(
            feature_dim=256,
            max_frames=16,
            max_objects_per_frame=4
        )
    
    @pytest.fixture
    def sample_detected_object(self):
        """Create sample detected object."""
        return DetectedObject(
            bbox=(10, 15, 8, 8),
            confidence=0.95,
            class_id=0
        )
    
    @pytest.fixture
    def sample_frame(self):
        """Create sample frame."""
        frame = np.zeros((64, 64, 3), dtype=np.float32)
        # White sphere at (14, 19) (center of bbox 10, 15, 8, 8)
        for i in range(11, 19):
            for j in range(16, 24):
                if (i - 14)**2 + (j - 19)**2 <= 16:
                    frame[i, j] = [1.0, 1.0, 1.0]
        return frame
    
    @pytest.fixture
    def sample_video_sequence(self):
        """Create sample video sequence (3 frames)."""
        frames = []
        for t in range(3):
            frame = np.zeros((64, 64, 3), dtype=np.float32)
            # Moving sphere
            center_x = 20 + t * 10
            for i in range(center_x - 4, center_x + 4):
                for j in range(28, 36):
                    if (i - center_x)**2 + (j - 32)**2 <= 16:
                        frame[i, j] = [1.0, 1.0, 1.0]
            frames.append(frame)
        return np.array(frames)
    
    def test_tokenizer_initialization(self, tokenizer):
        """Test tokenizer initializes correctly."""
        assert tokenizer is not None
        assert tokenizer.feature_dim == 256
        assert tokenizer.max_frames == 16
        assert tokenizer.max_objects_per_frame == 4
    
    def test_extract_object_patch(self, tokenizer, sample_frame, sample_detected_object):
        """Test extracting object patch from frame."""
        patch = tokenizer.extract_patch(sample_frame, sample_detected_object)
        
        # Check patch dimensions
        assert patch.shape == (8, 8, 3)  # bbox width x height x channels
        
        # Check patch contains object (should have white pixels)
        assert patch.max() > 0.5
    
    def test_encode_patch_to_features(self, tokenizer, sample_frame, sample_detected_object):
        """Test encoding patch to feature vector."""
        patch = tokenizer.extract_patch(sample_frame, sample_detected_object)
        features = tokenizer.encode_patch(patch)
        
        # Check feature dimensions
        assert features.shape == (256,)  # feature_dim
        
        # Check features are normalized
        assert torch.is_tensor(features)
    
    def test_create_object_token(self, tokenizer, sample_frame, sample_detected_object):
        """Test creating object token from detection."""
        token = tokenizer.create_token(
            frame=sample_frame,
            detected_object=sample_detected_object,
            frame_idx=0,
            track_id=1
        )
        
        # Check token type
        assert isinstance(token, ObjectToken)
        
        # Check token properties
        assert token.features.shape == (256,)
        assert token.frame_idx == 0
        assert token.track_id == 1
        assert token.bbox == sample_detected_object.bbox
        assert token.confidence == sample_detected_object.confidence
    
    def test_add_positional_encoding(self, tokenizer):
        """Test adding positional encoding to features."""
        features = torch.randn(256)
        frame_idx = 5
        
        encoded = tokenizer.add_positional_encoding(features, frame_idx)
        
        # Check dimensions preserved
        assert encoded.shape == features.shape
        
        # Check features changed (positional info added)
        assert not torch.allclose(encoded, features)
    
    def test_positional_encoding_uniqueness(self, tokenizer):
        """Test different frames get different positional encodings."""
        features = torch.randn(256)
        
        encoded_0 = tokenizer.add_positional_encoding(features, frame_idx=0)
        encoded_5 = tokenizer.add_positional_encoding(features, frame_idx=5)
        
        # Different frame indices should yield different encodings
        assert not torch.allclose(encoded_0, encoded_5)
    
    def test_tokenize_single_frame(self, tokenizer, sample_frame):
        """Test tokenizing single frame with multiple objects."""
        detections = [
            DetectedObject(bbox=(10, 10, 8, 8), confidence=0.9, class_id=0),
            DetectedObject(bbox=(30, 30, 8, 8), confidence=0.85, class_id=0)
        ]
        track_ids = [1, 1]  # Both belong to same track
        
        tokens = tokenizer.tokenize_frame(
            frame=sample_frame,
            detections=detections,
            track_ids=track_ids,
            frame_idx=0
        )
        
        # Should create 2 tokens
        assert len(tokens) == 2
        
        # Check all tokens have correct frame index
        assert all(t.frame_idx == 0 for t in tokens)
        
        # Check track IDs assigned
        assert all(t.track_id == 1 for t in tokens)
    
    def test_tokenize_video_sequence(self, tokenizer, sample_video_sequence):
        """Test tokenizing entire video sequence."""
        # Mock detections for each frame
        all_detections = [
            [DetectedObject(bbox=(20, 28, 8, 8), confidence=0.9, class_id=0)],
            [DetectedObject(bbox=(30, 28, 8, 8), confidence=0.9, class_id=0)],
            [DetectedObject(bbox=(40, 28, 8, 8), confidence=0.9, class_id=0)]
        ]
        all_track_ids = [[1], [1], [1]]  # Same track across frames
        
        token_sequence = tokenizer.tokenize_video(
            video=sample_video_sequence,
            detections_per_frame=all_detections,
            track_ids_per_frame=all_track_ids
        )
        
        # Should create 3 tokens (1 per frame)
        assert len(token_sequence) == 3
        
        # Check temporal ordering
        assert token_sequence[0].frame_idx == 0
        assert token_sequence[1].frame_idx == 1
        assert token_sequence[2].frame_idx == 2
        
        # Check all same track
        assert all(t.track_id == 1 for t in token_sequence)
    
    def test_tokenize_multiple_tracks(self, tokenizer):
        """Test tokenizing frame with multiple tracks."""
        frame = np.zeros((64, 64, 3), dtype=np.float32)
        
        detections = [
            DetectedObject(bbox=(10, 10, 8, 8), confidence=0.9, class_id=0),  # Track 1
            DetectedObject(bbox=(30, 30, 8, 8), confidence=0.85, class_id=0), # Track 2
            DetectedObject(bbox=(50, 10, 8, 8), confidence=0.8, class_id=0)   # Track 1
        ]
        track_ids = [1, 2, 1]
        
        tokens = tokenizer.tokenize_frame(
            frame=frame,
            detections=detections,
            track_ids=track_ids,
            frame_idx=0
        )
        
        # Should create 3 tokens
        assert len(tokens) == 3
        
        # Check track IDs preserved
        assert tokens[0].track_id == 1
        assert tokens[1].track_id == 2
        assert tokens[2].track_id == 1
    
    def test_create_sequence_tensor(self, tokenizer):
        """Test creating batched tensor from token sequence."""
        tokens = [
            ObjectToken(
                features=torch.randn(256),
                frame_idx=i,
                track_id=1,
                bbox=(10, 10, 8, 8),
                confidence=0.9
            )
            for i in range(5)
        ]
        
        sequence_tensor = tokenizer.tokens_to_tensor(tokens)
        
        # Check shape: (seq_len, feature_dim)
        assert sequence_tensor.shape == (5, 256)
    
    def test_padding_short_sequences(self, tokenizer):
        """Test padding sequences shorter than max_frames."""
        tokens = [
            ObjectToken(
                features=torch.randn(256),
                frame_idx=i,
                track_id=1,
                bbox=(10, 10, 8, 8),
                confidence=0.9
            )
            for i in range(5)
        ]
        
        # Tokenizer max_frames=16, but we only have 5
        sequence_tensor, mask = tokenizer.tokens_to_tensor_padded(tokens)
        
        # Check padded shape
        assert sequence_tensor.shape == (16, 256)
        
        # Check mask (first 5 True, rest False)
        assert mask[:5].all()
        assert not mask[5:].any()
    
    def test_truncate_long_sequences(self, tokenizer):
        """Test truncating sequences longer than max_frames."""
        tokens = [
            ObjectToken(
                features=torch.randn(256),
                frame_idx=i,
                track_id=1,
                bbox=(10, 10, 8, 8),
                confidence=0.9
            )
            for i in range(20)  # More than max_frames=16
        ]
        
        sequence_tensor = tokenizer.tokens_to_tensor(tokens)
        
        # Should truncate to max_frames
        assert sequence_tensor.shape == (16, 256)


class TestObjectToken:
    """Test ObjectToken data structure."""
    
    def test_object_token_creation(self):
        """Test creating ObjectToken."""
        features = torch.randn(256)
        token = ObjectToken(
            features=features,
            frame_idx=3,
            track_id=5,
            bbox=(10, 20, 8, 8),
            confidence=0.95
        )
        
        assert torch.equal(token.features, features)
        assert token.frame_idx == 3
        assert token.track_id == 5
        assert token.bbox == (10, 20, 8, 8)
        assert token.confidence == 0.95
    
    def test_object_token_representation(self):
        """Test string representation of token."""
        token = ObjectToken(
            features=torch.randn(256),
            frame_idx=2,
            track_id=7,
            bbox=(10, 10, 8, 8),
            confidence=0.88
        )
        
        repr_str = repr(token)
        assert 'frame_idx=2' in repr_str
        assert 'track_id=7' in repr_str
        assert '0.88' in repr_str


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

