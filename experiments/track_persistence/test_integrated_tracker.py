#!/usr/bin/env python3
"""
Tests for Integrated 3D Tracker with Persistence Filtering
"""

import pytest
import numpy as np
import torch
from pathlib import Path
import tempfile

from integrated_3d_tracker import (
    PersistenceFilter,
    Integrated3DTracker
)


class MockPersistenceFilter:
    """Mock persistence filter for testing."""
    
    def __init__(self, always_persistent=True):
        self.always_persistent = always_persistent
        self.predict_call_count = 0
    
    def predict(self, track_pixels):
        """Mock predict method."""
        self.predict_call_count += 1
        
        is_persistent = self.always_persistent
        confidence = 0.9 if is_persistent else 0.3
        attention = np.ones(track_pixels.shape[0]) / track_pixels.shape[0]
        
        return is_persistent, confidence, attention
    
    def get_explanation(self, attention_weights, track_duration):
        """Mock explanation method."""
        return f"Mock explanation for {track_duration} frames"


class TestIntegrated3DTracker:
    """Test Integrated3DTracker."""
    
    @pytest.fixture
    def mock_filter_keep_all(self):
        """Create a mock filter that keeps all tracks."""
        return MockPersistenceFilter(always_persistent=True)
    
    @pytest.fixture
    def mock_filter_reject_all(self):
        """Create a mock filter that rejects all tracks."""
        return MockPersistenceFilter(always_persistent=False)
    
    @pytest.fixture
    def sample_tracks(self):
        """Create sample track data."""
        # Create 3 sample tracks
        camera1_tracks = []
        camera2_tracks = []
        
        for i in range(3):
            # Each track has 5 frames
            track1 = {
                'frames': list(range(5)),
                'bboxes': [(100 + i*10 + j, 200 + j, 50, 50) for j in range(5)],
                'pixels': np.random.randint(0, 255, (5, 64, 64, 3), dtype=np.uint8)
            }
            
            track2 = {
                'frames': list(range(5)),
                'bboxes': [(150 + i*10 + j, 200 + j, 50, 50) for j in range(5)],
                'pixels': np.random.randint(0, 255, (5, 64, 64, 3), dtype=np.uint8)
            }
            
            camera1_tracks.append(track1)
            camera2_tracks.append(track2)
        
        return camera1_tracks, camera2_tracks
    
    def test_tracker_creation_without_filter(self):
        """Test creating tracker without persistence filter."""
        tracker = Integrated3DTracker(
            persistence_filter=None,
            use_filter=False
        )
        
        assert tracker.persistence_filter is None
        assert not tracker.use_filter
        assert tracker.P1 is not None
        assert tracker.P2 is not None
    
    def test_tracker_creation_with_filter(self, mock_filter_keep_all):
        """Test creating tracker with persistence filter."""
        tracker = Integrated3DTracker(
            persistence_filter=mock_filter_keep_all,
            use_filter=True
        )
        
        assert tracker.persistence_filter is not None
        assert tracker.use_filter
    
    def test_process_scene_without_filter(self, sample_tracks):
        """Test processing scene without filtering."""
        camera1_tracks, camera2_tracks = sample_tracks
        
        tracker = Integrated3DTracker(
            persistence_filter=None,
            use_filter=False
        )
        
        reconstructed_3d, decisions = tracker.process_scene(
            camera1_tracks,
            camera2_tracks,
            verbose=False
        )
        
        # Should keep all 3 tracks
        assert len(reconstructed_3d) == 3
        
        # Each track should have 5 3D points
        for points in reconstructed_3d:
            assert points.shape == (5, 3)
        
        # Check statistics
        stats = tracker.get_statistics()
        assert stats['total_tracks'] == 3
        assert stats['kept'] == 3
        assert stats['filtered_out'] == 0
    
    def test_process_scene_with_filter_keep_all(self, mock_filter_keep_all, sample_tracks):
        """Test processing scene with filter that keeps all tracks."""
        camera1_tracks, camera2_tracks = sample_tracks
        
        tracker = Integrated3DTracker(
            persistence_filter=mock_filter_keep_all,
            use_filter=True
        )
        
        reconstructed_3d, decisions = tracker.process_scene(
            camera1_tracks,
            camera2_tracks,
            verbose=False
        )
        
        # Should keep all 3 tracks
        assert len(reconstructed_3d) == 3
        assert len(decisions) == 3
        
        # Check filter was called
        assert mock_filter_keep_all.predict_call_count == 3
        
        # Check decisions
        for track_id, decision in decisions.items():
            assert decision['is_persistent']
            assert 'confidence' in decision
            assert 'attention_weights' in decision
            assert 'explanation' in decision
        
        # Check statistics
        stats = tracker.get_statistics()
        assert stats['total_tracks'] == 3
        assert stats['kept'] == 3
        assert stats['filtered_out'] == 0
    
    def test_process_scene_with_filter_reject_all(self, mock_filter_reject_all, sample_tracks):
        """Test processing scene with filter that rejects all tracks."""
        camera1_tracks, camera2_tracks = sample_tracks
        
        tracker = Integrated3DTracker(
            persistence_filter=mock_filter_reject_all,
            use_filter=True
        )
        
        reconstructed_3d, decisions = tracker.process_scene(
            camera1_tracks,
            camera2_tracks,
            verbose=False
        )
        
        # Should filter out all tracks
        assert len(reconstructed_3d) == 0
        assert len(decisions) == 3
        
        # Check filter was called
        assert mock_filter_reject_all.predict_call_count == 3
        
        # Check decisions
        for track_id, decision in decisions.items():
            assert not decision['is_persistent']
        
        # Check statistics
        stats = tracker.get_statistics()
        assert stats['total_tracks'] == 3
        assert stats['kept'] == 0
        assert stats['filtered_out'] == 3
    
    def test_mismatched_track_counts(self):
        """Test handling of mismatched track counts."""
        camera1_tracks = [
            {
                'frames': [0, 1],
                'bboxes': [(100, 200, 50, 50), (101, 201, 50, 50)],
                'pixels': np.random.randint(0, 255, (2, 64, 64, 3), dtype=np.uint8)
            }
        ]
        
        camera2_tracks = [
            {
                'frames': [0, 1],
                'bboxes': [(150, 200, 50, 50), (151, 201, 50, 50)],
                'pixels': np.random.randint(0, 255, (2, 64, 64, 3), dtype=np.uint8)
            },
            {
                'frames': [0, 1],
                'bboxes': [(160, 210, 50, 50), (161, 211, 50, 50)],
                'pixels': np.random.randint(0, 255, (2, 64, 64, 3), dtype=np.uint8)
            }
        ]
        
        tracker = Integrated3DTracker(
            persistence_filter=None,
            use_filter=False
        )
        
        # Should process min(len(cam1), len(cam2)) = 1 track
        reconstructed_3d, _ = tracker.process_scene(
            camera1_tracks,
            camera2_tracks,
            verbose=False
        )
        
        assert len(reconstructed_3d) == 1
    
    def test_statistics_accumulation(self, mock_filter_keep_all, sample_tracks):
        """Test that statistics accumulate across multiple scenes."""
        camera1_tracks, camera2_tracks = sample_tracks
        
        tracker = Integrated3DTracker(
            persistence_filter=mock_filter_keep_all,
            use_filter=True
        )
        
        # Process first scene
        tracker.process_scene(camera1_tracks, camera2_tracks, verbose=False)
        stats1 = tracker.get_statistics()
        
        # Process second scene
        tracker.process_scene(camera1_tracks, camera2_tracks, verbose=False)
        stats2 = tracker.get_statistics()
        
        # Statistics should accumulate
        assert stats2['total_tracks'] == stats1['total_tracks'] * 2
        assert stats2['kept'] == stats1['kept'] * 2
    
    def test_visualize_results(self, sample_tracks):
        """Test visualization method (check it doesn't crash)."""
        camera1_tracks, camera2_tracks = sample_tracks
        
        tracker = Integrated3DTracker(
            persistence_filter=None,
            use_filter=False
        )
        
        reconstructed_3d, _ = tracker.process_scene(
            camera1_tracks,
            camera2_tracks,
            verbose=False
        )
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test_viz.png"
            
            # Should not raise exception
            tracker.visualize_results(reconstructed_3d, str(output_path))
            
            # File should be created
            assert output_path.exists()
    
    def test_empty_tracks(self):
        """Test handling of empty track lists."""
        tracker = Integrated3DTracker(
            persistence_filter=None,
            use_filter=False
        )
        
        reconstructed_3d, decisions = tracker.process_scene(
            [],
            [],
            verbose=False
        )
        
        assert len(reconstructed_3d) == 0
        assert len(decisions) == 0
        
        stats = tracker.get_statistics()
        assert stats['total_tracks'] == 0
        assert stats['kept'] == 0


class TestMockPersistenceFilter:
    """Test the mock persistence filter itself."""
    
    def test_mock_filter_keep_all(self):
        """Test mock filter that keeps all tracks."""
        filter = MockPersistenceFilter(always_persistent=True)
        
        track_pixels = np.random.randint(0, 255, (10, 64, 64, 3), dtype=np.uint8)
        is_persistent, confidence, attention = filter.predict(track_pixels)
        
        assert is_persistent
        assert confidence > 0.5
        assert attention.shape == (10,)
        assert filter.predict_call_count == 1
    
    def test_mock_filter_reject_all(self):
        """Test mock filter that rejects all tracks."""
        filter = MockPersistenceFilter(always_persistent=False)
        
        track_pixels = np.random.randint(0, 255, (10, 64, 64, 3), dtype=np.uint8)
        is_persistent, confidence, attention = filter.predict(track_pixels)
        
        assert not is_persistent
        assert confidence < 0.5
        assert attention.shape == (10,)
        assert filter.predict_call_count == 1
    
    def test_mock_filter_explanation(self):
        """Test mock filter explanation."""
        filter = MockPersistenceFilter()
        
        attention = np.array([0.1, 0.2, 0.3, 0.4])
        explanation = filter.get_explanation(attention, 4)
        
        assert "Mock explanation" in explanation
        assert "4 frames" in explanation


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

