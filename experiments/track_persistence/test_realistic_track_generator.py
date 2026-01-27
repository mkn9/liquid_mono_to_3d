#!/usr/bin/env python3
"""
Tests for Realistic 2D Track Generator
"""

import pytest
import numpy as np
from pathlib import Path
import tempfile
import shutil
import json

from realistic_track_generator import (
    Realistic2DTrackGenerator,
    Track2D,
    generate_dataset
)


class TestTrack2D:
    """Test Track2D dataclass."""
    
    def test_track2d_creation(self):
        """Test creating a Track2D object."""
        track = Track2D(
            track_id=1,
            frames=[0, 1, 2],
            bboxes=[(10, 20, 30, 40), (11, 21, 31, 41), (12, 22, 32, 42)],
            confidences=[0.9, 0.85, 0.88],
            pixels=np.random.randint(0, 255, (3, 64, 64, 3), dtype=np.uint8),
            is_persistent=True,
            track_type='persistent',
            duration=3,
            camera_id=0
        )
        
        assert track.track_id == 1
        assert len(track.frames) == 3
        assert len(track.bboxes) == 3
        assert len(track.confidences) == 3
        assert track.pixels.shape == (3, 64, 64, 3)
        assert track.is_persistent
        assert track.track_type == 'persistent'
        assert track.duration == 3
        assert track.camera_id == 0
    
    def test_track2d_to_dict(self):
        """Test converting Track2D to dictionary."""
        track = Track2D(
            track_id=1,
            frames=[0, 1, 2],
            bboxes=[(10, 20, 30, 40), (11, 21, 31, 41), (12, 22, 32, 42)],
            confidences=[0.9, 0.85, 0.88],
            pixels=np.random.randint(0, 255, (3, 64, 64, 3), dtype=np.uint8),
            is_persistent=True,
            track_type='persistent',
            duration=3,
            camera_id=0
        )
        
        track_dict = track.to_dict()
        
        assert 'track_id' in track_dict
        assert 'frames' in track_dict
        assert 'bboxes' in track_dict
        assert 'confidences' in track_dict
        assert 'is_persistent' in track_dict
        assert 'track_type' in track_dict
        assert 'duration' in track_dict
        assert 'camera_id' in track_dict
        assert 'pixels' not in track_dict  # Should not include pixels


class TestRealistic2DTrackGenerator:
    """Test Realistic2DTrackGenerator class."""
    
    @pytest.fixture
    def generator(self):
        """Create a generator instance."""
        return Realistic2DTrackGenerator(seed=42)
    
    def test_generator_initialization(self, generator):
        """Test generator initialization."""
        assert generator.img_width == 1280
        assert generator.img_height == 720
        assert generator.num_frames == 50
        assert generator.rng is not None
    
    def test_generate_persistent_track(self, generator):
        """Test generating a persistent track."""
        track = generator.generate_persistent_track(
            track_id=1,
            camera_id=0,
            start_frame=0,
            min_duration=20,
            max_duration=30
        )
        
        assert track.track_id == 1
        assert track.camera_id == 0
        assert track.is_persistent
        assert track.track_type == 'persistent'
        assert 20 <= track.duration <= 30
        assert len(track.frames) == track.duration
        assert len(track.bboxes) == track.duration
        assert len(track.confidences) == track.duration
        assert track.pixels.shape[0] == track.duration
        assert track.pixels.shape[1:] == (64, 64, 3)
        
        # Check confidence values
        for conf in track.confidences:
            assert 0.85 <= conf <= 0.99
    
    def test_generate_brief_track(self, generator):
        """Test generating a brief track."""
        track = generator.generate_brief_track(
            track_id=2,
            camera_id=1,
            start_frame=10,
            min_duration=2,
            max_duration=5
        )
        
        assert track.track_id == 2
        assert track.camera_id == 1
        assert not track.is_persistent
        assert track.track_type == 'brief'
        assert 2 <= track.duration <= 5
        assert len(track.frames) == track.duration
        assert len(track.bboxes) == track.duration
        assert len(track.confidences) == track.duration
        assert track.pixels.shape[0] == track.duration
        
        # Check confidence values (should be lower than persistent)
        for conf in track.confidences:
            assert 0.3 <= conf <= 0.8
    
    def test_generate_noise_track(self, generator):
        """Test generating a noise track."""
        track = generator.generate_noise_track(
            track_id=3,
            camera_id=0,
            start_frame=20
        )
        
        assert track.track_id == 3
        assert track.camera_id == 0
        assert not track.is_persistent
        assert track.track_type == 'noise'
        assert track.duration == 1
        assert len(track.frames) == 1
        assert len(track.bboxes) == 1
        assert len(track.confidences) == 1
        assert track.pixels.shape[0] == 1
        
        # Check confidence values (should be low)
        assert 0.3 <= track.confidences[0] <= 0.6
    
    def test_generate_scene(self, generator):
        """Test generating a complete scene."""
        tracks = generator.generate_scene(
            scene_id=1,
            num_persistent=2,
            num_brief=3,
            num_noise=2,
            camera_id=0
        )
        
        assert len(tracks) == 7  # 2 + 3 + 2
        
        # Count track types
        persistent_count = sum(1 for t in tracks if t.track_type == 'persistent')
        brief_count = sum(1 for t in tracks if t.track_type == 'brief')
        noise_count = sum(1 for t in tracks if t.track_type == 'noise')
        
        assert persistent_count == 2
        assert brief_count == 3
        assert noise_count == 2
        
        # Check all tracks have same camera_id
        assert all(t.camera_id == 0 for t in tracks)
    
    def test_bbox_within_bounds(self, generator):
        """Test that bounding boxes stay within image bounds."""
        track = generator.generate_persistent_track(
            track_id=1,
            camera_id=0,
            start_frame=0
        )
        
        for bbox in track.bboxes:
            x, y, w, h = bbox
            assert 0 <= x <= generator.img_width
            assert 0 <= y <= generator.img_height
            assert x + w <= generator.img_width
            assert y + h <= generator.img_height
    
    def test_pixel_values_valid(self, generator):
        """Test that pixel values are in valid range."""
        track = generator.generate_persistent_track(
            track_id=1,
            camera_id=0,
            start_frame=0
        )
        
        assert track.pixels.dtype == np.uint8
        assert np.all(track.pixels >= 0)
        assert np.all(track.pixels <= 255)


class TestDatasetGeneration:
    """Test dataset generation."""
    
    def test_generate_small_dataset(self):
        """Test generating a small dataset."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            
            # Generate small dataset
            generate_dataset(
                output_dir=output_dir,
                num_scenes=5,
                persistent_ratio=0.6,
                brief_ratio=0.3,
                noise_ratio=0.1,
                seed=42
            )
            
            # Check files exist
            assert (output_dir / "tracks_metadata.json").exists()
            assert (output_dir / "dataset_summary.json").exists()
            
            # Load and check metadata
            with open(output_dir / "tracks_metadata.json", 'r') as f:
                metadata = json.load(f)
            
            assert len(metadata) > 0
            
            # Load and check summary
            with open(output_dir / "dataset_summary.json", 'r') as f:
                summary = json.load(f)
            
            assert summary['total_scenes'] == 5
            assert summary['total_tracks'] > 0
            assert 'num_persistent' in summary
            assert 'num_brief' in summary
            assert 'num_noise' in summary
            assert 'avg_persistent_duration' in summary
            assert 'avg_brief_duration' in summary
            
            # Check pixel files exist for at least one track
            first_track = metadata[0]
            pixel_file = output_dir / f"track_{first_track['track_id']}_pixels.npy"
            assert pixel_file.exists()
            
            # Load and check pixel file
            pixels = np.load(pixel_file)
            assert pixels.ndim == 4  # (T, H, W, 3)
            assert pixels.shape[1:] == (64, 64, 3)


def test_reproducibility():
    """Test that same seed produces same results."""
    gen1 = Realistic2DTrackGenerator(seed=123)
    gen2 = Realistic2DTrackGenerator(seed=123)
    
    track1 = gen1.generate_persistent_track(track_id=1, camera_id=0, start_frame=0)
    track2 = gen2.generate_persistent_track(track_id=1, camera_id=0, start_frame=0)
    
    assert track1.duration == track2.duration
    assert track1.frames == track2.frames
    assert np.allclose(track1.confidences, track2.confidences)
    
    # Bounding boxes should be very similar (within small numerical error)
    for bbox1, bbox2 in zip(track1.bboxes, track2.bboxes):
        assert all(abs(a - b) < 5 for a, b in zip(bbox1, bbox2))


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

