"""
TDD Tests for Transient Sphere Generator

Tests the generation of non-persistent spheres that overlay on existing trajectories.
"""

import pytest
import torch
import numpy as np
import json
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from generate_transient_dataset import (
    TransientSphereGenerator,
    augment_video_with_transients,
    load_existing_trajectory,
    save_augmented_sample
)


class TestTransientSphereGenerator:
    """Test the transient sphere generator."""
    
    def test_init(self):
        """Test generator initialization."""
        gen = TransientSphereGenerator(
            min_duration=1,
            max_duration=3,
            sphere_radius=0.05
        )
        assert gen.min_duration == 1
        assert gen.max_duration == 3
        assert gen.sphere_radius == 0.05
    
    def test_generate_transient_parameters(self):
        """Test generation of transient sphere parameters."""
        gen = TransientSphereGenerator(min_duration=1, max_duration=3)
        
        # Generate parameters for a 16-frame video
        params = gen.generate_transient_parameters(num_frames=16, num_transients=3)
        
        # Check we got 3 transients
        assert len(params) == 3
        
        # Check each transient has required fields
        for t in params:
            assert 'start_frame' in t
            assert 'duration' in t
            assert 'trajectory_type' in t
            assert 'start_position' in t
            
            # Check duration is in range [1, 3]
            assert 1 <= t['duration'] <= 3
            
            # Check start_frame is valid
            assert 0 <= t['start_frame'] < 16
            
            # Check trajectory type is valid
            assert t['trajectory_type'] in ['linear', 'circular', 'helical', 'parabolic']
    
    def test_render_sphere(self):
        """Test rendering a sphere on a video frame."""
        gen = TransientSphereGenerator()
        
        # Create a blank frame (64x64)
        frame = torch.zeros(3, 64, 64)
        
        # Render a sphere at center
        position = np.array([0.0, 0.0, 0.0])
        rendered_frame = gen.render_sphere(frame, position)
        
        # Check frame shape unchanged
        assert rendered_frame.shape == (3, 64, 64)
        
        # Check some pixels were modified (sphere was drawn)
        assert not torch.allclose(rendered_frame, frame)
    
    def test_generate_transient_trajectory(self):
        """Test generating a short transient trajectory."""
        gen = TransientSphereGenerator()
        
        start_pos = np.array([0.0, 0.0, 0.0])
        trajectory = gen.generate_transient_trajectory(
            trajectory_type='linear',
            start_position=start_pos,
            num_frames=3
        )
        
        # Check we got 3 positions
        assert len(trajectory) == 3
        
        # Check each position is a 3D point
        for pos in trajectory:
            assert pos.shape == (3,)


class TestVideoAugmentation:
    """Test video augmentation with transients."""
    
    def test_augment_video_with_transients(self):
        """Test augmenting a video with transient spheres."""
        # Create a fake video (16 frames, 3 channels, 64x64)
        video = torch.randn(16, 3, 64, 64)
        
        # Define transient parameters
        transients = [
            {
                'start_frame': 5,
                'duration': 2,
                'trajectory_type': 'linear',
                'start_position': np.array([0.0, 0.0, 0.0])
            }
        ]
        
        # Augment video
        augmented_video, metadata = augment_video_with_transients(video, transients)
        
        # Check output shape
        assert augmented_video.shape == video.shape
        
        # Check metadata
        assert 'num_transients' in metadata
        assert metadata['num_transients'] == 1
        assert 'transient_frames' in metadata
        assert len(metadata['transient_frames']) > 0


class TestDataLoading:
    """Test loading existing trajectory data."""
    
    def test_load_existing_trajectory(self, tmp_path):
        """Test loading an existing trajectory sample."""
        # Create a fake trajectory file
        video = torch.randn(16, 3, 64, 64)
        metadata = {
            'trajectory_type': 'linear',
            'trajectory_class': 0
        }
        
        video_path = tmp_path / "traj_00000.pt"
        metadata_path = tmp_path / "traj_00000.json"
        
        torch.save(video, video_path)
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f)
        
        # Load it back
        loaded_video, loaded_metadata = load_existing_trajectory(str(video_path))
        
        # Check loaded data
        assert loaded_video.shape == video.shape
        assert loaded_metadata['trajectory_type'] == 'linear'


class TestSaving:
    """Test saving augmented samples."""
    
    def test_save_augmented_sample(self, tmp_path):
        """Test saving an augmented video sample."""
        video = torch.randn(16, 3, 64, 64)
        metadata = {
            'trajectory_type': 'linear',
            'num_transients': 2,
            'transient_frames': [5, 6, 10, 11, 12]
        }
        
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        
        # Save sample
        save_augmented_sample(video, metadata, output_dir, sample_idx=42)
        
        # Check files were created
        video_path = output_dir / "augmented_traj_00042.pt"
        metadata_path = output_dir / "augmented_traj_00042.json"
        
        assert video_path.exists()
        assert metadata_path.exists()
        
        # Load and verify
        loaded_video = torch.load(video_path)
        with open(metadata_path, 'r') as f:
            loaded_metadata = json.load(f)
        
        assert loaded_video.shape == video.shape
        assert loaded_metadata['num_transients'] == 2


class TestCheckpointing:
    """Test checkpointing and resume functionality."""
    
    def test_checkpoint_creation(self, tmp_path):
        """Test creating a checkpoint file."""
        from generate_transient_dataset import save_checkpoint
        
        checkpoint = {
            'last_completed_idx': 500,
            'total_samples': 10000,
            'timestamp': '2026-01-26_00:00:00'
        }
        
        save_checkpoint(checkpoint, tmp_path)
        
        # Check checkpoint file exists
        checkpoint_file = tmp_path / "checkpoint.json"
        assert checkpoint_file.exists()
        
        # Load and verify
        with open(checkpoint_file, 'r') as f:
            loaded = json.load(f)
        
        assert loaded['last_completed_idx'] == 500
    
    def test_checkpoint_resume(self, tmp_path):
        """Test resuming from a checkpoint."""
        from generate_transient_dataset import load_checkpoint, save_checkpoint
        
        # Create a checkpoint
        checkpoint = {
            'last_completed_idx': 1500,
            'total_samples': 10000
        }
        save_checkpoint(checkpoint, tmp_path)
        
        # Load it back
        loaded = load_checkpoint(tmp_path)
        
        assert loaded is not None
        assert loaded['last_completed_idx'] == 1500

