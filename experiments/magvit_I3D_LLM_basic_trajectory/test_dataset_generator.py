"""
Test module for dataset generation.

CRITICAL REQUIREMENT: Dataset must contain IMAGES, not coordinate shortcuts.
This ensures we're building a TRUE vision-language model.

All tests follow Red → Green → Refactor TDD cycle.
All random tests use explicit seeds for determinism.
All numeric comparisons use explicit tolerances.
"""

import numpy as np
import torch
import pytest
from numpy.testing import assert_allclose
from pathlib import Path

# These imports will fail initially (RED phase - expected!)
from dataset_generator import (
    generate_dataset,
    generate_linear_trajectory,
    generate_circular_trajectory,
    generate_helical_trajectory,
    generate_parabolic_trajectory,
    augment_trajectory
)


class TestDatasetGeneratorInvariants:
    """Invariant tests: properties that must always hold."""
    
    def test_dataset_contains_images_not_coordinates(self):
        """CRITICAL: Dataset must contain IMAGES (N,T,C,H,W), not coordinates.
        
        This is the core test enforcing TRUE vision modeling.
        A dataset of coordinates is NOT a vision dataset.
        
        Specification:
        - videos: (N, T, 3, H, W) - RGB video frames
        - labels: (N,) - class labels
        - NOT: videos (N, T, 2) - coordinate shortcuts
        
        Seed: 42 (deterministic generation)
        """
        dataset = generate_dataset(
            num_samples=20,  # Small for test speed
            frames_per_video=8,
            image_size=(32, 32),
            augmentation=False,
            seed=42
        )
        
        # Must contain 'videos' key with image tensors
        assert 'videos' in dataset, "Dataset must have 'videos' key"
        videos = dataset['videos']
        
        # Must be 5D tensor: (N, T, C, H, W)
        assert isinstance(videos, torch.Tensor), "Videos must be torch.Tensor"
        assert videos.ndim == 5, f"Expected 5D tensor (N,T,C,H,W), got {videos.ndim}D"
        assert videos.shape == (20, 8, 3, 32, 32), \
            f"Expected (20,8,3,32,32), got {videos.shape}"
        
        # Verify it's image data (pixel values in [0, 1])
        assert videos.min() >= 0.0, "Image values must be >= 0"
        assert videos.max() <= 1.0, "Image values must be <= 1"
        
        # Should NOT be coordinates (would have very different properties)
        assert videos.shape[2] == 3, "Must have 3 RGB channels, not 2 coordinate dimensions"
    
    def test_dataset_has_correct_number_of_samples(self):
        """Dataset must generate exactly the requested number of samples.
        
        Seed: 42
        """
        num_samples = 40
        dataset = generate_dataset(
            num_samples=num_samples,
            frames_per_video=8,
            image_size=(32, 32),
            augmentation=False,
            seed=42
        )
        
        assert len(dataset['videos']) == num_samples, \
            f"Expected {num_samples} samples, got {len(dataset['videos'])}"
        assert len(dataset['labels']) == num_samples, \
            f"Expected {num_samples} labels, got {len(dataset['labels'])}"
    
    def test_dataset_has_balanced_classes(self):
        """Dataset must have roughly balanced distribution of trajectory types.
        
        With 4 classes, each should have ~25% of samples.
        
        Seed: 42
        Tolerance: Within 5% of balanced (for small datasets)
        """
        num_samples = 80  # Divisible by 4
        dataset = generate_dataset(
            num_samples=num_samples,
            frames_per_video=8,
            image_size=(32, 32),
            augmentation=False,
            seed=42
        )
        
        labels = dataset['labels']
        
        # Count each class
        class_counts = {}
        for label in labels:
            label_int = int(label)
            class_counts[label_int] = class_counts.get(label_int, 0) + 1
        
        # Should have 4 classes
        assert len(class_counts) == 4, f"Expected 4 classes, got {len(class_counts)}"
        
        # Each class should have ~25% (20 samples out of 80)
        expected_per_class = num_samples // 4
        for class_id, count in class_counts.items():
            assert abs(count - expected_per_class) <= 4, \
                f"Class {class_id} has {count} samples, expected ~{expected_per_class} (±4)"
    
    def test_dataset_videos_are_finite(self):
        """All video frames must be finite (no NaN/Inf).
        
        Seed: 42
        """
        dataset = generate_dataset(
            num_samples=10,
            frames_per_video=8,
            image_size=(32, 32),
            augmentation=False,
            seed=42
        )
        
        videos = dataset['videos']
        assert torch.all(torch.isfinite(videos)), \
            "Dataset contains NaN or Inf values"
    
    def test_dataset_includes_ground_truth_3d(self):
        """Dataset must include 3D ground truth for evaluation.
        
        Seed: 42
        Note: 10 samples with 4 classes = 10//4 = 2 per class = 8 total samples
        """
        dataset = generate_dataset(
            num_samples=10,
            frames_per_video=8,
            image_size=(32, 32),
            augmentation=False,
            seed=42
        )
        
        assert 'trajectory_3d' in dataset, "Must include 3D ground truth"
        traj_3d = dataset['trajectory_3d']
        
        # Shape: (N, T, 3) where N = (num_samples // 4) * 4 = 8 for input of 10
        actual_samples = (10 // 4) * 4
        assert traj_3d.shape == (actual_samples, 8, 3), \
            f"Expected ({actual_samples}, 8, 3) for 3D trajectories, got {traj_3d.shape}"


class TestTrajectoryGenerators:
    """Test individual trajectory generation functions."""
    
    def test_linear_trajectory_generation(self):
        """Test linear trajectory generation.
        
        Specification:
        - Linear motion: position = start + t * (end - start)
        - Should move in straight line
        - Deterministic with fixed seed
        
        Seed: 42
        """
        rng = np.random.default_rng(42)
        
        trajectory = generate_linear_trajectory(
            num_frames=16,
            rng=rng
        )
        
        # Shape: (16, 3)
        assert trajectory.shape == (16, 3), \
            f"Expected (16, 3), got {trajectory.shape}"
        
        # Should move in roughly straight line (small deviations from noise)
        start = trajectory[0]
        end = trajectory[-1]
        direction = end - start
        direction = direction / np.linalg.norm(direction)
        
        # Check linearity: all points should be close to the line
        for i, point in enumerate(trajectory):
            vec_to_point = point - start
            projection_length = np.dot(vec_to_point, direction)
            projected_point = start + projection_length * direction
            distance_from_line = np.linalg.norm(point - projected_point)
            
            # Allow some deviation due to noise
            assert distance_from_line < 0.1, \
                f"Point {i} too far from line: {distance_from_line:.3f}"
    
    def test_circular_trajectory_generation(self):
        """Test circular trajectory generation.
        
        Specification:
        - Circular motion: x = r*cos(θ), y = r*sin(θ), z = constant
        - Should maintain roughly constant radius
        
        Seed: 42
        """
        rng = np.random.default_rng(42)
        
        trajectory = generate_circular_trajectory(
            num_frames=16,
            rng=rng
        )
        
        assert trajectory.shape == (16, 3)
        
        # Center should be near origin
        center = trajectory.mean(axis=0)
        
        # All points should be roughly same distance from center
        radii = np.linalg.norm(trajectory - center, axis=1)
        mean_radius = radii.mean()
        
        # Radius variation should be small (noise accounts for some variation)
        radius_std = radii.std()
        assert radius_std < 0.1, \
            f"Radius variation too large: std={radius_std:.3f}"
    
    def test_helical_trajectory_generation(self):
        """Test helical trajectory generation.
        
        Specification:
        - Helical motion: circular in XY plane + linear in Z
        - x = r*cos(θ), y = r*sin(θ), z = linear(t)
        
        Seed: 42
        """
        rng = np.random.default_rng(42)
        
        trajectory = generate_helical_trajectory(
            num_frames=16,
            rng=rng
        )
        
        assert trajectory.shape == (16, 3)
        
        # Z should increase/decrease roughly linearly
        z_values = trajectory[:, 2]
        z_diffs = np.diff(z_values)
        
        # Check that Z changes consistently (not oscillating)
        mean_z_diff = z_diffs.mean()
        assert abs(mean_z_diff) > 0.01, "Z should change over time"
        
        # XY should form roughly circular pattern
        xy_center = trajectory[:, :2].mean(axis=0)
        xy_radii = np.linalg.norm(trajectory[:, :2] - xy_center, axis=1)
        xy_radius_std = xy_radii.std()
        assert xy_radius_std < 0.15, \
            f"XY pattern should be roughly circular, got std={xy_radius_std:.3f}"
    
    def test_parabolic_trajectory_generation(self):
        """Test parabolic trajectory generation.
        
        Specification:
        - Parabolic motion: x = t, y = at² + bt + c, z = dt² + et + f
        - Should show quadratic behavior
        
        Seed: 42
        """
        rng = np.random.default_rng(42)
        
        trajectory = generate_parabolic_trajectory(
            num_frames=16,
            rng=rng
        )
        
        assert trajectory.shape == (16, 3)
        
        # Should have quadratic-like acceleration
        # Second derivative should be roughly constant
        for dim in range(3):
            values = trajectory[:, dim]
            first_diff = np.diff(values)
            second_diff = np.diff(first_diff)
            
            # Second derivative std should be relatively small
            # (indicates consistent acceleration)
            second_diff_std = second_diff.std()
            # Allow larger variation due to noise
            assert second_diff_std < 0.5, \
                f"Dimension {dim} doesn't show parabolic behavior: {second_diff_std:.3f}"


class TestAugmentation:
    """Test data augmentation functionality."""
    
    def test_augmentation_creates_variations(self):
        """Augmentation must create different versions of same trajectory.
        
        Seed: 42 (different augmentations with different sub-seeds)
        """
        rng = np.random.default_rng(42)
        
        # Generate base trajectory
        base_trajectory = generate_linear_trajectory(num_frames=16, rng=rng)
        
        # Create augmented versions
        aug1 = augment_trajectory(base_trajectory, rng=np.random.default_rng(1))
        aug2 = augment_trajectory(base_trajectory, rng=np.random.default_rng(2))
        
        # Augmented trajectories should differ from base
        diff1 = np.linalg.norm(aug1 - base_trajectory)
        diff2 = np.linalg.norm(aug2 - base_trajectory)
        
        assert diff1 > 0.01, "Augmentation should modify trajectory"
        assert diff2 > 0.01, "Augmentation should modify trajectory"
        
        # Different augmentations should differ from each other
        diff_aug = np.linalg.norm(aug1 - aug2)
        assert diff_aug > 0.01, "Different augmentations should produce different results"
    
    def test_augmentation_preserves_trajectory_shape(self):
        """Augmentation must preserve trajectory type characteristics.
        
        For circular trajectory, should remain roughly circular after augmentation.
        
        Seed: 42
        """
        rng = np.random.default_rng(42)
        
        # Generate circular trajectory
        circular = generate_circular_trajectory(num_frames=16, rng=rng)
        
        # Augment
        augmented = augment_trajectory(circular, rng=np.random.default_rng(100))
        
        # Should still be roughly circular (similar radius variance)
        center_orig = circular.mean(axis=0)
        center_aug = augmented.mean(axis=0)
        
        radii_orig = np.linalg.norm(circular - center_orig, axis=1)
        radii_aug = np.linalg.norm(augmented - center_aug, axis=1)
        
        std_orig = radii_orig.std()
        std_aug = radii_aug.std()
        
        # Augmentation shouldn't drastically change circularity
        assert abs(std_aug - std_orig) < 0.2, \
            f"Augmentation changed circularity too much: {std_orig:.3f} → {std_aug:.3f}"


class TestDatasetSaveLoad:
    """Test dataset persistence."""
    
    def test_dataset_saves_and_loads_correctly(self, tmp_path):
        """Dataset must save to disk and load back identically.
        
        Seed: 42
        """
        # Generate dataset
        original = generate_dataset(
            num_samples=20,
            frames_per_video=8,
            image_size=(32, 32),
            augmentation=False,
            seed=42
        )
        
        # Save to temp file
        save_path = tmp_path / "test_dataset.npz"
        np.savez_compressed(
            save_path,
            videos=original['videos'].numpy(),
            labels=original['labels'].numpy(),
            trajectory_3d=original['trajectory_3d']
        )
        
        # Load back
        loaded = np.load(save_path)
        
        # Verify all keys exist
        assert 'videos' in loaded
        assert 'labels' in loaded
        assert 'trajectory_3d' in loaded
        
        # Verify data matches
        assert_allclose(loaded['videos'], original['videos'].numpy(), rtol=1e-10)
        assert_allclose(loaded['labels'], original['labels'].numpy(), rtol=1e-10)
        assert_allclose(loaded['trajectory_3d'], original['trajectory_3d'], rtol=1e-10)


class TestDatasetMetadata:
    """Test dataset metadata and descriptions."""
    
    def test_dataset_includes_metadata(self):
        """Dataset should include equations and descriptions.
        
        Seed: 42
        """
        dataset = generate_dataset(
            num_samples=20,
            frames_per_video=8,
            image_size=(32, 32),
            augmentation=False,
            seed=42
        )
        
        # Should have equations and descriptions
        assert 'equations' in dataset, "Dataset must include symbolic equations"
        assert 'descriptions' in dataset, "Dataset must include natural language descriptions"
        
        # Should have one per sample
        assert len(dataset['equations']) == 20
        assert len(dataset['descriptions']) == 20
    
    def test_equations_are_valid_strings(self):
        """Equations must be valid string representations.
        
        Seed: 42
        """
        dataset = generate_dataset(
            num_samples=20,
            frames_per_video=8,
            image_size=(32, 32),
            augmentation=False,
            seed=42
        )
        
        for i, eq in enumerate(dataset['equations']):
            assert isinstance(eq, str), f"Equation {i} must be string, got {type(eq)}"
            assert len(eq) > 0, f"Equation {i} is empty"
    
    def test_descriptions_are_valid_strings(self):
        """Descriptions must be valid natural language strings.
        
        Seed: 42
        """
        dataset = generate_dataset(
            num_samples=20,
            frames_per_video=8,
            image_size=(32, 32),
            augmentation=False,
            seed=42
        )
        
        for i, desc in enumerate(dataset['descriptions']):
            assert isinstance(desc, str), f"Description {i} must be string, got {type(desc)}"
            assert len(desc) > 10, f"Description {i} too short: {len(desc)} chars"


class TestReproducibility:
    """Test deterministic generation with seeds."""
    
    def test_same_seed_produces_identical_datasets(self):
        """Same seed must produce identical datasets.
        
        Seed: 42 (used twice)
        """
        dataset1 = generate_dataset(
            num_samples=20,
            frames_per_video=8,
            image_size=(32, 32),
            augmentation=False,
            seed=42
        )
        
        dataset2 = generate_dataset(
            num_samples=20,
            frames_per_video=8,
            image_size=(32, 32),
            augmentation=False,
            seed=42
        )
        
        # Videos should be identical
        assert torch.allclose(dataset1['videos'], dataset2['videos'], rtol=1e-10), \
            "Same seed should produce identical videos"
        
        # Labels should be identical
        assert torch.all(dataset1['labels'] == dataset2['labels']), \
            "Same seed should produce identical labels"
    
    def test_different_seed_produces_different_datasets(self):
        """Different seeds must produce different datasets.
        
        Seeds: 42 vs 43
        """
        dataset1 = generate_dataset(
            num_samples=20,
            frames_per_video=8,
            image_size=(32, 32),
            augmentation=False,
            seed=42
        )
        
        dataset2 = generate_dataset(
            num_samples=20,
            frames_per_video=8,
            image_size=(32, 32),
            augmentation=False,
            seed=43
        )
        
        # Videos should differ
        diff = torch.abs(dataset1['videos'] - dataset2['videos']).mean()
        assert diff > 0.01, \
            f"Different seeds should produce different videos, diff={diff:.4f}"

