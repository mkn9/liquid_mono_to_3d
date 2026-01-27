"""
Test parallel dataset generation for trajectory videos.

Following TDD: Tests written FIRST (RED phase).
Tests ensure parallel generation produces identical results to sequential.
"""

import numpy as np
import torch
import pytest
from pathlib import Path
import tempfile
import shutil

# These imports will fail initially (RED phase - expected!)
from parallel_dataset_generator import (
    generate_dataset_parallel,
    generate_single_class_dataset,
    merge_class_datasets,
    validate_merged_dataset
)
from dataset_generator import generate_dataset


class TestParallelDatasetGeneratorInvariants:
    """Invariant tests: properties that must always hold."""
    
    def test_parallel_output_shape_matches_sequential(self):
        """Parallel generation must produce same shapes as sequential.
        
        Specification:
        - Input: num_samples=80 (20 per class)
        - Output: Same shapes as sequential generation
        - Invariant: Shape consistency regardless of generation method
        
        Seed: 42 (deterministic)
        """
        num_samples = 80
        
        # Sequential generation
        seq_dataset = generate_dataset(
            num_samples=num_samples,
            frames_per_video=8,
            image_size=(32, 32),
            augmentation=False,
            seed=42
        )
        
        # Parallel generation
        par_dataset = generate_dataset_parallel(
            num_samples=num_samples,
            frames_per_video=8,
            image_size=(32, 32),
            augmentation=False,
            seed=42,
            num_workers=4
        )
        
        # Verify shapes match
        assert seq_dataset['videos'].shape == par_dataset['videos'].shape, \
            f"Video shapes differ: {seq_dataset['videos'].shape} vs {par_dataset['videos'].shape}"
        assert seq_dataset['labels'].shape == par_dataset['labels'].shape, \
            f"Label shapes differ"
        assert seq_dataset['trajectory_3d'].shape == par_dataset['trajectory_3d'].shape, \
            f"Trajectory shapes differ"
    
    def test_parallel_class_distribution_balanced(self):
        """Parallel generation must maintain balanced class distribution.
        
        Invariant: Each class has equal number of samples (±1 for rounding)
        
        Seed: 42
        """
        num_samples = 80  # Divisible by 4
        
        dataset = generate_dataset_parallel(
            num_samples=num_samples,
            frames_per_video=8,
            image_size=(32, 32),
            augmentation=False,
            seed=42,
            num_workers=4
        )
        
        labels = dataset['labels'].numpy()
        expected_per_class = num_samples // 4
        
        for class_id in range(4):
            count = (labels == class_id).sum()
            assert abs(count - expected_per_class) <= 1, \
                f"Class {class_id} has {count} samples, expected {expected_per_class}"
    
    def test_parallel_videos_are_finite(self):
        """All generated videos must contain finite values (no NaN/Inf).
        
        Invariant: Video tensors must be finite
        """
        dataset = generate_dataset_parallel(
            num_samples=20,
            frames_per_video=8,
            image_size=(32, 32),
            augmentation=False,
            seed=42,
            num_workers=2
        )
        
        videos = dataset['videos']
        assert torch.all(torch.isfinite(videos)), \
            "Videos contain NaN or Inf values"


class TestParallelDatasetGeneratorGolden:
    """Golden tests: verify specific scenarios with known outcomes."""
    
    def test_single_class_generation_produces_correct_class(self):
        """Single class generation must produce only specified class.
        
        Scenario: Generate 20 samples of class 1 (circular)
        Expected: All labels are 1, all trajectories are circular
        
        Seed: 42
        Tolerance: Exact match for labels
        """
        class_id = 1  # Circular
        num_samples = 20
        
        dataset = generate_single_class_dataset(
            class_id=class_id,
            num_samples=num_samples,
            frames_per_video=8,
            image_size=(32, 32),
            augmentation=False,
            seed=42
        )
        
        # All labels must be class_id
        labels = dataset['labels'].numpy()
        assert np.all(labels == class_id), \
            f"Expected all labels to be {class_id}, got {np.unique(labels)}"
        
        # Correct number of samples
        assert len(labels) == num_samples, \
            f"Expected {num_samples} samples, got {len(labels)}"
    
    def test_merge_preserves_all_samples(self):
        """Merging class datasets must preserve all samples.
        
        Scenario: Generate 4 class datasets (10 each), merge
        Expected: Merged dataset has 40 samples total
        
        Seed: 42
        """
        # Generate 4 separate class datasets
        class_datasets = []
        for class_id in range(4):
            dataset = generate_single_class_dataset(
                class_id=class_id,
                num_samples=10,
                frames_per_video=8,
                image_size=(32, 32),
                augmentation=False,
                seed=42 + class_id
            )
            class_datasets.append(dataset)
        
        # Merge
        merged = merge_class_datasets(class_datasets, shuffle_seed=42)
        
        # Verify total count
        assert len(merged['videos']) == 40, \
            f"Expected 40 samples after merge, got {len(merged['videos'])}"
        
        # Verify each class present
        labels = merged['labels'].numpy()
        for class_id in range(4):
            count = (labels == class_id).sum()
            assert count == 10, \
                f"Class {class_id} has {count} samples, expected 10"
    
    def test_parallel_generation_deterministic_with_seed(self):
        """Parallel generation must be deterministic with fixed seed.
        
        Scenario: Generate same dataset twice with same seed
        Expected: Identical results
        
        Seed: 42 (used twice)
        Tolerance: Exact match for videos and labels
        """
        dataset1 = generate_dataset_parallel(
            num_samples=40,
            frames_per_video=8,
            image_size=(32, 32),
            augmentation=False,
            seed=42,
            num_workers=4
        )
        
        dataset2 = generate_dataset_parallel(
            num_samples=40,
            frames_per_video=8,
            image_size=(32, 32),
            augmentation=False,
            seed=42,
            num_workers=4
        )
        
        # Videos should be identical
        torch.testing.assert_close(
            dataset1['videos'],
            dataset2['videos'],
            rtol=1e-10,
            atol=1e-10,
            msg="Videos differ between runs with same seed"
        )
        
        # Labels should be identical
        assert torch.all(dataset1['labels'] == dataset2['labels']), \
            "Labels differ between runs with same seed"


class TestParallelDatasetValidation:
    """Tests for dataset validation after merge."""
    
    def test_validate_merged_dataset_passes_for_valid_data(self):
        """Validation must pass for correctly merged dataset.
        
        Scenario: Generate and merge dataset, validate
        Expected: Validation passes without errors
        """
        dataset = generate_dataset_parallel(
            num_samples=40,
            frames_per_video=8,
            image_size=(32, 32),
            augmentation=False,
            seed=42,
            num_workers=4
        )
        
        # Should not raise any exceptions
        is_valid = validate_merged_dataset(dataset, expected_samples=40)
        assert is_valid, "Validation failed for valid dataset"
    
    def test_validate_detects_missing_samples(self):
        """Validation must detect when samples are missing.
        
        Scenario: Create dataset with fewer samples than expected
        Expected: Validation fails
        """
        dataset = generate_dataset_parallel(
            num_samples=30,
            frames_per_video=8,
            image_size=(32, 32),
            augmentation=False,
            seed=42,
            num_workers=4
        )
        
        # Should detect mismatch
        is_valid = validate_merged_dataset(dataset, expected_samples=40)
        assert not is_valid, "Validation should fail for missing samples"
    
    def test_validate_detects_shape_mismatches(self):
        """Validation must detect shape inconsistencies.
        
        Scenario: Create dataset with inconsistent shapes
        Expected: Validation fails
        """
        # Generate valid dataset
        dataset = generate_dataset_parallel(
            num_samples=40,
            frames_per_video=8,
            image_size=(32, 32),
            augmentation=False,
            seed=42,
            num_workers=4
        )
        
        # Corrupt trajectory shape
        dataset['trajectory_3d'] = dataset['trajectory_3d'][:30]  # Wrong size
        
        # Should detect mismatch
        is_valid = validate_merged_dataset(dataset, expected_samples=40)
        assert not is_valid, "Validation should fail for shape mismatch"


class TestParallelPerformance:
    """Performance tests to verify speedup."""
    
    def test_parallel_faster_than_sequential_for_large_dataset(self):
        """Parallel generation should be faster for large datasets.
        
        Scenario: Generate 200 samples both ways, compare time
        Expected: Parallel is at least 2× faster
        
        Note: This is a performance test, not strict requirement
        """
        import time
        
        num_samples = 200
        
        # Sequential
        start = time.time()
        seq_dataset = generate_dataset(
            num_samples=num_samples,
            frames_per_video=8,
            image_size=(32, 32),
            augmentation=False,
            seed=42
        )
        seq_time = time.time() - start
        
        # Parallel
        start = time.time()
        par_dataset = generate_dataset_parallel(
            num_samples=num_samples,
            frames_per_video=8,
            image_size=(32, 32),
            augmentation=False,
            seed=42,
            num_workers=4
        )
        par_time = time.time() - start
        
        speedup = seq_time / par_time
        
        # Expect at least 2× speedup (conservative, accounting for overhead)
        assert speedup >= 2.0, \
            f"Parallel speedup is only {speedup:.2f}×, expected >= 2.0×"
        
        print(f"\nPerformance: Sequential={seq_time:.2f}s, Parallel={par_time:.2f}s, Speedup={speedup:.2f}×")

