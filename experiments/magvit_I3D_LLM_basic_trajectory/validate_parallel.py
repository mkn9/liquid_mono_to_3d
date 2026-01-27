#!/usr/bin/env python3
"""
Quick validation script for parallel dataset generation.
Tests basic functionality without full pytest framework.
"""

import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from parallel_dataset_generator import generate_dataset_parallel
from dataset_generator import generate_dataset
import torch
import numpy as np


def validate_parallel_generation():
    """Validate parallel generation produces correct output."""
    print("="*70)
    print("PARALLEL DATASET GENERATION VALIDATION")
    print("="*70)
    print()
    
    num_samples = 80  # Small for quick validation
    
    # Test 1: Generate dataset in parallel
    print("Test 1: Parallel Generation")
    print("-" * 70)
    start = time.time()
    parallel_dataset = generate_dataset_parallel(
        num_samples=num_samples,
        frames_per_video=8,
        image_size=(32, 32),
        augmentation=False,
        seed=42,
        num_workers=4
    )
    parallel_time = time.time() - start
    print(f"✓ Generated {num_samples} samples in {parallel_time:.2f}s")
    print(f"  Videos shape: {parallel_dataset['videos'].shape}")
    print(f"  Labels shape: {parallel_dataset['labels'].shape}")
    print(f"  Trajectories shape: {parallel_dataset['trajectory_3d'].shape}")
    print()
    
    # Test 2: Verify class balance
    print("Test 2: Class Balance")
    print("-" * 70)
    labels = parallel_dataset['labels'].numpy()
    for class_id in range(4):
        count = (labels == class_id).sum()
        percentage = count / len(labels) * 100
        print(f"  Class {class_id}: {count} samples ({percentage:.1f}%)")
    
    expected_per_class = num_samples // 4
    all_balanced = all(
        abs((labels == i).sum() - expected_per_class) <= 1
        for i in range(4)
    )
    if all_balanced:
        print("✓ Classes are balanced")
    else:
        print("✗ Classes are NOT balanced")
        return False
    print()
    
    # Test 3: Verify finite values
    print("Test 3: Value Validation")
    print("-" * 70)
    if torch.all(torch.isfinite(parallel_dataset['videos'])):
        print("✓ All video values are finite (no NaN/Inf)")
    else:
        print("✗ Videos contain NaN or Inf")
        return False
    
    if np.all(np.isfinite(parallel_dataset['trajectory_3d'])):
        print("✓ All trajectory values are finite")
    else:
        print("✗ Trajectories contain NaN or Inf")
        return False
    print()
    
    # Test 4: Compare with sequential (shape only, for speed)
    print("Test 4: Sequential Comparison (Shapes)")
    print("-" * 70)
    start = time.time()
    sequential_dataset = generate_dataset(
        num_samples=num_samples,
        frames_per_video=8,
        image_size=(32, 32),
        augmentation=False,
        seed=42
    )
    sequential_time = time.time() - start
    
    if parallel_dataset['videos'].shape == sequential_dataset['videos'].shape:
        print(f"✓ Shapes match: {parallel_dataset['videos'].shape}")
    else:
        print(f"✗ Shape mismatch!")
        print(f"  Parallel: {parallel_dataset['videos'].shape}")
        print(f"  Sequential: {sequential_dataset['videos'].shape}")
        return False
    
    speedup = sequential_time / parallel_time
    print(f"✓ Sequential time: {sequential_time:.2f}s")
    print(f"✓ Parallel time: {parallel_time:.2f}s")
    print(f"✓ Speedup: {speedup:.2f}×")
    print()
    
    # Test 5: Determinism check
    print("Test 5: Determinism")
    print("-" * 70)
    dataset2 = generate_dataset_parallel(
        num_samples=40,  # Smaller for speed
        frames_per_video=8,
        image_size=(32, 32),
        augmentation=False,
        seed=42,
        num_workers=4
    )
    
    dataset3 = generate_dataset_parallel(
        num_samples=40,
        frames_per_video=8,
        image_size=(32, 32),
        augmentation=False,
        seed=42,
        num_workers=4
    )
    
    if torch.allclose(dataset2['videos'], dataset3['videos'], rtol=1e-6, atol=1e-6):
        print("✓ Parallel generation is deterministic with same seed")
    else:
        print("✗ Parallel generation is NOT deterministic")
        return False
    print()
    
    print("="*70)
    print("✅ ALL VALIDATION TESTS PASSED")
    print("="*70)
    print()
    print(f"Summary:")
    print(f"  - Parallel generation works correctly")
    print(f"  - {speedup:.2f}× speedup over sequential")
    print(f"  - All values finite and valid")
    print(f"  - Deterministic with seed")
    print(f"  - Classes balanced")
    print()
    return True


if __name__ == "__main__":
    try:
        success = validate_parallel_generation()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ VALIDATION FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

