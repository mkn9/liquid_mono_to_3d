#!/usr/bin/env python3
"""
QUICK TDD validation for parallel dataset generation.
Uses TINY dataset (20 samples) to complete in <1 minute.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from parallel_dataset_generator import generate_dataset_parallel
from dataset_generator import generate_dataset
import torch
import numpy as np
import time

print("="*70)
print("QUICK TDD VALIDATION - PARALLEL DATASET GENERATION")
print("="*70)
print()

# Test 1: Basic parallel generation
print("Test 1: Generate 20 samples with 4 workers")
print("-"*70)
start = time.time()
dataset = generate_dataset_parallel(
    num_samples=20,
    frames_per_video=8,
    image_size=(32, 32),
    augmentation=False,
    seed=42,
    num_workers=4
)
elapsed = time.time() - start
print(f"✓ Generated in {elapsed:.2f}s")
print(f"  Shape: {dataset['videos'].shape}")
assert dataset['videos'].shape == (20, 8, 3, 32, 32), "Wrong shape!"
print("✓ Shape correct")
print()

# Test 2: Class balance
print("Test 2: Class Balance")
print("-"*70)
labels = dataset['labels'].numpy()
for i in range(4):
    count = (labels == i).sum()
    print(f"  Class {i}: {count} samples")
    assert count == 5, f"Class {i} has {count}, expected 5"
print("✓ All classes balanced (5 each)")
print()

# Test 3: Finite values
print("Test 3: Value Validation")
print("-"*70)
assert torch.all(torch.isfinite(dataset['videos'])), "Videos have NaN/Inf!"
print("✓ All values finite")
print()

# Test 4: Determinism
print("Test 4: Determinism Check")
print("-"*70)
dataset2 = generate_dataset_parallel(
    num_samples=20,
    frames_per_video=8,
    image_size=(32, 32),
    augmentation=False,
    seed=42,
    num_workers=4
)
if torch.allclose(dataset['videos'], dataset2['videos'], rtol=1e-6):
    print("✓ Deterministic with same seed")
else:
    print("✗ NOT deterministic!")
    sys.exit(1)
print()

# Test 5: Compare with sequential
print("Test 5: Sequential Comparison")
print("-"*70)
seq_start = time.time()
seq_dataset = generate_dataset(
    num_samples=20,
    frames_per_video=8,
    image_size=(32, 32),
    augmentation=False,
    seed=42
)
seq_elapsed = time.time() - seq_start
print(f"  Sequential: {seq_elapsed:.2f}s")
print(f"  Parallel: {elapsed:.2f}s")
speedup = seq_elapsed / elapsed
print(f"  Speedup: {speedup:.2f}×")

if dataset['videos'].shape == seq_dataset['videos'].shape:
    print("✓ Shapes match sequential")
else:
    print("✗ Shape mismatch!")
    sys.exit(1)
print()

print("="*70)
print("✅ ALL VALIDATION TESTS PASSED")
print("="*70)
print()
print("TDD Evidence:")
print(f"  - Parallel generation works correctly")
print(f"  - {speedup:.2f}× faster than sequential")
print(f"  - All values finite and valid")
print(f"  - Deterministic with seed")
print(f"  - Classes balanced")
print()
print("✅ READY FOR 30K GENERATION")
print()

