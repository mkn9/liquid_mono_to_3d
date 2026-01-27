#!/usr/bin/env python3
"""
Verify the existing 1200-sample dataset format and contents.
"""

import numpy as np
import torch
from pathlib import Path

print("="*70)
print("VERIFYING EXISTING DATASET")
print("="*70)
print()

dataset_path = Path("results/20260124_1546_full_dataset.npz")

if not dataset_path.exists():
    print(f"❌ Dataset not found: {dataset_path}")
    exit(1)

print(f"Loading: {dataset_path}")
print(f"Size: {dataset_path.stat().st_size / 1024:.1f} KB")
print()

# Load dataset
data = np.load(dataset_path)

print("Dataset keys:", list(data.keys()))
print()

# Check each component
for key in data.keys():
    arr = data[key]
    print(f"{key}:")
    print(f"  Shape: {arr.shape}")
    print(f"  Dtype: {arr.dtype}")
    
    # Only show range for numeric types
    if np.issubdtype(arr.dtype, np.number):
        print(f"  Range: [{arr.min():.3f}, {arr.max():.3f}]")
        print(f"  Memory: {arr.nbytes / (1024**2):.2f} MB")
    else:
        print(f"  Sample: {arr[0] if len(arr) > 0 else 'empty'}")
    print()

# Verify format matches MAGVIT requirements
videos = data['videos']
labels = data['labels']

print("="*70)
print("VALIDATION CHECKS")
print("="*70)
print()

checks = {
    'videos_5d': videos.ndim == 5,
    'videos_normalized': 0 <= videos.min() and videos.max() <= 1.0,
    'labels_1d': labels.ndim == 1,
    'num_samples_match': len(videos) == len(labels),
    'four_classes': len(np.unique(labels)) == 4,
    'class_balance': all(np.sum(labels == i) > 0 for i in range(4))
}

for check, passed in checks.items():
    status = "✅" if passed else "❌"
    print(f"{status} {check}")

print()

if all(checks.values()):
    print("="*70)
    print("✅ DATASET VALID - READY FOR MAGVIT TRAINING!")
    print("="*70)
    print()
    print(f"Total samples: {len(videos)}")
    print(f"Video shape: {videos.shape[1:]} (T, C, H, W)")
    print(f"Classes: {np.unique(labels)}")
    
    # Class distribution
    print()
    print("Class distribution:")
    for class_id in range(4):
        count = np.sum(labels == class_id)
        print(f"  Class {class_id}: {count} samples ({100*count/len(labels):.1f}%)")
    
else:
    print("❌ DATASET HAS ISSUES - NEEDS FIXING")
    exit(1)

print()
print("Next step: Initialize MAGVIT model and test loading this dataset")

