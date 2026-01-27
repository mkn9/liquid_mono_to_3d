#!/usr/bin/env python3
"""
Quick test to verify the optimized generator works.
Generate just 100 samples to validate basic functionality.
"""

import time
from parallel_dataset_generator_with_checkpoints import generate_dataset_parallel_with_checkpoints

print("="*70)
print("QUICK VALIDATION TEST - 100 samples")
print("="*70)
print()

start = time.time()

try:
    dataset = generate_dataset_parallel_with_checkpoints(
        num_samples=100,
        checkpoint_interval=50,  # 2 checkpoints
        frames_per_video=8,
        image_size=(32, 32),
        augmentation=False,  # Disabled for speed
        seed=42,
        num_workers=4,
        output_dir='test_output',
        grayscale=True
    )
    
    elapsed = time.time() - start
    
    print()
    print("="*70)
    print("✅ SUCCESS!")
    print("="*70)
    print(f"Generated: {len(dataset['videos'])} samples")
    print(f"Time: {elapsed:.1f} seconds")
    print(f"Shape: {dataset['videos'].shape}")
    print(f"Estimated 30K time: {elapsed * 300:.1f} seconds = {elapsed * 300 / 60:.1f} minutes")
    print()
    
except Exception as e:
    print()
    print("="*70)
    print("❌ FAILED!")
    print("="*70)
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()

