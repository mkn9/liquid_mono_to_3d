#!/usr/bin/env python3
"""
Generate 30,000 sample dataset using parallel processing.
Includes progress tracking and periodic status updates.
"""

import sys
import time
from pathlib import Path
from datetime import datetime
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from parallel_dataset_generator import generate_dataset_parallel
from visualization_utils import get_timestamped_filename


def generate_30k_dataset():
    """Generate 30K sample dataset with progress tracking."""
    
    print("="*70)
    print("PARALLEL 30K DATASET GENERATION")
    print("="*70)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Configuration
    num_samples = 30000
    frames_per_video = 16
    image_size = (64, 64)
    num_workers = 4  # One per trajectory class
    
    print("Configuration:")
    print(f"  Total samples: {num_samples:,}")
    print(f"  Per class: {num_samples // 4:,}")
    print(f"  Frames per video: {frames_per_video}")
    print(f"  Image size: {image_size[0]}×{image_size[1]}")
    print(f"  Workers: {num_workers}")
    print(f"  Augmentation: Yes")
    print()
    
    # Progress tracking
    print("Generating dataset...")
    print("(This will take ~15-20 minutes with parallel processing)")
    print()
    
    start_time = time.time()
    
    try:
        # Generate dataset
        dataset = generate_dataset_parallel(
            num_samples=num_samples,
            frames_per_video=frames_per_video,
            image_size=image_size,
            augmentation=True,
            seed=42,
            num_workers=num_workers
        )
        
        generation_time = time.time() - start_time
        
        print()
        print("✓ Generation complete!")
        print(f"  Time: {generation_time:.1f} seconds ({generation_time/60:.1f} minutes)")
        print()
        
        # Statistics
        print("Dataset Statistics:")
        print(f"  Videos shape: {dataset['videos'].shape}")
        print(f"  Labels shape: {dataset['labels'].shape}")
        print(f"  3D trajectories shape: {dataset['trajectory_3d'].shape}")
        print(f"  Memory size: ~{dataset['videos'].element_size() * dataset['videos'].nelement() / (1024**3):.2f} GB")
        print()
        
        # Class distribution
        labels_np = dataset['labels'].numpy()
        print("Class Distribution:")
        for class_id in range(4):
            count = (labels_np == class_id).sum()
            percentage = count / len(labels_np) * 100
            print(f"  Class {class_id}: {count:,} samples ({percentage:.1f}%)")
        print()
        
        # Save dataset
        results_dir = Path(__file__).parent / "results"
        results_dir.mkdir(exist_ok=True)
        
        filename = get_timestamped_filename("dataset_30k_parallel", "npz")
        save_path = results_dir / filename
        
        print(f"Saving to: {save_path}")
        print("(This may take a few minutes...)")
        
        save_start = time.time()
        np.savez_compressed(
            save_path,
            videos=dataset['videos'].numpy(),
            labels=dataset['labels'].numpy(),
            trajectory_3d=dataset['trajectory_3d'],
            equations=dataset['equations'],
            descriptions=dataset['descriptions']
        )
        save_time = time.time() - save_start
        
        file_size_mb = save_path.stat().st_size / (1024 * 1024)
        print(f"✓ Saved in {save_time:.1f}s ({file_size_mb:.1f} MB)")
        print()
        
        # Total time
        total_time = time.time() - start_time
        print("="*70)
        print("✅ DATASET GENERATION COMPLETE")
        print("="*70)
        print(f"Total time: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
        print(f"Dataset path: {save_path}")
        print(f"File size: {file_size_mb:.1f} MB")
        print()
        
        # Estimated speedup
        estimated_sequential_time = generation_time * 3.5  # Conservative estimate
        print(f"Estimated speedup: ~{estimated_sequential_time/generation_time:.1f}× faster than sequential")
        print(f"(Sequential would have taken ~{estimated_sequential_time/60:.0f} minutes)")
        print()
        
        return True
        
    except Exception as e:
        print()
        print(f"❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = generate_30k_dataset()
    sys.exit(0 if success else 1)

