#!/usr/bin/env python3
"""
Generate full 1,200 sample dataset for training.

This script generates the complete dataset with augmentation:
- 1,200 base samples (300 per trajectory type)
- With augmentation enabled
- Saves to results/ with timestamp
"""

import sys
from pathlib import Path
import numpy as np
import argparse
from datetime import datetime

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent))

from dataset_generator import generate_dataset
from visualization_utils import get_timestamped_filename


def main():
    parser = argparse.ArgumentParser(description="Generate full trajectory dataset")
    parser.add_argument("--num-samples", type=int, default=1200,
                        help="Number of samples (default: 1200)")
    parser.add_argument("--frames", type=int, default=16,
                        help="Frames per video (default: 16)")
    parser.add_argument("--image-size", type=int, default=64,
                        help="Image size (default: 64)")
    parser.add_argument("--augmentation", action="store_true", default=True,
                        help="Enable augmentation (default: True)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    args = parser.parse_args()
    
    print("=" * 70)
    print("MAGVIT I3D LLM Dataset Generation")
    print("=" * 70)
    print(f"Samples: {args.num_samples}")
    print(f"Frames per video: {args.frames}")
    print(f"Image size: {args.image_size}x{args.image_size}")
    print(f"Augmentation: {args.augmentation}")
    print(f"Seed: {args.seed}")
    print()
    
    # Generate dataset
    print("Generating dataset...")
    start_time = datetime.now()
    
    dataset = generate_dataset(
        num_samples=args.num_samples,
        frames_per_video=args.frames,
        image_size=(args.image_size, args.image_size),
        augmentation=args.augmentation,
        seed=args.seed
    )
    
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    print(f"✓ Dataset generated in {duration:.1f} seconds")
    print()
    
    # Print statistics
    print("Dataset Statistics:")
    print(f"  Videos shape: {dataset['videos'].shape}")
    print(f"  Labels shape: {dataset['labels'].shape}")
    print(f"  3D trajectories shape: {dataset['trajectory_3d'].shape}")
    print(f"  Number of equations: {len(dataset['equations'])}")
    print(f"  Number of descriptions: {len(dataset['descriptions'])}")
    print()
    
    # Class distribution
    labels_np = dataset['labels'].numpy()
    print("Class Distribution:")
    for class_id in range(4):
        count = (labels_np == class_id).sum()
        print(f"  Class {class_id}: {count} samples ({count/len(labels_np)*100:.1f}%)")
    print()
    
    # Save dataset
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)
    
    filename = get_timestamped_filename("full_dataset", "npz")
    save_path = results_dir / filename
    
    print(f"Saving to: {save_path}")
    np.savez_compressed(
        save_path,
        videos=dataset['videos'].numpy(),
        labels=dataset['labels'].numpy(),
        trajectory_3d=dataset['trajectory_3d'],
        equations=dataset['equations'],
        descriptions=dataset['descriptions']
    )
    
    file_size_mb = save_path.stat().st_size / (1024 * 1024)
    print(f"✓ Saved ({file_size_mb:.1f} MB)")
    print()
    
    print("=" * 70)
    print("Dataset generation complete!")
    print(f"Dataset path: {save_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()

