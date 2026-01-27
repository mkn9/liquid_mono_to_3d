"""
Batch Augment Dataset with Transient Spheres

Processes all existing trajectory samples and augments them with non-persistent spheres.

Author: AI Assistant
Date: 2026-01-26
"""

import torch
import numpy as np
import json
from pathlib import Path
from datetime import datetime
import time
import argparse
from generate_transient_dataset import (
    TransientSphereGenerator,
    augment_video_with_transients,
    load_existing_trajectory,
    save_augmented_sample,
    save_checkpoint,
    load_checkpoint
)


def create_progress_file(output_dir: Path, message: str):
    """Create/update progress file visible on MacBook."""
    progress_file = output_dir / "PROGRESS.txt"
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(progress_file, 'a') as f:
        f.write(f"[{timestamp}] {message}\n")
    print(f"[PROGRESS] {message}")


def sync_to_macbook(output_dir: Path):
    """
    Sync results to MacBook for real-time visibility.
    Note: This creates files that will be synced by external script.
    """
    sync_marker = output_dir / "SYNC_REQUESTED.txt"
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(sync_marker, 'w') as f:
        f.write(f"Sync requested at {timestamp}\n")


def main():
    parser = argparse.ArgumentParser(description="Batch augment trajectory dataset")
    parser.add_argument('--source_dir', type=str, 
                       default='/home/ubuntu/mono_to_3d/experiments/trajectory_video_understanding/data/trajectory_dataset_10k',
                       help='Directory containing original trajectories')
    parser.add_argument('--output_dir', type=str,
                       default='/home/ubuntu/mono_to_3d/experiments/trajectory_video_understanding/persistence_augmented_dataset/output',
                       help='Output directory for augmented data')
    parser.add_argument('--num_samples', type=int, default=10000,
                       help='Number of samples to process')
    parser.add_argument('--transients_per_video', type=int, default=3,
                       help='Average number of transient spheres per video')
    parser.add_argument('--checkpoint_interval', type=int, default=100,
                       help='Save checkpoint every N samples')
    parser.add_argument('--sync_interval', type=int, default=500,
                       help='Sync to MacBook every N samples')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("PERSISTENCE-AUGMENTED DATASET GENERATION")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  Source: {args.source_dir}")
    print(f"  Output: {args.output_dir}")
    print(f"  Samples: {args.num_samples}")
    print(f"  Transients/video: {args.transients_per_video}")
    print(f"  Checkpoint interval: {args.checkpoint_interval}")
    print(f"  Sync interval: {args.sync_interval}")
    print("=" * 80)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize progress tracking
    create_progress_file(output_dir, f"Starting augmentation of {args.num_samples} samples")
    
    # Check for existing checkpoint
    checkpoint = load_checkpoint(output_dir)
    start_idx = 0
    if checkpoint:
        start_idx = checkpoint['last_completed_idx'] + 1
        create_progress_file(output_dir, f"Resuming from checkpoint: sample {start_idx}")
        print(f"\n✅ Resuming from checkpoint: sample {start_idx}")
    
    # Initialize generator
    generator = TransientSphereGenerator(min_duration=1, max_duration=3, sphere_radius=0.05)
    
    # Statistics
    stats = {
        'total_processed': 0,
        'total_transients_added': 0,
        'errors': 0,
        'start_time': time.time()
    }
    
    # Process samples
    source_dir = Path(args.source_dir)
    
    for idx in range(start_idx, args.num_samples):
        try:
            # Load existing trajectory
            video_path = source_dir / f"traj_{idx:05d}.pt"
            
            if not video_path.exists():
                print(f"⚠️  Warning: {video_path} not found, skipping...")
                continue
            
            video, orig_metadata = load_existing_trajectory(str(video_path))
            
            # Generate transient parameters
            # Randomize number of transients (1 to 2x average)
            num_transients = np.random.randint(1, args.transients_per_video * 2 + 1)
            num_frames = video.shape[0]
            
            transient_params = generator.generate_transient_parameters(
                num_frames=num_frames,
                num_transients=num_transients
            )
            
            # Augment video
            augmented_video, augment_metadata = augment_video_with_transients(
                video, transient_params
            )
            
            # Merge metadata
            full_metadata = {
                **orig_metadata,
                **augment_metadata,
                'augmented': True,
                'source_sample_idx': idx
            }
            
            # Save augmented sample
            save_augmented_sample(augmented_video, full_metadata, output_dir, idx)
            
            # Update statistics
            stats['total_processed'] += 1
            stats['total_transients_added'] += num_transients
            
            # Periodic checkpoint
            if (idx + 1) % args.checkpoint_interval == 0:
                checkpoint_data = {
                    'last_completed_idx': idx,
                    'total_samples': args.num_samples,
                    'timestamp': datetime.now().isoformat(),
                    'statistics': stats
                }
                save_checkpoint(checkpoint_data, output_dir)
                
                elapsed = time.time() - stats['start_time']
                rate = stats['total_processed'] / elapsed
                remaining = (args.num_samples - idx - 1) / rate if rate > 0 else 0
                
                progress_msg = (
                    f"Checkpoint: {idx+1}/{args.num_samples} samples "
                    f"({100*(idx+1)/args.num_samples:.1f}%) | "
                    f"Rate: {rate:.1f} samples/sec | "
                    f"ETA: {remaining/60:.1f} min | "
                    f"Transients added: {stats['total_transients_added']}"
                )
                create_progress_file(output_dir, progress_msg)
            
            # Periodic sync marker
            if (idx + 1) % args.sync_interval == 0:
                sync_to_macbook(output_dir)
                create_progress_file(output_dir, f"Sync marker created at sample {idx+1}")
            
            # Progress dots
            if (idx + 1) % 10 == 0:
                print(f".", end="", flush=True)
            if (idx + 1) % 100 == 0:
                print(f" {idx+1}", flush=True)
        
        except Exception as e:
            print(f"\n❌ Error processing sample {idx}: {e}")
            stats['errors'] += 1
            create_progress_file(output_dir, f"ERROR at sample {idx}: {e}")
            
            # Save checkpoint on error
            checkpoint_data = {
                'last_completed_idx': idx - 1,  # Last successful
                'total_samples': args.num_samples,
                'timestamp': datetime.now().isoformat(),
                'statistics': stats,
                'last_error': str(e)
            }
            save_checkpoint(checkpoint_data, output_dir)
            
            # Continue processing (don't stop on single error)
            continue
    
    # Final summary
    elapsed = time.time() - stats['start_time']
    
    print("\n" + "=" * 80)
    print("GENERATION COMPLETE")
    print("=" * 80)
    print(f"\nStatistics:")
    print(f"  Total processed: {stats['total_processed']}")
    print(f"  Total transients added: {stats['total_transients_added']}")
    print(f"  Errors: {stats['errors']}")
    print(f"  Elapsed time: {elapsed/60:.2f} minutes")
    print(f"  Average rate: {stats['total_processed']/elapsed:.2f} samples/sec")
    print(f"\nOutput directory: {output_dir}")
    
    # Save final summary
    summary = {
        'completed': True,
        'timestamp': datetime.now().isoformat(),
        'statistics': stats,
        'elapsed_time_minutes': elapsed / 60,
        'configuration': vars(args)
    }
    
    with open(output_dir / "GENERATION_SUMMARY.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    create_progress_file(output_dir, "✅ GENERATION COMPLETE")
    sync_to_macbook(output_dir)
    
    print("=" * 80)


if __name__ == "__main__":
    main()

