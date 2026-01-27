#!/usr/bin/env python3
"""
Show Dataset Examples

Displays examples from the dataset showing:
1. Brief/non-persistent objects (appear and disappear) - WOULD BE FILTERED
2. Persistent objects (stay throughout) - WOULD BE KEPT AS TRACKS
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_dataset(data_dir: Path):
    """Load dataset."""
    videos = np.load(data_dir / 'videos.npy')
    with open(data_dir / 'metadata.json', 'r') as f:
        metadata = json.load(f)
    
    logger.info(f"✓ Loaded {len(videos)} videos from {data_dir}")
    return videos, metadata


def find_examples(videos, metadata, num_each=2):
    """Find good examples of each type."""
    examples = {
        'persistent': [],
        'brief': [],
        'noise': []
    }
    
    for idx, meta in enumerate(metadata):
        label = meta['label']
        
        if label == 'persistent' and len(examples['persistent']) < num_each:
            examples['persistent'].append({'idx': idx, 'video': videos[idx], 'meta': meta})
            logger.info(f"  Found persistent example {len(examples['persistent'])}")
        
        elif label == 'brief' and len(examples['brief']) < num_each:
            examples['brief'].append({'idx': idx, 'video': videos[idx], 'meta': meta})
            logger.info(f"  Found brief example {len(examples['brief'])}")
        
        elif label == 'noise' and len(examples['noise']) < num_each:
            examples['noise'].append({'idx': idx, 'video': videos[idx], 'meta': meta})
            logger.info(f"  Found noise example {len(examples['noise'])}")
        
        if all(len(examples[k]) >= num_each for k in examples):
            break
    
    return examples


def visualize_comparison(examples, output_path: Path):
    """Create side-by-side comparison of persistent vs non-persistent."""
    
    fig = plt.figure(figsize=(16, 12))
    fig.suptitle('Track Persistence System - Dataset Examples\n'
                'Model classifies tracks to filter noise and keep valid persistent tracks',
                fontsize=14, fontweight='bold')
    
    # Create layout: 2 columns (non-persistent vs persistent), 5 rows (frames)
    gs = fig.add_gridspec(6, 2, hspace=0.4, wspace=0.3, 
                         left=0.05, right=0.95, top=0.93, bottom=0.05)
    
    # LEFT COLUMN: Non-persistent (brief detection)
    if examples['brief']:
        example = examples['brief'][0]
        video = example['video']
        
        # Show frames 0, 5, 10, 15, 20
        frame_indices = [0, 5, 10, 15, 20]
        
        for i, frame_idx in enumerate(frame_indices):
            ax = fig.add_subplot(gs[i, 0])
            ax.imshow(video[frame_idx])
            ax.set_title(f'Frame {frame_idx}', fontsize=11, color='red')
            ax.axis('off')
            # Red border
            for spine in ['top', 'bottom', 'left', 'right']:
                ax.spines[spine].set_color('red')
                ax.spines[spine].set_linewidth(3)
                ax.spines[spine].set_visible(True)
        
        # Add description
        ax_desc = fig.add_subplot(gs[5, 0])
        ax_desc.text(0.5, 0.5,
                    '✗ NON-PERSISTENT DETECTION\n\n'
                    'Object appears briefly (1-3 frames)\n'
                    'then disappears\n\n'
                    '→ MODEL FILTERS OUT as noise\n'
                    '→ NOT included in 3D reconstruction',
                    ha='center', va='center', fontsize=11,
                    color='red', fontweight='bold',
                    bbox=dict(boxstyle='round,pad=1', facecolor='mistyrose', 
                             edgecolor='red', linewidth=2))
        ax_desc.axis('off')
    
    # RIGHT COLUMN: Persistent track
    if examples['persistent']:
        example = examples['persistent'][0]
        video = example['video']
        
        # Show frames 0, 5, 10, 15, 20
        frame_indices = [0, 5, 10, 15, 20]
        
        for i, frame_idx in enumerate(frame_indices):
            ax = fig.add_subplot(gs[i, 1])
            ax.imshow(video[frame_idx])
            ax.set_title(f'Frame {frame_idx}', fontsize=11, color='green')
            ax.axis('off')
            # Green border
            for spine in ['top', 'bottom', 'left', 'right']:
                ax.spines[spine].set_color('green')
                ax.spines[spine].set_linewidth(3)
                ax.spines[spine].set_visible(True)
        
        # Add description
        ax_desc = fig.add_subplot(gs[5, 1])
        ax_desc.text(0.5, 0.5,
                    '✓ PERSISTENT TRACK\n\n'
                    'Object present throughout\n'
                    'entire video sequence\n\n'
                    '→ MODEL KEEPS as valid track\n'
                    '→ USED for 3D reconstruction',
                    ha='center', va='center', fontsize=11,
                    color='green', fontweight='bold',
                    bbox=dict(boxstyle='round,pad=1', facecolor='honeydew',
                             edgecolor='green', linewidth=2))
        ax_desc.axis('off')
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"✓ Saved comparison to {output_path}")


def create_detailed_examples(examples, output_dir: Path):
    """Create detailed visualizations for each type."""
    
    # Brief detection example
    if examples['brief']:
        example = examples['brief'][0]
        video = example['video']
        
        fig, axes = plt.subplots(3, 5, figsize=(15, 9))
        fig.suptitle('NON-PERSISTENT: Brief Detection (Would Be Filtered Out)',
                    fontsize=14, fontweight='bold', color='red')
        
        for i in range(15):
            row, col = i // 5, i % 5
            axes[row, col].imshow(video[i])
            axes[row, col].set_title(f'Frame {i}', fontsize=9)
            axes[row, col].axis('off')
        
        # Add explanation
        fig.text(0.5, 0.02,
                'Object appears briefly then vanishes → System correctly FILTERS OUT as noise/artifact',
                ha='center', fontsize=11, color='red', fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
        
        plt.tight_layout(rect=[0, 0.04, 1, 0.96])
        plt.savefig(output_dir / 'example_brief_detection.png', dpi=150, bbox_inches='tight')
        plt.close()
        logger.info(f"✓ Saved brief detection example")
    
    # Persistent track example
    if examples['persistent']:
        example = examples['persistent'][0]
        video = example['video']
        
        fig, axes = plt.subplots(3, 5, figsize=(15, 9))
        fig.suptitle('PERSISTENT: Valid Track (Would Be Kept)',
                    fontsize=14, fontweight='bold', color='green')
        
        for i in range(15):
            row, col = i // 5, i % 5
            axes[row, col].imshow(video[i])
            axes[row, col].set_title(f'Frame {i}', fontsize=9)
            axes[row, col].axis('off')
        
        # Add explanation
        fig.text(0.5, 0.02,
                'Object consistently present throughout sequence → System correctly KEEPS as valid track for 3D reconstruction',
                ha='center', fontsize=11, color='green', fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
        
        plt.tight_layout(rect=[0, 0.04, 1, 0.96])
        plt.savefig(output_dir / 'example_persistent_track.png', dpi=150, bbox_inches='tight')
        plt.close()
        logger.info(f"✓ Saved persistent track example")


def main():
    """Main visualization script."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Show dataset examples')
    parser.add_argument('--data', type=str, required=True,
                       help='Path to dataset directory')
    parser.add_argument('--output', type=str, default='output/dataset_examples',
                       help='Output directory')
    
    args = parser.parse_args()
    
    data_dir = Path(args.data)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("="*70)
    logger.info("Track Persistence Dataset Examples")
    logger.info("="*70)
    
    # Load data
    logger.info("\n📦 Loading dataset...")
    videos, metadata = load_dataset(data_dir)
    
    # Find examples
    logger.info("\n🔍 Finding examples...")
    examples = find_examples(videos, metadata)
    
    # Create visualizations
    logger.info("\n🎨 Creating visualizations...")
    
    # Main comparison
    comparison_path = output_dir / 'comparison_persistent_vs_nonpersistent.png'
    visualize_comparison(examples, comparison_path)
    
    # Detailed examples
    create_detailed_examples(examples, output_dir)
    
    logger.info("\n" + "="*70)
    logger.info("✅ VISUALIZATION COMPLETE")
    logger.info("="*70)
    logger.info(f"\nOutputs saved to: {output_dir}")
    logger.info(f"\nMain comparison: {comparison_path}")
    logger.info(f"View: open {comparison_path}")


if __name__ == '__main__':
    main()

