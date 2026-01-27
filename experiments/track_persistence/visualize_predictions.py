#!/usr/bin/env python3
"""
Visualize Track Persistence Predictions

Shows specific examples of:
1. Non-persistent objects (brief/noise) that were correctly filtered out
2. Persistent objects that were correctly identified as valid tracks
"""

import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
import json
from typing import Dict, List, Tuple
import logging

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_model_and_data(model_checkpoint: Path, data_dir: Path):
    """Load trained model and test dataset."""
    from experiments.track_persistence.models.modular_system import (
        ModularTrackingPipeline,
        SimpleStatisticalExtractor,
        SimpleTransformerSequence,
        PersistenceClassificationHead
    )
    
    # Create model architecture
    feature_extractor = SimpleStatisticalExtractor(output_dim=16)
    sequence_model = SimpleTransformerSequence(
        input_dim=16,
        hidden_dim=256,
        num_layers=4,
        num_heads=8,
        dropout=0.1
    )
    task_head = PersistenceClassificationHead(
        input_dim=256,
        hidden_dim=128,
        num_classes=2
    )
    
    model = ModularTrackingPipeline(
        feature_extractor=feature_extractor,
        sequence_model=sequence_model,
        task_head=task_head
    )
    
    # Load checkpoint
    checkpoint = torch.load(model_checkpoint, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    logger.info(f"✓ Loaded model from {model_checkpoint}")
    
    # Load dataset
    videos = np.load(data_dir / 'videos.npy')
    with open(data_dir / 'metadata.json', 'r') as f:
        metadata = json.load(f)
    
    logger.info(f"✓ Loaded {len(videos)} videos from {data_dir}")
    
    return model, videos, metadata


def predict_single_video(model, video: np.ndarray) -> Tuple[int, float]:
    """
    Run prediction on a single video.
    
    Returns:
        (predicted_class, confidence)
        predicted_class: 0 = non-persistent, 1 = persistent
        confidence: probability of the predicted class
    """
    video_tensor = torch.from_numpy(video).float().unsqueeze(0)  # Add batch dimension
    
    with torch.no_grad():
        logits = model(video_tensor)
        probs = torch.softmax(logits, dim=1)
        predicted_class = torch.argmax(logits, dim=1).item()
        confidence = probs[0, predicted_class].item()
    
    return predicted_class, confidence


def find_example_videos(model, videos: np.ndarray, metadata: List[Dict], 
                        num_examples: int = 2) -> Dict:
    """
    Find good examples of correct predictions for both classes.
    
    Returns:
        Dictionary with examples for each category
    """
    examples = {
        'non_persistent_correct': [],  # Correctly filtered out
        'persistent_correct': []        # Correctly identified as tracks
    }
    
    logger.info("\n🔍 Searching for good examples...")
    
    # Use a subset for faster searching (e.g., first 100 samples)
    search_limit = min(100, len(videos))
    
    for idx in range(search_limit):
        video = videos[idx]
        meta = metadata[idx]
        true_label = meta['label']
        
        # Get prediction
        pred_class, confidence = predict_single_video(model, video)
        
        # Determine true class (0 = non-persistent, 1 = persistent)
        if true_label in ['persistent', 'mixed']:
            true_class = 1
        else:
            true_class = 0
        
        # Check if prediction is correct
        if pred_class == true_class:
            if pred_class == 0 and len(examples['non_persistent_correct']) < num_examples:
                examples['non_persistent_correct'].append({
                    'idx': idx,
                    'video': video,
                    'metadata': meta,
                    'confidence': confidence,
                    'label': true_label
                })
                logger.info(f"  ✓ Found non-persistent example {len(examples['non_persistent_correct'])}/{num_examples}: "
                          f"{true_label} (confidence: {confidence:.3f})")
            
            elif pred_class == 1 and len(examples['persistent_correct']) < num_examples:
                examples['persistent_correct'].append({
                    'idx': idx,
                    'video': video,
                    'metadata': meta,
                    'confidence': confidence,
                    'label': true_label
                })
                logger.info(f"  ✓ Found persistent example {len(examples['persistent_correct'])}/{num_examples}: "
                          f"{true_label} (confidence: {confidence:.3f})")
        
        # Stop if we have enough examples
        if (len(examples['non_persistent_correct']) >= num_examples and 
            len(examples['persistent_correct']) >= num_examples):
            break
    
    return examples


def visualize_example(example: Dict, output_path: Path, example_type: str):
    """
    Create visualization showing frames and prediction.
    
    Args:
        example: Dictionary with video, metadata, and prediction
        output_path: Where to save the visualization
        example_type: 'non_persistent' or 'persistent'
    """
    video = example['video']  # Shape: (T, H, W, C)
    metadata = example['metadata']
    confidence = example['confidence']
    label = example['label']
    
    num_frames = video.shape[0]
    
    # Select frames to display (show every 5th frame for brevity)
    frame_indices = list(range(0, num_frames, 5))
    if frame_indices[-1] != num_frames - 1:
        frame_indices.append(num_frames - 1)  # Always show last frame
    
    num_display_frames = len(frame_indices)
    
    # Create figure
    fig, axes = plt.subplots(2, (num_display_frames + 1) // 2, 
                            figsize=(3 * ((num_display_frames + 1) // 2), 6))
    axes = axes.flatten()
    
    # Plot frames
    for i, frame_idx in enumerate(frame_indices):
        frame = video[frame_idx]
        axes[i].imshow(frame)
        axes[i].set_title(f'Frame {frame_idx}', fontsize=10)
        axes[i].axis('off')
        
        # Add border color based on type
        if example_type == 'non_persistent':
            # Red border for filtered out
            rect = patches.Rectangle((0, 0), frame.shape[1]-1, frame.shape[0]-1,
                                    linewidth=3, edgecolor='red', facecolor='none')
        else:
            # Green border for kept as track
            rect = patches.Rectangle((0, 0), frame.shape[1]-1, frame.shape[0]-1,
                                    linewidth=3, edgecolor='green', facecolor='none')
        axes[i].add_patch(rect)
    
    # Hide unused subplots
    for i in range(num_display_frames, len(axes)):
        axes[i].axis('off')
    
    # Add title
    if example_type == 'non_persistent':
        title = f'NON-PERSISTENT TRACK - FILTERED OUT ✗\n'
        title += f'Category: {label} | Model Confidence: {confidence:.1%}\n'
        title += f'Decision: Correctly rejected as noise/brief detection'
        color = 'red'
    else:
        title = f'PERSISTENT TRACK - KEPT AS VALID TRACK ✓\n'
        title += f'Category: {label} | Model Confidence: {confidence:.1%}\n'
        title += f'Decision: Correctly identified as valid persistent track'
        color = 'green'
    
    fig.suptitle(title, fontsize=12, fontweight='bold', color=color, y=0.98)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"  ✓ Saved visualization to {output_path}")


def create_summary_figure(examples: Dict, output_path: Path):
    """Create a summary figure showing both types side-by-side."""
    
    fig = plt.figure(figsize=(16, 10))
    
    # Create grid
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.2)
    
    # Title
    fig.suptitle('Track Persistence Model - Prediction Examples', 
                fontsize=16, fontweight='bold', y=0.98)
    
    # Left column: Non-persistent examples
    if examples['non_persistent_correct']:
        example = examples['non_persistent_correct'][0]
        video = example['video']
        
        # Show 4 key frames
        frame_indices = [0, 8, 16, 24]
        for i, frame_idx in enumerate(frame_indices):
            ax = fig.add_subplot(gs[i//2, 0])
            ax.imshow(video[frame_idx])
            ax.set_title(f'Non-Persistent Frame {frame_idx}\n'
                        f'({example["label"]})', fontsize=10, color='red')
            ax.axis('off')
            rect = patches.Rectangle((0, 0), video.shape[2]-1, video.shape[1]-1,
                                    linewidth=3, edgecolor='red', facecolor='none')
            ax.add_patch(rect)
        
        # Add decision text
        ax_text_left = fig.add_subplot(gs[2, 0])
        ax_text_left.text(0.5, 0.5, 
                         f'✗ FILTERED OUT\n'
                         f'Confidence: {example["confidence"]:.1%}\n'
                         f'Brief/noise detection removed',
                         ha='center', va='center', fontsize=12, 
                         color='red', fontweight='bold',
                         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        ax_text_left.axis('off')
    
    # Right column: Persistent examples
    if examples['persistent_correct']:
        example = examples['persistent_correct'][0]
        video = example['video']
        
        # Show 4 key frames
        frame_indices = [0, 8, 16, 24]
        for i, frame_idx in enumerate(frame_indices):
            ax = fig.add_subplot(gs[i//2, 1])
            ax.imshow(video[frame_idx])
            ax.set_title(f'Persistent Frame {frame_idx}\n'
                        f'({example["label"]})', fontsize=10, color='green')
            ax.axis('off')
            rect = patches.Rectangle((0, 0), video.shape[2]-1, video.shape[1]-1,
                                    linewidth=3, edgecolor='green', facecolor='none')
            ax.add_patch(rect)
        
        # Add decision text
        ax_text_right = fig.add_subplot(gs[2, 1])
        ax_text_right.text(0.5, 0.5, 
                          f'✓ KEPT AS VALID TRACK\n'
                          f'Confidence: {example["confidence"]:.1%}\n'
                          f'Persistent object retained',
                          ha='center', va='center', fontsize=12,
                          color='green', fontweight='bold',
                          bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
        ax_text_right.axis('off')
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"  ✓ Saved summary figure to {output_path}")


def main():
    """Main visualization script."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Visualize track persistence predictions')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--data', type=str, required=True,
                       help='Path to dataset directory')
    parser.add_argument('--output', type=str, default='output/prediction_examples',
                       help='Output directory for visualizations')
    parser.add_argument('--num-examples', type=int, default=2,
                       help='Number of examples per category')
    
    args = parser.parse_args()
    
    checkpoint_path = Path(args.checkpoint)
    data_dir = Path(args.data)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("="*70)
    logger.info("Track Persistence Prediction Visualization")
    logger.info("="*70)
    
    # Load model and data
    logger.info("\n📦 Loading model and data...")
    model, videos, metadata = load_model_and_data(checkpoint_path, data_dir)
    
    # Find good examples
    examples = find_example_videos(model, videos, metadata, args.num_examples)
    
    # Create visualizations
    logger.info("\n🎨 Creating visualizations...")
    
    # Individual examples
    for i, example in enumerate(examples['non_persistent_correct']):
        output_path = output_dir / f'example_non_persistent_{i+1}.png'
        visualize_example(example, output_path, 'non_persistent')
    
    for i, example in enumerate(examples['persistent_correct']):
        output_path = output_dir / f'example_persistent_{i+1}.png'
        visualize_example(example, output_path, 'persistent')
    
    # Summary figure
    summary_path = output_dir / 'prediction_examples_summary.png'
    create_summary_figure(examples, summary_path)
    
    logger.info("\n" + "="*70)
    logger.info("✅ VISUALIZATION COMPLETE")
    logger.info("="*70)
    logger.info(f"\nOutputs saved to: {output_dir}")
    logger.info(f"  • {len(examples['non_persistent_correct'])} non-persistent examples")
    logger.info(f"  • {len(examples['persistent_correct'])} persistent examples")
    logger.info(f"  • 1 summary figure")
    logger.info(f"\nView summary: open {summary_path}")


if __name__ == '__main__':
    main()

