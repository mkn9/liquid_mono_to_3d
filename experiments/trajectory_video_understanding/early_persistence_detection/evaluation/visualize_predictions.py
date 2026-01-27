"""
Visualize Model Predictions - Correct and Incorrect Classifications

Shows example frames from videos that were correctly and incorrectly classified
by the MagVIT early persistence detection model.
"""

import torch
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
import argparse
import sys
from typing import List, Tuple

# Add necessary paths
sys.path.insert(0, str(Path(__file__).parent.parent / 'models'))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'branch_4_magvit'))

from early_persistence_classifier import EarlyPersistenceClassifier
from feature_extractor import MagVITExtractor


def load_model(model_path: Path, device: str = 'cpu') -> EarlyPersistenceClassifier:
    """Load trained model."""
    model = EarlyPersistenceClassifier(
        feature_extractor='magvit',
        early_stop_frame=4,
        confidence_threshold=0.9
    )
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    print(f"✅ Model loaded from {model_path}")
    return model


def load_sample(video_path: Path) -> Tuple[torch.Tensor, dict, int]:
    """Load a single video sample with metadata."""
    video = torch.load(video_path)
    
    # Ensure correct shape
    if video.dim() == 5:
        video = video.squeeze(0)
    
    # Load metadata
    json_path = video_path.with_suffix('.json')
    with open(json_path, 'r') as f:
        metadata = json.load(f)
    
    # Create label
    transient_frames = metadata.get('transient_frames', [])
    total_frames = video.shape[0]
    label = 1 if len(transient_frames) / total_frames < 0.2 else 0  # 1=persistent, 0=transient
    
    return video, metadata, label


def predict_sample(model: EarlyPersistenceClassifier, video: torch.Tensor, device: str) -> Tuple[int, float]:
    """Get prediction for a single video."""
    video = video.unsqueeze(0).to(device)  # Add batch dim
    
    with torch.no_grad():
        outputs = model(video)
        logits = outputs['logits'][0]
        probs = torch.softmax(logits, dim=0)
        predicted_class = torch.argmax(probs).item()
        confidence = probs[predicted_class].item()
    
    return predicted_class, confidence


def visualize_frames(video: torch.Tensor, metadata: dict, true_label: int, 
                     pred_label: int, confidence: float, output_path: Path,
                     sample_idx: int):
    """
    Visualize key frames from a video.
    
    Shows: first frame, frames with transients (if any), and last frame.
    """
    T, C, H, W = video.shape
    transient_frames = metadata.get('transient_frames', [])
    
    # Select frames to show
    frames_to_show = [0]  # First frame
    
    # Add some transient frames if they exist
    if transient_frames:
        transient_samples = sorted(transient_frames)[:3]  # Up to 3 transient frames
        frames_to_show.extend(transient_samples)
    else:
        # If no transients, show middle frame
        frames_to_show.append(T // 2)
    
    frames_to_show.append(T - 1)  # Last frame
    frames_to_show = sorted(list(set(frames_to_show)))[:5]  # Max 5 frames
    
    # Create figure
    fig, axes = plt.subplots(1, len(frames_to_show), figsize=(4 * len(frames_to_show), 4))
    if len(frames_to_show) == 1:
        axes = [axes]
    
    # Labels
    true_label_str = "Persistent" if true_label == 1 else "Transient"
    pred_label_str = "Persistent" if pred_label == 1 else "Transient"
    correct = "✅ CORRECT" if true_label == pred_label else "❌ INCORRECT"
    
    fig.suptitle(f"Sample {sample_idx} | True: {true_label_str} | Predicted: {pred_label_str} ({confidence:.1%}) | {correct}",
                 fontsize=14, fontweight='bold', 
                 color='green' if true_label == pred_label else 'red')
    
    for idx, frame_num in enumerate(frames_to_show):
        ax = axes[idx]
        
        # Get frame and convert to displayable format
        frame = video[frame_num].permute(1, 2, 0).cpu().numpy()
        frame = np.clip(frame, 0, 1)  # Ensure valid range
        
        ax.imshow(frame)
        
        # Title with frame number
        title = f"Frame {frame_num}"
        if frame_num in transient_frames:
            title += " (TRANSIENT)"
            # Add red border for transient frames
            rect = patches.Rectangle((0, 0), W-1, H-1, linewidth=3, 
                                    edgecolor='red', facecolor='none')
            ax.add_patch(rect)
        
        ax.set_title(title, fontsize=10)
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Visualize model predictions")
    parser.add_argument('--model', type=Path, required=True, help='Path to trained model')
    parser.add_argument('--data_dir', type=Path, required=True, help='Path to augmented dataset')
    parser.add_argument('--output_dir', type=Path, default=Path('./prediction_visualizations'), 
                       help='Output directory for visualizations')
    parser.add_argument('--num_correct', type=int, default=10, help='Number of correct examples to show')
    parser.add_argument('--num_incorrect', type=int, default=10, help='Number of incorrect examples to show')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Create output directories
    args.output_dir.mkdir(parents=True, exist_ok=True)
    correct_dir = args.output_dir / 'correct'
    incorrect_dir = args.output_dir / 'incorrect'
    correct_dir.mkdir(exist_ok=True)
    incorrect_dir.mkdir(exist_ok=True)
    
    print("=" * 80)
    print("VISUALIZING MODEL PREDICTIONS")
    print("=" * 80)
    
    # Load model
    model = load_model(args.model, args.device)
    
    # Get all samples
    all_samples = sorted(list(args.data_dir.glob("augmented_traj_*.pt")))
    print(f"\n📊 Found {len(all_samples)} samples")
    
    # Randomly sample some for evaluation
    sample_indices = np.random.choice(len(all_samples), 
                                     min(500, len(all_samples)), 
                                     replace=False)
    
    correct_examples = []
    incorrect_examples = []
    
    print("\n🔍 Evaluating samples...")
    for idx in sample_indices:
        video_path = all_samples[idx]
        video, metadata, true_label = load_sample(video_path)
        pred_label, confidence = predict_sample(model, video, args.device)
        
        example = {
            'video': video,
            'metadata': metadata,
            'true_label': true_label,
            'pred_label': pred_label,
            'confidence': confidence,
            'sample_idx': idx
        }
        
        if true_label == pred_label:
            correct_examples.append(example)
        else:
            incorrect_examples.append(example)
        
        if len(correct_examples) >= args.num_correct and len(incorrect_examples) >= args.num_incorrect:
            break
    
    print(f"\n✅ Correct predictions: {len(correct_examples)}")
    print(f"❌ Incorrect predictions: {len(incorrect_examples)}")
    
    # Visualize correct examples
    print(f"\n📊 Generating visualizations for {min(args.num_correct, len(correct_examples))} correct predictions...")
    for i, example in enumerate(correct_examples[:args.num_correct]):
        output_path = correct_dir / f"correct_{i:03d}_sample_{example['sample_idx']:05d}.png"
        visualize_frames(
            example['video'],
            example['metadata'],
            example['true_label'],
            example['pred_label'],
            example['confidence'],
            output_path,
            example['sample_idx']
        )
        print(f"  ✅ {i+1}/{min(args.num_correct, len(correct_examples))}: {output_path.name}")
    
    # Visualize incorrect examples
    print(f"\n📊 Generating visualizations for {min(args.num_incorrect, len(incorrect_examples))} incorrect predictions...")
    for i, example in enumerate(incorrect_examples[:args.num_incorrect]):
        output_path = incorrect_dir / f"incorrect_{i:03d}_sample_{example['sample_idx']:05d}.png"
        visualize_frames(
            example['video'],
            example['metadata'],
            example['true_label'],
            example['pred_label'],
            example['confidence'],
            output_path,
            example['sample_idx']
        )
        print(f"  ❌ {i+1}/{min(args.num_incorrect, len(incorrect_examples))}: {output_path.name}")
    
    # Summary
    print("\n" + "=" * 80)
    print("📊 SUMMARY")
    print("=" * 80)
    print(f"Total samples evaluated: {len(sample_indices)}")
    print(f"Correct: {len(correct_examples)} ({100*len(correct_examples)/len(sample_indices):.1f}%)")
    print(f"Incorrect: {len(incorrect_examples)} ({100*len(incorrect_examples)/len(sample_indices):.1f}%)")
    print(f"\n📁 Visualizations saved to:")
    print(f"  Correct: {correct_dir}")
    print(f"  Incorrect: {incorrect_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()

