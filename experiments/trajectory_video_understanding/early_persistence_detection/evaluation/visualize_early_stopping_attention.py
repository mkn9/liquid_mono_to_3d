#!/usr/bin/env python3
"""
Visualize Attention Patterns During Early Stopping

Shows how attention evolves over frames and how it relates to early stopping decisions.
"""

import torch
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import argparse
import json
import sys

# Add models to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'models'))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'branch_4_magvit'))

from early_persistence_classifier import EarlyPersistenceClassifier, get_early_decision


def load_model(model_path: str, device: str = 'cpu'):
    """Load trained model."""
    state_dict = torch.load(model_path, map_location=device)
    
    model = EarlyPersistenceClassifier(
        feature_extractor='magvit',
        early_stop_frame=4,
        confidence_threshold=0.9,
        feature_dim=256
    )
    
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    
    return model


def load_sample_data(data_dir: Path, num_samples: int = 30):
    """Load sample videos for analysis."""
    video_files = sorted(list(data_dir.glob("augmented_traj_*.pt")))[:num_samples]
    
    samples = []
    for video_path in video_files:
        video = torch.load(video_path)
        if video.dim() == 5:
            video = video.squeeze(0)
        
        json_path = video_path.with_suffix('.json')
        with open(json_path, 'r') as f:
            metadata = json.load(f)
        
        transient_frames = metadata.get('transient_frames', [])
        total_frames = video.shape[0]
        transient_ratio = len(transient_frames) / total_frames if total_frames > 0 else 0
        label = 1 if transient_ratio < 0.2 else 0  # 1=persistent, 0=transient
        
        samples.append({
            'video': video,
            'label': label,
            'transient_frames': transient_frames,
            'sample_id': int(video_path.stem.split('_')[-1])
        })
    
    return samples


def extract_frame_by_frame_attention(model, video, device='cpu'):
    """
    Extract attention weights and confidence for each frame as model processes sequentially.
    """
    video = video.unsqueeze(0).to(device)
    B, T, C, H, W = video.shape
    
    frame_attentions = []
    frame_confidences = []
    
    model.eval()
    with torch.no_grad():
        # Process incrementally, frame by frame
        for t in range(1, min(T, 16) + 1):
            video_partial = video[:, :t]  # Only first t frames
            
            # Get features
            features = model.extractor(video_partial)
            
            # Get attention from tokenizer
            if hasattr(model.extractor, 'tokenizer'):
                _, attn_weights = model.extractor.tokenizer(
                    features, features, features, need_weights=True, average_attn_weights=True
                )
                
                if attn_weights.dim() > 2:
                    attn_weights = attn_weights.squeeze()
                
                # Average attention across all queries for each key (frame)
                avg_attn = attn_weights.mean(dim=0).cpu().numpy()
                frame_attentions.append(avg_attn)
            
            # Get confidence at this frame
            outputs = model(video_partial)
            logits = outputs['logits'][0]
            probs = torch.softmax(logits, dim=0)
            confidence = probs.max().item()
            frame_confidences.append(confidence)
    
    return frame_attentions, frame_confidences


def plot_temporal_attention_evolution(samples, model, device, output_dir):
    """Plot how attention evolves over time for different early stopping scenarios."""
    
    # Categorize samples by decision frame
    early_stop_2 = []
    early_stop_3_4 = []
    full_process = []
    
    for sample in samples:
        video = sample['video']
        decision, conf, frame_idx = get_early_decision(model, video.unsqueeze(0).to(device))
        
        if frame_idx <= 2:
            early_stop_2.append(sample)
        elif frame_idx <= 4:
            early_stop_3_4.append(sample)
        else:
            full_process.append(sample)
    
    print(f"  Categorized: {len(early_stop_2)} @ frame≤2, {len(early_stop_3_4)} @ frame 3-4, {len(full_process)} full")
    
    # Plot 1: Temporal attention evolution for different categories
    fig, axes = plt.subplots(3, 1, figsize=(14, 12))
    
    categories = [
        (early_stop_2[:5], "Early Stop (Frame ≤2)", axes[0], 'green'),
        (early_stop_3_4[:5], "Early Stop (Frame 3-4)", axes[1], 'orange'),
        (full_process[:5], "Full Processing (Frame >4)", axes[2], 'red')
    ]
    
    for samples_cat, title, ax, color in categories:
        for sample in samples_cat:
            frame_attentions, _ = extract_frame_by_frame_attention(
                model, sample['video'], device
            )
            
            # Plot maximum attention at each timestep
            max_attentions = [attn.max() if attn.size > 0 else 0 for attn in frame_attentions]
            ax.plot(range(1, len(max_attentions) + 1), max_attentions, 
                   alpha=0.6, linewidth=2, color=color)
        
        ax.axvline(4, color='red', linestyle='--', linewidth=2, alpha=0.5, 
                  label='Early Stop Threshold')
        ax.set_xlabel('Frame Number', fontsize=11)
        ax.set_ylabel('Max Attention Weight', fontsize=11)
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'temporal_attention_evolution.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    return output_dir / 'temporal_attention_evolution.png'


def plot_attention_confidence_timeline(samples, model, device, output_dir):
    """Plot attention and confidence evolving together over frames."""
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    
    for idx, sample in enumerate(samples[:6]):
        frame_attentions, frame_confidences = extract_frame_by_frame_attention(
            model, sample['video'], device
        )
        
        transient_frames = set(sample['transient_frames'])
        decision, conf, frame_idx = get_early_decision(model, sample['video'].unsqueeze(0).to(device))
        
        ax = axes[idx]
        ax2 = ax.twinx()
        
        # Plot max attention per frame
        max_attentions = [attn.max() if attn.size > 0 else 0 for attn in frame_attentions]
        frames = list(range(1, len(max_attentions) + 1))
        
        # Color code by transient/persistent
        colors = ['red' if (f-1) in transient_frames else 'green' for f in frames]
        ax.scatter(frames, max_attentions, c=colors, s=100, alpha=0.6, 
                  label='Attention (Red=Transient)', zorder=3)
        ax.plot(frames, max_attentions, 'b-', alpha=0.3, linewidth=1)
        
        # Plot confidence
        ax2.plot(frames, frame_confidences, 'purple', linewidth=2, 
                label='Confidence', alpha=0.7)
        
        # Mark decision frame
        ax.axvline(frame_idx, color='red', linestyle='--', linewidth=2, 
                  label=f'Decision @ frame {frame_idx}')
        
        ax.set_xlabel('Frame', fontsize=10)
        ax.set_ylabel('Max Attention', fontsize=10, color='blue')
        ax2.set_ylabel('Confidence', fontsize=10, color='purple')
        ax.set_title(f"Sample {sample['sample_id']} - {decision.capitalize()}", 
                    fontsize=11, fontweight='bold')
        ax.grid(alpha=0.3)
        ax.tick_params(axis='y', labelcolor='blue')
        ax2.tick_params(axis='y', labelcolor='purple')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'attention_confidence_timeline.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    return output_dir / 'attention_confidence_timeline.png'


def plot_attention_heatmap_with_decision(samples, model, device, output_dir):
    """Show attention heatmap with decision frame marked."""
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    
    for idx, sample in enumerate(samples[:6]):
        frame_attentions, _ = extract_frame_by_frame_attention(
            model, sample['video'], device
        )
        
        decision, conf, frame_idx = get_early_decision(model, sample['video'].unsqueeze(0).to(device))
        transient_frames = sample['transient_frames']
        
        # Create attention matrix over time
        max_len = len(frame_attentions)
        attention_matrix = np.zeros((max_len, max_len))
        
        for t, attn in enumerate(frame_attentions):
            attn_len = attn.size if attn.ndim == 0 else attn.shape[0]
            attention_matrix[t, :attn_len] = attn
        
        ax = axes[idx]
        im = ax.imshow(attention_matrix, cmap='viridis', aspect='auto')
        
        # Mark transient frames
        for tf in transient_frames:
            if tf < max_len:
                ax.axvline(tf + 0.5, color='red', linestyle=':', linewidth=1, alpha=0.7)
                ax.axhline(tf + 0.5, color='red', linestyle=':', linewidth=1, alpha=0.7)
        
        # Mark decision frame
        if frame_idx <= max_len:
            ax.axhline(frame_idx - 0.5, color='yellow', linestyle='--', linewidth=3, 
                      label=f'Decision @ frame {frame_idx}')
        
        ax.set_xlabel('Key Frame (attended TO)', fontsize=9)
        ax.set_ylabel('Processing Step', fontsize=9)
        ax.set_title(f"Sample {sample['sample_id']} - {decision.capitalize()}\nDecision @ frame {frame_idx}", 
                    fontsize=10, fontweight='bold')
        plt.colorbar(im, ax=ax, label='Attention')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'attention_heatmap_with_decision.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    return output_dir / 'attention_heatmap_with_decision.png'


def main():
    parser = argparse.ArgumentParser(description='Visualize early stopping attention patterns')
    parser.add_argument('--model', type=str, required=True, help='Path to trained model')
    parser.add_argument('--data', type=str, required=True, help='Path to dataset directory')
    parser.add_argument('--output', type=str, default='./early_stopping_attention',
                       help='Output directory')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--num_samples', type=int, default=30,
                       help='Number of samples to analyze')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("EARLY STOPPING ATTENTION VISUALIZATION")
    print("=" * 80)
    print(f"Model: {args.model}")
    print(f"Data: {args.data}")
    print(f"Output: {args.output}")
    print("=" * 80)
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model
    print("\n📦 Loading model...")
    model = load_model(args.model, device=args.device)
    print("✅ Model loaded")
    
    # Load sample data
    print(f"\n📂 Loading {args.num_samples} samples...")
    samples = load_sample_data(Path(args.data), num_samples=args.num_samples)
    print(f"✅ Loaded {len(samples)} samples")
    
    # Generate visualizations
    print("\n🎨 Generating visualizations...")
    
    print("  1. Temporal attention evolution...")
    plot1 = plot_temporal_attention_evolution(samples, model, args.device, output_dir)
    print(f"     ✅ {plot1.name}")
    
    print("  2. Attention-confidence timeline...")
    plot2 = plot_attention_confidence_timeline(samples, model, args.device, output_dir)
    print(f"     ✅ {plot2.name}")
    
    print("  3. Attention heatmap with decision markers...")
    plot3 = plot_attention_heatmap_with_decision(samples, model, args.device, output_dir)
    print(f"     ✅ {plot3.name}")
    
    print("\n" + "=" * 80)
    print("VISUALIZATION COMPLETE")
    print("=" * 80)
    print(f"\nGenerated files:")
    print(f"  • temporal_attention_evolution.png")
    print(f"  • attention_confidence_timeline.png")
    print(f"  • attention_heatmap_with_decision.png")
    print(f"\nAll saved to: {output_dir}")
    print("=" * 80)


if __name__ == '__main__':
    main()

