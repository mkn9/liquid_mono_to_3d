#!/usr/bin/env python3
"""Generate Attention Heatmap Visualizations.

Creates visualizations showing:
1. Attention heatmaps for individual samples
2. Attention focus on persistent vs transient frames
3. Aggregate attention analysis
"""

import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import argparse
import json


def load_attention_data(attention_file):
    """Load saved attention weights."""
    return torch.load(attention_file)


def generate_attention_heatmap(attention, transient_frames, sample_id, output_path):
    """Generate attention heatmap for a single sample."""
    # attention is (T, T) - query frames x key frames
    # Squeeze any extra dimensions
    if attention.dim() > 2:
        attention = attention.squeeze()
    attention_np = attention.cpu().numpy()
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Create heatmap
    sns.heatmap(attention_np, cmap='viridis', ax=ax,
                cbar_kws={'label': 'Attention Weight'},
                square=True, vmin=0, vmax=attention_np.max())
    
    # Mark transient frames with red lines
    for frame_idx in transient_frames:
        if frame_idx < attention_np.shape[0]:
            # Vertical line (key frames)
            ax.axvline(frame_idx + 0.5, color='red', linestyle='--', linewidth=2, alpha=0.7)
            # Horizontal line (query frames)
            ax.axhline(frame_idx + 0.5, color='red', linestyle='--', linewidth=2, alpha=0.7)
    
    ax.set_xlabel('Key Frame (What the model attends TO)', fontsize=12)
    ax.set_ylabel('Query Frame (What the model attends FROM)', fontsize=12)
    ax.set_title(f'Attention Pattern - Sample {sample_id}\n(Red lines = Transient frames)', 
                 fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return output_path


def analyze_attention_efficiency(attention_weights_list):
    """Analyze how much attention is given to persistent vs transient frames."""
    
    persistent_attention_sum = 0
    transient_attention_sum = 0
    total_frames = 0
    
    for sample in attention_weights_list:
        attention = sample['attention']
        # Squeeze any extra dimensions
        if attention.dim() > 2:
            attention = attention.squeeze()
        attention = attention.cpu().numpy()
        transient_frames = set(sample['transient_frames'])
        num_frames = attention.shape[0]
        
        # Average attention across all query frames
        avg_attention_per_key = attention.mean(axis=0)  # (T,)
        
        for frame_idx in range(num_frames):
            if frame_idx in transient_frames:
                transient_attention_sum += avg_attention_per_key[frame_idx]
            else:
                persistent_attention_sum += avg_attention_per_key[frame_idx]
            total_frames += 1
    
    avg_persistent_attention = persistent_attention_sum / total_frames if total_frames > 0 else 0
    avg_transient_attention = transient_attention_sum / total_frames if total_frames > 0 else 0
    
    attention_ratio = avg_persistent_attention / avg_transient_attention if avg_transient_attention > 0 else float('inf')
    
    return {
        'avg_attention_persistent_frames': float(avg_persistent_attention),
        'avg_attention_transient_frames': float(avg_transient_attention),
        'attention_ratio_persistent_to_transient': float(attention_ratio),
        'interpretation': f"Model pays {attention_ratio:.2f}x more attention to persistent frames"
    }


def generate_aggregate_attention_plot(attention_weights_list, output_path):
    """Generate aggregate plot showing attention distribution."""
    
    persistent_attentions = []
    transient_attentions = []
    
    for sample in attention_weights_list:
        attention = sample['attention']
        # Squeeze any extra dimensions
        if attention.dim() > 2:
            attention = attention.squeeze()
        attention = attention.cpu().numpy()
        transient_frames = set(sample['transient_frames'])
        num_frames = attention.shape[0]
        
        avg_attention_per_key = attention.mean(axis=0)
        
        for frame_idx in range(num_frames):
            if frame_idx in transient_frames:
                transient_attentions.append(avg_attention_per_key[frame_idx])
            else:
                persistent_attentions.append(avg_attention_per_key[frame_idx])
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Histogram
    axes[0].hist(persistent_attentions, bins=50, alpha=0.7, label='Persistent Frames', color='green')
    axes[0].hist(transient_attentions, bins=50, alpha=0.7, label='Transient Frames', color='red')
    axes[0].set_xlabel('Attention Weight')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Attention Distribution by Frame Type')
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    # Box plot
    data = [persistent_attentions, transient_attentions]
    axes[1].boxplot(data, labels=['Persistent', 'Transient'])
    axes[1].set_ylabel('Attention Weight')
    axes[1].set_title('Attention Weight Comparison')
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return output_path


def main():
    parser = argparse.ArgumentParser(description='Visualize attention patterns')
    parser.add_argument('--attention_data', type=str, required=True, 
                       help='Path to attention_weights.pt file')
    parser.add_argument('--output', type=str, default='./attention_visualizations',
                       help='Output directory')
    parser.add_argument('--num_samples', type=int, default=20,
                       help='Number of individual heatmaps to generate')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("ATTENTION VISUALIZATION")
    print("=" * 80)
    print(f"Attention data: {args.attention_data}")
    print(f"Output: {args.output}")
    print("=" * 80)
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load attention data
    print("\n📦 Loading attention data...")
    attention_weights_list = load_attention_data(args.attention_data)
    print(f"✅ Loaded attention for {len(attention_weights_list)} samples")
    
    # Generate individual heatmaps
    print(f"\n🎨 Generating {min(args.num_samples, len(attention_weights_list))} individual heatmaps...")
    for i, sample in enumerate(attention_weights_list[:args.num_samples]):
        output_path = output_dir / f"attention_heatmap_sample_{sample['sample_id']:05d}.png"
        generate_attention_heatmap(
            sample['attention'],
            sample['transient_frames'],
            sample['sample_id'],
            output_path
        )
        if (i + 1) % 5 == 0:
            print(f"  Generated {i + 1}/{min(args.num_samples, len(attention_weights_list))} heatmaps...")
    
    print(f"✅ Generated {min(args.num_samples, len(attention_weights_list))} heatmaps")
    
    # Analyze attention efficiency
    print("\n📊 Analyzing attention efficiency...")
    efficiency_analysis = analyze_attention_efficiency(attention_weights_list)
    
    # Save analysis
    with open(output_dir / 'attention_efficiency_analysis.json', 'w') as f:
        json.dump(efficiency_analysis, f, indent=2)
    
    print("✅ Attention efficiency analysis saved")
    
    # Generate aggregate plot
    print("\n📈 Generating aggregate attention distribution plot...")
    aggregate_plot_path = output_dir / "attention_distribution_aggregate.png"
    generate_aggregate_attention_plot(attention_weights_list, aggregate_plot_path)
    print(f"✅ Aggregate plot saved: {aggregate_plot_path}")
    
    # Print summary
    print("\n" + "=" * 80)
    print("ATTENTION ANALYSIS SUMMARY")
    print("=" * 80)
    print(f"Avg attention on persistent frames: {efficiency_analysis['avg_attention_persistent_frames']:.4f}")
    print(f"Avg attention on transient frames: {efficiency_analysis['avg_attention_transient_frames']:.4f}")
    print(f"Attention ratio (persistent/transient): {efficiency_analysis['attention_ratio_persistent_to_transient']:.2f}x")
    print(f"\n{efficiency_analysis['interpretation']}")
    print("=" * 80)
    print(f"\nAll results saved to: {output_dir}")


if __name__ == '__main__':
    main()

