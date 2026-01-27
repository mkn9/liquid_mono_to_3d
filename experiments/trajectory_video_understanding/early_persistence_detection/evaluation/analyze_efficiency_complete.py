#!/usr/bin/env python3
"""Efficiency Analysis - Visualize Compute Savings and Decision Patterns.

Generates:
1. Decision frame histogram
2. Compute savings analysis
3. Time-to-decision plots
4. Early stopping statistics
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import argparse


def load_efficiency_metrics(metrics_file):
    """Load efficiency metrics from JSON."""
    with open(metrics_file, 'r') as f:
        return json.load(f)


def plot_decision_frame_histogram(efficiency_data, output_path):
    """Plot histogram of decision frames."""
    decision_frames = [e['decision_frame'] for e in efficiency_data]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    bins = list(range(1, max(decision_frames) + 2))
    ax.hist(decision_frames, bins=bins, edgecolor='black', alpha=0.7, color='steelblue')
    
    # Mark early stop threshold
    ax.axvline(4.5, color='red', linestyle='--', linewidth=2, label='Early Stop Threshold (Frame 4)')
    
    ax.set_xlabel('Decision Frame', fontsize=12)
    ax.set_ylabel('Number of Samples', fontsize=12)
    ax.set_title('Distribution of Decision Frames\n(How early does the model make confident decisions?)', 
                 fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return output_path


def plot_compute_savings(efficiency_data, output_path):
    """Plot compute savings analysis."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Extract data
    compute_used = [e['compute_used'] * 100 for e in efficiency_data]
    compute_saved = [e['compute_saved'] * 100 for e in efficiency_data]
    early_stops = [e['early_stop'] for e in efficiency_data]
    
    # 1. Compute used distribution
    axes[0, 0].hist(compute_used, bins=20, edgecolor='black', alpha=0.7, color='coral')
    axes[0, 0].set_xlabel('Compute Used (%)', fontsize=11)
    axes[0, 0].set_ylabel('Frequency', fontsize=11)
    axes[0, 0].set_title('Compute Usage Distribution', fontsize=12, fontweight='bold')
    axes[0, 0].grid(alpha=0.3, axis='y')
    axes[0, 0].axvline(np.mean(compute_used), color='red', linestyle='--', linewidth=2, 
                       label=f'Mean: {np.mean(compute_used):.1f}%')
    axes[0, 0].legend()
    
    # 2. Compute saved distribution
    axes[0, 1].hist(compute_saved, bins=20, edgecolor='black', alpha=0.7, color='green')
    axes[0, 1].set_xlabel('Compute Saved (%)', fontsize=11)
    axes[0, 1].set_ylabel('Frequency', fontsize=11)
    axes[0, 1].set_title('Compute Savings Distribution', fontsize=12, fontweight='bold')
    axes[0, 1].grid(alpha=0.3, axis='y')
    axes[0, 1].axvline(np.mean(compute_saved), color='red', linestyle='--', linewidth=2,
                       label=f'Mean: {np.mean(compute_saved):.1f}%')
    axes[0, 1].legend()
    
    # 3. Early stop vs full processing
    early_stop_count = sum(early_stops)
    full_process_count = len(early_stops) - early_stop_count
    
    axes[1, 0].bar(['Early Stop\n(≤4 frames)', 'Full Processing\n(>4 frames)'],
                   [early_stop_count, full_process_count],
                   color=['green', 'orange'], edgecolor='black', alpha=0.7)
    axes[1, 0].set_ylabel('Number of Samples', fontsize=11)
    axes[1, 0].set_title('Early Stopping Statistics', fontsize=12, fontweight='bold')
    axes[1, 0].grid(alpha=0.3, axis='y')
    
    # Add percentages
    total = early_stop_count + full_process_count
    axes[1, 0].text(0, early_stop_count + 5, f'{100*early_stop_count/total:.1f}%', 
                    ha='center', fontweight='bold')
    axes[1, 0].text(1, full_process_count + 5, f'{100*full_process_count/total:.1f}%',
                    ha='center', fontweight='bold')
    
    # 4. Scatter: Decision frame vs Compute saved
    decision_frames = [e['decision_frame'] for e in efficiency_data]
    axes[1, 1].scatter(decision_frames, compute_saved, alpha=0.5, s=30)
    axes[1, 1].set_xlabel('Decision Frame', fontsize=11)
    axes[1, 1].set_ylabel('Compute Saved (%)', fontsize=11)
    axes[1, 1].set_title('Decision Frame vs Compute Savings', fontsize=12, fontweight='bold')
    axes[1, 1].grid(alpha=0.3)
    axes[1, 1].axvline(4, color='red', linestyle='--', linewidth=1, alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return output_path


def plot_inference_time_analysis(efficiency_data, output_path):
    """Plot inference time analysis."""
    inference_times = [e['inference_time_ms'] for e in efficiency_data]
    decision_frames = [e['decision_frame'] for e in efficiency_data]
    early_stops = [e['early_stop'] for e in efficiency_data]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 1. Inference time distribution
    axes[0].hist(inference_times, bins=30, edgecolor='black', alpha=0.7, color='purple')
    axes[0].set_xlabel('Inference Time (ms)', fontsize=11)
    axes[0].set_ylabel('Frequency', fontsize=11)
    axes[0].set_title('Inference Time Distribution', fontsize=12, fontweight='bold')
    axes[0].grid(alpha=0.3, axis='y')
    axes[0].axvline(np.mean(inference_times), color='red', linestyle='--', linewidth=2,
                    label=f'Mean: {np.mean(inference_times):.2f}ms')
    axes[0].legend()
    
    # 2. Inference time by early stop
    early_stop_times = [t for t, es in zip(inference_times, early_stops) if es]
    full_process_times = [t for t, es in zip(inference_times, early_stops) if not es]
    
    data = [early_stop_times, full_process_times]
    axes[1].boxplot(data, labels=['Early Stop', 'Full Process'])
    axes[1].set_ylabel('Inference Time (ms)', fontsize=11)
    axes[1].set_title('Inference Time: Early Stop vs Full Process', fontsize=12, fontweight='bold')
    axes[1].grid(alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return output_path


def compute_efficiency_statistics(efficiency_data):
    """Compute comprehensive efficiency statistics."""
    decision_frames = [e['decision_frame'] for e in efficiency_data]
    compute_saved = [e['compute_saved'] * 100 for e in efficiency_data]
    inference_times = [e['inference_time_ms'] for e in efficiency_data]
    early_stops = [e['early_stop'] for e in efficiency_data]
    
    stats = {
        'total_samples': len(efficiency_data),
        'early_stop_count': sum(early_stops),
        'early_stop_rate': sum(early_stops) / len(early_stops) if early_stops else 0,
        'avg_decision_frame': float(np.mean(decision_frames)),
        'median_decision_frame': float(np.median(decision_frames)),
        'avg_compute_saved_percent': float(np.mean(compute_saved)),
        'median_compute_saved_percent': float(np.median(compute_saved)),
        'total_compute_saved_percent': float(np.sum(compute_saved)),
        'avg_inference_time_ms': float(np.mean(inference_times)),
        'median_inference_time_ms': float(np.median(inference_times)),
        'speedup_factor': len(efficiency_data) * 16 / sum(decision_frames) if sum(decision_frames) > 0 else 1
    }
    
    return stats


def main():
    parser = argparse.ArgumentParser(description='Analyze efficiency metrics')
    parser.add_argument('--metrics', type=str, required=True,
                       help='Path to efficiency_metrics.json')
    parser.add_argument('--output', type=str, default='./efficiency_analysis',
                       help='Output directory')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("EFFICIENCY ANALYSIS")
    print("=" * 80)
    print(f"Metrics: {args.metrics}")
    print(f"Output: {args.output}")
    print("=" * 80)
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load metrics
    print("\n📦 Loading efficiency metrics...")
    efficiency_data = load_efficiency_metrics(args.metrics)
    print(f"✅ Loaded metrics for {len(efficiency_data)} samples")
    
    # Generate plots
    print("\n📊 Generating visualizations...")
    
    print("  1. Decision frame histogram...")
    plot_decision_frame_histogram(efficiency_data, output_dir / "decision_frame_histogram.png")
    
    print("  2. Compute savings analysis...")
    plot_compute_savings(efficiency_data, output_dir / "compute_savings_analysis.png")
    
    print("  3. Inference time analysis...")
    plot_inference_time_analysis(efficiency_data, output_dir / "inference_time_analysis.png")
    
    print("✅ Visualizations complete")
    
    # Compute statistics
    print("\n📈 Computing efficiency statistics...")
    stats = compute_efficiency_statistics(efficiency_data)
    
    # Save statistics
    with open(output_dir / "efficiency_statistics.json", 'w') as f:
        json.dump(stats, f, indent=2)
    
    print("✅ Statistics saved")
    
    # Print summary
    print("\n" + "=" * 80)
    print("EFFICIENCY SUMMARY")
    print("=" * 80)
    print(f"Total Samples: {stats['total_samples']}")
    print(f"Early Stops: {stats['early_stop_count']} ({stats['early_stop_rate']:.1%})")
    print(f"Avg Decision Frame: {stats['avg_decision_frame']:.2f}")
    print(f"Avg Compute Saved: {stats['avg_compute_saved_percent']:.1f}%")
    print(f"Avg Inference Time: {stats['avg_inference_time_ms']:.2f}ms")
    print(f"Overall Speedup: {stats['speedup_factor']:.2f}x")
    print("=" * 80)
    print(f"\nAll results saved to: {output_dir}")


if __name__ == '__main__':
    main()

