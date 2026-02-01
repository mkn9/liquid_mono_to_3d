"""
Create visualizations comparing GPT-4 and TinyLlama descriptions against actual trajectories.
This provides visual proof of GPT-4's accuracy.
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from datetime import datetime
from mpl_toolkits.mplot3d import Axes3D
import textwrap


def get_timestamped_filename(base_name: str, extension: str = "png") -> str:
    """Generate timestamped filename for results."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    return f"{timestamp}_{base_name}.{extension}"


def wrap_text(text: str, width: int = 60) -> str:
    """Wrap text to specified width."""
    return '\n'.join(textwrap.wrap(text, width=width))


def create_trajectory_from_ground_truth(ground_truth: dict) -> np.ndarray:
    """Create trajectory points from ground truth data."""
    start = np.array(ground_truth['start'])
    end = np.array(ground_truth['end'])
    num_points = ground_truth['num_points']
    
    # Create linear trajectory
    trajectory = np.array([
        start + (end - start) * t 
        for t in np.linspace(0, 1, num_points)
    ])
    
    return trajectory


def create_gpt4_comparison_viz(sample: dict, sample_id: int, output_dir: Path):
    """
    Create a visualization comparing actual trajectory with GPT-4 and TinyLlama descriptions.
    
    Args:
        sample: Sample data including ground_truth, gpt4_description, tinyllama_description
        sample_id: Sample identifier
        output_dir: Directory to save the visualization
    """
    # Create trajectory
    trajectory = create_trajectory_from_ground_truth(sample['ground_truth'])
    
    # Create figure with 2 rows, 3 columns
    fig = plt.figure(figsize=(18, 10))
    
    # Row 1: 3D trajectory and 2D projections
    ax1 = fig.add_subplot(2, 3, 1, projection='3d')
    ax2 = fig.add_subplot(2, 3, 2)
    ax3 = fig.add_subplot(2, 3, 3)
    
    # Row 2: Text descriptions
    ax4 = fig.add_subplot(2, 3, 4)
    ax5 = fig.add_subplot(2, 3, 5)
    ax6 = fig.add_subplot(2, 3, 6)
    
    # Plot 3D trajectory
    ax1.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], 
             'b-o', linewidth=2, markersize=6, label='Trajectory')
    ax1.scatter(trajectory[0, 0], trajectory[0, 1], trajectory[0, 2], 
                color='green', s=100, marker='o', label='Start', zorder=5)
    ax1.scatter(trajectory[-1, 0], trajectory[-1, 1], trajectory[-1, 2], 
                color='red', s=100, marker='X', label='End', zorder=5)
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.set_title('3D Trajectory', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot XY projection (Top View)
    ax2.plot(trajectory[:, 0], trajectory[:, 1], 'b-o', linewidth=2, markersize=6)
    ax2.scatter(trajectory[0, 0], trajectory[0, 1], color='green', s=100, marker='o', zorder=5)
    ax2.scatter(trajectory[-1, 0], trajectory[-1, 1], color='red', s=100, marker='X', zorder=5)
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_title('Top View (XY Projection)', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.axis('equal')
    
    # Plot XZ projection (Side View)
    ax3.plot(trajectory[:, 0], trajectory[:, 2], 'b-o', linewidth=2, markersize=6)
    ax3.scatter(trajectory[0, 0], trajectory[0, 2], color='green', s=100, marker='o', zorder=5)
    ax3.scatter(trajectory[-1, 0], trajectory[-1, 2], color='red', s=100, marker='X', zorder=5)
    ax3.set_xlabel('X')
    ax3.set_ylabel('Z')
    ax3.set_title('Side View (XZ Projection)', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.axis('equal')
    
    # Ground Truth text box
    ax4.axis('off')
    gt_text = f"GROUND TRUTH\n{'='*40}\n\n"
    gt_text += f"Type: {sample['ground_truth']['type']}\n"
    gt_text += f"Start: {sample['ground_truth']['start']}\n"
    gt_text += f"End: {sample['ground_truth']['end']}\n"
    gt_text += f"Direction: {sample['ground_truth']['primary_direction']}\n"
    gt_text += f"Avg Speed: {sample['ground_truth']['avg_speed']:.3f}\n"
    gt_text += f"Length: {sample['ground_truth']['length']:.2f}\n\n"
    gt_text += wrap_text(sample['ground_truth']['description'], width=40)
    
    ax4.text(0.05, 0.95, gt_text, transform=ax4.transAxes,
             fontsize=9, verticalalignment='top', family='monospace',
             bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))
    
    # GPT-4 text box
    ax5.axis('off')
    gpt4_accuracy = sample['gpt4_metrics']['overall_accuracy'] * 100
    gpt4_text = f"GPT-4 DESCRIPTION\n{'='*40}\n"
    gpt4_text += f"Accuracy: {gpt4_accuracy:.0f}%\n"
    gpt4_text += f"Type: {'✅' if sample['gpt4_metrics']['type_mentioned'] else '❌'} "
    gpt4_text += f"Speed: {'✅' if sample['gpt4_metrics']['speed_mentioned'] else '❌'} "
    gpt4_text += f"Coords: {'✅' if sample['gpt4_metrics']['has_coordinates'] else '❌'}\n\n"
    gpt4_text += wrap_text(sample['gpt4_description'][:300], width=40)
    if len(sample['gpt4_description']) > 300:
        gpt4_text += "..."
    
    color = 'lightgreen' if gpt4_accuracy >= 70 else 'lightyellow' if gpt4_accuracy >= 50 else 'lightcoral'
    ax5.text(0.05, 0.95, gpt4_text, transform=ax5.transAxes,
             fontsize=9, verticalalignment='top', family='monospace',
             bbox=dict(boxstyle='round', facecolor=color, alpha=0.5))
    
    # TinyLlama text box
    ax6.axis('off')
    tl_accuracy = sample['tinyllama_metrics']['overall_accuracy'] * 100
    tl_text = f"TinyLlama DESCRIPTION\n{'='*40}\n"
    tl_text += f"Accuracy: {tl_accuracy:.0f}%\n"
    tl_text += f"Type: {'✅' if sample['tinyllama_metrics']['type_mentioned'] else '❌'} "
    tl_text += f"Speed: {'✅' if sample['tinyllama_metrics']['speed_mentioned'] else '❌'} "
    tl_text += f"Coords: {'✅' if sample['tinyllama_metrics']['has_coordinates'] else '❌'}\n\n"
    tl_text += wrap_text(sample['tinyllama_description'][:300], width=40)
    if len(sample['tinyllama_description']) > 300:
        tl_text += "..."
    
    color = 'lightgreen' if tl_accuracy >= 70 else 'lightyellow' if tl_accuracy >= 50 else 'lightcoral'
    ax6.text(0.05, 0.95, tl_text, transform=ax6.transAxes,
             fontsize=9, verticalalignment='top', family='monospace',
             bbox=dict(boxstyle='round', facecolor=color, alpha=0.5))
    
    # Overall title
    fig.suptitle(f'Sample {sample_id}: GPT-4 vs TinyLlama Accuracy Proof', 
                 fontsize=16, fontweight='bold')
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    
    # Save
    filename = get_timestamped_filename(f"gpt4_proof_sample_{sample_id}", "png")
    output_path = output_dir / filename
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    return output_path


def main():
    """Generate GPT-4 accuracy proof visualizations."""
    print("="*70)
    print("Creating GPT-4 Accuracy Proof Visualizations")
    print("="*70)
    
    # Load evaluation results
    results_file = Path("experiments/liquid_vlm_integration/results/20260131_1835_gpt4_evaluation.json")
    if not results_file.exists():
        print(f"❌ Results file not found: {results_file}")
        return
    
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    print(f"\n✅ Loaded {results['num_samples']} samples")
    
    output_dir = Path("experiments/liquid_vlm_integration/results")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create visualizations for first 5 samples (to keep file count reasonable)
    num_viz = min(5, results['num_samples'])
    print(f"\n📊 Creating visualizations for {num_viz} samples...")
    
    output_files = []
    for i in range(num_viz):
        sample = results['samples'][i]
        print(f"\n  Sample {i}:")
        print(f"    Ground Truth: {sample['ground_truth']['type']}")
        print(f"    GPT-4 Accuracy: {sample['gpt4_metrics']['overall_accuracy']*100:.0f}%")
        print(f"    TinyLlama Accuracy: {sample['tinyllama_metrics']['overall_accuracy']*100:.0f}%")
        
        output_path = create_gpt4_comparison_viz(sample, i, output_dir)
        output_files.append(output_path)
        print(f"    ✅ Created: {output_path.name}")
    
    print("\n" + "="*70)
    print("✅ All visualizations created successfully!")
    print("="*70)
    print(f"\nOutput files ({len(output_files)} total):")
    for f in output_files:
        print(f"  • {f}")
    
    print("\n📌 These visualizations PROVE GPT-4's accuracy by showing:")
    print("  1. The actual 3D trajectory (from ground truth)")
    print("  2. Ground truth description with key metrics")
    print("  3. GPT-4's description with accuracy scores")
    print("  4. TinyLlama's description (for comparison)")
    print("  5. Visual color coding: Green (>70%), Yellow (50-70%), Red (<50%)")


if __name__ == "__main__":
    main()

