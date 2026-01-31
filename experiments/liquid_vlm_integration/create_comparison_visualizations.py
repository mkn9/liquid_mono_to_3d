#!/usr/bin/env python3
"""
Create GPT-4 vs TinyLlama Comparison Visualizations
Shows accuracy improvements and example comparisons

Following TDD and output naming conventions (YYYYMMDD_HHMM_description.ext)
"""

import json
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
from pathlib import Path
from datetime import datetime
import numpy as np


def get_output_filename(base_name: str, extension: str = "png") -> str:
    """Generate timestamped filename."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    return f"{timestamp}_{base_name}.{extension}"


def load_results() -> dict:
    """Load latest GPT-4 evaluation results."""
    results_dir = Path("results")
    eval_files = sorted(results_dir.glob("*_gpt4_evaluation.json"), reverse=True)
    
    if not eval_files:
        raise FileNotFoundError("No GPT-4 evaluation results found")
    
    with open(eval_files[0], 'r') as f:
        return json.load(f)


def create_accuracy_comparison_bar_chart(results: dict):
    """Create bar chart comparing TinyLlama vs GPT-4 accuracy."""
    tinyllama_avg = results['tinyllama_avg_accuracy']
    gpt4_avg = results['gpt4_avg_accuracy']
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    models = ['TinyLlama', 'GPT-4']
    accuracies = [tinyllama_avg * 100, gpt4_avg * 100]
    colors = ['#FF6B6B', '#4ECDC4']
    
    bars = ax.bar(models, accuracies, color=colors, alpha=0.8, edgecolor='black', linewidth=2)
    
    # Add value labels on bars
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{acc:.1f}%',
                ha='center', va='bottom', fontsize=14, fontweight='bold')
    
    ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax.set_title('VLM Accuracy Comparison: TinyLlama vs GPT-4\nTrajectory Description Task',
                 fontsize=14, fontweight='bold')
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add improvement annotation
    improvement = (gpt4_avg - tinyllama_avg) * 100
    ax.annotate(f'+{improvement:.1f}%\nimprovement',
                xy=(1, gpt4_avg * 100), xytext=(1.3, (tinyllama_avg + gpt4_avg) * 50),
                arrowprops=dict(arrowstyle='->', color='green', lw=2),
                fontsize=12, fontweight='bold', color='green')
    
    plt.tight_layout()
    
    output_path = Path('results') / get_output_filename('accuracy_comparison')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Created: {output_path}")
    return output_path


def create_metrics_breakdown(results: dict):
    """Create breakdown of metrics (type, direction, coordinates, speed)."""
    # Aggregate metrics across all samples
    metrics = ['Type', 'Direction', 'Coordinates', 'Speed']
    
    tinyllama_scores = [
        np.mean([s['tinyllama_metrics']['type_mentioned'] for s in results['samples']]) * 100,
        np.mean([s['tinyllama_metrics']['direction_mentioned'] for s in results['samples']]) * 100,
        np.mean([s['tinyllama_metrics']['has_coordinates'] for s in results['samples']]) * 100,
        np.mean([s['tinyllama_metrics']['speed_mentioned'] for s in results['samples']]) * 100
    ]
    
    gpt4_scores = [
        np.mean([s['gpt4_metrics']['type_mentioned'] for s in results['samples']]) * 100,
        np.mean([s['gpt4_metrics']['direction_mentioned'] for s in results['samples']]) * 100,
        np.mean([s['gpt4_metrics']['has_coordinates'] for s in results['samples']]) * 100,
        np.mean([s['gpt4_metrics']['speed_mentioned'] for s in results['samples']]) * 100
    ]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    bars1 = ax.bar(x - width/2, tinyllama_scores, width, label='TinyLlama', 
                   color='#FF6B6B', alpha=0.8, edgecolor='black')
    bars2 = ax.bar(x + width/2, gpt4_scores, width, label='GPT-4', 
                   color='#4ECDC4', alpha=0.8, edgecolor='black')
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.0f}%',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax.set_xlabel('Metric Category', fontsize=12, fontweight='bold')
    ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax.set_title('Metric Breakdown: TinyLlama vs GPT-4',
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend(fontsize=11)
    ax.set_ylim(0, 110)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    output_path = Path('results') / get_output_filename('metrics_breakdown')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Created: {output_path}")
    return output_path


def create_enhanced_metrics_comparison(results: dict):
    """Create comparison of enhanced metrics (BLEU, ROUGE, Semantic)."""
    # Calculate average enhanced metrics
    tinyllama_bleu = np.mean([s['tinyllama_enhanced_metrics']['bleu'] for s in results['samples']])
    tinyllama_rouge = np.mean([s['tinyllama_enhanced_metrics']['rouge_l'] for s in results['samples']])
    tinyllama_semantic = np.mean([s['tinyllama_enhanced_metrics']['semantic_similarity'] for s in results['samples']])
    
    gpt4_bleu = np.mean([s['gpt4_enhanced_metrics']['bleu'] for s in results['samples']])
    gpt4_rouge = np.mean([s['gpt4_enhanced_metrics']['rouge_l'] for s in results['samples']])
    gpt4_semantic = np.mean([s['gpt4_enhanced_metrics']['semantic_similarity'] for s in results['samples']])
    
    metrics = ['BLEU', 'ROUGE-L', 'Semantic\nSimilarity']
    tinyllama_scores = [tinyllama_bleu, tinyllama_rouge, tinyllama_semantic]
    gpt4_scores = [gpt4_bleu, gpt4_rouge, gpt4_semantic]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    bars1 = ax.bar(x - width/2, tinyllama_scores, width, label='TinyLlama', 
                   color='#FF6B6B', alpha=0.8, edgecolor='black')
    bars2 = ax.bar(x + width/2, gpt4_scores, width, label='GPT-4', 
                   color='#4ECDC4', alpha=0.8, edgecolor='black')
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax.set_xlabel('Enhanced Metric', fontsize=12, fontweight='bold')
    ax.set_ylabel('Score (0-1, higher is better)', fontsize=12, fontweight='bold')
    ax.set_title('Enhanced Metrics Comparison: TinyLlama vs GPT-4',
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend(fontsize=11)
    ax.set_ylim(0, 1.0)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    output_path = Path('results') / get_output_filename('enhanced_metrics_comparison')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Created: {output_path}")
    return output_path


def main():
    """Generate all comparison visualizations."""
    print("="*70)
    print("Creating GPT-4 vs TinyLlama Comparison Visualizations")
    print("="*70)
    print()
    
    # Load results
    print("📂 Loading evaluation results...")
    results = load_results()
    print(f"✅ Loaded {len(results['samples'])} samples")
    print()
    
    # Create visualizations
    print("📊 Generating visualizations...")
    paths = []
    
    paths.append(create_accuracy_comparison_bar_chart(results))
    paths.append(create_metrics_breakdown(results))
    paths.append(create_enhanced_metrics_comparison(results))
    
    print()
    print("="*70)
    print("✅ All visualizations created successfully!")
    print("="*70)
    print()
    print("Output files:")
    for i, path in enumerate(paths, 1):
        print(f"  {i}. {path}")
    print()
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())

