"""
Fast attention visualization to validate transformer behavior.

Generate heatmaps showing:
1. High attention on persistent (white) objects
2. Low attention on transient (red) objects
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys
import json
import argparse

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.fast_dataset import FastObjectDataset, collate_fn
from src.fast_object_transformer import FastObjectTransformer


def load_model(checkpoint_path, device):
    """Load trained model."""
    model = FastObjectTransformer(
        feature_dim=256,
        num_heads=8,
        num_layers=2,
        dropout=0.1
    ).to(device)
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model


def visualize_sample_attention(
    model,
    tokens,
    labels,
    mask,
    metadata,
    device,
    output_path
):
    """
    Visualize attention for a single sample.
    
    Creates visualization showing:
    - Attention matrix heatmap
    - Per-object attention scores
    - Comparison of persistent vs transient attention
    """
    tokens = tokens.unsqueeze(0).to(device)  # Add batch dimension
    mask = mask.unsqueeze(0).to(device)
    
    # Get predictions and attention
    with torch.no_grad():
        predictions, confidence, attention = model.predict(
            tokens,
            src_key_padding_mask=~mask
        )
    
    # Extract data
    seq_len = mask.sum().item()
    attention = attention[0].cpu().numpy()  # (num_heads, seq_len, seq_len)
    attention = attention[:, :seq_len, :seq_len]  # Remove padding
    
    labels = labels[:seq_len].numpy()
    predictions = predictions[0, :seq_len].cpu().numpy()
    confidence = confidence[0, :seq_len].cpu().numpy()
    
    # Average attention across heads
    avg_attention = attention.mean(axis=0)  # (seq_len, seq_len)
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # 1. Attention matrix heatmap
    ax = axes[0, 0]
    sns.heatmap(avg_attention, cmap='viridis', ax=ax, cbar_kws={'label': 'Attention Weight'})
    ax.set_title(f'Attention Matrix (Sample {metadata["sample_id"]})', fontsize=12, fontweight='bold')
    ax.set_xlabel('Object Index (Key)')
    ax.set_ylabel('Object Index (Query)')
    
    # Add persistent/transient labels
    for i, label in enumerate(labels):
        color = 'white' if label == 0 else 'red'
        ax.text(i, -0.5, '●', color=color, ha='center', fontsize=10)
        ax.text(-0.5, i, '●', color=color, va='center', fontsize=10)
    
    # 2. Per-object incoming attention (how much attention each object receives)
    ax = axes[0, 1]
    incoming_attention = avg_attention.sum(axis=0)  # Sum over queries
    
    colors = ['green' if label == 0 else 'red' for label in labels]
    bars = ax.bar(range(seq_len), incoming_attention, color=colors, alpha=0.7)
    ax.set_title('Incoming Attention per Object', fontsize=12, fontweight='bold')
    ax.set_xlabel('Object Index')
    ax.set_ylabel('Total Attention Received')
    ax.axhline(incoming_attention.mean(), color='black', linestyle='--', label='Average')
    ax.legend()
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='green', alpha=0.7, label='Persistent (White Sphere)'),
        Patch(facecolor='red', alpha=0.7, label='Transient (Red Sphere)')
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    
    # 3. Attention comparison: Persistent vs Transient
    ax = axes[1, 0]
    
    persistent_mask = labels == 0
    transient_mask = labels == 1
    
    persistent_attention = incoming_attention[persistent_mask]
    transient_attention = incoming_attention[transient_mask]
    
    data_to_plot = []
    labels_plot = []
    if len(persistent_attention) > 0:
        data_to_plot.append(persistent_attention)
        labels_plot.append(f'Persistent\n(n={len(persistent_attention)})')
    if len(transient_attention) > 0:
        data_to_plot.append(transient_attention)
        labels_plot.append(f'Transient\n(n={len(transient_attention)})')
    
    if data_to_plot:
        bp = ax.boxplot(data_to_plot, labels=labels_plot, patch_artist=True)
        bp['boxes'][0].set_facecolor('green' if len(data_to_plot) > 0 else 'gray')
        if len(data_to_plot) > 1:
            bp['boxes'][1].set_facecolor('red')
        
        ax.set_title('Attention Distribution by Object Type', fontsize=12, fontweight='bold')
        ax.set_ylabel('Incoming Attention')
        ax.grid(axis='y', alpha=0.3)
    
    # 4. Predictions vs Ground Truth
    ax = axes[1, 1]
    
    x = np.arange(seq_len)
    width = 0.35
    
    ax.bar(x - width/2, labels, width, label='Ground Truth', color='gray', alpha=0.5)
    prediction_colors = ['green' if pred == 0 else 'red' for pred in predictions]
    ax.bar(x + width/2, predictions, width, label='Prediction', color=prediction_colors, alpha=0.7)
    
    ax.set_title('Classification Results', fontsize=12, fontweight='bold')
    ax.set_xlabel('Object Index')
    ax.set_ylabel('Class (0=Persistent, 1=Transient)')
    ax.set_yticks([0, 1])
    ax.set_yticklabels(['Persistent', 'Transient'])
    ax.legend()
    
    # Add accuracy
    accuracy = (predictions == labels).mean()
    ax.text(0.02, 0.98, f'Accuracy: {accuracy:.2%}', 
            transform=ax.transAxes, va='top', 
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    # Compute metrics
    metrics = {
        'sample_id': metadata['sample_id'],
        'num_objects': seq_len,
        'num_persistent': int(persistent_mask.sum()),
        'num_transient': int(transient_mask.sum()),
        'accuracy': float(accuracy),
        'avg_persistent_attention': float(persistent_attention.mean()) if len(persistent_attention) > 0 else 0,
        'avg_transient_attention': float(transient_attention.mean()) if len(transient_attention) > 0 else 0,
        'attention_ratio': float(persistent_attention.mean() / transient_attention.mean()) if len(transient_attention) > 0 and transient_attention.mean() > 0 else 0
    }
    
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='results/attention_viz')
    parser.add_argument('--num_samples', type=int, default=10)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    
    print(f"Visualizing attention on {args.device}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model
    print("Loading model...")
    model = load_model(args.checkpoint, args.device)
    
    # Load validation dataset
    print("Loading validation dataset...")
    dataset = FastObjectDataset(
        data_root=args.data_root,
        max_samples=args.num_samples,
        split='val'
    )
    
    # Visualize samples
    print(f"Visualizing {min(args.num_samples, len(dataset))} samples...")
    all_metrics = []
    
    for i in range(min(args.num_samples, len(dataset))):
        tokens, labels, mask, metadata = dataset[i]
        
        output_path = output_dir / f'attention_sample_{i:03d}_{metadata["sample_id"]}.png'
        
        print(f"  [{i+1}/{args.num_samples}] {metadata['sample_id']}: "
              f"{metadata['num_persistent']} persistent, {metadata['num_transient']} transient")
        
        metrics = visualize_sample_attention(
            model, tokens, labels, mask, metadata,
            args.device, output_path
        )
        
        all_metrics.append(metrics)
        
        print(f"    → Accuracy: {metrics['accuracy']:.2%}, "
              f"Attention ratio (P/T): {metrics['attention_ratio']:.2f}x")
    
    # Save aggregate metrics
    with open(output_dir / 'attention_metrics.json', 'w') as f:
        json.dump(all_metrics, f, indent=2)
    
    # Print summary
    print(f"\n{'='*60}")
    print("ATTENTION VALIDATION SUMMARY")
    print(f"{'='*60}")
    
    avg_accuracy = np.mean([m['accuracy'] for m in all_metrics])
    avg_ratio = np.mean([m['attention_ratio'] for m in all_metrics if m['attention_ratio'] > 0])
    
    print(f"Average Accuracy: {avg_accuracy:.2%}")
    print(f"Average Attention Ratio (Persistent/Transient): {avg_ratio:.2f}x")
    print(f"\nVisualizations saved to: {output_dir}")
    print(f"{'='*60}")
    
    # Check if attention is working correctly
    if avg_ratio > 1.5:
        print("✅ SUCCESS: Transformer attends MORE to persistent objects!")
    elif avg_ratio < 0.67:
        print("⚠️  WARNING: Transformer attends MORE to transient objects (unexpected)")
    else:
        print("⚠️  WARNING: No clear attention preference detected")


if __name__ == '__main__':
    main()

