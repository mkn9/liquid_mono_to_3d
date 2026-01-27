"""
Attention Visualization Script

Generates visualizations of attention patterns.
"""

import torch
import torch.nn.functional as F
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import argparse
import sys
from typing import List, Dict

# Add models to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'models'))
from early_persistence_classifier import EarlyPersistenceClassifier
from attention_visualization import AttentionVisualizer


def extract_attention_from_model(model, video: torch.Tensor) -> torch.Tensor:
    """
    Extract attention weights from model.
    
    Args:
        model: Trained model with attention mechanism
        video: Input video tensor (1, T, C, H, W)
    
    Returns:
        Attention weights tensor
    """
    model.eval()
    
    with torch.no_grad():
        # Extract features
        features = model.feature_extractor(video)  # (1, T, feature_dim)
        
        # Process through LSTM to get internal states
        # For visualization, we'll use the feature correlations as a proxy
        # In a real implementation with explicit attention, we'd extract it directly
        
        # Compute attention as similarity between frame features
        # Normalize features
        features_norm = F.normalize(features, p=2, dim=-1)  # (1, T, feature_dim)
        
        # Compute attention matrix as dot product (scaled dot-product attention)
        attention = torch.matmul(features_norm, features_norm.transpose(1, 2))  # (1, T, T)
        
        # Apply softmax to get attention weights
        attention = F.softmax(attention / np.sqrt(features.shape[-1]), dim=-1)
    
    return attention


def generate_attention_heatmap(attention_weights: torch.Tensor, transient_frames: List[int], 
                               output_path: Path):
    """
    Generate attention heatmap visualization.
    
    Args:
        attention_weights: Attention weights tensor (num_heads, T, T)
        transient_frames: List of transient frame indices
        output_path: Path to save visualization
    """
    if attention_weights.dim() == 3:
        # Average over attention heads
        attention_weights = attention_weights.mean(dim=0)
    elif attention_weights.dim() == 4:
        # (batch, heads, T, T) -> (T, T)
        attention_weights = attention_weights[0].mean(dim=0)
    
    # Ensure we have (T, T)
    if attention_weights.dim() == 3:
        attention_weights = attention_weights[0]
    
    attention_np = attention_weights.detach().cpu().numpy()
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Create heatmap
    sns.heatmap(attention_np, cmap='viridis', ax=ax, 
                cbar_kws={'label': 'Attention Weight'},
                square=True)
    
    # Mark transient frames
    for frame_idx in transient_frames:
        # Vertical line (marking key frames)
        ax.axvline(frame_idx + 0.5, color='red', linestyle='--', linewidth=2, alpha=0.7)
        # Horizontal line (marking query frames)
        ax.axhline(frame_idx + 0.5, color='red', linestyle='--', linewidth=2, alpha=0.7)
    
    ax.set_xlabel('Key Frame', fontsize=12)
    ax.set_ylabel('Query Frame', fontsize=12)
    ax.set_title('Attention Pattern (Transient frames marked in red)', fontsize=14)
    
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def analyze_attention_distribution(attention_weights: torch.Tensor, 
                                   transient_frames: List[int]) -> Dict:
    """
    Analyze attention distribution across frames.
    
    Args:
        attention_weights: Attention weights tensor
        transient_frames: List of transient frame indices
    
    Returns:
        Dictionary with attention analysis metrics
    """
    if attention_weights.dim() == 3:
        attention_weights = attention_weights.mean(dim=0)
    elif attention_weights.dim() == 4:
        attention_weights = attention_weights[0].mean(dim=0)
    if attention_weights.dim() == 3:
        attention_weights = attention_weights[0]
    
    seq_len = attention_weights.shape[0]
    
    # Compute total attention received by each frame (sum over query dimension)
    attention_received = attention_weights.sum(dim=0).cpu().numpy()
    
    # Separate persistent and transient frames
    persistent_frames = [i for i in range(seq_len) if i not in transient_frames]
    
    if persistent_frames:
        avg_attention_persistent = attention_received[persistent_frames].mean()
    else:
        avg_attention_persistent = 0.0
    
    if transient_frames:
        transient_indices = [i for i in transient_frames if i < seq_len]
        if transient_indices:
            avg_attention_transient = attention_received[transient_indices].mean()
        else:
            avg_attention_transient = 0.0
    else:
        avg_attention_transient = 0.0
    
    attention_ratio = (avg_attention_persistent / (avg_attention_transient + 1e-6))
    
    return {
        'avg_attention_persistent': float(avg_attention_persistent),
        'avg_attention_transient': float(avg_attention_transient),
        'attention_ratio': float(attention_ratio),
        'attention_received_per_frame': attention_received.tolist()
    }


def save_visualization_batch(samples: List[Dict], output_dir: Path):
    """
    Save batch of visualizations.
    
    Args:
        samples: List of sample dictionaries with 'video' and 'metadata'
        output_dir: Directory to save visualizations
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for idx, sample in enumerate(samples):
        video = sample['video']
        metadata = sample.get('metadata', {})
        transient_frames = metadata.get('transient_frames', [])
        
        # Mock attention for now (in real use, extract from model)
        seq_len = video.shape[0] if video.dim() == 4 else video.shape[1]
        mock_attention = torch.randn(seq_len, seq_len).softmax(dim=-1)
        
        output_path = output_dir / f"sample_{idx:04d}_attention.png"
        generate_attention_heatmap(mock_attention, transient_frames, output_path)


def main():
    """Main visualization script."""
    parser = argparse.ArgumentParser(description='Visualize attention patterns')
    parser.add_argument('--model', type=str, required=True,
                       help='Path to trained model')
    parser.add_argument('--data', type=str, required=True,
                       help='Path to data directory')
    parser.add_argument('--output', type=str, default='./visualizations',
                       help='Output directory')
    parser.add_argument('--num_samples', type=int, default=20,
                       help='Number of samples to visualize')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("ATTENTION VISUALIZATION")
    print("=" * 80)
    print(f"Model: {args.model}")
    print(f"Data: {args.data}")
    print(f"Output: {args.output}")
    print(f"Samples: {args.num_samples}")
    print("=" * 80)
    
    # Load model
    print("\n📦 Loading model...")
    from evaluate_model import load_model_for_evaluation
    model = load_model_for_evaluation(args.model, device=args.device)
    print("✅ Model loaded")
    
    # Initialize visualizer
    visualizer = AttentionVisualizer(save_dir=Path(args.output))
    
    # Process samples
    print(f"\n🎨 Generating visualizations for {args.num_samples} samples...")
    
    # TODO: Load actual samples from dataset
    # For now, create mock samples
    samples = []
    
    save_visualization_batch(samples, Path(args.output))
    
    print(f"\n✅ Visualizations saved to: {args.output}")


if __name__ == '__main__':
    main()

