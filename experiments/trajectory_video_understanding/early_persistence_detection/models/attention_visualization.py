"""
Attention Visualization Module

Visualizes attention weights to show low weights on transient frames.
Creates heatmaps and analysis of attention distribution.
"""

import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Optional


class AttentionVisualizer:
    """Visualizes attention weights from transformer models."""
    
    def __init__(self, num_heads: int = 4, save_dir: str = './visualizations'):
        """
        Initialize attention visualizer.
        
        Args:
            num_heads: Number of attention heads
            save_dir: Directory to save visualizations
        """
        self.num_heads = num_heads
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
    
    def visualize_sample(self, attention_weights: torch.Tensor,
                        transient_frames: List[int],
                        sample_id: str = "sample") -> Path:
        """
        Visualize attention for a single sample.
        
        Args:
            attention_weights: Attention tensor (num_heads, seq_len, seq_len)
            transient_frames: List of transient frame indices
            sample_id: Sample identifier
            
        Returns:
            Path to saved visualization
        """
        output_path = self.save_dir / f"{sample_id}_attention.png"
        plot_attention_heatmap(attention_weights, transient_frames, output_path)
        return output_path
    
    def analyze_attention_efficiency(self, attention_weights: torch.Tensor,
                                    transient_frames: List[int]) -> Dict:
        """
        Analyze attention efficiency (focus on persistent vs transient).
        
        Args:
            attention_weights: Attention tensor
            transient_frames: List of transient frame indices
            
        Returns:
            Dictionary with efficiency metrics
        """
        # Average attention across heads
        avg_attention = attention_weights.mean(dim=0)  # (seq_len, seq_len)
        
        # Get attention each frame receives (column-wise mean)
        frame_attention = avg_attention.mean(dim=0).cpu().numpy()  # (seq_len,)
        
        seq_len = len(frame_attention)
        persistent_frames = [i for i in range(seq_len) if i not in transient_frames]
        
        # Calculate averages
        if persistent_frames:
            avg_attention_persistent = float(np.mean([frame_attention[i] for i in persistent_frames]))
        else:
            avg_attention_persistent = 0.0
        
        if transient_frames:
            avg_attention_transient = float(np.mean([frame_attention[i] for i in transient_frames]))
        else:
            avg_attention_transient = 0.0
        
        # Attention ratio (should be > 1 for good efficiency)
        if avg_attention_transient > 0:
            attention_ratio = avg_attention_persistent / avg_attention_transient
        else:
            attention_ratio = float('inf') if avg_attention_persistent > 0 else 1.0
        
        analysis = {
            'avg_attention_persistent': avg_attention_persistent,
            'avg_attention_transient': avg_attention_transient,
            'attention_ratio': attention_ratio,
            'frame_attention': frame_attention.tolist(),
            'transient_frames': transient_frames,
            'persistent_frames': persistent_frames
        }
        
        return analysis


def extract_attention_weights(attention_tensor: torch.Tensor) -> torch.Tensor:
    """
    Extract and format attention weights.
    
    Args:
        attention_tensor: Raw attention tensor (num_heads, seq_len, seq_len)
        
    Returns:
        Formatted attention weights
    """
    # Ensure it's on CPU and detached
    attention_weights = attention_tensor.detach().cpu()
    
    # Normalize if needed
    if attention_weights.max() > 1.0:
        attention_weights = torch.softmax(attention_weights, dim=-1)
    
    return attention_weights


def plot_attention_heatmap(attention_weights: torch.Tensor,
                           transient_frames: List[int],
                           output_path: Path):
    """
    Plot attention heatmap with transient frames highlighted.
    
    Args:
        attention_weights: Attention tensor (num_heads, seq_len, seq_len)
        transient_frames: List of transient frame indices
        output_path: Path to save figure
    """
    # Average across attention heads
    avg_attention = attention_weights.mean(dim=0).cpu().numpy()  # (seq_len, seq_len)
    
    seq_len = avg_attention.shape[0]
    
    # Create figure with subplots
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Full attention matrix
    ax1 = axes[0]
    sns.heatmap(avg_attention, cmap='viridis', ax=ax1, cbar_kws={'label': 'Attention Weight'})
    ax1.set_title('Attention Matrix (Averaged Across Heads)')
    ax1.set_xlabel('Key Frame')
    ax1.set_ylabel('Query Frame')
    
    # Highlight transient frames
    for frame_idx in transient_frames:
        if frame_idx < seq_len:
            ax1.axhline(y=frame_idx + 0.5, color='red', linewidth=2, alpha=0.5)
            ax1.axvline(x=frame_idx + 0.5, color='red', linewidth=2, alpha=0.5)
    
    # Plot 2: Average attention per frame
    ax2 = axes[1]
    frame_attention = avg_attention.mean(axis=0)
    frames = np.arange(seq_len)
    
    colors = ['red' if i in transient_frames else 'blue' for i in range(seq_len)]
    ax2.bar(frames, frame_attention, color=colors, alpha=0.7)
    ax2.set_title('Average Attention Received Per Frame')
    ax2.set_xlabel('Frame Index')
    ax2.set_ylabel('Average Attention Weight')
    ax2.legend(['Transient', 'Persistent'], loc='upper right')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def save_attention_analysis(analysis: Dict, output_file: Path):
    """
    Save attention analysis to JSON file.
    
    Args:
        analysis: Analysis dictionary
        output_file: Output JSON file path
    """
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump(analysis, f, indent=2)
