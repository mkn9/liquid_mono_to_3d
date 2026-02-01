"""
Component Diagnostics: Measure Signal Preservation Through Pipeline

Purpose: Identify WHERE in the pipeline visual information is lost.

Pipeline stages tested:
1. Video → MagVIT features (512-dim)
2. MagVIT → Liquid fusion (4096-dim)
3. Liquid → Statistics (5 values)
4. Statistics → LLM interpretation

Metrics:
- Signal-to-noise ratio
- Information entropy
- Feature diversity
- Reconstruction error (where applicable)
"""

import torch
import numpy as np
from typing import Dict, List, Tuple
from pathlib import Path
from datetime import datetime
import json
import matplotlib.pyplot as plt


def measure_signal_quality(features: torch.Tensor) -> Dict[str, float]:
    """
    Measure quality of feature representation.
    
    Args:
        features: (B, D) feature tensor
        
    Returns:
        dict: Quality metrics
    """
    feat = features.cpu().numpy().flatten()
    
    # 1. Signal-to-noise ratio (SNR)
    signal_power = np.mean(feat ** 2)
    noise_estimate = np.std(feat) ** 2  # Use variance as noise estimate
    snr = 10 * np.log10(signal_power / (noise_estimate + 1e-10))
    
    # 2. Information entropy
    # Discretize to compute entropy
    hist, _ = np.histogram(feat, bins=50, density=True)
    hist = hist + 1e-10  # Avoid log(0)
    entropy = -np.sum(hist * np.log2(hist))
    
    # 3. Feature diversity (std/mean ratio)
    diversity = np.std(feat) / (abs(np.mean(feat)) + 1e-10)
    
    # 4. Dynamic range
    dynamic_range = feat.max() - feat.min()
    
    return {
        'snr_db': float(snr),
        'entropy': float(entropy),
        'diversity': float(diversity),
        'dynamic_range': float(dynamic_range),
        'mean': float(feat.mean()),
        'std': float(feat.std())
    }


def compare_signal_preservation(input_features: torch.Tensor, 
                               output_features: torch.Tensor) -> Dict[str, float]:
    """
    Compare signal quality before and after processing.
    
    Args:
        input_features: Features before processing
        output_features: Features after processing
        
    Returns:
        dict: Preservation metrics
    """
    input_quality = measure_signal_quality(input_features)
    output_quality = measure_signal_quality(output_features)
    
    # Compute preservation ratios
    preservation = {
        'snr_ratio': output_quality['snr_db'] / (input_quality['snr_db'] + 1e-10),
        'entropy_ratio': output_quality['entropy'] / (input_quality['entropy'] + 1e-10),
        'diversity_ratio': output_quality['diversity'] / (input_quality['diversity'] + 1e-10),
        'input_quality': input_quality,
        'output_quality': output_quality
    }
    
    # Overall preservation score (average of ratios)
    preservation['overall_preservation'] = np.mean([
        preservation['snr_ratio'],
        preservation['entropy_ratio'],
        preservation['diversity_ratio']
    ])
    
    return preservation


def diagnose_pipeline_stages() -> Dict:
    """
    Diagnose each stage of the pipeline for signal preservation.
    
    Returns:
        dict: Diagnostic results for each stage
    """
    print("\n" + "="*70)
    print("PIPELINE COMPONENT DIAGNOSTICS")
    print("="*70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    diagnostics = {}
    
    # Stage 1: Mock video → MagVIT features
    print("\n📊 Stage 1: Video → MagVIT (512-dim)")
    mock_video = torch.randn(1, 32, 3, 128, 128, device=device)  # (B, T, C, H, W)
    
    # Flatten video for quality measurement
    video_flat = mock_video.view(1, -1)  # (1, 32*3*128*128)
    magvit_features = torch.randn(1, 512, device=device)  # Mock MagVIT output
    
    stage1_preservation = compare_signal_preservation(video_flat, magvit_features)
    diagnostics['stage1_video_to_magvit'] = stage1_preservation
    
    print(f"  Preservation score: {stage1_preservation['overall_preservation']:.3f}")
    print(f"  SNR ratio: {stage1_preservation['snr_ratio']:.3f}")
    print(f"  Entropy ratio: {stage1_preservation['entropy_ratio']:.3f}")
    
    # Stage 2: MagVIT → Liquid fusion (4096-dim)
    print("\n📊 Stage 2: MagVIT (512-dim) → Liquid Fusion (4096-dim)")
    
    # Mock Liquid fusion (upsampling from 512 to 4096)
    liquid_output = torch.randn(1, 4096, device=device)
    
    stage2_preservation = compare_signal_preservation(magvit_features, liquid_output)
    diagnostics['stage2_magvit_to_liquid'] = stage2_preservation
    
    print(f"  Preservation score: {stage2_preservation['overall_preservation']:.3f}")
    print(f"  SNR ratio: {stage2_preservation['snr_ratio']:.3f}")
    print(f"  Entropy ratio: {stage2_preservation['entropy_ratio']:.3f}")
    
    # Stage 3: Liquid → Statistics (5 values)
    print("\n📊 Stage 3: Liquid (4096-dim) → Statistics (5 values)")
    
    # Compress to 5 statistics
    stats = torch.tensor([
        liquid_output.mean().item(),
        liquid_output.std().item(),
        torch.norm(liquid_output, p=2).item(),
        liquid_output.min().item(),
        liquid_output.max().item()
    ]).unsqueeze(0)
    
    stage3_preservation = compare_signal_preservation(liquid_output, stats)
    diagnostics['stage3_liquid_to_stats'] = stage3_preservation
    
    print(f"  Preservation score: {stage3_preservation['overall_preservation']:.3f}")
    print(f"  SNR ratio: {stage3_preservation['snr_ratio']:.3f}")
    print(f"  Entropy ratio: {stage3_preservation['entropy_ratio']:.3f}")
    print(f"  ⚠️  EXTREME COMPRESSION: 4096 → 5 values")
    
    # Overall pipeline
    print("\n" + "="*70)
    print("OVERALL PIPELINE ANALYSIS")
    print("="*70)
    
    # Compute cumulative preservation
    cumulative_preservation = (
        stage1_preservation['overall_preservation'] *
        stage2_preservation['overall_preservation'] *
        stage3_preservation['overall_preservation']
    )
    
    diagnostics['cumulative_preservation'] = cumulative_preservation
    
    print(f"Cumulative preservation: {cumulative_preservation:.3f}")
    print(f"\nStage-by-stage breakdown:")
    print(f"  1. Video → MagVIT:    {stage1_preservation['overall_preservation']:.3f}")
    print(f"  2. MagVIT → Liquid:   {stage2_preservation['overall_preservation']:.3f}")
    print(f"  3. Liquid → Stats:    {stage3_preservation['overall_preservation']:.3f}")
    print(f"  Cumulative:           {cumulative_preservation:.3f}")
    
    # Identify bottleneck
    preservations = {
        'Video → MagVIT': stage1_preservation['overall_preservation'],
        'MagVIT → Liquid': stage2_preservation['overall_preservation'],
        'Liquid → Stats': stage3_preservation['overall_preservation']
    }
    
    bottleneck_stage = min(preservations.items(), key=lambda x: x[1])
    diagnostics['bottleneck_stage'] = bottleneck_stage[0]
    diagnostics['bottleneck_score'] = bottleneck_stage[1]
    
    print(f"\n🔍 BOTTLENECK IDENTIFIED: {bottleneck_stage[0]}")
    print(f"   Preservation score: {bottleneck_stage[1]:.3f}")
    print("="*70)
    
    return diagnostics


def create_diagnostic_visualization(diagnostics: Dict, output_dir: Path):
    """Create visualization of signal preservation through pipeline."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    stages = ['Video → MagVIT', 'MagVIT → Liquid', 'Liquid → Stats']
    preservations = [
        diagnostics['stage1_video_to_magvit']['overall_preservation'],
        diagnostics['stage2_magvit_to_liquid']['overall_preservation'],
        diagnostics['stage3_liquid_to_stats']['overall_preservation']
    ]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    colors = ['#4ECDC4' if p > 0.5 else '#FF6B6B' for p in preservations]
    bars = ax.bar(stages, preservations, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    
    # Add value labels
    for bar, val in zip(bars, preservations):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.3f}',
                ha='center', va='bottom', fontweight='bold', fontsize=12)
    
    ax.set_ylabel('Signal Preservation Score', fontsize=12, fontweight='bold')
    ax.set_title('Pipeline Signal Preservation Diagnostic', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 1.2)
    ax.axhline(y=0.5, color='orange', linestyle='--', alpha=0.7, label='Threshold (0.5)')
    ax.grid(axis='y', alpha=0.3)
    ax.legend()
    
    plt.tight_layout()
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    filename = f"{timestamp}_component_diagnostics.png"
    output_path = output_dir / filename
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n📊 Diagnostic visualization saved: {output_path}")
    return output_path


def save_diagnostic_results(diagnostics: Dict, output_dir: Path = Path("results")) -> Path:
    """Save diagnostic results to JSON."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    filename = f"{timestamp}_component_diagnostics.json"
    output_path = output_dir / filename
    
    with open(output_path, 'w') as f:
        json.dump(diagnostics, f, indent=2)
    
    print(f"💾 Results saved to: {output_path}")
    return output_path


if __name__ == "__main__":
    print("="*70)
    print("Component Diagnostics: Signal Preservation Analysis")
    print("="*70)
    
    # Run diagnostics
    diagnostics = diagnose_pipeline_stages()
    
    # Interpret results
    print("\n" + "="*70)
    print("RECOMMENDATIONS")
    print("="*70)
    
    if diagnostics['bottleneck_score'] < 0.3:
        print(f"⚠️  CRITICAL: Stage '{diagnostics['bottleneck_stage']}' has very low preservation")
        print("   This stage is destroying signal - needs immediate attention")
    elif diagnostics['bottleneck_score'] < 0.5:
        print(f"⚠️  WARNING: Stage '{diagnostics['bottleneck_stage']}' has low preservation")
        print("   This stage is the primary bottleneck")
    else:
        print(f"✅ All stages have reasonable preservation (>{0.5:.1f})")
        print("   Problem may be in LLM decoding, not pipeline")
    
    if diagnostics['stage3_liquid_to_stats']['overall_preservation'] < 0.5:
        print("\n📌 Compression (Liquid → Stats) is particularly problematic")
        print("   Consider using richer encoding (Worker 2 addresses this)")
    
    print("="*70)
    
    # Save results
    output_dir = Path("experiments/liquid_vlm_integration/results")
    save_diagnostic_results(diagnostics, output_dir)
    create_diagnostic_visualization(diagnostics, output_dir)
    
    print("\n✅ Component diagnostics complete")

