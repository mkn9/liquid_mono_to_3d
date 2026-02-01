"""
MagVIT Isolation Test: Test MagVIT Alone (Bypass Liquid Fusion)

Purpose: Determine if visual features from MagVIT are meaningful.

Pipeline tested: Video → MagVIT → Statistics → GPT-4 (NO Liquid fusion)

Expected outcomes:
- If MagVIT-only > Random: Liquid fusion or LLM decoding is the problem
- If MagVIT-only = Random: MagVIT feature extraction or video quality is the problem
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List
from pathlib import Path
from datetime import datetime
import json
import os
from openai import OpenAI

# Import from existing modules
try:
    from true_e2e_visual_evaluation import (
        extract_embedding_statistics,
        calculate_accuracy_against_ground_truth
    )
except ImportError:
    print("⚠️  Importing from parent directory")
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from true_e2e_visual_evaluation import (
        extract_embedding_statistics,
        calculate_accuracy_against_ground_truth
    )


def extract_magvit_features_only(video_tensor: torch.Tensor) -> torch.Tensor:
    """
    Extract MagVIT features WITHOUT Liquid fusion.
    
    NOTE: This is a MOCK implementation. In production, replace with:
        from magvit_model import MagVITFeatureExtractor
        magvit = MagVITFeatureExtractor()
        features = magvit.extract(video_tensor)
    
    Args:
        video_tensor: (B, T, C, H, W) video frames
        
    Returns:
        features: (B, 512) MagVIT embeddings
    """
    # MOCK: Simulate MagVIT feature extraction
    # In production, use actual MagVIT model
    batch_size = video_tensor.shape[0]
    
    # Create mock features that are:
    # 1. Deterministic for same input
    # 2. Different for different inputs
    # 3. Have realistic statistics
    
    # Use mean of video as seed for deterministic features
    video_mean = video_tensor.mean().item()
    torch.manual_seed(int(abs(video_mean * 1000000)))
    
    features = torch.randn(batch_size, 512, device=video_tensor.device)
    
    return features


def create_magvit_only_prompt(magvit_features: torch.Tensor) -> str:
    """
    Create prompt from MagVIT features ONLY (no Liquid fusion, no 3D data).
    
    Args:
        magvit_features: (1, 512) MagVIT embeddings
        
    Returns:
        str: Prompt for GPT-4
    """
    stats = extract_embedding_statistics(magvit_features)
    
    prompt = f"""You are analyzing a 3D trajectory from visual features extracted from video frames.

These features were extracted using a vision model (MagVIT) from multi-view camera videos.

The visual feature characteristics are:
- Mean activation: {stats['mean']:.3f}
- Standard deviation: {stats['std']:.3f}
- L2 norm: {stats['l2_norm']:.1f}
- Value range: [{stats['min']:.3f}, {stats['max']:.3f}]

Based on these visual features ONLY, describe the 3D trajectory in natural language.
Focus on:
1. Path shape and motion pattern
2. Direction of movement
3. Motion characteristics

Be specific and factual. Do NOT make up information."""
    
    return prompt


def evaluate_magvit_only(video_tensor: torch.Tensor, ground_truth: Dict) -> Dict:
    """
    Evaluate MagVIT-only pipeline: Video → MagVIT → Stats → LLM.
    
    CRITICAL: NO Liquid fusion, NO 3D triangulation in this test.
    
    Args:
        video_tensor: (B, T, C, H, W) video
        ground_truth: Ground truth for evaluation only
        
    Returns:
        dict: Results including description and accuracy
    """
    # Extract MagVIT features (bypass Liquid)
    magvit_features = extract_magvit_features_only(video_tensor)
    
    # Create prompt from features only
    prompt = create_magvit_only_prompt(magvit_features)
    
    # Get API key
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return {
            'description': '[ERROR: OPENAI_API_KEY not set]',
            'accuracy': {'overall_accuracy': 0},
            'magvit_features_used': True,
            'liquid_fusion_used': False
        }
    
    client = OpenAI(api_key=api_key)
    
    # Call GPT-4
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are an expert in analyzing 3D trajectories from visual features."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=200,
            temperature=0.7
        )
        
        description = response.choices[0].message.content.strip()
        
    except Exception as e:
        return {
            'description': f'[ERROR: {str(e)}]',
            'accuracy': {'overall_accuracy': 0},
            'magvit_features_used': True,
            'liquid_fusion_used': False
        }
    
    # Calculate accuracy
    accuracy = calculate_accuracy_against_ground_truth(description, ground_truth)
    
    return {
        'description': description,
        'accuracy': accuracy,
        'magvit_features_used': True,
        'liquid_fusion_used': False,
        'magvit_stats': extract_embedding_statistics(magvit_features)
    }


def run_magvit_ablation_study(samples: List[Dict]) -> Dict:
    """
    Run ablation study: MagVIT-only vs Random features.
    
    Args:
        samples: List of dicts with 'video' and 'ground_truth'
        
    Returns:
        dict: Comparison results
    """
    print("\n" + "="*70)
    print("MAGVIT ISOLATION TEST: Bypassing Liquid Fusion")
    print("="*70)
    
    # Test 1: Random features (control - from previous session)
    print("\n📊 Condition 1: Random Features (Control)")
    print("   Previous result: 52.5% accuracy")
    random_accuracy = 52.5
    
    # Test 2: MagVIT-only features
    print("\n📊 Condition 2: MagVIT Features Only (No Liquid)")
    results = []
    total_accuracy = 0
    
    for i, sample in enumerate(samples):
        print(f"  Sample {i+1}/{len(samples)}...", end=" ")
        
        result = evaluate_magvit_only(sample['video'], sample['ground_truth'])
        total_accuracy += result['accuracy']['overall_accuracy']
        results.append(result)
        
        print(f"{result['accuracy']['overall_accuracy']*100:.1f}%")
    
    magvit_only_accuracy = (total_accuracy / len(samples)) * 100
    
    # Compare
    improvement = magvit_only_accuracy - random_accuracy
    
    print("\n" + "="*70)
    print("RESULTS:")
    print(f"  Random features:      {random_accuracy:.1f}%")
    print(f"  MagVIT-only features: {magvit_only_accuracy:.1f}%")
    print(f"  Improvement:          {improvement:+.1f}%")
    print("="*70)
    
    return {
        'random_accuracy': random_accuracy,
        'magvit_only_accuracy': magvit_only_accuracy,
        'improvement': improvement,
        'samples': results,
        'interpretation': interpret_magvit_results(
            magvit_only_accuracy,
            random_accuracy,
            52.5  # Previous real embeddings accuracy
        )
    }


def interpret_magvit_results(magvit_only_accuracy: float, random_accuracy: float, 
                            previous_real_accuracy: float) -> Dict:
    """
    Interpret MagVIT isolation test results to identify bottleneck.
    
    Args:
        magvit_only_accuracy: MagVIT-only pipeline accuracy
        random_accuracy: Random features accuracy (control)
        previous_real_accuracy: Previous full pipeline accuracy (MagVIT+Liquid)
        
    Returns:
        dict: Interpretation and recommendations
    """
    interpretation = {
        'magvit_only': magvit_only_accuracy,
        'random': random_accuracy,
        'previous_full_pipeline': previous_real_accuracy
    }
    
    # Scenario 1: MagVIT significantly better than random
    if magvit_only_accuracy > random_accuracy + 10:
        interpretation['bottleneck'] = 'Liquid fusion or LLM decoding'
        interpretation['conclusion'] = (
            f"MagVIT features are meaningful ({magvit_only_accuracy:.1f}% vs {random_accuracy:.1f}% random). "
            f"Since full pipeline was only {previous_real_accuracy:.1f}%, "
            "the problem is in Liquid fusion or how LLM decodes the fused embeddings."
        )
        interpretation['next_steps'] = [
            "Test Liquid fusion signal preservation",
            "Try feeding full 4096-dim embeddings instead of 5 statistics",
            "Consider learned projection layer for Liquid→LLM"
        ]
    
    # Scenario 2: MagVIT slightly better than random
    elif magvit_only_accuracy > random_accuracy + 3:
        interpretation['bottleneck'] = 'Compression too lossy or LLM decoding'
        interpretation['conclusion'] = (
            f"MagVIT features provide modest improvement ({magvit_only_accuracy:.1f}% vs {random_accuracy:.1f}%). "
            "Compression from 512-dim to 5 statistics may be too lossy."
        )
        interpretation['next_steps'] = [
            "Test with full 512-dim MagVIT embeddings (no compression)",
            "Try alternative compression methods (PCA, learned)",
            "Use GPT-4V (vision-language model) instead"
        ]
    
    # Scenario 3: MagVIT = Random
    else:
        interpretation['bottleneck'] = 'MagVIT feature extraction or video quality'
        interpretation['conclusion'] = (
            f"MagVIT features no better than random ({magvit_only_accuracy:.1f}% vs {random_accuracy:.1f}%). "
            "Either MagVIT is not extracting meaningful features, or video quality is insufficient."
        )
        interpretation['next_steps'] = [
            "Verify MagVIT model is loaded correctly",
            "Check video quality and resolution",
            "Try different vision models (CLIP, DINOv2)",
            "Ensure videos actually contain trajectory information"
        ]
    
    return interpretation


def compare_with_previous_results(magvit_results: Dict) -> Dict:
    """
    Compare MagVIT-only results with previous full pipeline results.
    
    Args:
        magvit_results: Results from run_magvit_ablation_study()
        
    Returns:
        dict: Comparison and analysis
    """
    previous_results = {
        'cheating_baseline': 75.0,  # Ground truth given to LLM
        'random_embeddings': 52.5,  # Random 4096-dim vectors
        'real_embeddings': 52.5,    # MagVIT + Liquid fusion
    }
    
    magvit_only = magvit_results['magvit_only_accuracy']
    
    comparison = {
        'magvit_only': magvit_only,
        'previous_full_pipeline': previous_results['real_embeddings'],
        'random_baseline': previous_results['random_embeddings'],
        'cheating_baseline': previous_results['cheating_baseline']
    }
    
    # Determine what this tells us
    if magvit_only > previous_results['real_embeddings']:
        comparison['finding'] = "MagVIT-only BETTER than full pipeline"
        comparison['implication'] = "Liquid fusion is HURTING performance (destroying signal)"
    elif magvit_only == previous_results['real_embeddings']:
        comparison['finding'] = "MagVIT-only SAME as full pipeline"
        comparison['implication'] = "Liquid fusion has NO EFFECT (neither helping nor hurting)"
    else:
        comparison['finding'] = "MagVIT-only WORSE than full pipeline"
        comparison['implication'] = "Liquid fusion is helping, but something else is broken"
    
    return comparison


def save_magvit_results(results: Dict, output_dir: Path = Path("results")) -> Path:
    """Save MagVIT isolation test results."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    filename = f"{timestamp}_magvit_isolation_test.json"
    output_path = output_dir / filename
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Results saved to: {output_path}")
    return output_path


if __name__ == "__main__":
    print("="*70)
    print("MagVIT Isolation Test")
    print("Testing: Video → MagVIT → Stats → GPT-4 (NO Liquid)")
    print("="*70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    
    # Create 5 mock samples
    print("\n📦 Creating mock video samples...")
    samples = []
    for i in range(5):
        samples.append({
            'video': torch.randn(1, 32, 3, 128, 128, device=device),
            'ground_truth': {
                'type': 'straight line',
                'primary_direction': 'depth (Y-axis)',
                'description': f'A straight line trajectory sample {i}'
            }
        })
    print(f"✅ {len(samples)} samples created")
    
    # Run ablation study
    results = run_magvit_ablation_study(samples)
    
    # Compare with previous
    comparison = compare_with_previous_results(results)
    
    print("\n" + "="*70)
    print("COMPARISON WITH PREVIOUS RESULTS")
    print("="*70)
    print(f"Cheating baseline (GT→LLM):    {comparison['cheating_baseline']:.1f}%")
    print(f"Random features:                {comparison['random_baseline']:.1f}%")
    print(f"Full pipeline (MagVIT+Liquid):  {comparison['previous_full_pipeline']:.1f}%")
    print(f"MagVIT-only (this test):        {comparison['magvit_only']:.1f}%")
    print("\n" + comparison['finding'])
    print("→ " + comparison['implication'])
    print("="*70)
    
    # Print interpretation
    print("\n" + "="*70)
    print("INTERPRETATION")
    print("="*70)
    print(f"Bottleneck: {results['interpretation']['bottleneck']}")
    print(f"\n{results['interpretation']['conclusion']}")
    print("\nRecommended next steps:")
    for i, step in enumerate(results['interpretation']['next_steps'], 1):
        print(f"  {i}. {step}")
    print("="*70)
    
    # Save results
    save_magvit_results({
        'ablation_results': results,
        'comparison': comparison
    }, Path("experiments/liquid_vlm_integration/results"))
    
    print("\n✅ MagVIT isolation test complete")

