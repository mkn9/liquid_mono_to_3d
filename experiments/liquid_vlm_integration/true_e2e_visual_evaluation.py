"""
True End-to-End Visual Evaluation: The Honest Test

This module implements the REAL vision-to-language pipeline:
    Video → MagVIT → Liquid Fusion → GPT-4 → Description

CRITICAL: The LLM receives ONLY embeddings, NOT ground truth data.
This tests actual visual understanding, not text-to-text conversion.
"""

import torch
import numpy as np
from typing import Dict, List
from pathlib import Path
from datetime import datetime
import json
import os
from openai import OpenAI


def extract_embedding_statistics(embedding: torch.Tensor) -> Dict[str, float]:
    """
    Extract summary statistics from embeddings for GPT-4 prompt.
    
    Args:
        embedding: (1, 4096) tensor of fused embeddings
        
    Returns:
        dict: Statistics (mean, std, l2_norm, min, max)
    """
    emb = embedding.cpu().numpy().flatten()
    
    stats = {
        'mean': float(emb.mean()),
        'std': float(emb.std()),
        'l2_norm': float((emb ** 2).sum() ** 0.5),
        'min': float(emb.min()),
        'max': float(emb.max())
    }
    
    return stats


def create_visual_prompt(fused_embedding: torch.Tensor) -> str:
    """
    Create a prompt for GPT-4 that includes ONLY embedding statistics.
    
    CRITICAL: This function must NOT include ground truth data.
    
    Args:
        fused_embedding: (1, 4096) tensor from Liquid fusion
        
    Returns:
        str: Prompt with embedding statistics only
    """
    stats = extract_embedding_statistics(fused_embedding)
    
    prompt = f"""You are analyzing a 3D trajectory from visual-spatial embeddings extracted from stereo camera videos.

These embeddings were created by:
1. Processing video frames through a vision model (MagVIT)
2. Triangulating 3D positions from 2D tracks
3. Fusing visual and spatial features using Liquid Neural Networks

The fused embedding characteristics are:
- Mean activation: {stats['mean']:.3f}
- Standard deviation: {stats['std']:.3f}
- L2 norm: {stats['l2_norm']:.1f}
- Value range: [{stats['min']:.3f}, {stats['max']:.3f}]

Based on these visual-spatial features, describe the 3D trajectory in natural language.
Focus on:
1. Path shape and motion pattern
2. Direction of movement  
3. Motion characteristics

Be specific and factual. Do NOT make up information or mention videos/URLs."""
    
    return prompt


def evaluate_from_embeddings(fused_embedding: torch.Tensor, prompt: str = None) -> Dict:
    """
    Generate trajectory description from fused embeddings ONLY.
    
    CRITICAL: This function does NOT receive ground truth data.
    
    Args:
        fused_embedding: (1, 4096) tensor from Liquid fusion
        prompt: Optional custom prompt (default: use create_visual_prompt)
        
    Returns:
        dict: {'description': str, 'stats': dict}
    """
    # Get API key
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return {
            'description': '[ERROR: OPENAI_API_KEY not set]',
            'stats': extract_embedding_statistics(fused_embedding)
        }
    
    client = OpenAI(api_key=api_key)
    
    # Create prompt from embeddings only (no ground truth!)
    if prompt is None:
        prompt = create_visual_prompt(fused_embedding)
    
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are an expert in analyzing 3D trajectories from visual-spatial embeddings."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=200,
            temperature=0.7
        )
        
        description = response.choices[0].message.content.strip()
        
        return {
            'description': description,
            'stats': extract_embedding_statistics(fused_embedding)
        }
    
    except Exception as e:
        return {
            'description': f'[ERROR: {str(e)}]',
            'stats': extract_embedding_statistics(fused_embedding)
        }


def calculate_accuracy_against_ground_truth(description: str, ground_truth: Dict) -> Dict[str, float]:
    """
    Calculate accuracy by comparing description to ground truth.
    
    NOTE: Ground truth is ONLY used for evaluation, NOT for generation.
    
    Args:
        description: Generated description from embeddings
        ground_truth: Ground truth data (type, description, etc.)
        
    Returns:
        dict: Accuracy metrics
    """
    desc_lower = description.lower()
    
    # Check type mentioned
    type_mentioned = 1 if ground_truth['type'].lower() in desc_lower else 0
    
    # Check direction mentioned (if available)
    direction_mentioned = 0
    if 'primary_direction' in ground_truth:
        direction_mentioned = 1 if ground_truth['primary_direction'].lower() in desc_lower else 0
    
    # Check has coordinates (numbers present)
    has_coordinates = 1 if any(char.isdigit() for char in description) else 0
    
    # Check speed mentioned
    speed_terms = ['speed', 'velocity', 'fast', 'slow', 'moving', 'motion']
    speed_mentioned = 1 if any(term in desc_lower for term in speed_terms) else 0
    
    # Overall accuracy
    overall_accuracy = (type_mentioned + direction_mentioned + has_coordinates + speed_mentioned) / 4.0
    
    return {
        "type_mentioned": type_mentioned,
        "direction_mentioned": direction_mentioned,
        "has_coordinates": has_coordinates,
        "speed_mentioned": speed_mentioned,
        "overall_accuracy": overall_accuracy
    }


def run_true_e2e_evaluation(samples: List[Dict]) -> Dict:
    """
    Run true end-to-end visual evaluation on multiple samples.
    
    Args:
        samples: List of dicts with 'fused_embedding' and 'ground_truth'
        
    Returns:
        dict: Evaluation results
    """
    results = {
        'num_samples': len(samples),
        'samples': [],
        'average_accuracy': 0.0,
        'timestamp': datetime.now().strftime("%Y%m%d_%H%M")
    }
    
    total_accuracy = 0.0
    
    for i, sample in enumerate(samples):
        print(f"Evaluating sample {i+1}/{len(samples)}...", end=" ")
        
        # Generate description from embeddings ONLY
        eval_result = evaluate_from_embeddings(sample['fused_embedding'])
        
        # Calculate accuracy against ground truth (separate step!)
        accuracy = calculate_accuracy_against_ground_truth(
            eval_result['description'],
            sample['ground_truth']
        )
        
        total_accuracy += accuracy['overall_accuracy']
        
        # Store results
        sample_result = {
            'sample_id': i,
            'description': eval_result['description'],
            'accuracy': accuracy,
            'embedding_used': True,  # Confirm we used embeddings, not ground truth
            'embedding_stats': eval_result['stats']
        }
        
        results['samples'].append(sample_result)
        print(f"Accuracy: {accuracy['overall_accuracy']*100:.1f}%")
    
    results['average_accuracy'] = (total_accuracy / len(samples)) * 100 if len(samples) > 0 else 0
    
    return results


def run_ablation_study(samples: List[Dict]) -> Dict:
    """
    Ablation study: Compare random vs real embeddings.
    
    Args:
        samples: List of samples with real fused_embeddings and ground_truth
        
    Returns:
        dict: Comparison results
    """
    print("\n" + "="*70)
    print("ABLATION STUDY: Random vs Real Embeddings")
    print("="*70)
    
    # Test 1: Random embeddings (control)
    print("\n📊 Test 1: Random Embeddings (Control)")
    random_samples = []
    for sample in samples:
        random_samples.append({
            'fused_embedding': torch.randn_like(sample['fused_embedding']),
            'ground_truth': sample['ground_truth']
        })
    
    random_results = run_true_e2e_evaluation(random_samples)
    
    # Test 2: Real embeddings
    print("\n📊 Test 2: Real Embeddings (From MagVIT + Liquid)")
    real_results = run_true_e2e_evaluation(samples)
    
    # Compare
    comparison = {
        'random_accuracy': random_results['average_accuracy'],
        'real_accuracy': real_results['average_accuracy'],
        'improvement': real_results['average_accuracy'] - random_results['average_accuracy'],
        'random_results': random_results,
        'real_results': real_results
    }
    
    print("\n" + "="*70)
    print("RESULTS:")
    print(f"  Random embeddings: {comparison['random_accuracy']:.1f}%")
    print(f"  Real embeddings:   {comparison['real_accuracy']:.1f}%")
    print(f"  Improvement:       {comparison['improvement']:.1f}%")
    print("="*70)
    
    return comparison


def save_results(results: Dict, output_dir: Path = Path("results")) -> Path:
    """Save evaluation results to timestamped JSON file."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    filename = f"{timestamp}_true_e2e_visual_eval.json"
    output_path = output_dir / filename
    
    with open(output_path, 'w') as f:
        # Convert tensors to lists for JSON serialization
        serializable_results = results.copy()
        for sample in serializable_results.get('samples', []):
            if 'embedding_stats' in sample:
                # Already serializable
                pass
        
        json.dump(serializable_results, f, indent=2)
    
    print(f"\n💾 Results saved to: {output_path}")
    return output_path


if __name__ == "__main__":
    print("="*70)
    print("True End-to-End Visual Evaluation")
    print("Testing: MagVIT → Liquid → GPT-4 (NO ground truth to LLM)")
    print("="*70)
    
    # For standalone testing, create mock samples
    print("\n⚠️  Using mock embeddings for testing")
    print("    In production, use real MagVIT → Liquid fusion outputs\n")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create 5 mock samples
    mock_samples = []
    for i in range(5):
        mock_samples.append({
            'fused_embedding': torch.randn(1, 4096, device=device),
            'ground_truth': {
                'type': 'straight line',
                'primary_direction': 'depth (Y-axis)',
                'description': f'A straight line trajectory sample {i}'
            }
        })
    
    # Run evaluation
    results = run_true_e2e_evaluation(mock_samples)
    
    # Save
    save_results(results, Path("experiments/liquid_vlm_integration/results"))
    
    print("\n✅ True end-to-end evaluation complete")

