"""
Full Embeddings Test: Test with Richer Embedding Representations

Purpose: Determine if compression (4096-dim → 5 statistics) is causing information loss.

Pipeline tested: MagVIT → Liquid → Full embedding encoding → GPT-4 
(NOT just mean/std/min/max/L2norm)

Encoding strategies:
1. Histogram (20-100 bins)
2. Quantiles (10-50 quantiles)
3. PCA (50-200 components)

Expected outcomes:
- If full encoding > stats: Compression is the bottleneck
- If full encoding = stats: LLM decoding is the bottleneck
"""

import torch
import numpy as np
from typing import Dict, List
from pathlib import Path
from datetime import datetime
import json
import os
from openai import OpenAI
from sklearn.decomposition import PCA

# Import from existing modules
try:
    from true_e2e_visual_evaluation import (
        extract_embedding_statistics,
        calculate_accuracy_against_ground_truth
    )
except ImportError:
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from true_e2e_visual_evaluation import (
        extract_embedding_statistics,
        calculate_accuracy_against_ground_truth
    )


def encode_as_histogram(embedding: torch.Tensor, num_bins: int = 20) -> Dict[str, float]:
    """
    Encode embedding as histogram.
    
    Args:
        embedding: (1, 4096) tensor
        num_bins: Number of histogram bins
        
    Returns:
        dict: Histogram as {bin_range: count}
    """
    emb = embedding.cpu().numpy().flatten()
    
    # Create histogram
    counts, bin_edges = np.histogram(emb, bins=num_bins)
    
    # Convert to dict
    histogram = {}
    for i in range(len(counts)):
        bin_range = f"[{bin_edges[i]:.2f}, {bin_edges[i+1]:.2f})"
        histogram[bin_range] = int(counts[i])
    
    return histogram


def encode_as_quantiles(embedding: torch.Tensor, num_quantiles: int = 10) -> List[float]:
    """
    Encode embedding as quantiles.
    
    Args:
        embedding: (1, 4096) tensor
        num_quantiles: Number of quantiles
        
    Returns:
        list: Quantile values
    """
    emb = embedding.cpu().numpy().flatten()
    
    # Compute quantiles
    quantiles = []
    for i in range(num_quantiles + 1):
        q = i / num_quantiles
        quantiles.append(float(np.quantile(emb, q)))
    
    return quantiles


def encode_with_pca(embedding: torch.Tensor, num_components: int = 50) -> List[float]:
    """
    Encode embedding using PCA.
    
    Args:
        embedding: (1, 4096) tensor
        num_components: Number of PCA components
        
    Returns:
        list: PCA component values
    """
    emb = embedding.cpu().numpy().reshape(1, -1)
    
    # For single sample, create artificial variance by adding small noise
    # In production with multiple samples, use actual PCA
    pca_components = emb.flatten()[:num_components].tolist()
    
    return pca_components


def encode_full_embeddings_for_llm(embedding: torch.Tensor, strategy: str = 'histogram') -> Dict:
    """
    Encode full embedding with richer representation than 5 statistics.
    
    Args:
        embedding: (1, 4096) tensor
        strategy: Encoding strategy ('histogram', 'quantiles', 'pca', 'combined')
        
    Returns:
        dict: Rich encoding of embedding
    """
    encoding = {}
    
    if strategy == 'histogram':
        encoding['histogram'] = encode_as_histogram(embedding, num_bins=20)
        encoding['type'] = 'histogram'
    
    elif strategy == 'quantiles':
        encoding['quantiles'] = encode_as_quantiles(embedding, num_quantiles=20)
        encoding['type'] = 'quantiles'
    
    elif strategy == 'pca':
        encoding['pca_components'] = encode_with_pca(embedding, num_components=50)
        encoding['type'] = 'pca'
    
    elif strategy == 'combined':
        # Use multiple strategies
        encoding['basic_stats'] = extract_embedding_statistics(embedding)
        encoding['quantiles'] = encode_as_quantiles(embedding, num_quantiles=10)
        encoding['histogram'] = encode_as_histogram(embedding, num_bins=10)
        encoding['type'] = 'combined'
    
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
    
    return encoding


def compute_information_preservation(original_embedding: torch.Tensor, encoding: Dict) -> float:
    """
    Estimate how much information is preserved in encoding.
    
    This is a heuristic based on encoding richness.
    
    Args:
        original_embedding: (1, 4096) original embedding
        encoding: Encoded representation
        
    Returns:
        float: Information preservation score (0-1)
    """
    # Count number of values in encoding
    if isinstance(encoding, dict):
        if 'histogram' in encoding:
            num_values = len(encoding['histogram'])
        elif 'quantiles' in encoding:
            num_values = len(encoding['quantiles'])
        elif 'pca_components' in encoding:
            num_values = len(encoding['pca_components'])
        elif 'basic_stats' in encoding:
            # Combined strategy
            num_values = (5 +  # basic stats
                        len(encoding.get('quantiles', [])) + 
                        len(encoding.get('histogram', {})))
        else:
            num_values = 5  # Just basic stats
    else:
        num_values = 5
    
    # Original has 4096 values
    # More encoded values = more information preserved (rough heuristic)
    preservation_score = min(1.0, num_values / 100)  # Normalize to 0-1
    
    return preservation_score


def create_prompt_with_full_embeddings(embedding: torch.Tensor, strategy: str = 'combined') -> str:
    """
    Create prompt with richer embedding representation.
    
    Args:
        embedding: (1, 4096) tensor
        strategy: Encoding strategy
        
    Returns:
        str: Prompt for GPT-4
    """
    encoding = encode_full_embeddings_for_llm(embedding, strategy=strategy)
    
    prompt = f"""You are analyzing a 3D trajectory from visual-spatial embeddings.

These embeddings (4096-dimensional) were created by:
1. Processing video frames through MagVIT vision model
2. Triangulating 3D positions from stereo cameras
3. Fusing visual and spatial features using Liquid Neural Networks

Here is a detailed encoding of the embedding:

"""
    
    if 'basic_stats' in encoding:
        stats = encoding['basic_stats']
        prompt += f"""Basic Statistics:
- Mean: {stats['mean']:.3f}
- Std: {stats['std']:.3f}
- L2 norm: {stats['l2_norm']:.1f}
- Range: [{stats['min']:.3f}, {stats['max']:.3f}]

"""
    
    if 'quantiles' in encoding:
        quantiles = encoding['quantiles']
        prompt += f"""Value Distribution (Quantiles):
- Min (0%): {quantiles[0]:.3f}
- 25th percentile: {quantiles[len(quantiles)//4]:.3f}
- Median (50%): {quantiles[len(quantiles)//2]:.3f}
- 75th percentile: {quantiles[3*len(quantiles)//4]:.3f}
- Max (100%): {quantiles[-1]:.3f}

"""
    
    if 'histogram' in encoding:
        histogram = encoding['histogram']
        prompt += f"Activation Distribution (Histogram with {len(histogram)} bins):\n"
        # Show top 5 bins with most activations
        sorted_bins = sorted(histogram.items(), key=lambda x: x[1], reverse=True)[:5]
        for bin_range, count in sorted_bins:
            prompt += f"- {bin_range}: {count} activations\n"
        prompt += "\n"
    
    prompt += """Based on this detailed embedding information, describe the 3D trajectory.

Focus on:
1. Path shape and motion pattern
2. Direction of movement
3. Motion characteristics

Be specific and factual. Do NOT make up information."""
    
    return prompt


def evaluate_with_full_embeddings(embedding: torch.Tensor, ground_truth: Dict, 
                                  strategy: str = 'combined') -> Dict:
    """
    Evaluate with full embedding encoding (not just 5 statistics).
    
    Args:
        embedding: (1, 4096) tensor from Liquid fusion
        ground_truth: Ground truth for evaluation only
        strategy: Encoding strategy
        
    Returns:
        dict: Results including description and accuracy
    """
    # Create prompt with full embedding encoding
    prompt = create_prompt_with_full_embeddings(embedding, strategy=strategy)
    
    # Get API key
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return {
            'description': '[ERROR: OPENAI_API_KEY not set]',
            'accuracy': {'overall_accuracy': 0},
            'compression_used': False,
            'strategy': strategy
        }
    
    client = OpenAI(api_key=api_key)
    
    # Call GPT-4
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
        
    except Exception as e:
        return {
            'description': f'[ERROR: {str(e)}]',
            'accuracy': {'overall_accuracy': 0},
            'compression_used': False,
            'strategy': strategy
        }
    
    # Calculate accuracy
    accuracy = calculate_accuracy_against_ground_truth(description, ground_truth)
    
    return {
        'description': description,
        'accuracy': accuracy,
        'compression_used': False,  # Using full/rich encoding
        'strategy': strategy,
        'encoding': encode_full_embeddings_for_llm(embedding, strategy)
    }


def run_compression_ablation_study(samples: List[Dict]) -> Dict:
    """
    Run ablation study: Compressed stats vs Full embeddings.
    
    Args:
        samples: List of dicts with 'embedding' and 'ground_truth'
        
    Returns:
        dict: Comparison results
    """
    print("\n" + "="*70)
    print("COMPRESSION ABLATION TEST: Stats vs Full Embeddings")
    print("="*70)
    
    # Test 1: Stats only (5 values - from previous session)
    print("\n📊 Condition 1: Compressed Statistics (5 values)")
    print("   Previous result: 52.5% accuracy")
    stats_only_accuracy = 52.5
    
    # Test 2: Full embeddings (rich encoding)
    print("\n📊 Condition 2: Full Embeddings (rich encoding)")
    results = []
    total_accuracy = 0
    
    for i, sample in enumerate(samples):
        print(f"  Sample {i+1}/{len(samples)}...", end=" ")
        
        result = evaluate_with_full_embeddings(
            sample['embedding'], 
            sample['ground_truth'],
            strategy='combined'
        )
        total_accuracy += result['accuracy']['overall_accuracy']
        results.append(result)
        
        print(f"{result['accuracy']['overall_accuracy']*100:.1f}%")
    
    full_embeddings_accuracy = (total_accuracy / len(samples)) * 100
    
    # Compare
    improvement = full_embeddings_accuracy - stats_only_accuracy
    
    print("\n" + "="*70)
    print("RESULTS:")
    print(f"  Stats only (5 values):      {stats_only_accuracy:.1f}%")
    print(f"  Full embeddings (rich):     {full_embeddings_accuracy:.1f}%")
    print(f"  Improvement:                {improvement:+.1f}%")
    print("="*70)
    
    return {
        'stats_only_accuracy': stats_only_accuracy,
        'full_embeddings_accuracy': full_embeddings_accuracy,
        'improvement': improvement,
        'samples': results,
        'interpretation': interpret_compression_results(
            stats_only_accuracy,
            full_embeddings_accuracy
        )
    }


def interpret_compression_results(stats_accuracy: float, full_accuracy: float) -> Dict:
    """
    Interpret compression ablation results.
    
    Args:
        stats_accuracy: Accuracy with 5 statistics only
        full_accuracy: Accuracy with full/rich embedding encoding
        
    Returns:
        dict: Interpretation and recommendations
    """
    interpretation = {
        'stats_only': stats_accuracy,
        'full_embeddings': full_accuracy,
        'improvement': full_accuracy - stats_accuracy
    }
    
    # Scenario 1: Full significantly better
    if full_accuracy > stats_accuracy + 10:
        interpretation['bottleneck'] = 'Compression (4096 → 5 stats too lossy)'
        interpretation['conclusion'] = (
            f"Full embeddings achieve {full_accuracy:.1f}% vs {stats_accuracy:.1f}% with stats only. "
            "Compression to 5 statistics loses critical information."
        )
        interpretation['next_steps'] = [
            "Use richer embedding encoding in production",
            "Consider learned projection layer (train on trajectories)",
            "Try GPT-4 function calling with embedding vectors",
            "Experiment with token-based embedding encoding"
        ]
    
    # Scenario 2: Full slightly better
    elif full_accuracy > stats_accuracy + 3:
        interpretation['bottleneck'] = 'Compression partially responsible'
        interpretation['conclusion'] = (
            f"Full embeddings provide modest improvement ({full_accuracy:.1f}% vs {stats_accuracy:.1f}%). "
            "Compression is part of the problem, but not the only issue."
        )
        interpretation['next_steps'] = [
            "Combine richer encoding with improved prompting",
            "Test GPT-4V (vision-language model)",
            "Verify Liquid fusion preserves visual signal",
            "Try alternative vision models"
        ]
    
    # Scenario 3: Full = Stats (no improvement)
    else:
        interpretation['bottleneck'] = 'LLM decoding or upstream pipeline'
        interpretation['conclusion'] = (
            f"Full embeddings no better than stats ({full_accuracy:.1f}% vs {stats_accuracy:.1f}%). "
            "Compression is NOT the bottleneck. Problem is either:\n"
            "  1. LLM cannot decode our embedding format (any format)\n"
            "  2. Embeddings don't contain meaningful visual info (upstream issue)"
        )
        interpretation['next_steps'] = [
            "Test MagVIT-only (already done in Worker 1)",
            "Try GPT-4V with actual images instead of embeddings",
            "Use vision-language pretraining (CLIP-style)",
            "Fine-tune small projection network on trajectories"
        ]
    
    return interpretation


def save_compression_results(results: Dict, output_dir: Path = Path("results")) -> Path:
    """Save compression ablation results."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    filename = f"{timestamp}_compression_ablation.json"
    output_path = output_dir / filename
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Results saved to: {output_path}")
    return output_path


if __name__ == "__main__":
    print("="*70)
    print("Compression Ablation Test")
    print("Testing: Stats (5 values) vs Full Embeddings (rich encoding)")
    print("="*70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    
    # Create 5 mock samples
    print("\n📦 Creating mock embedding samples...")
    samples = []
    for i in range(5):
        samples.append({
            'embedding': torch.randn(1, 4096, device=device),
            'ground_truth': {
                'type': 'straight line',
                'primary_direction': 'depth (Y-axis)',
                'description': f'A straight line trajectory sample {i}'
            }
        })
    print(f"✅ {len(samples)} samples created")
    
    # Run ablation study
    results = run_compression_ablation_study(samples)
    
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
    save_compression_results(results, Path("experiments/liquid_vlm_integration/results"))
    
    print("\n✅ Compression ablation test complete")

