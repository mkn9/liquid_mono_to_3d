#!/usr/bin/env python3
"""
GPT-4 Baseline Evaluation
Runs GPT-4 on existing trajectory samples and compares with TinyLlama

Following TDD and output naming conventions (YYYYMMDD_HHMM_description.ext)
"""

import json
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, List
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from enhanced_metrics import evaluate_all_metrics
from openai import OpenAI


def get_output_filename(base_name: str, extension: str = "json") -> str:
    """
    Generate timestamped filename for results.
    
    Args:
        base_name: Base name for file
        extension: File extension (default: json)
        
    Returns:
        str: Filename with format YYYYMMDD_HHMM_basename.ext
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    return f"{timestamp}_{base_name}.{extension}"


def load_existing_results() -> Dict:
    """
    Load existing VLM evaluation results.
    
    Returns:
        dict: Existing results with samples and ground truth
    """
    results_dir = Path("results")
    
    # Find most recent evaluation file
    eval_files = sorted(results_dir.glob("*_vlm_evaluation.json"), reverse=True)
    
    if not eval_files:
        raise FileNotFoundError("No existing evaluation results found")
    
    latest_file = eval_files[0]
    print(f"📂 Loading existing results from: {latest_file}")
    
    with open(latest_file, 'r') as f:
        results = json.load(f)
    
    return results


def calculate_enhanced_metrics(reference: str, candidate: str) -> Dict[str, float]:
    """
    Calculate enhanced metrics (BLEU, ROUGE-L, Semantic Similarity).
    
    Args:
        reference: Ground truth description
        candidate: Generated description
        
    Returns:
        dict: Dictionary with enhanced metrics
    """
    return evaluate_all_metrics(reference, candidate)


def generate_gpt4_description(ground_truth: Dict, client: OpenAI) -> str:
    """
    Generate description using GPT-4.
    
    Args:
        ground_truth: Ground truth data with trajectory information
        client: OpenAI client instance
        
    Returns:
        str: Generated description
    """
    # Create structured prompt based on ground truth data
    # Use the improved prompting approach from Worker 2
    prompt = f"""You are analyzing a 3D trajectory from stereo camera tracking.

Trajectory characteristics:
- Type: {ground_truth['type']}
- Start position: {ground_truth['start']}
- End position: {ground_truth['end']}
- Primary direction: {ground_truth['primary_direction']}
- Average speed: {ground_truth['avg_speed']:.3f} units/frame
- Path length: {ground_truth['length']:.3f} units

Describe ONLY what you observe about:
1. Path shape and motion pattern
2. Direction of movement
3. Start and end positions with approximate coordinates
4. Motion characteristics (speed, smoothness)

Be factual and specific. Use only the data provided above.
Do NOT mention videos, URLs, or make up information."""
    
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are an expert in analyzing 3D trajectories from multi-view camera tracking."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=200,
            temperature=0.7
        )
        
        description = response.choices[0].message.content.strip()
        return description
    except Exception as e:
        print(f"⚠️  GPT-4 generation failed: {e}")
        return f"[ERROR: {str(e)}]"


def calculate_accuracy_metrics(ground_truth: Dict, description: str) -> Dict[str, float]:
    """
    Calculate basic accuracy metrics.
    
    Args:
        ground_truth: Ground truth data
        description: Generated description
        
    Returns:
        dict: Accuracy metrics
    """
    desc_lower = description.lower()
    
    # Check type mentioned
    type_mentioned = 1 if ground_truth['type'].lower() in desc_lower else 0
    
    # Check direction mentioned
    direction_mentioned = 1 if ground_truth['primary_direction'].lower() in desc_lower else 0
    
    # Check has coordinates (numbers present)
    has_coordinates = 1 if any(char.isdigit() for char in description) else 0
    
    # Check speed mentioned
    speed_terms = ['speed', 'velocity', 'fast', 'slow', 'moving']
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


def run_gpt4_evaluation() -> Dict:
    """
    Run GPT-4 baseline evaluation on existing samples.
    
    Returns:
        dict: Updated results with GPT-4 descriptions and metrics
    """
    print("="*70)
    print("GPT-4 Baseline Evaluation")
    print("="*70)
    print()
    
    # Check API key
    api_key = os.environ.get('OPENAI_API_KEY')
    if not api_key:
        print("❌ OPENAI_API_KEY not set in environment")
        print("   Please set it in ~/.zshrc or ~/.bashrc")
        return None
    
    print(f"✅ API key found: {api_key[:20]}... (length: {len(api_key)})")
    print()
    
    # Load existing results
    results = load_existing_results()
    print(f"✅ Loaded {len(results['samples'])} samples")
    print()
    
    # Initialize OpenAI client
    print("🤖 Initializing GPT-4 client...")
    client = OpenAI(api_key=api_key)
    print("✅ GPT-4 client ready")
    print()
    
    # Process each sample
    print("🔄 Generating GPT-4 descriptions...")
    for i, sample in enumerate(results['samples']):
        print(f"  Sample {i+1}/{len(results['samples'])}...", end=" ", flush=True)
        
        # Generate GPT-4 description
        gpt4_description = generate_gpt4_description(sample['ground_truth'], client)
        
        # Calculate basic metrics
        gpt4_metrics = calculate_accuracy_metrics(sample['ground_truth'], gpt4_description)
        
        # Calculate enhanced metrics
        enhanced_metrics = calculate_enhanced_metrics(
            sample['ground_truth']['description'],
            gpt4_description
        )
        
        # Update sample
        sample['gpt4_description'] = gpt4_description
        sample['gpt4_metrics'] = gpt4_metrics
        sample['gpt4_enhanced_metrics'] = enhanced_metrics
        
        # Also calculate enhanced metrics for TinyLlama if not present
        if 'tinyllama_enhanced_metrics' not in sample:
            sample['tinyllama_enhanced_metrics'] = calculate_enhanced_metrics(
                sample['ground_truth']['description'],
                sample['tinyllama_description']
            )
        
        print(f"✅ (accuracy: {gpt4_metrics['overall_accuracy']:.1%})")
    
    print()
    
    # Calculate aggregate statistics
    tinyllama_avg = sum(s['tinyllama_metrics']['overall_accuracy'] for s in results['samples']) / len(results['samples'])
    gpt4_avg = sum(s['gpt4_metrics']['overall_accuracy'] for s in results['samples']) / len(results['samples'])
    
    print("="*70)
    print("📊 Results Summary")
    print("="*70)
    print(f"TinyLlama Average Accuracy: {tinyllama_avg:.1%}")
    print(f"GPT-4 Average Accuracy:     {gpt4_avg:.1%}")
    print(f"Improvement:                {(gpt4_avg - tinyllama_avg):.1%}")
    print("="*70)
    print()
    
    # Update results metadata
    results['gpt4_evaluation_timestamp'] = datetime.now().strftime("%Y%m%d_%H%M")
    results['tinyllama_avg_accuracy'] = tinyllama_avg
    results['gpt4_avg_accuracy'] = gpt4_avg
    
    # Save updated results
    output_file = Path("results") / get_output_filename("gpt4_evaluation")
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"💾 Results saved to: {output_file}")
    print()
    
    return results


def main():
    """Main execution."""
    try:
        results = run_gpt4_evaluation()
        if results:
            print("✅ GPT-4 evaluation complete!")
            return 0
        else:
            print("❌ GPT-4 evaluation failed")
            return 1
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

