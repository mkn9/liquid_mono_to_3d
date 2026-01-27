#!/usr/bin/env python3
"""Complete Model Evaluation with Attention Extraction and Efficiency Metrics.

This script:
1. Loads trained model
2. Evaluates on validation data
3. Extracts attention weights
4. Computes efficiency metrics
5. Generates visualizations
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import argparse
import time
from typing import List, Tuple, Dict
from tqdm import tqdm

# Add models to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'models'))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'branch_4_magvit'))

from early_persistence_classifier import EarlyPersistenceClassifier, get_early_decision
from efficiency_metrics import EfficiencyMetrics


class AttentionExtractorWrapper(nn.Module):
    """Wrapper to extract attention weights during forward pass."""
    
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.attention_weights = None
        
    def forward(self, video):
        """Forward with attention extraction."""
        # Get features from MagVIT extractor
        features = self.model.extractor.extract(video)
        
        # Extract attention from the tokenizer (MultiheadAttention)
        if hasattr(self.model.extractor, 'tokenizer'):
            attn_output, attn_weights = self.model.extractor.tokenizer(
                features, features, features, need_weights=True, average_attn_weights=True
            )
            self.attention_weights = attn_weights
        
        # Continue with rest of model
        return self.model(video)


def load_model_with_attention(model_path: str, device: str = 'cpu'):
    """Load model with attention extraction capability."""
    state_dict = torch.load(model_path, map_location=device)
    
    model = EarlyPersistenceClassifier(
        feature_extractor='magvit',
        early_stop_frame=4,
        confidence_threshold=0.9,
        feature_dim=256
    )
    
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    
    # Wrap with attention extractor
    wrapper = AttentionExtractorWrapper(model)
    wrapper.to(device)
    wrapper.eval()
    
    return wrapper


def load_validation_data(data_dir: Path, num_samples: int = None):
    """Load validation dataset."""
    print(f"Loading data from: {data_dir}")
    
    # Find all video files
    video_files = sorted(list(data_dir.glob("augmented_traj_*.pt")))
    
    if num_samples:
        video_files = video_files[:num_samples]
    
    print(f"Found {len(video_files)} samples")
    
    dataset = []
    for video_path in tqdm(video_files, desc="Loading videos"):
        try:
            video = torch.load(video_path)
            if video.dim() == 5:
                video = video.squeeze(0)
            
            # Load metadata
            json_path = video_path.with_suffix('.json')
            with open(json_path, 'r') as f:
                metadata = json.load(f)
            
            # Create label
            transient_frames = metadata.get('transient_frames', [])
            total_frames = video.shape[0]
            transient_ratio = len(transient_frames) / total_frames if total_frames > 0 else 0
            label = 1 if transient_ratio < 0.2 else 0  # 1=persistent, 0=transient
            
            dataset.append({
                'video': video,
                'label': label,
                'metadata': metadata,
                'sample_id': int(video_path.stem.split('_')[-1])
            })
        except Exception as e:
            print(f"Error loading {video_path}: {e}")
            continue
    
    return dataset


def evaluate_with_attention_and_efficiency(model_wrapper, dataset, device='cuda'):
    """Evaluate model and collect attention + efficiency metrics."""
    
    y_true = []
    y_pred = []
    confidences = []
    decision_frames = []
    attention_weights_list = []
    efficiency_data = []
    
    model_wrapper.eval()
    
    for sample in tqdm(dataset, desc="Evaluating"):
        video = sample['video'].unsqueeze(0).to(device)
        true_label = sample['label']
        
        with torch.no_grad():
            # Get prediction with attention
            start_time = time.time()
            outputs = model_wrapper(video)
            inference_time = time.time() - start_time
            
            logits = outputs['logits'][0]
            probs = torch.softmax(logits, dim=0)
            predicted_class = torch.argmax(probs).item()
            confidence = probs[predicted_class].item()
            
            # Get early decision
            decision, conf, frame_idx = get_early_decision(model_wrapper.model, video)
            
            # Store results
            y_true.append(true_label)
            y_pred.append(predicted_class)
            confidences.append(confidence)
            decision_frames.append(frame_idx)
            
            # Store attention weights if available
            if model_wrapper.attention_weights is not None:
                attention_weights_list.append({
                    'sample_id': sample['sample_id'],
                    'attention': model_wrapper.attention_weights.cpu(),
                    'transient_frames': sample['metadata'].get('transient_frames', [])
                })
            
            # Efficiency data
            total_frames = video.shape[1]
            frames_processed = frame_idx
            compute_used = frames_processed / total_frames
            compute_saved = 1.0 - compute_used
            
            efficiency_data.append({
                'sample_id': sample['sample_id'],
                'decision_frame': frame_idx,
                'total_frames': total_frames,
                'compute_used': compute_used,
                'compute_saved': compute_saved,
                'inference_time_ms': inference_time * 1000,
                'confidence': conf,
                'early_stop': frame_idx <= 4
            })
    
    # Compute metrics
    correct = sum([1 for t, p in zip(y_true, y_pred) if t == p])
    accuracy = correct / len(y_true) if y_true else 0.0
    
    early_stops = sum([1 for d in decision_frames if d <= 4])
    early_stop_rate = early_stops / len(decision_frames) if decision_frames else 0.0
    
    avg_decision_frame = np.mean(decision_frames) if decision_frames else 0.0
    avg_compute_saved = np.mean([e['compute_saved'] for e in efficiency_data]) if efficiency_data else 0.0
    avg_inference_time = np.mean([e['inference_time_ms'] for e in efficiency_data]) if efficiency_data else 0.0
    
    return {
        'accuracy': accuracy,
        'early_stop_rate': early_stop_rate,
        'avg_decision_frame': avg_decision_frame,
        'avg_compute_saved': avg_compute_saved,
        'avg_inference_time_ms': avg_inference_time,
        'total_samples': len(dataset),
        'correct_predictions': correct,
        'y_true': y_true,
        'y_pred': y_pred,
        'confidences': confidences,
        'decision_frames': decision_frames,
        'attention_weights': attention_weights_list,
        'efficiency_data': efficiency_data
    }


def compute_confusion_matrix(y_true, y_pred):
    """Compute confusion matrix."""
    cm = np.zeros((2, 2), dtype=int)
    for true, pred in zip(y_true, y_pred):
        cm[true][pred] += 1
    return cm


def save_results(results, output_dir):
    """Save evaluation results."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save metrics (without attention weights and efficiency data for JSON)
    metrics = {
        'accuracy': results['accuracy'],
        'early_stop_rate': results['early_stop_rate'],
        'avg_decision_frame': results['avg_decision_frame'],
        'avg_compute_saved': results['avg_compute_saved'],
        'avg_inference_time_ms': results['avg_inference_time_ms'],
        'total_samples': results['total_samples'],
        'correct_predictions': results['correct_predictions'],
        'confusion_matrix': compute_confusion_matrix(results['y_true'], results['y_pred']).tolist()
    }
    
    with open(output_dir / 'evaluation_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Save efficiency data
    with open(output_dir / 'efficiency_metrics.json', 'w') as f:
        json.dump(results['efficiency_data'], f, indent=2)
    
    # Save attention weights (as torch file for later visualization)
    torch.save(results['attention_weights'], output_dir / 'attention_weights.pt')
    
    print(f"\n✅ Results saved to: {output_dir}")
    print(f"  - evaluation_metrics.json")
    print(f"  - efficiency_metrics.json")
    print(f"  - attention_weights.pt")


def main():
    parser = argparse.ArgumentParser(description='Complete model evaluation')
    parser.add_argument('--model', type=str, required=True, help='Path to trained model')
    parser.add_argument('--data', type=str, required=True, help='Path to dataset directory')
    parser.add_argument('--output', type=str, default='./evaluation_results', help='Output directory')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--num_samples', type=int, default=None, help='Number of samples to evaluate')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("COMPLETE MODEL EVALUATION WITH ATTENTION & EFFICIENCY")
    print("=" * 80)
    print(f"Model: {args.model}")
    print(f"Data: {args.data}")
    print(f"Output: {args.output}")
    print(f"Device: {args.device}")
    print("=" * 80)
    
    # Load model
    print("\n📦 Loading model with attention extraction...")
    model = load_model_with_attention(args.model, device=args.device)
    print("✅ Model loaded")
    
    # Load validation data
    print("\n📂 Loading validation data...")
    dataset = load_validation_data(Path(args.data), num_samples=args.num_samples)
    print(f"✅ Loaded {len(dataset)} samples")
    
    # Evaluate
    print("\n🔍 Evaluating with attention & efficiency tracking...")
    results = evaluate_with_attention_and_efficiency(model, dataset, device=args.device)
    print("✅ Evaluation complete")
    
    # Save results
    print("\n💾 Saving results...")
    save_results(results, args.output)
    
    # Print summary
    print("\n" + "=" * 80)
    print("EVALUATION SUMMARY")
    print("=" * 80)
    print(f"Accuracy: {results['accuracy']:.2%}")
    print(f"Early Stop Rate: {results['early_stop_rate']:.2%}")
    print(f"Avg Decision Frame: {results['avg_decision_frame']:.2f}")
    print(f"Avg Compute Saved: {results['avg_compute_saved']:.2%}")
    print(f"Avg Inference Time: {results['avg_inference_time_ms']:.2f}ms")
    print(f"Samples with attention: {len(results['attention_weights'])}")
    print("=" * 80)


if __name__ == '__main__':
    main()

