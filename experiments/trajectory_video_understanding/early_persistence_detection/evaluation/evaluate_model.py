"""
Model Evaluation Script

Evaluates trained model on test dataset and generates performance metrics.
"""

import torch
import torch.nn as nn
from pathlib import Path
import json
import numpy as np
import sys
import argparse
import time
from typing import List, Tuple, Dict

# Add models to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'models'))
from early_persistence_classifier import EarlyPersistenceClassifier, get_early_decision
from efficiency_metrics import EfficiencyMetrics


def load_model_for_evaluation(model_path: str, device: str = 'cpu'):
    """
    Load trained model for evaluation.
    
    Args:
        model_path: Path to saved model checkpoint
        device: Device to load model on ('cpu' or 'cuda')
    
    Returns:
        Loaded model in eval mode
    """
    # Load state dict directly (not a checkpoint with metadata)
    state_dict = torch.load(model_path, map_location=device)
    
    # Reconstruct model with default parameters used in training
    model = EarlyPersistenceClassifier(
        feature_extractor='magvit',
        early_stop_frame=4,
        confidence_threshold=0.9,
        feature_dim=256
    )
    
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    
    return model


def evaluate_on_test_set(model, test_samples: List[Tuple[torch.Tensor, int]], 
                         device: str = 'cpu') -> Dict:
    """
    Evaluate model on test dataset.
    
    Args:
        model: Trained model
        test_samples: List of (video, label) tuples
        device: Device for inference
    
    Returns:
        Dictionary of evaluation metrics
    """
    model.eval()
    
    correct = 0
    total = 0
    early_stops = 0
    decision_frames = []
    compute_savings_list = []
    
    y_true = []
    y_pred = []
    
    with torch.no_grad():
        for video, label in test_samples:
            video = video.unsqueeze(0).to(device)  # Add batch dim
            
            # Track efficiency
            metrics_tracker = EfficiencyMetrics(num_frames=video.shape[1])
            
            # Get prediction
            decision, confidence, decision_frame = get_early_decision(
                model, video, metrics_tracker=metrics_tracker
            )
            
            predicted_label = 1 if decision == "persistent" else 0
            
            y_true.append(label)
            y_pred.append(predicted_label)
            
            if predicted_label == label:
                correct += 1
            total += 1
            
            decision_frames.append(decision_frame)
            
            if decision_frame < video.shape[1] - 1:
                early_stops += 1
            
            compute_savings = metrics_tracker.compute_compute_savings(
                metrics_tracker.processed_frames
            )
            compute_savings_list.append(compute_savings)
    
    accuracy = correct / total if total > 0 else 0.0
    early_stop_rate = early_stops / total if total > 0 else 0.0
    avg_decision_frame = np.mean(decision_frames) if decision_frames else 0.0
    avg_compute_savings = np.mean(compute_savings_list) if compute_savings_list else 0.0
    
    return {
        'accuracy': accuracy,
        'early_stop_rate': early_stop_rate,
        'avg_decision_frame': avg_decision_frame,
        'avg_compute_savings': avg_compute_savings,
        'total_samples': total,
        'correct_predictions': correct,
        'y_true': y_true,
        'y_pred': y_pred
    }


def compute_confusion_matrix(y_true: List[int], y_pred: List[int]) -> np.ndarray:
    """
    Compute confusion matrix.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
    
    Returns:
        2x2 confusion matrix as numpy array
    """
    cm = np.zeros((2, 2), dtype=int)
    
    for true, pred in zip(y_true, y_pred):
        cm[true][pred] += 1
    
    return cm


def save_evaluation_report(metrics: Dict, output_file: Path):
    """
    Save evaluation report to file.
    
    Args:
        metrics: Dictionary of evaluation metrics
        output_file: Path to output JSON file
    """
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert numpy types to Python types for JSON serialization
    serializable_metrics = {}
    for key, value in metrics.items():
        if isinstance(value, (np.integer, np.floating)):
            serializable_metrics[key] = float(value)
        elif isinstance(value, (list, np.ndarray)):
            if isinstance(value, np.ndarray):
                serializable_metrics[key] = value.tolist()
            else:
                serializable_metrics[key] = value
        else:
            serializable_metrics[key] = value
    
    with open(output_file, 'w') as f:
        json.dump(serializable_metrics, f, indent=2)


def main():
    """Main evaluation script."""
    parser = argparse.ArgumentParser(description='Evaluate trained model')
    parser.add_argument('--model', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--test_data', type=str, required=True,
                       help='Path to test dataset directory')
    parser.add_argument('--output', type=str, default='./evaluation_results',
                       help='Output directory for results')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device for inference')
    parser.add_argument('--num_samples', type=int, default=None,
                       help='Number of test samples to evaluate (None for all)')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("MODEL EVALUATION")
    print("=" * 80)
    print(f"Model: {args.model}")
    print(f"Test data: {args.test_data}")
    print(f"Output: {args.output}")
    print(f"Device: {args.device}")
    print("=" * 80)
    
    # Load model
    print("\n📦 Loading model...")
    model = load_model_for_evaluation(args.model, device=args.device)
    print("✅ Model loaded successfully")
    
    # Load test data
    print("\n📂 Loading test data...")
    # TODO: Implement actual dataset loading
    # For now, create mock test samples
    test_samples = []
    print(f"✅ Loaded {len(test_samples)} test samples")
    
    # Evaluate
    print("\n🔍 Evaluating...")
    start_time = time.time()
    results = evaluate_on_test_set(model, test_samples, device=args.device)
    elapsed = time.time() - start_time
    
    print(f"✅ Evaluation complete in {elapsed:.2f}s")
    
    # Compute confusion matrix
    cm = compute_confusion_matrix(results['y_true'], results['y_pred'])
    results['confusion_matrix'] = cm
    
    # Save results
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    report_file = output_dir / "evaluation_metrics.json"
    save_evaluation_report(results, report_file)
    
    print("\n" + "=" * 80)
    print("EVALUATION RESULTS")
    print("=" * 80)
    print(f"Accuracy: {results['accuracy']:.2%}")
    print(f"Early Stop Rate: {results['early_stop_rate']:.2%}")
    print(f"Avg Decision Frame: {results['avg_decision_frame']:.2f}")
    print(f"Avg Compute Savings: {results['avg_compute_savings']:.2%}")
    print(f"\nConfusion Matrix:")
    print(cm)
    print("=" * 80)
    print(f"\n✅ Results saved to: {report_file}")


if __name__ == '__main__':
    main()

