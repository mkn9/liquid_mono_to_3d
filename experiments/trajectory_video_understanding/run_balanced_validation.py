"""
Run balanced validation across all 4 classes.
Dataset structure:
- Linear: indices 0-2499
- Circular: indices 2500-4999
- Helical: indices 5000-7499
- Parabolic: indices 7500-9999

We'll sample from validation portion of each class.
"""

import torch
import sys
from pathlib import Path

# Reuse the validation inference script but with custom sampling
sys.path.insert(0, str(Path(__file__).parent))
from run_validation_inference import (
    load_model, load_validation_data, run_inference,
    compute_metrics, plot_confusion_matrix, visualize_predictions
)
import json
from datetime import datetime


def main():
    print("="*80)
    print("MagVIT Balanced Validation - All 4 Classes")
    print("="*80)
    
    model_path = Path("/home/ubuntu/mono_to_3d/parallel_training/worker_magvit/results/validation/final_model.pt")
    data_dir = Path("/home/ubuntu/mono_to_3d/data/10k_trajectories")
    output_dir = Path("validation_results_balanced")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}\n")
    
    # Load model once
    model = load_model(model_path, device=device)
    
    # Sample from validation portion of each class (20% from end of each class range)
    # Linear: 2000-2499 (last 500)
    # Circular: 4500-4999 (last 500)
    # Helical: 7000-7499 (last 500)
    # Parabolic: 9500-9999 (last 500)
    
    class_ranges = {
        'Linear': (2000, 125),      # 125 samples from validation portion
        'Circular': (4500, 125),
        'Helical': (7000, 125),
        'Parabolic': (9500, 125)
    }
    
    all_videos = []
    all_labels = []
    all_positions = []
    all_metadata = []
    
    for class_name, (start_idx, num_samples) in class_ranges.items():
        print(f"Loading {num_samples} samples from {class_name} (indices {start_idx}-{start_idx+num_samples-1})...")
        videos, labels, positions, metadata = load_validation_data(
            data_dir, start_idx=start_idx, num_samples=num_samples
        )
        all_videos.append(videos)
        all_labels.append(labels)
        all_positions.append(positions)
        all_metadata.extend(metadata)
    
    # Combine all classes
    all_videos = torch.cat(all_videos, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    all_positions = torch.cat(all_positions, dim=0)
    
    print(f"\n✅ Total samples loaded: {len(all_videos)}")
    print(f"   Class distribution: 125 samples × 4 classes = 500 total\n")
    
    # Run inference
    pred_labels, probs, pred_positions = run_inference(model, all_videos, device=device)
    
    # Compute metrics
    metrics = compute_metrics(all_labels.numpy(), pred_labels,
                             all_positions.numpy(), pred_positions)
    
    # Save metrics
    timestamp = datetime.now().strftime('%Y%m%d_%H%M')
    metrics_path = output_dir / f'{timestamp}_balanced_validation_metrics.json'
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"\n✅ Metrics saved: {metrics_path}")
    
    # Generate confusion matrix
    class_names = ['Linear', 'Circular', 'Helical', 'Parabolic']
    confusion_path = output_dir / f'{timestamp}_balanced_confusion_matrix.png'
    plot_confusion_matrix(np.array(metrics['confusion_matrix']), class_names, confusion_path)
    
    # Visualize predictions
    visualize_predictions(all_videos, all_labels.numpy(), pred_labels, probs,
                         all_positions.numpy(), pred_positions,
                         all_metadata, output_dir, num_examples=8)
    
    # Print summary
    print("\n" + "="*80)
    print("BALANCED VALIDATION RESULTS SUMMARY")
    print("="*80)
    print(f"Overall Accuracy:     {metrics['overall_accuracy']*100:.2f}%")
    print(f"Samples Evaluated:    {metrics['num_samples']}")
    print(f"Correct Predictions:  {metrics['num_correct']}")
    print(f"Incorrect Predictions: {metrics['num_incorrect']}")
    print()
    print("Per-Class Accuracy:")
    for class_name, data in metrics['per_class_accuracy'].items():
        print(f"  {class_name:<12} {data['accuracy']*100:>6.2f}% ({data['correct']}/{data['total']})")
    print()
    print(f"Position Prediction Error:")
    print(f"  RMSE: {metrics['position_rmse']:.4f}")
    print(f"  MAE:  {metrics['position_mae']:.4f}")
    print()
    print(f"Results saved to: {output_dir}/")
    print("="*80)


if __name__ == "__main__":
    import numpy as np
    main()

