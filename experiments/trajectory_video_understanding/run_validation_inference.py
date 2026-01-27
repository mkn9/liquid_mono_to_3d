"""
Validation Inference Script
Load trained MagVIT model and run detailed inference on validation set.
Generates:
1. Per-sample predictions
2. Confusion matrix
3. Per-class accuracy
4. Visualization of example predictions
"""

import torch
import torch.nn as nn
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
import sys

# Add shared modules and feature extractor to path
sys.path.insert(0, str(Path(__file__).parent / 'shared'))
sys.path.insert(0, str(Path(__file__).parent))
from unified_model import UnifiedModel
from feature_extractor import MagVITExtractor


def load_model(model_path, device='cpu'):
    """Load trained MagVIT model."""
    print(f"Loading model from: {model_path}")
    
    # Create model
    extractor = MagVITExtractor(feature_dim=256)
    model = UnifiedModel(extractor, num_classes=4)
    
    # Load weights
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    
    print(f"✅ Model loaded on {device}")
    return model


def load_validation_data(data_dir, start_idx=8000, num_samples=100):
    """Load validation samples (indices 8000-9999)."""
    print(f"\nLoading {num_samples} validation samples (indices {start_idx}-{start_idx+num_samples-1})...")
    
    data_path = Path(data_dir)
    videos = []
    labels = []
    positions = []
    metadata_list = []
    
    class_map = {'Linear': 0, 'Circular': 1, 'Helical': 2, 'Parabolic': 3}
    
    for i in range(start_idx, start_idx + num_samples):
        video_file = data_path / 'videos' / f'traj_{i:05d}.pt'
        label_file = data_path / 'labels' / f'traj_{i:05d}.json'
        
        if not video_file.exists() or not label_file.exists():
            print(f"⚠️  Skipping {i}: files not found")
            continue
        
        # Load video
        video = torch.load(video_file, map_location='cpu')
        videos.append(video)
        
        # Load metadata
        with open(label_file) as f:
            metadata = json.load(f)
        
        labels.append(class_map[metadata['class']])
        positions.append(metadata['positions'][-1])  # Last position
        metadata_list.append(metadata)
    
    videos = torch.stack(videos)
    labels = torch.tensor(labels, dtype=torch.long)
    positions = torch.tensor(positions, dtype=torch.float32)
    
    print(f"✅ Loaded {len(videos)} validation samples")
    return videos, labels, positions, metadata_list


def run_inference(model, videos, device='cpu'):
    """Run model inference on validation data."""
    print("\nRunning inference...")
    
    model.eval()
    all_predictions = []
    all_probs = []
    all_pred_positions = []
    
    batch_size = 32
    num_batches = (len(videos) + batch_size - 1) // batch_size
    
    with torch.no_grad():
        for i in range(num_batches):
            start = i * batch_size
            end = min((i + 1) * batch_size, len(videos))
            batch = videos[start:end].to(device)
            
            output = model(batch)
            
            # Classification
            probs = torch.softmax(output['classification'], dim=1)
            preds = torch.argmax(probs, dim=1)
            
            all_predictions.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_pred_positions.extend(output['prediction'].cpu().numpy())
    
    print(f"✅ Inference complete on {len(videos)} samples")
    
    return np.array(all_predictions), np.array(all_probs), np.array(all_pred_positions)


def compute_metrics(true_labels, pred_labels, true_positions, pred_positions):
    """Compute detailed validation metrics."""
    print("\nComputing metrics...")
    
    # Overall accuracy
    overall_acc = np.mean(true_labels == pred_labels)
    
    # Per-class accuracy
    class_names = ['Linear', 'Circular', 'Helical', 'Parabolic']
    per_class_acc = {}
    
    for i, class_name in enumerate(class_names):
        mask = true_labels == i
        if mask.sum() > 0:
            class_acc = np.mean(pred_labels[mask] == i)
            per_class_acc[class_name] = {
                'accuracy': float(class_acc),
                'correct': int((pred_labels[mask] == i).sum()),
                'total': int(mask.sum())
            }
    
    # Confusion matrix
    confusion = np.zeros((4, 4), dtype=int)
    for true, pred in zip(true_labels, pred_labels):
        confusion[true, pred] += 1
    
    # Position errors
    position_mse = float(np.mean((true_positions - pred_positions) ** 2))
    position_mae = float(np.mean(np.abs(true_positions - pred_positions)))
    position_rmse = float(np.sqrt(position_mse))
    
    metrics = {
        'overall_accuracy': float(overall_acc),
        'per_class_accuracy': per_class_acc,
        'confusion_matrix': confusion.tolist(),
        'position_mse': position_mse,
        'position_mae': position_mae,
        'position_rmse': position_rmse,
        'num_samples': len(true_labels),
        'num_correct': int((true_labels == pred_labels).sum()),
        'num_incorrect': int((true_labels != pred_labels).sum())
    }
    
    print(f"✅ Overall Accuracy: {overall_acc*100:.2f}%")
    print(f"✅ Position RMSE: {position_rmse:.4f}")
    
    return metrics


def plot_confusion_matrix(confusion, class_names, output_path):
    """Generate confusion matrix visualization."""
    plt.figure(figsize=(10, 8))
    
    # Normalize by row (true labels)
    confusion_norm = confusion.astype(float) / confusion.sum(axis=1, keepdims=True)
    
    sns.heatmap(confusion_norm, annot=True, fmt='.2%', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                vmin=0, vmax=1, cbar_kws={'label': 'Accuracy'})
    
    plt.title('MagVIT Confusion Matrix (Validation Set)', fontsize=14, fontweight='bold')
    plt.xlabel('Predicted Class', fontsize=12)
    plt.ylabel('True Class', fontsize=12)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Confusion matrix saved: {output_path}")


def visualize_predictions(videos, true_labels, pred_labels, probs, true_positions, pred_positions, 
                         metadata_list, output_dir, num_examples=6):
    """Visualize example predictions."""
    class_names = ['Linear', 'Circular', 'Helical', 'Parabolic']
    
    # Select examples: 3 correct, 3 incorrect (if any)
    correct_indices = np.where(true_labels == pred_labels)[0]
    incorrect_indices = np.where(true_labels != pred_labels)[0]
    
    num_correct = min(3, len(correct_indices))
    num_incorrect = min(3, len(incorrect_indices))
    
    selected_indices = []
    if num_correct > 0:
        selected_indices.extend(np.random.choice(correct_indices, num_correct, replace=False))
    if num_incorrect > 0:
        selected_indices.extend(np.random.choice(incorrect_indices, num_incorrect, replace=False))
    
    if len(selected_indices) == 0:
        print("⚠️  No examples to visualize")
        return
    
    # Create visualization
    fig, axes = plt.subplots(len(selected_indices), 4, figsize=(16, 4*len(selected_indices)))
    if len(selected_indices) == 1:
        axes = axes.reshape(1, -1)
    
    for row, idx in enumerate(selected_indices):
        video = videos[idx]
        true_class = class_names[true_labels[idx]]
        pred_class = class_names[pred_labels[idx]]
        confidence = probs[idx][pred_labels[idx]]
        is_correct = true_labels[idx] == pred_labels[idx]
        
        # Show 4 frames from the video
        frame_indices = [0, 5, 10, 15]
        for col, frame_idx in enumerate(frame_indices):
            frame = video[frame_idx].permute(1, 2, 0).numpy()
            frame = (frame - frame.min()) / (frame.max() - frame.min() + 1e-8)
            
            axes[row, col].imshow(frame)
            axes[row, col].axis('off')
            
            if col == 0:
                status = "✅ CORRECT" if is_correct else "❌ WRONG"
                color = 'green' if is_correct else 'red'
                axes[row, col].set_title(
                    f"{status}\nTrue: {true_class}\nPred: {pred_class} ({confidence*100:.1f}%)",
                    fontsize=10, color=color, fontweight='bold'
                )
            else:
                axes[row, col].set_title(f"Frame {frame_idx}", fontsize=10)
    
    plt.tight_layout()
    output_path = output_dir / f'20260125_validation_predictions_{len(selected_indices)}_examples.png'
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Prediction visualization saved: {output_path}")


def main():
    """Main validation inference pipeline."""
    print("="*80)
    print("MagVIT Validation Inference - Detailed Analysis")
    print("="*80)
    
    # Paths
    model_path = Path("sequential_results_20260125_2148_FULL/magvit/final_model.pt")
    data_dir = Path("/Users/mike/Dropbox/Documents/Machine_Learning/LLM_and_Multimodal_Models/repos/mono_to_3d/data/10k_trajectories")
    output_dir = Path("validation_results")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    # Load model
    model = load_model(model_path, device=device)
    
    # Load validation data (sample 200 from validation set)
    videos, true_labels, true_positions, metadata_list = load_validation_data(
        data_dir, start_idx=8000, num_samples=200
    )
    
    # Run inference
    pred_labels, probs, pred_positions = run_inference(model, videos, device=device)
    
    # Compute metrics
    metrics = compute_metrics(true_labels.numpy(), pred_labels, 
                             true_positions.numpy(), pred_positions)
    
    # Save metrics
    timestamp = datetime.now().strftime('%Y%m%d_%H%M')
    metrics_path = output_dir / f'{timestamp}_validation_metrics.json'
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"\n✅ Metrics saved: {metrics_path}")
    
    # Generate confusion matrix
    class_names = ['Linear', 'Circular', 'Helical', 'Parabolic']
    confusion_path = output_dir / f'{timestamp}_confusion_matrix.png'
    plot_confusion_matrix(np.array(metrics['confusion_matrix']), class_names, confusion_path)
    
    # Visualize predictions
    visualize_predictions(videos, true_labels.numpy(), pred_labels, probs,
                         true_positions.numpy(), pred_positions,
                         metadata_list, output_dir, num_examples=6)
    
    # Print summary
    print("\n" + "="*80)
    print("VALIDATION RESULTS SUMMARY")
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
    main()

