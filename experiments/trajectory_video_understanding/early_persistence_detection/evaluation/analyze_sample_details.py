#!/usr/bin/env python3
"""Analyze detailed information for visualized samples."""

import json
import torch
from pathlib import Path
import sys

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent / 'models'))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'branch_4_magvit'))

from early_persistence_classifier import EarlyPersistenceClassifier


def load_model(model_path, device='cpu'):
    """Load trained model."""
    model = EarlyPersistenceClassifier(
        feature_extractor='magvit',
        early_stop_frame=4,
        confidence_threshold=0.9
    )
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model


def analyze_sample(sample_id, data_dir, model, device):
    """Analyze a single sample."""
    # Find the video file
    video_path = data_dir / f"augmented_traj_{sample_id:05d}.pt"
    json_path = data_dir / f"augmented_traj_{sample_id:05d}.json"
    
    if not video_path.exists() or not json_path.exists():
        return None
    
    # Load metadata
    with open(json_path, 'r') as f:
        metadata = json.load(f)
    
    # Load video and predict
    video = torch.load(video_path)
    if video.dim() == 5:
        video = video.squeeze(0)
    
    video_batch = video.unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(video_batch)
        logits = outputs['logits'][0]
        probs = torch.softmax(logits, dim=0)
        predicted_class = torch.argmax(probs).item()
        confidence = probs[predicted_class].item()
    
    # Calculate true label
    transient_frames = metadata.get('transient_frames', [])
    total_frames = video.shape[0]
    transient_ratio = len(transient_frames) / total_frames if total_frames > 0 else 0
    true_label = 1 if transient_ratio < 0.2 else 0  # 1=persistent, 0=transient
    
    return {
        'sample_id': sample_id,
        'trajectory_class': metadata.get('class', 'unknown'),
        'total_frames': total_frames,
        'transient_frames': len(transient_frames),
        'transient_ratio': transient_ratio,
        'true_label': 'Persistent' if true_label == 1 else 'Transient',
        'predicted_label': 'Persistent' if predicted_class == 1 else 'Transient',
        'confidence': confidence,
        'correct': true_label == predicted_class
    }


def main():
    # Sample IDs from the visualizations
    correct_samples = [6252, 4684, 1731, 4742, 4521, 6340, 576, 5202, 6363, 439]
    incorrect_samples = [3999, 4640, 6590, 4819, 9655, 1315, 8575, 4903, 5527, 6743]
    
    data_dir = Path('/Users/mike/Dropbox/Documents/Machine_Learning/LLM_and_Multimodal_Models/repos/mono_to_3d/experiments/trajectory_video_understanding/persistence_augmented_dataset/output_samples')
    model_path = Path('/Users/mike/Dropbox/Documents/Machine_Learning/LLM_and_Multimodal_Models/repos/mono_to_3d/experiments/trajectory_video_understanding/early_persistence_detection/results/best_model.pt')
    
    print("Loading model...")
    model = load_model(model_path, device='cpu')
    print("Model loaded\n")
    
    print("=" * 100)
    print("CORRECT PREDICTIONS (10 samples)")
    print("=" * 100)
    print(f"{'ID':<6} {'Trajectory':<12} {'True':<12} {'Predicted':<12} {'Confidence':<12} {'Transient %':<15}")
    print("-" * 100)
    
    for sample_id in correct_samples:
        info = analyze_sample(sample_id, data_dir, model, 'cpu')
        if info:
            print(f"{info['sample_id']:<6} "
                  f"{info['trajectory_class']:<12} "
                  f"{info['true_label']:<12} "
                  f"{info['predicted_label']:<12} "
                  f"{info['confidence']*100:>6.1f}%      "
                  f"{info['transient_ratio']*100:>6.1f}%")
    
    print("\n" + "=" * 100)
    print("INCORRECT PREDICTIONS (10 samples)")
    print("=" * 100)
    print(f"{'ID':<6} {'Trajectory':<12} {'True':<12} {'Predicted':<12} {'Confidence':<12} {'Transient %':<15}")
    print("-" * 100)
    
    for sample_id in incorrect_samples:
        info = analyze_sample(sample_id, data_dir, model, 'cpu')
        if info:
            print(f"{info['sample_id']:<6} "
                  f"{info['trajectory_class']:<12} "
                  f"{info['true_label']:<12} "
                  f"{info['predicted_label']:<12} "
                  f"{info['confidence']*100:>6.1f}%      "
                  f"{info['transient_ratio']*100:>6.1f}%")
    
    print("\n" + "=" * 100)
    
    # Save detailed report
    output_file = Path('/Users/mike/Dropbox/Documents/Machine_Learning/LLM_and_Multimodal_Models/repos/mono_to_3d/experiments/trajectory_video_understanding/early_persistence_detection/results/prediction_visualizations/DETAILED_ANALYSIS.txt')
    
    with open(output_file, 'w') as f:
        f.write("DETAILED ANALYSIS OF VISUALIZED SAMPLES\n")
        f.write("=" * 100 + "\n\n")
        
        f.write("CORRECT PREDICTIONS (10 samples)\n")
        f.write("=" * 100 + "\n")
        f.write(f"{'ID':<6} {'Trajectory':<12} {'True':<12} {'Predicted':<12} {'Confidence':<12} {'Transient %':<15}\n")
        f.write("-" * 100 + "\n")
        
        for sample_id in correct_samples:
            info = analyze_sample(sample_id, data_dir, model, 'cpu')
            if info:
                f.write(f"{info['sample_id']:<6} "
                       f"{info['trajectory_class']:<12} "
                       f"{info['true_label']:<12} "
                       f"{info['predicted_label']:<12} "
                       f"{info['confidence']*100:>6.1f}%      "
                       f"{info['transient_ratio']*100:>6.1f}%\n")
        
        f.write("\n" + "=" * 100 + "\n")
        f.write("INCORRECT PREDICTIONS (10 samples)\n")
        f.write("=" * 100 + "\n")
        f.write(f"{'ID':<6} {'Trajectory':<12} {'True':<12} {'Predicted':<12} {'Confidence':<12} {'Transient %':<15}\n")
        f.write("-" * 100 + "\n")
        
        for sample_id in incorrect_samples:
            info = analyze_sample(sample_id, data_dir, model, 'cpu')
            if info:
                f.write(f"{info['sample_id']:<6} "
                       f"{info['trajectory_class']:<12} "
                       f"{info['true_label']:<12} "
                       f"{info['predicted_label']:<12} "
                       f"{info['confidence']*100:>6.1f}%      "
                       f"{info['transient_ratio']*100:>6.1f}%\n")
    
    print(f"\nDetailed analysis saved to: {output_file}")


if __name__ == "__main__":
    main()
