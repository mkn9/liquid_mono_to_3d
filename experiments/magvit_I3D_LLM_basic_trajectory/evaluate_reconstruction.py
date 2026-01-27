#!/usr/bin/env python3
"""
Evaluate MAGVIT Reconstruction Quality

Loads trained model and visualizes reconstruction quality on test samples.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from train_magvit import create_model

print("="*70)
print("MAGVIT RECONSTRUCTION QUALITY EVALUATION")
print("="*70)
print()

# Setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# Load dataset
print("Loading dataset...")
dataset_path = Path("results/20260125_0304_dataset_200_validated.npz")
data = np.load(dataset_path)

videos = torch.from_numpy(data['videos']).float()  # (N, T, C, H, W)
labels = torch.from_numpy(data['labels']).long()
trajectory_3d = data['trajectory_3d']

# Convert to MAGVIT format: (N, C, T, H, W)
videos_magvit = videos.permute(0, 2, 1, 3, 4)

print(f"✅ Dataset loaded: {len(videos)} samples")
print(f"   Video shape: {videos.shape}")
print()

# Load best model
print("Loading best model checkpoint...")
checkpoint_path = Path("results/magvit_training/20260125_0329_best_model.pt")

try:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    print(f"✅ Checkpoint loaded")
    print(f"   Epoch: {checkpoint['epoch']}")
    print(f"   Loss: {checkpoint['loss']:.6f}")
    print()
    
    # Create and load model
    model = create_model(image_size=64, init_dim=64, use_fsq=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    print(f"✅ Model loaded and ready")
    print()
    
except Exception as e:
    print(f"❌ Failed to load model: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Select test videos (last 10 samples - not used in training if 80/20 split)
test_indices = list(range(150, 160))
test_videos = videos_magvit[test_indices].to(device)
test_labels = labels[test_indices]
test_traj = trajectory_3d[test_indices]

print(f"Selected {len(test_indices)} test videos:")
print(f"  Indices: {test_indices}")
print(f"  Labels: {test_labels.tolist()}")
print()

# Reconstruct
print("Reconstructing videos...")
with torch.no_grad():
    codes = model.encode(test_videos)
    reconstructed = model.decode(codes)

print(f"✅ Reconstruction complete")
print(f"   Codes shape: {codes.shape}")
print(f"   Reconstructed shape: {reconstructed.shape}")
print()

# Compute metrics
test_videos_np = test_videos.cpu().numpy()
reconstructed_np = reconstructed.cpu().numpy()

mse = np.mean((test_videos_np - reconstructed_np) ** 2)
mse_per_sample = np.mean((test_videos_np - reconstructed_np) ** 2, axis=(1,2,3,4))

# PSNR (Peak Signal-to-Noise Ratio)
# Assuming pixel values in [0, 1]
psnr_per_sample = 20 * np.log10(1.0 / np.sqrt(mse_per_sample))
psnr_mean = np.mean(psnr_per_sample)

print("Reconstruction Metrics:")
print(f"  MSE: {mse:.6f}")
print(f"  PSNR: {psnr_mean:.2f} dB")
print(f"  PSNR range: [{psnr_per_sample.min():.2f}, {psnr_per_sample.max():.2f}] dB")
print()

# Compute per-class metrics
class_names = ['Linear', 'Circular', 'Helical', 'Parabolic']
print("Per-Class Metrics:")
for class_id in range(4):
    class_mask = test_labels == class_id
    if class_mask.sum() > 0:
        class_mse = mse_per_sample[class_mask].mean()
        class_psnr = psnr_per_sample[class_mask].mean()
        print(f"  {class_names[class_id]:10s}: MSE={class_mse:.6f}, PSNR={class_psnr:.2f} dB")
print()

# Visualization 1: Side-by-side comparison for each video
print("Creating visualizations...")
timestamp = datetime.now().strftime('%Y%m%d_%H%M')

# Convert back to (N, T, C, H, W) for visualization
test_videos_vis = test_videos.cpu().permute(0, 2, 1, 3, 4)
reconstructed_vis = reconstructed.cpu().permute(0, 2, 1, 3, 4)

# Show 4 samples with multiple frames
num_samples_to_show = 4
frames_to_show = [0, 5, 10, 15]  # Show frames 0, 5, 10, 15

fig, axes = plt.subplots(num_samples_to_show, len(frames_to_show)*2, 
                         figsize=(20, 10))
fig.suptitle('MAGVIT Reconstruction Quality - Original (top) vs Reconstructed (bottom)', 
             fontsize=16, y=0.98)

for sample_idx in range(num_samples_to_show):
    for frame_idx, frame_num in enumerate(frames_to_show):
        # Original
        original_frame = test_videos_vis[sample_idx, frame_num].numpy().transpose(1, 2, 0)
        original_frame = np.clip(original_frame, 0, 1)
        
        ax_orig = axes[sample_idx, frame_idx*2]
        ax_orig.imshow(original_frame)
        ax_orig.axis('off')
        
        if sample_idx == 0:
            ax_orig.set_title(f'Original\nFrame {frame_num}', fontsize=10)
        if frame_idx == 0:
            label = test_labels[sample_idx].item()
            mse_val = mse_per_sample[sample_idx]
            psnr_val = psnr_per_sample[sample_idx]
            ax_orig.set_ylabel(
                f'{class_names[label]}\nMSE:{mse_val:.4f}\nPSNR:{psnr_val:.1f}dB', 
                fontsize=9, rotation=0, ha='right', va='center'
            )
        
        # Reconstructed
        recon_frame = reconstructed_vis[sample_idx, frame_num].numpy().transpose(1, 2, 0)
        recon_frame = np.clip(recon_frame, 0, 1)
        
        ax_recon = axes[sample_idx, frame_idx*2 + 1]
        ax_recon.imshow(recon_frame)
        ax_recon.axis('off')
        
        if sample_idx == 0:
            ax_recon.set_title(f'Reconstructed\nFrame {frame_num}', fontsize=10)

plt.tight_layout()
output_path = Path(f"results/{timestamp}_reconstruction_comparison.png")
output_path.parent.mkdir(exist_ok=True)
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"✅ Saved: {output_path}")

# Visualization 2: Error heatmaps
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
fig.suptitle('Reconstruction Error Heatmaps (Frame 8)', fontsize=16)

for sample_idx in range(min(4, num_samples_to_show)):
    # Original
    orig_frame = test_videos_vis[sample_idx, 8].numpy().transpose(1, 2, 0)
    axes[0, sample_idx].imshow(orig_frame, vmin=0, vmax=1)
    axes[0, sample_idx].set_title(f'{class_names[test_labels[sample_idx]]} - Original', 
                                   fontsize=10)
    axes[0, sample_idx].axis('off')
    
    # Error (absolute difference, averaged across channels)
    recon_frame = reconstructed_vis[sample_idx, 8].numpy().transpose(1, 2, 0)
    error = np.abs(orig_frame - recon_frame).mean(axis=2)
    
    im = axes[1, sample_idx].imshow(error, cmap='hot', vmin=0, vmax=0.1)
    axes[1, sample_idx].set_title(f'Error (mean: {error.mean():.4f})', fontsize=10)
    axes[1, sample_idx].axis('off')
    plt.colorbar(im, ax=axes[1, sample_idx], fraction=0.046)

plt.tight_layout()
error_path = Path(f"results/{timestamp}_reconstruction_errors.png")
plt.savefig(error_path, dpi=150, bbox_inches='tight')
print(f"✅ Saved: {error_path}")

# Visualization 3: All 10 test samples (single frame)
fig, axes = plt.subplots(3, 10, figsize=(20, 6))
fig.suptitle('All 10 Test Samples - Original / Reconstructed / Error (Frame 8)', 
             fontsize=14)

for idx in range(10):
    # Original
    orig = test_videos_vis[idx, 8].numpy().transpose(1, 2, 0)
    axes[0, idx].imshow(np.clip(orig, 0, 1))
    axes[0, idx].axis('off')
    if idx == 0:
        axes[0, idx].set_ylabel('Original', rotation=0, ha='right', va='center')
    axes[0, idx].set_title(f'{class_names[test_labels[idx]]}', fontsize=8)
    
    # Reconstructed
    recon = reconstructed_vis[idx, 8].numpy().transpose(1, 2, 0)
    axes[1, idx].imshow(np.clip(recon, 0, 1))
    axes[1, idx].axis('off')
    if idx == 0:
        axes[1, idx].set_ylabel('Recon', rotation=0, ha='right', va='center')
    
    # Error
    error = np.abs(orig - recon).mean(axis=2)
    im = axes[2, idx].imshow(error, cmap='hot', vmin=0, vmax=0.1)
    axes[2, idx].axis('off')
    if idx == 0:
        axes[2, idx].set_ylabel('Error', rotation=0, ha='right', va='center')
    axes[2, idx].text(32, 58, f'{error.mean():.3f}', 
                      ha='center', va='top', fontsize=7, 
                      bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

plt.tight_layout()
all_samples_path = Path(f"results/{timestamp}_all_test_samples.png")
plt.savefig(all_samples_path, dpi=150, bbox_inches='tight')
print(f"✅ Saved: {all_samples_path}")

print()
print("="*70)
print("EVALUATION COMPLETE")
print("="*70)
print()

print("Summary:")
print(f"  Test samples: {len(test_indices)}")
print(f"  Mean MSE: {mse:.6f}")
print(f"  Mean PSNR: {psnr_mean:.2f} dB")
print()
print("Visualizations created:")
print(f"  1. {output_path}")
print(f"  2. {error_path}")
print(f"  3. {all_samples_path}")
print()

# Save metrics
metrics = {
    "timestamp": timestamp,
    "test_indices": test_indices,
    "test_labels": test_labels.tolist(),
    "mse_overall": float(mse),
    "psnr_overall": float(psnr_mean),
    "mse_per_sample": mse_per_sample.tolist(),
    "psnr_per_sample": psnr_per_sample.tolist(),
    "per_class": {}
}

for class_id in range(4):
    class_mask = test_labels == class_id
    if class_mask.sum() > 0:
        metrics["per_class"][class_names[class_id]] = {
            "mse": float(mse_per_sample[class_mask].mean()),
            "psnr": float(psnr_per_sample[class_mask].mean())
        }

import json
metrics_path = Path(f"results/{timestamp}_reconstruction_metrics.json")
with open(metrics_path, 'w') as f:
    json.dump(metrics, f, indent=2)

print(f"✅ Metrics saved: {metrics_path}")
print()

# Quality assessment
print("Quality Assessment:")
if psnr_mean > 30:
    print("  🎉 EXCELLENT - Reconstructions are very high quality (PSNR > 30 dB)")
elif psnr_mean > 25:
    print("  ✅ GOOD - Reconstructions are good quality (PSNR 25-30 dB)")
elif psnr_mean > 20:
    print("  ⚠️ FAIR - Reconstructions are acceptable (PSNR 20-25 dB)")
else:
    print("  ❌ POOR - Reconstructions need improvement (PSNR < 20 dB)")

print()
print("Interpretation:")
print("  - PSNR > 30 dB: Near-perfect reconstruction")
print("  - PSNR 25-30 dB: Good quality, minor artifacts")
print("  - PSNR 20-25 dB: Acceptable, visible differences")
print("  - PSNR < 20 dB: Poor quality, significant loss")
print()
print("="*70)

