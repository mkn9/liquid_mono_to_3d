#!/usr/bin/env python3
"""
Diagnose MAGVIT codes to understand why classification is failing.

This script analyzes the encoded MAGVIT representations to determine if
they contain class-discriminative information.
"""

import torch
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import sys

sys.path.insert(0, str(Path(__file__).parent))

from classify_magvit import encode_dataset_to_codes, load_classification_data

print("="*70)
print("MAGVIT CODE DIAGNOSTICS")
print("="*70)
print()

# Load and encode dataset
dataset_path = "results/20260125_0304_dataset_200_validated.npz"
model_checkpoint = sorted(list(Path("results/magvit_training").glob("*_best_model.pt")))[-1]

print("Encoding dataset...")
codes, labels = encode_dataset_to_codes(dataset_path, str(model_checkpoint), batch_size=8, device='cpu')

print(f"✅ Codes shape: {codes.shape}")
print(f"✅ Labels shape: {labels.shape}")
print()

# Analyze per-class statistics
print("="*70)
print("PER-CLASS CODE STATISTICS")
print("="*70)

class_names = {0: "Linear", 1: "Circular", 2: "Helical", 3: "Parabolic"}
num_classes = 4

for class_id in range(num_classes):
    class_mask = (labels == class_id)
    class_codes = codes[class_mask]
    
    mean = class_codes.mean(dim=0)
    std = class_codes.std(dim=0)
    
    print(f"\n{class_names[class_id]}:")
    print(f"  Samples: {class_mask.sum()}")
    print(f"  Mean code mean: {mean.mean():.6f}")
    print(f"  Mean code std:  {std.mean():.6f}")
    print(f"  Code range: [{class_codes.min():.6f}, {class_codes.max():.6f}]")

# Check if codes are separable
print("\n" + "="*70)
print("INTER-CLASS DISTANCE ANALYSIS")
print("="*70)
print()

# Compute class centroids
centroids = []
for class_id in range(num_classes):
    class_mask = (labels == class_id)
    centroid = codes[class_mask].mean(dim=0)
    centroids.append(centroid)

# Compute pairwise distances
print("Pairwise centroid distances (L2):")
for i in range(num_classes):
    for j in range(i+1, num_classes):
        dist = torch.norm(centroids[i] - centroids[j]).item()
        print(f"  {class_names[i]:10} <-> {class_names[j]:10}: {dist:.6f}")

# Compute within-class variance
print("\nWithin-class variance:")
for class_id in range(num_classes):
    class_mask = (labels == class_id)
    class_codes = codes[class_mask]
    centroid = centroids[class_id]
    variance = torch.mean(torch.norm(class_codes - centroid, dim=1)).item()
    print(f"  {class_names[class_id]:10}: {variance:.6f}")

# Visualize with PCA
print("\n" + "="*70)
print("VISUALIZATION")
print("="*70)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# PCA
print("Computing PCA...")
pca = PCA(n_components=2)
codes_pca = pca.fit_transform(codes.numpy())

ax = axes[0]
for class_id in range(num_classes):
    class_mask = (labels == class_id).numpy()
    ax.scatter(codes_pca[class_mask, 0], codes_pca[class_mask, 1], 
               label=class_names[class_id], alpha=0.6, s=50)
ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} var)')
ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} var)')
ax.set_title('PCA of MAGVIT Codes')
ax.legend()
ax.grid(True, alpha=0.3)

# t-SNE
print("Computing t-SNE...")
tsne = TSNE(n_components=2, random_state=42, perplexity=30)
codes_tsne = tsne.fit_transform(codes.numpy())

ax = axes[1]
for class_id in range(num_classes):
    class_mask = (labels == class_id).numpy()
    ax.scatter(codes_tsne[class_mask, 0], codes_tsne[class_mask, 1],
               label=class_names[class_id], alpha=0.6, s=50)
ax.set_xlabel('t-SNE 1')
ax.set_ylabel('t-SNE 2')
ax.set_title('t-SNE of MAGVIT Codes')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
output_path = Path("results/classification/code_visualization.png")
plt.savefig(output_path)
print(f"\n✅ Visualization saved: {output_path}")

# Check code diversity
print("\n" + "="*70)
print("CODE DIVERSITY CHECK")
print("="*70)

# Check if all codes are similar
code_diffs = []
for i in range(len(codes)):
    for j in range(i+1, len(codes)):
        diff = torch.norm(codes[i] - codes[j]).item()
        code_diffs.append(diff)

code_diffs = np.array(code_diffs)
print(f"Mean pairwise distance: {code_diffs.mean():.6f}")
print(f"Std pairwise distance:  {code_diffs.std():.6f}")
print(f"Min pairwise distance:  {code_diffs.min():.6f}")
print(f"Max pairwise distance:  {code_diffs.max():.6f}")

if code_diffs.std() < 0.1:
    print("\n⚠️  WARNING: Very low code diversity!")
    print("   MAGVIT codes are nearly identical across samples.")
    print("   This explains classification failure.")

print("\n" + "="*70)
print("DIAGNOSIS COMPLETE")
print("="*70)

