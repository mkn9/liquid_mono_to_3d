"""
Visualization utilities for trajectory data and model results.

Provides plotting functions with timestamped outputs.
"""

import matplotlib.pyplot as plt
import numpy as np
import torch
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple, List


def get_timestamped_filename(base_name: str, extension: str = "png") -> str:
    """Generate timestamped filename for results.
    
    Args:
        base_name: Descriptive name
        extension: File extension (without dot)
    
    Returns:
        str: Filename in format YYYYMMDD_HHMM_base_name.extension
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    return f"{timestamp}_{base_name}.{extension}"


def plot_trajectory_3d(
    trajectory: np.ndarray,
    title: str = "3D Trajectory",
    save_path: Optional[Path] = None
) -> plt.Figure:
    """Plot 3D trajectory.
    
    Args:
        trajectory: 3D points, shape (T, 3)
        title: Plot title
        save_path: Optional path to save figure
    
    Returns:
        matplotlib Figure
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], 'b-', linewidth=2)
    ax.scatter(trajectory[0, 0], trajectory[0, 1], trajectory[0, 2], 
               c='g', s=100, marker='o', label='Start')
    ax.scatter(trajectory[-1, 0], trajectory[-1, 1], trajectory[-1, 2], 
               c='r', s=100, marker='s', label='End')
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)
    ax.legend()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_classification_results(
    confusion_matrix: np.ndarray,
    class_names: List[str],
    save_path: Optional[Path] = None
) -> plt.Figure:
    """Plot confusion matrix.
    
    Args:
        confusion_matrix: Confusion matrix, shape (N, N)
        class_names: Names of classes
        save_path: Optional path to save figure
    
    Returns:
        matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    im = ax.imshow(confusion_matrix, cmap='Blues')
    
    # Add colorbar
    plt.colorbar(im, ax=ax)
    
    # Set ticks
    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)
    
    # Rotate x labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    
    # Add text annotations
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            text = ax.text(j, i, int(confusion_matrix[i, j]),
                          ha="center", va="center", color="black")
    
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_forecasting_results(
    predictions: np.ndarray,
    ground_truth: np.ndarray,
    save_path: Optional[Path] = None
) -> plt.Figure:
    """Plot forecasting predictions vs ground truth.
    
    Args:
        predictions: Predicted trajectories, shape (N, T, 3)
        ground_truth: Ground truth trajectories, shape (N, T, 3)
        save_path: Optional path to save figure
    
    Returns:
        matplotlib Figure
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    dims = ['X', 'Y', 'Z']
    
    for dim_idx, (ax, dim_name) in enumerate(zip(axes, dims)):
        # Plot first few samples
        num_samples = min(5, len(predictions))
        
        for i in range(num_samples):
            ax.plot(ground_truth[i, :, dim_idx], 'b-', alpha=0.3, linewidth=2)
            ax.plot(predictions[i, :, dim_idx], 'r--', alpha=0.3, linewidth=2)
        
        # Add legend once
        if dim_idx == 0:
            ax.plot([], [], 'b-', label='Ground Truth', linewidth=2)
            ax.plot([], [], 'r--', label='Prediction', linewidth=2)
            ax.legend()
        
        ax.set_xlabel('Time Step')
        ax.set_ylabel(f'{dim_name} Position')
        ax.set_title(f'{dim_name} Dimension')
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('Forecasting Results: Predictions vs Ground Truth')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_training_curves(
    train_losses: List[float],
    val_losses: List[float],
    train_accuracies: Optional[List[float]] = None,
    val_accuracies: Optional[List[float]] = None,
    save_path: Optional[Path] = None
) -> plt.Figure:
    """Plot training and validation curves.
    
    Args:
        train_losses: Training losses per epoch
        val_losses: Validation losses per epoch
        train_accuracies: Optional training accuracies
        val_accuracies: Optional validation accuracies
        save_path: Optional path to save figure
    
    Returns:
        matplotlib Figure
    """
    if train_accuracies is not None:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    else:
        fig, ax1 = plt.subplots(1, 1, figsize=(10, 5))
        ax2 = None
    
    # Plot losses
    epochs = range(1, len(train_losses) + 1)
    ax1.plot(epochs, train_losses, 'b-', label='Train Loss', linewidth=2)
    ax1.plot(epochs, val_losses, 'r-', label='Val Loss', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot accuracies if provided
    if ax2 is not None and train_accuracies is not None:
        ax2.plot(epochs, train_accuracies, 'b-', label='Train Acc', linewidth=2)
        ax2.plot(epochs, val_accuracies, 'r-', label='Val Acc', linewidth=2)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.set_title('Training and Validation Accuracy')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig

