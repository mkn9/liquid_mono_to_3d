#!/usr/bin/env python3
"""
MAGVIT Trajectory Classification

Trains a classifier on MAGVIT-encoded trajectory representations to classify
trajectories into one of four types: Linear, Circular, Helical, or Parabolic.

This validates whether MAGVIT encodings preserve class-discriminative information.

Architecture:
    Video -> MAGVIT Encoder -> Codes -> MLP Classifier -> Class Prediction

Training Process:
    1. Load pretrained MAGVIT model
    2. Encode all videos to code representations
    3. Train MLP classifier on codes
    4. Evaluate classification accuracy
    5. Save best model and metrics
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from datetime import datetime
import json
import sys
from typing import Dict, Tuple, Optional
import time

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from train_magvit import create_model as create_magvit_model


def load_classification_data(dataset_path: str) -> Dict[str, torch.Tensor]:
    """
    Load dataset for classification.
    
    Args:
        dataset_path: Path to .npz dataset file
    
    Returns:
        Dictionary with 'videos' and 'labels'
    """
    data = np.load(dataset_path)
    
    videos = torch.from_numpy(data['videos']).float()
    labels = torch.from_numpy(data['labels']).long()
    
    return {
        'videos': videos,
        'labels': labels
    }


def encode_dataset_to_codes(
    dataset_path: str,
    model_checkpoint: str,
    batch_size: int = 8,
    device: str = 'cpu'
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Encode all videos in dataset to MAGVIT codes.
    
    Args:
        dataset_path: Path to .npz dataset file
        model_checkpoint: Path to trained MAGVIT model checkpoint
        batch_size: Batch size for encoding
        device: Device to use ('cpu' or 'cuda')
    
    Returns:
        Tuple of (codes, labels) where codes is (N, feature_dim)
    """
    print("Loading dataset...")
    data = load_classification_data(dataset_path)
    videos = data['videos'].to(device)  # (N, T, C, H, W)
    labels = data['labels']
    
    print(f"Dataset loaded: {len(videos)} videos")
    print(f"Video shape: {videos.shape}")
    
    # Load MAGVIT model
    print("Loading MAGVIT model...")
    checkpoint = torch.load(model_checkpoint, map_location='cpu')
    
    # Assuming 64x64 images, 64 init_dim, FSQ quantization
    model = create_magvit_model(image_size=64, init_dim=64, use_fsq=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"MAGVIT model loaded from epoch {checkpoint['epoch']}")
    
    # Encode all videos
    print("Encoding videos to codes...")
    all_codes = []
    
    with torch.no_grad():
        for i in range(0, len(videos), batch_size):
            batch_videos = videos[i:i+batch_size]
            
            # Convert from (B, T, C, H, W) to (B, C, T, H, W)
            batch_videos = batch_videos.permute(0, 2, 1, 3, 4)
            
            # Encode to codes
            codes = model.encode(batch_videos)  # (B, D, H', W', T') or (B, D, T', H', W')
            
            # Apply spatial average pooling to reduce dimensionality
            # Keep temporal dimension to preserve motion information
            # Assume codes shape is (B, D, T, H, W) or similar
            if codes.dim() == 5:
                # Pool over spatial dimensions only, keep channel and temporal
                batch_codes_pooled = codes.mean(dim=[-2, -1])  # (B, D, T)
                # Flatten to (B, D*T)
                batch_codes_flat = batch_codes_pooled.flatten(start_dim=1)
            else:
                # Fallback: global average pooling
                batch_codes_flat = codes.mean(dim=list(range(2, codes.dim())))
            
            all_codes.append(batch_codes_flat.cpu())
            
            if (i // batch_size + 1) % 5 == 0:
                print(f"  Encoded {i+len(batch_videos)}/{len(videos)} videos", flush=True)
    
    all_codes = torch.cat(all_codes, dim=0)  # (N, feature_dim)
    
    print(f"✅ Encoding complete: {all_codes.shape}")
    
    return all_codes, labels


def split_dataset(
    codes: torch.Tensor,
    labels: torch.Tensor,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    seed: int = 42
) -> Dict[str, torch.Tensor]:
    """
    Split dataset into train/val/test sets with stratification.
    
    Args:
        codes: Encoded representations (N, feature_dim)
        labels: Class labels (N,)
        train_ratio: Fraction for training
        val_ratio: Fraction for validation
        seed: Random seed
    
    Returns:
        Dictionary with train/val/test splits
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    num_samples = len(codes)
    num_train = int(num_samples * train_ratio)
    num_val = int(num_samples * val_ratio)
    num_test = num_samples - num_train - num_val
    
    # Create stratified splits (ensure each class is represented)
    num_classes = labels.max().item() + 1
    
    train_indices = []
    val_indices = []
    test_indices = []
    
    for class_id in range(num_classes):
        class_mask = (labels == class_id)
        class_indices = torch.where(class_mask)[0].numpy()
        
        np.random.shuffle(class_indices)
        
        n_class = len(class_indices)
        n_train = int(n_class * train_ratio)
        n_val = int(n_class * val_ratio)
        
        train_indices.extend(class_indices[:n_train])
        val_indices.extend(class_indices[n_train:n_train+n_val])
        test_indices.extend(class_indices[n_train+n_val:])
    
    # Shuffle indices
    np.random.shuffle(train_indices)
    np.random.shuffle(val_indices)
    np.random.shuffle(test_indices)
    
    return {
        'train_codes': codes[train_indices],
        'train_labels': labels[train_indices],
        'val_codes': codes[val_indices],
        'val_labels': labels[val_indices],
        'test_codes': codes[test_indices],
        'test_labels': labels[test_indices]
    }


class TrajectoryClassifier(nn.Module):
    """
    MLP classifier for trajectory classification from MAGVIT codes.
    
    Architecture:
        Input -> [Linear -> BatchNorm -> ReLU -> Dropout]+ -> Output
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: list,
        num_classes: int,
        dropout: float = 0.3
    ):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, num_classes))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input codes (B, feature_dim)
        
        Returns:
            Logits (B, num_classes)
        """
        return self.network(x)


def train_one_epoch(
    model: nn.Module,
    train_codes: torch.Tensor,
    train_labels: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    batch_size: int,
    device: str = 'cpu'
) -> float:
    """
    Train for one epoch.
    
    Args:
        model: Classifier model
        train_codes: Training codes
        train_labels: Training labels
        optimizer: Optimizer
        criterion: Loss function
        batch_size: Batch size
        device: Device
    
    Returns:
        Average training loss
    """
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    # Shuffle data
    indices = torch.randperm(len(train_codes))
    train_codes = train_codes[indices]
    train_labels = train_labels[indices]
    
    for i in range(0, len(train_codes), batch_size):
        batch_codes = train_codes[i:i+batch_size].to(device)
        batch_labels = train_labels[i:i+batch_size].to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        logits = model(batch_codes)
        loss = criterion(logits, batch_labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
    
    return total_loss / num_batches


def validate(
    model: nn.Module,
    val_codes: torch.Tensor,
    val_labels: torch.Tensor,
    criterion: nn.Module,
    batch_size: int,
    device: str = 'cpu'
) -> Dict[str, float]:
    """
    Validate model.
    
    Args:
        model: Classifier model
        val_codes: Validation codes
        val_labels: Validation labels
        criterion: Loss function
        batch_size: Batch size
        device: Device
    
    Returns:
        Dictionary with loss, accuracy, per_class_accuracy
    """
    model.eval()
    total_loss = 0.0
    num_batches = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for i in range(0, len(val_codes), batch_size):
            batch_codes = val_codes[i:i+batch_size].to(device)
            batch_labels = val_labels[i:i+batch_size].to(device)
            
            # Forward pass
            logits = model(batch_codes)
            loss = criterion(logits, batch_labels)
            
            total_loss += loss.item()
            num_batches += 1
            
            # Predictions
            preds = torch.argmax(logits, dim=1)
            all_preds.append(preds.cpu())
            all_labels.append(batch_labels.cpu())
    
    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    
    # Overall accuracy
    accuracy = (all_preds == all_labels).float().mean().item()
    
    # Per-class accuracy
    num_classes = all_labels.max().item() + 1
    per_class_acc = {}
    for class_id in range(num_classes):
        class_mask = (all_labels == class_id)
        if class_mask.sum() > 0:
            class_correct = (all_preds[class_mask] == all_labels[class_mask]).float().mean().item()
            per_class_acc[class_id] = class_correct
    
    return {
        'loss': total_loss / num_batches,
        'accuracy': accuracy,
        'per_class_accuracy': per_class_acc
    }


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    train_loss: float,
    val_loss: float,
    val_accuracy: float,
    checkpoint_path: str
):
    """Save classifier checkpoint."""
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'train_loss': train_loss,
        'val_loss': val_loss,
        'val_accuracy': val_accuracy
    }, checkpoint_path)


def load_checkpoint(
    checkpoint_path: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None
) -> Dict:
    """Load classifier checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return checkpoint


def update_progress(
    progress_path: str,
    epoch: int,
    train_loss: float,
    val_loss: float,
    val_accuracy: float,
    best_accuracy: float
):
    """Update progress file."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    with open(progress_path, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("MAGVIT CLASSIFICATION TRAINING PROGRESS\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Last Update: {timestamp}\n")
        f.write(f"Epoch: {epoch}\n")
        f.write(f"Train Loss: {train_loss:.6f}\n")
        f.write(f"Val Loss: {val_loss:.6f}\n")
        f.write(f"Val Accuracy: {val_accuracy:.4f} ({val_accuracy*100:.2f}%)\n")
        f.write(f"Best Accuracy: {best_accuracy:.4f} ({best_accuracy*100:.2f}%)\n")
        f.write("\n" + "=" * 70 + "\n")
    
    sys.stdout.flush()


def train_classifier(
    dataset_path: str,
    magvit_checkpoint: str,
    output_dir: str,
    epochs: int = 100,
    batch_size: int = 16,
    learning_rate: float = 0.001,
    hidden_dims: list = None,
    dropout: float = 0.3,
    device: str = None
) -> Dict:
    """
    Train trajectory classifier on MAGVIT codes.
    
    Args:
        dataset_path: Path to dataset .npz file
        magvit_checkpoint: Path to trained MAGVIT checkpoint
        output_dir: Output directory for checkpoints and logs
        epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Learning rate
        hidden_dims: Hidden layer dimensions (default: [512, 256, 128])
        dropout: Dropout rate
        device: Device ('cpu', 'cuda', or None for auto)
    
    Returns:
        Dictionary with training results
    """
    if hidden_dims is None:
        hidden_dims = [512, 256, 128]
    
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M')
    
    print("=" * 70)
    print("MAGVIT TRAJECTORY CLASSIFICATION TRAINING")
    print("=" * 70)
    print(f"Device: {device}")
    print(f"Dataset: {dataset_path}")
    print(f"MAGVIT Checkpoint: {magvit_checkpoint}")
    print(f"Output: {output_dir}")
    print(f"Epochs: {epochs}")
    print(f"Batch Size: {batch_size}")
    print(f"Learning Rate: {learning_rate}")
    print(f"Hidden Dims: {hidden_dims}")
    print(f"Dropout: {dropout}")
    print()
    
    # Step 1: Encode dataset to codes
    print("=" * 70)
    print("STEP 1: ENCODING DATASET")
    print("=" * 70)
    codes, labels = encode_dataset_to_codes(dataset_path, magvit_checkpoint, batch_size=8, device=device)
    print(f"✅ Codes shape: {codes.shape}")
    print(f"✅ Labels shape: {labels.shape}")
    print()
    
    # Step 2: Split dataset
    print("=" * 70)
    print("STEP 2: SPLITTING DATASET")
    print("=" * 70)
    splits = split_dataset(codes, labels, train_ratio=0.7, val_ratio=0.15)
    print(f"Train: {len(splits['train_codes'])} samples")
    print(f"Val:   {len(splits['val_codes'])} samples")
    print(f"Test:  {len(splits['test_codes'])} samples")
    print()
    
    # Step 3: Create classifier
    print("=" * 70)
    print("STEP 3: CREATING CLASSIFIER")
    print("=" * 70)
    input_dim = codes.shape[1]
    num_classes = 4
    
    classifier = TrajectoryClassifier(
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        num_classes=num_classes,
        dropout=dropout
    ).to(device)
    
    num_params = sum(p.numel() for p in classifier.parameters())
    print(f"Classifier created: {num_params:,} parameters")
    print(f"Input dim: {input_dim}")
    print(f"Hidden dims: {hidden_dims}")
    print(f"Output classes: {num_classes}")
    print()
    
    # Step 4: Training setup
    # Add weight decay for regularization
    optimizer = torch.optim.Adam(classifier.parameters(), lr=learning_rate, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    best_accuracy = 0.0
    train_history = []
    
    progress_path = output_path / "PROGRESS.txt"
    
    # Step 5: Training loop
    print("=" * 70)
    print("STEP 4: TRAINING")
    print("=" * 70)
    
    for epoch in range(epochs):
        epoch_start = time.time()
        
        # Train
        train_loss = train_one_epoch(
            classifier,
            splits['train_codes'],
            splits['train_labels'],
            optimizer,
            criterion,
            batch_size,
            device
        )
        
        # Validate
        val_metrics = validate(
            classifier,
            splits['val_codes'],
            splits['val_labels'],
            criterion,
            batch_size,
            device
        )
        
        val_loss = val_metrics['loss']
        val_accuracy = val_metrics['accuracy']
        
        epoch_time = time.time() - epoch_start
        
        # Update history
        train_history.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'val_accuracy': val_accuracy,
            'per_class_accuracy': val_metrics['per_class_accuracy'],
            'epoch_time': epoch_time
        })
        
        # Print progress
        if epoch % 5 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch:3d}/{epochs} | "
                  f"Train Loss: {train_loss:.4f} | "
                  f"Val Loss: {val_loss:.4f} | "
                  f"Val Acc: {val_accuracy:.4f} ({val_accuracy*100:.2f}%) | "
                  f"Time: {epoch_time:.1f}s",
                  flush=True)
        
        # Save best model
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            best_checkpoint_path = output_path / f"{timestamp}_best_classifier.pt"
            save_checkpoint(
                classifier,
                optimizer,
                epoch,
                train_loss,
                val_loss,
                val_accuracy,
                str(best_checkpoint_path)
            )
            print(f"  ✅ New best accuracy: {best_accuracy:.4f} (saved)", flush=True)
        
        # Update progress file
        update_progress(
            str(progress_path),
            epoch,
            train_loss,
            val_loss,
            val_accuracy,
            best_accuracy
        )
        
        # Save checkpoint every 20 epochs
        if (epoch + 1) % 20 == 0:
            checkpoint_path = output_path / f"{timestamp}_checkpoint_epoch_{epoch}.pt"
            save_checkpoint(
                classifier,
                optimizer,
                epoch,
                train_loss,
                val_loss,
                val_accuracy,
                str(checkpoint_path)
            )
    
    print()
    print("=" * 70)
    print("STEP 5: FINAL EVALUATION")
    print("=" * 70)
    
    # Load best model for final evaluation
    best_checkpoint = load_checkpoint(str(best_checkpoint_path), classifier)
    
    # Test set evaluation
    test_metrics = validate(
        classifier,
        splits['test_codes'],
        splits['test_labels'],
        criterion,
        batch_size,
        device
    )
    
    print(f"Best Validation Accuracy: {best_accuracy:.4f} ({best_accuracy*100:.2f}%)")
    print(f"Test Accuracy: {test_metrics['accuracy']:.4f} ({test_metrics['accuracy']*100:.2f}%)")
    print(f"Test Loss: {test_metrics['loss']:.4f}")
    print()
    print("Per-Class Test Accuracy:")
    class_names = {0: "Linear", 1: "Circular", 2: "Helical", 3: "Parabolic"}
    for class_id, acc in test_metrics['per_class_accuracy'].items():
        print(f"  {class_names[class_id]:<10}: {acc:.4f} ({acc*100:.2f}%)")
    print()
    
    # Save final model
    final_checkpoint_path = output_path / f"{timestamp}_final_classifier.pt"
    save_checkpoint(
        classifier,
        optimizer,
        epochs - 1,
        train_history[-1]['train_loss'],
        test_metrics['loss'],
        test_metrics['accuracy'],
        str(final_checkpoint_path)
    )
    
    # Save training history
    history_path = output_path / f"{timestamp}_training_history.json"
    with open(history_path, 'w') as f:
        json.dump({
            'train_history': train_history,
            'best_val_accuracy': best_accuracy,
            'test_metrics': {
                'accuracy': test_metrics['accuracy'],
                'loss': test_metrics['loss'],
                'per_class_accuracy': {class_names[k]: v for k, v in test_metrics['per_class_accuracy'].items()}
            },
            'config': {
                'epochs': epochs,
                'batch_size': batch_size,
                'learning_rate': learning_rate,
                'hidden_dims': hidden_dims,
                'dropout': dropout
            }
        }, f, indent=2)
    
    print(f"✅ Training complete!")
    print(f"   Best model: {best_checkpoint_path}")
    print(f"   Final model: {final_checkpoint_path}")
    print(f"   History: {history_path}")
    print("=" * 70)
    
    return {
        'best_accuracy': best_accuracy,
        'final_accuracy': test_metrics['accuracy'],
        'train_history': train_history,
        'test_metrics': test_metrics
    }


if __name__ == "__main__":
    # Run classification training
    dataset_path = "results/20260125_0304_dataset_200_validated.npz"
    model_checkpoint = sorted(list(Path("results/magvit_training").glob("*_best_model.pt")))[-1]
    
    results = train_classifier(
        dataset_path=dataset_path,
        magvit_checkpoint=str(model_checkpoint),
        output_dir="results/classification",
        epochs=100,
        batch_size=16,
        learning_rate=0.001
    )
    
    print(f"\n✅ Classification accuracy: {results['final_accuracy']:.2%}")

