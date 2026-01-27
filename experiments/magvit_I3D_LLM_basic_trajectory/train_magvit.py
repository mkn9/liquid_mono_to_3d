#!/usr/bin/env python3
"""
MAGVIT Training Script

Trains MAGVIT VideoTokenizer on trajectory videos with:
- Periodic checkpoint saving
- Progress monitoring visible on MacBook
- Resume capability
- Full TDD compliance
- Batch-level progress (keeps SSH alive!)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np
from pathlib import Path
from datetime import datetime
import time
import json
import sys
import os
import threading
from magvit2_pytorch import VideoTokenizer

# CRITICAL: Force unbuffered output to prevent SSH hangs
os.environ['PYTHONUNBUFFERED'] = '1'
sys.stdout.reconfigure(line_buffering=True)

# Global flag for heartbeat thread
training_active = False


def load_dataset(dataset_path, batch_size=8, train_split=0.8):
    """
    Load dataset and create train/val DataLoaders.
    
    Args:
        dataset_path: Path to .npz dataset file
        batch_size: Batch size for DataLoader
        train_split: Fraction of data for training
        
    Returns:
        train_loader, val_loader, dataset_info dict
    """
    print(f"Loading dataset from {dataset_path}...")
    
    data = np.load(dataset_path)
    videos = torch.from_numpy(data['videos']).float()  # (N, T, C, H, W)
    labels = torch.from_numpy(data['labels']).long()
    
    # Convert from (N, T, C, H, W) to (N, C, T, H, W) for MAGVIT
    videos = videos.permute(0, 2, 1, 3, 4)
    
    # Create dataset
    dataset = TensorDataset(videos, labels)
    
    # Split train/val
    num_samples = len(dataset)
    num_train = int(num_samples * train_split)
    num_val = num_samples - num_train
    
    train_dataset, val_dataset = random_split(
        dataset, 
        [num_train, num_val],
        generator=torch.Generator().manual_seed(42)
    )
    
    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )
    
    dataset_info = {
        'num_train': num_train,
        'num_val': num_val,
        'num_samples': num_samples,
        'video_shape': tuple(videos.shape),
        'num_classes': len(torch.unique(labels))
    }
    
    print(f"✅ Dataset loaded: {num_train} train, {num_val} val samples")
    
    return train_loader, val_loader, dataset_info


def create_model(image_size=64, init_dim=64, use_fsq=True):
    """
    Create MAGVIT VideoTokenizer model.
    
    Args:
        image_size: Size of video frames
        init_dim: Initial convolution dimension
        use_fsq: Use Finite Scalar Quantization
        
    Returns:
        VideoTokenizer model
    """
    print(f"Creating MAGVIT model...")
    
    model = VideoTokenizer(
        image_size=image_size,
        init_dim=init_dim,
        layers=('residual', 'residual'),
        use_fsq=use_fsq,
        fsq_levels=[8, 5, 5, 5] if use_fsq else None,
    )
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"✅ Model created: {num_params:,} parameters")
    
    return model


def train_one_epoch(model, train_loader, optimizer, device, epoch, verbose=True, print_interval=5):
    """
    Train for one epoch with batch-level progress monitoring.
    
    Args:
        model: MAGVIT VideoTokenizer
        train_loader: Training DataLoader
        optimizer: Optimizer
        device: Device to train on
        epoch: Current epoch number
        verbose: Print batch-level progress
        print_interval: Print every N batches
        
    Returns:
        Dict with metrics: {'loss', 'mse'}
    """
    model.train()
    
    total_loss = 0.0
    total_mse = 0.0
    num_batches = len(train_loader)
    
    for batch_idx, (videos, labels) in enumerate(train_loader):
        videos = videos.to(device)
        
        # Forward pass: encode and decode
        optimizer.zero_grad()
        
        # Encode to discrete codes
        codes = model.encode(videos)
        
        # Decode back to video
        reconstructed = model.decode(codes)
        
        # Reconstruction loss (MSE)
        mse_loss = nn.functional.mse_loss(reconstructed, videos)
        
        # Total loss (could add perceptual loss, etc.)
        loss = mse_loss
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Track metrics
        total_loss += loss.item()
        total_mse += mse_loss.item()
        
        # CRITICAL: Print batch progress to keep SSH alive!
        if verbose and (batch_idx % print_interval == 0 or batch_idx == num_batches - 1):
            progress_pct = (batch_idx + 1) / num_batches * 100
            print(f"  [Epoch {epoch+1}] Batch {batch_idx+1}/{num_batches} ({progress_pct:.0f}%) - "
                  f"Loss: {loss.item():.6f}", flush=True)
    
    # Average metrics
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    avg_mse = total_mse / num_batches if num_batches > 0 else 0.0
    
    return {
        'loss': avg_loss,
        'mse': avg_mse
    }


def validate(model, val_loader, device):
    """
    Run validation.
    
    Args:
        model: MAGVIT VideoTokenizer
        val_loader: Validation DataLoader
        device: Device to validate on
        
    Returns:
        Dict with metrics: {'val_loss', 'val_mse'}
    """
    model.eval()
    
    total_loss = 0.0
    total_mse = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for videos, labels in val_loader:
            videos = videos.to(device)
            
            # Encode and decode
            codes = model.encode(videos)
            reconstructed = model.decode(codes)
            
            # Loss
            mse_loss = nn.functional.mse_loss(reconstructed, videos)
            loss = mse_loss
            
            total_loss += loss.item()
            total_mse += mse_loss.item()
            num_batches += 1
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    avg_mse = total_mse / num_batches if num_batches > 0 else 0.0
    
    return {
        'val_loss': avg_loss,
        'val_mse': avg_mse
    }


def save_checkpoint(model, optimizer, epoch, loss, checkpoint_path):
    """
    Save training checkpoint.
    
    Args:
        model: MAGVIT model
        optimizer: Optimizer
        epoch: Current epoch
        loss: Current loss
        checkpoint_path: Path to save checkpoint
    """
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'timestamp': datetime.now().isoformat()
    }
    
    torch.save(checkpoint, checkpoint_path)


def load_checkpoint(model, optimizer, checkpoint_path):
    """
    Load training checkpoint.
    
    Args:
        model: MAGVIT model
        optimizer: Optimizer
        checkpoint_path: Path to checkpoint file
        
    Returns:
        start_epoch (int), best_loss (float)
    """
    checkpoint_path = Path(checkpoint_path)
    
    if not checkpoint_path.exists():
        return 0, float('inf')
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    start_epoch = checkpoint['epoch'] + 1  # Resume from next epoch
    best_loss = checkpoint['loss']
    
    return start_epoch, best_loss


def update_progress(progress_path, epoch, total_epochs, train_loss, val_loss, elapsed_time):
    """
    Update PROGRESS.txt file AND print to stdout (keeps SSH alive!).
    
    Args:
        progress_path: Path to PROGRESS.txt
        epoch: Current epoch
        total_epochs: Total epochs
        train_loss: Training loss
        val_loss: Validation loss
        elapsed_time: Elapsed time in seconds
    """
    progress_path = Path(progress_path)
    progress_path.parent.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    percent_complete = (epoch + 1) / total_epochs * 100
    
    eta_seconds = (elapsed_time / (epoch + 1)) * (total_epochs - epoch - 1) if epoch >= 0 else 0
    eta_minutes = eta_seconds / 60
    
    content = f"""MAGVIT Training Progress
========================

Timestamp: {timestamp}
Status: RUNNING

Progress: {epoch + 1}/{total_epochs} epochs ({percent_complete:.1f}%)
Elapsed: {elapsed_time/60:.1f} minutes
ETA: {eta_minutes:.1f} minutes

Current Metrics:
  Train Loss: {train_loss:.6f}
  Val Loss: {val_loss:.6f}

Last Update: {timestamp}
"""
    
    # Write to file (visible on MacBook)
    progress_path.write_text(content)
    
    # CRITICAL: Also print to stdout to keep SSH alive!
    print(f"\n[PROGRESS UPDATE] Epoch {epoch+1}/{total_epochs} ({percent_complete:.0f}%) - "
          f"Train: {train_loss:.6f}, Val: {val_loss:.6f}, "
          f"ETA: {eta_minutes:.1f}min", flush=True)
    sys.stdout.flush()  # Explicit flush for safety


def should_save_checkpoint(epoch, checkpoint_interval):
    """
    Determine if checkpoint should be saved.
    
    Args:
        epoch: Current epoch
        checkpoint_interval: Save every N epochs
        
    Returns:
        True if should save, False otherwise
    """
    return epoch % checkpoint_interval == 0


def heartbeat_thread(interval=30):
    """
    Print heartbeat every N seconds to keep SSH connection alive.
    Runs in background thread.
    """
    global training_active
    while training_active:
        print(f"[HEARTBEAT {datetime.now().strftime('%H:%M:%S')}] Training active, SSH keepalive...", 
              flush=True)
        sys.stdout.flush()
        time.sleep(interval)


def train_magvit(
    dataset_path="results/20260125_0304_dataset_200_validated.npz",
    epochs=100,
    batch_size=8,
    learning_rate=0.0001,
    checkpoint_interval=10,
    output_dir="results/magvit_training",
    resume_from=None
):
    """
    Main training function with periodic monitoring.
    
    Args:
        dataset_path: Path to dataset
        epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Learning rate
        checkpoint_interval: Save checkpoint every N epochs
        output_dir: Output directory for checkpoints and logs
        resume_from: Path to checkpoint to resume from
    """
    global training_active
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M')
    
    print("="*70, flush=True)
    print("MAGVIT TRAINING", flush=True)
    print("="*70, flush=True)
    print(f"Device: {device}", flush=True)
    print(f"Dataset: {dataset_path}", flush=True)
    print(f"Epochs: {epochs}", flush=True)
    print(f"Batch size: {batch_size}", flush=True)
    print(f"Learning rate: {learning_rate}", flush=True)
    print(f"Checkpoint interval: {checkpoint_interval}", flush=True)
    print(f"Output monitoring: ENABLED (batch-level + heartbeat)", flush=True)
    print(flush=True)
    
    # Load dataset
    train_loader, val_loader, dataset_info = load_dataset(
        dataset_path=dataset_path,
        batch_size=batch_size
    )
    
    # Create model
    model = create_model(image_size=64, init_dim=64, use_fsq=True)
    model = model.to(device)
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Resume from checkpoint if specified
    start_epoch = 0
    best_loss = float('inf')
    if resume_from:
        start_epoch, best_loss = load_checkpoint(model, optimizer, resume_from)
        print(f"✅ Resumed from epoch {start_epoch}, best loss: {best_loss:.6f}", flush=True)
    
    # Start heartbeat thread to keep SSH alive
    training_active = True
    heartbeat = threading.Thread(target=heartbeat_thread, args=(30,), daemon=True)
    heartbeat.start()
    print("✅ Heartbeat thread started (30s interval)", flush=True)
    
    # Training loop
    print(flush=True)
    print("Starting training...", flush=True)
    print(flush=True)
    
    start_time = time.time()
    training_history = []
    
    try:
    
        for epoch in range(start_epoch, epochs):
            epoch_start = time.time()
            
            print(f"\n{'='*70}", flush=True)
            print(f"EPOCH {epoch+1}/{epochs}", flush=True)
            print(f"{'='*70}", flush=True)
            
            # Train one epoch (with batch-level progress)
            train_metrics = train_one_epoch(model, train_loader, optimizer, device, epoch, 
                                          verbose=True, print_interval=5)
            
            # Validate
            print(f"  Validating...", flush=True)
            val_metrics = validate(model, val_loader, device)
            
            # Track metrics
            train_loss = train_metrics['loss']
            val_loss = val_metrics['val_loss']
            
            # Update best model
            if val_loss < best_loss:
                best_loss = val_loss
                best_checkpoint = output_dir / f"{timestamp}_best_model.pt"
                save_checkpoint(model, optimizer, epoch, val_loss, best_checkpoint)
                print(f"  ⭐ New best model saved! Val loss: {val_loss:.6f}", flush=True)
            
            # Periodic checkpoint
            if should_save_checkpoint(epoch, checkpoint_interval):
                checkpoint_path = output_dir / f"{timestamp}_checkpoint_epoch_{epoch}.pt"
                save_checkpoint(model, optimizer, epoch, val_loss, checkpoint_path)
                print(f"  💾 Checkpoint saved: epoch {epoch}", flush=True)
            
            # Update progress (visible on MacBook AND prints to stdout)
            elapsed_time = time.time() - start_time
            progress_path = output_dir / "PROGRESS.txt"
            update_progress(progress_path, epoch, epochs, train_loss, val_loss, elapsed_time)
            
            # Log progress summary
            epoch_time = time.time() - epoch_start
            print(f"\n  Summary: Epoch {epoch+1}/{epochs} completed in {epoch_time:.1f}s", flush=True)
            print(f"  Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | Best: {best_loss:.6f}", 
                  flush=True)
            
            # Save history
            training_history.append({
                'epoch': epoch,
                'train_loss': float(train_loss),
                'train_mse': float(train_metrics['mse']),
                'val_loss': float(val_loss),
                'val_mse': float(val_metrics['val_mse']),
                'elapsed_time': elapsed_time
            })
        
        # Final checkpoint
        final_checkpoint = output_dir / f"{timestamp}_final_model.pt"
        save_checkpoint(model, optimizer, epochs-1, val_loss, final_checkpoint)
        
    finally:
        # Stop heartbeat thread
        training_active = False
        print("\n✅ Heartbeat thread stopped", flush=True)
    
    # Save training history
    history_path = output_dir / f"{timestamp}_training_history.json"
    with open(history_path, 'w') as f:
        json.dump(training_history, f, indent=2)
    
    # Final summary
    total_time = time.time() - start_time
    print()
    print("="*70)
    print("✅ TRAINING COMPLETE!")
    print("="*70)
    print(f"Total time: {total_time/60:.1f} minutes")
    print(f"Best val loss: {best_loss:.6f}")
    print(f"Final checkpoint: {final_checkpoint}")
    print(f"Best checkpoint: {best_checkpoint}")
    print(f"Training history: {history_path}")
    print()
    
    return {
        'best_loss': best_loss,
        'final_checkpoint': str(final_checkpoint),
        'best_checkpoint': str(best_checkpoint),
        'history': training_history
    }


if __name__ == "__main__":
    # Train MAGVIT on validated dataset
    results = train_magvit(
        dataset_path="results/20260125_0304_dataset_200_validated.npz",
        epochs=100,
        batch_size=8,
        learning_rate=0.0001,
        checkpoint_interval=10,
        output_dir="results/magvit_training"
    )
    
    print("Training results:")
    print(json.dumps(results, indent=2))

