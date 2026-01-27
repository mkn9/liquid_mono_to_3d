#!/usr/bin/env python3
"""
Training Script for MagVIT Feature Extractor
=============================================

Trains the MagVIT-based trajectory video understanding model.
Uses UnifiedModel with I3D extractor for multi-task learning:
- Task 1: Trajectory classification (4 classes)
- Task 2: Future position prediction (x, y, z)

Usage:
    python train.py --config config_validation.yaml
    python train.py --config config_full_training.yaml --resume results/checkpoint_epoch_10.pt
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import yaml
import json
from pathlib import Path
from datetime import datetime
import sys
import time
import argparse

# Add shared modules to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'shared'))
from unified_model import UnifiedModel, compute_loss, accuracy, prediction_error
from feature_extractor import MagVITExtractor


def create_dataloader(data_dir, batch_size, split='train'):
    """Create dataloader from generated dataset."""
    data_path = Path(data_dir)
    
    # Load videos and labels
    videos = []
    labels = []
    positions = []
    
    video_files = sorted((data_path / 'videos').glob('traj_*.pt'))
    label_files = sorted((data_path / 'labels').glob('traj_*.json'))
    
    # Simple 80/20 train/val split
    total = len(video_files)
    train_size = int(0.8 * total)
    
    if split == 'train':
        files = list(zip(video_files[:train_size], label_files[:train_size]))
    else:
        files = list(zip(video_files[train_size:], label_files[train_size:]))
    
    class_map = {'Linear': 0, 'Circular': 1, 'Helical': 2, 'Parabolic': 3}
    
    for video_file, label_file in files:
        video = torch.load(video_file)
        with open(label_file) as f:
            metadata = json.load(f)
        
        # Get last position as "future" position for prediction
        positions_array = metadata['positions']
        future_pos = torch.tensor(positions_array[-1], dtype=torch.float32)
        
        videos.append(video)
        labels.append(class_map[metadata['class']])
        positions.append(future_pos)
    
    videos = torch.stack(videos)
    labels = torch.tensor(labels, dtype=torch.long)
    positions = torch.stack(positions)
    
    dataset = TensorDataset(videos, labels, positions)
    return DataLoader(dataset, batch_size=batch_size, shuffle=(split == 'train'))


def update_progress(output_dir, epoch, total_epochs, train_loss, val_loss, val_acc, elapsed_time, start_time):
    """Update PROGRESS.txt for monitoring."""
    progress_path = Path(output_dir) / 'PROGRESS.txt'
    
    percent = 100 * (epoch + 1) / total_epochs
    eta = (elapsed_time / (epoch + 1)) * (total_epochs - epoch - 1)
    
    content = f"""MagVIT Trajectory Training Progress
========================================

Branch: MagVIT (Tokenization)
Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Progress: Epoch {epoch + 1}/{total_epochs} ({percent:.1f}%)

Metrics:
  Train Loss: {train_loss:.6f}
  Val Loss: {val_loss:.6f}
  Val Accuracy: {val_acc:.2%}

Timing:
  Elapsed: {elapsed_time/60:.1f} min
  ETA: {eta/60:.1f} min
  Avg Time/Epoch: {elapsed_time/(epoch+1):.1f}s

Last Update: {datetime.now().isoformat()}
"""
    
    progress_path.write_text(content)
    
    # Also print for SSH keep-alive
    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Epoch {epoch+1}/{total_epochs} ({percent:.1f}%) - "
          f"Loss: {val_loss:.4f}, Acc: {val_acc:.2%}, ETA: {eta/60:.1f}min", flush=True)
    sys.stdout.flush()


def save_checkpoint(model, optimizer, epoch, metrics, output_dir):
    """Save training checkpoint."""
    checkpoint_path = Path(output_dir) / f'checkpoint_epoch_{epoch+1}.pt'
    
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics,
        'timestamp': datetime.now().isoformat()
    }, checkpoint_path)
    
    print(f"💾 Checkpoint saved: {checkpoint_path}")


def train(config_path):
    """Main training loop."""
    # Load config
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    print("=" * 70)
    print("MagVIT Trajectory Video Training")
    print("=" * 70)
    print(f"Config: {config_path}")
    print(f"Output: {config['output_dir']}")
    print("=" * 70)
    
    # Create output directory
    output_dir = Path(config['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create model
    print("\n1. Creating MagVIT model...")
    extractor = MagVITExtractor(feature_dim=config['feature_dim'])
    model = UnifiedModel(extractor, num_classes=4)
    
    # Move to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    print(f"✅ Model created on {device}")
    
    # Create dataloaders
    print("\n2. Loading dataset...")
    train_loader = create_dataloader(config['data_dir'], config['batch_size'], 'train')
    val_loader = create_dataloader(config['data_dir'], config['batch_size'], 'val')
    print(f"✅ Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    
    # Resume if specified
    start_epoch = 0
    if 'resume_from' in config and config['resume_from']:
        print(f"\n3. Resuming from {config['resume_from']}...")
        checkpoint = torch.load(config['resume_from'])
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"✅ Resumed from epoch {start_epoch}")
    
    # Training loop
    print(f"\n4. Starting training for {config['epochs']} epochs...")
    start_time = time.time()
    
    for epoch in range(start_epoch, start_epoch + config['epochs']):
        # Train
        model.train()
        train_loss = 0.0
        
        for videos, labels, positions in train_loader:
            videos = videos.to(device)
            labels = labels.to(device)
            positions = positions.to(device)
            
            output = model(videos)
            targets = {'class_label': labels, 'future_position': positions}
            
            total_loss, _, _ = compute_loss(output, targets)
            
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
            train_loss += total_loss.item()
        
        train_loss /= len(train_loader)
        
        # Validate
        model.eval()
        val_loss = 0.0
        val_acc = 0.0
        
        with torch.no_grad():
            for videos, labels, positions in val_loader:
                videos = videos.to(device)
                labels = labels.to(device)
                positions = positions.to(device)
                
                output = model(videos)
                targets = {'class_label': labels, 'future_position': positions}
                
                total_loss, _, _ = compute_loss(output, targets)
                acc = accuracy(output, targets)
                
                val_loss += total_loss.item()
                val_acc += acc
        
        val_loss /= len(val_loader)
        val_acc /= len(val_loader)
        
        # Update progress
        elapsed = time.time() - start_time
        update_progress(output_dir, epoch, start_epoch + config['epochs'], 
                       train_loss, val_loss, val_acc, elapsed, start_time)
        
        # Save checkpoint
        if (epoch + 1) % config['checkpoint_interval'] == 0:
            metrics = {
                'train_loss': train_loss,
                'val_loss': val_loss,
                'val_acc': val_acc,
                'elapsed_time': elapsed
            }
            save_checkpoint(model, optimizer, epoch, metrics, output_dir)
    
    # Save final model
    final_path = output_dir / 'final_model.pt'
    torch.save(model.state_dict(), final_path)
    print(f"\n✅ Training complete! Final model: {final_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train MagVIT trajectory model')
    parser.add_argument('--config', type=str, required=True, help='Path to config YAML')
    args = parser.parse_args()
    
    train(args.config)

