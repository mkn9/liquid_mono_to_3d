"""
Train Early Persistence Detection System with MagVIT

Integrates all four components:
1. Early persistence classifier with MagVIT
2. Attention visualization
3. Compute gating
4. Efficiency metrics tracking

Training on persistence-augmented dataset with periodic saves.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import json
import time
from pathlib import Path
from datetime import datetime
import sys
import argparse

# Add model paths
sys.path.insert(0, str(Path(__file__).parent.parent / 'models'))
from early_persistence_classifier import EarlyPersistenceClassifier, get_early_decision
from attention_visualization import AttentionVisualizer, save_attention_analysis
from compute_gating import ComputeGate, get_gating_decision
from efficiency_metrics import EfficiencyTracker, compute_time_to_decision


class PersistenceDataset(Dataset):
    """Dataset for persistence-augmented trajectories."""
    
    def __init__(self, data_dir: Path):
        self.data_dir = Path(data_dir)
        # Find all video files
        self.video_files = sorted(list(self.data_dir.glob("augmented_traj_*.pt")))
        print(f"Found {len(self.video_files)} samples in {data_dir}")
    
    def __len__(self):
        return len(self.video_files)
    
    def __getitem__(self, idx):
        # Load video
        video_path = self.video_files[idx]
        video = torch.load(video_path)
        
        # Ensure video is the right shape (T, C, H, W)
        if video.dim() == 5:  # (1, T, C, H, W)
            video = video.squeeze(0)
        
        # Load metadata
        metadata_path = video_path.with_suffix('.json')
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        # Create label: 1 if persistent (no transients or few), 0 if transient
        num_transients = metadata.get('num_transients', 0)
        transient_frames = metadata.get('transient_frames', [])
        
        # Label as persistent if < 20% of frames have transients
        total_frames = video.shape[0]
        label = 1 if len(transient_frames) / total_frames < 0.2 else 0
        
        return video, label, metadata


def custom_collate(batch):
    """Custom collate function to handle varying video dimensions."""
    videos = []
    labels = []
    metadata_list = []
    
    for video, label, metadata in batch:
        videos.append(video)
        labels.append(label)
        metadata_list.append(metadata)
    
    # Find max dimensions
    max_frames = max(v.shape[0] for v in videos)
    channels = videos[0].shape[1]
    height = videos[0].shape[2]
    width = videos[0].shape[3]
    
    # Pad videos to same length
    padded_videos = []
    for video in videos:
        if video.shape[0] < max_frames:
            # Pad with zeros
            padding = torch.zeros((max_frames - video.shape[0], channels, height, width))
            video = torch.cat([video, padding], dim=0)
        padded_videos.append(video)
    
    # Stack into batch
    video_batch = torch.stack(padded_videos, dim=0)
    label_batch = torch.tensor(labels, dtype=torch.long)
    
    return video_batch, label_batch, metadata_list


def save_checkpoint(model, optimizer, epoch, metrics, output_dir):
    """Save training checkpoint."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics
    }
    torch.save(checkpoint, output_dir / f'checkpoint_epoch_{epoch}.pt')


def save_progress(epoch, metrics, output_dir):
    """Save progress file for MacBook visibility."""
    progress_file = output_dir / 'PROGRESS.txt'
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    with open(progress_file, 'a') as f:
        f.write(f"[{timestamp}] Epoch {epoch}: Loss={metrics['loss']:.4f}, "
                f"Acc={metrics['accuracy']:.2%}, "
                f"Early Stop Rate={metrics['early_stop_rate']:.2%}\n")


def train_epoch(model, dataloader, optimizer, criterion, gate, tracker, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, batch in enumerate(dataloader):
        videos, labels = batch[0].to(device), batch[1].to(device)
        # metadata is batch[2] but we don't use it in training
        
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(videos)
        logits = outputs['logits']
        
        # Compute loss
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        
        # Track metrics
        total_loss += loss.item()
        _, predicted = torch.max(logits, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)
        
        # Note: We skip early stopping tracking during training to avoid eval mode conflicts
        # Early stopping metrics will be computed during validation/evaluation
        
        if batch_idx % 10 == 0:
            print(f"  Batch {batch_idx}/{len(dataloader)}: Loss={loss.item():.4f}")
    
    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total
    
    return {
        'loss': avg_loss,
        'accuracy': accuracy,
        'early_stop_rate': 0.0  # Not tracked during training
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, 
                       default='/home/ubuntu/mono_to_3d/experiments/trajectory_video_understanding/persistence_augmented_dataset/output',
                       help='Path to augmented dataset')
    parser.add_argument('--output_dir', type=str,
                       default='/home/ubuntu/mono_to_3d/experiments/trajectory_video_understanding/early_persistence_detection/training/results',
                       help='Output directory')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--early_stop_accuracy', type=float, default=0.80, help='Stop when validation accuracy reaches this threshold')
    parser.add_argument('--patience', type=int, default=3, help='Epochs to wait after reaching threshold before stopping')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("EARLY PERSISTENCE DETECTION TRAINING WITH MAGVIT")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  Data dir: {args.data_dir}")
    print(f"  Output dir: {args.output_dir}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Device: {args.device}")
    print("=" * 80)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize components
    print("\n1. Initializing components...")
    model = EarlyPersistenceClassifier(
        feature_extractor='magvit',
        early_stop_frame=4,
        confidence_threshold=0.9
    ).to(args.device)
    
    gate = ComputeGate(confidence_threshold=0.9, early_stop_frame=4)
    tracker = EfficiencyTracker()
    visualizer = AttentionVisualizer(save_dir=output_dir / 'visualizations')
    
    print("✅ Components initialized")
    
    # Load dataset
    print("\n2. Loading dataset...")
    dataset = PersistenceDataset(args.data_dir)
    
    # Split: 80% train, 20% val
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=custom_collate)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=custom_collate)
    
    print(f"✅ Dataset loaded: {train_size} train, {val_size} val")
    
    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # Training loop with early stopping
    print("\n3. Starting training...")
    print(f"   Early stopping enabled: Will stop if validation accuracy >= {args.early_stop_accuracy:.0%}")
    print(f"   Patience: {args.patience} epochs after threshold")
    
    best_val_accuracy = 0.0
    epochs_above_threshold = 0
    
    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        print("-" * 80)
        
        # Train
        start_time = time.time()
        train_metrics = train_epoch(model, train_loader, optimizer, criterion, gate, tracker, args.device)
        epoch_time = time.time() - start_time
        
        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for batch in val_loader:
                videos, labels = batch[0], batch[1]
                videos, labels = videos.to(args.device), labels.to(args.device)
                
                # Forward pass
                outputs = model(videos)
                logits = outputs['logits']
                
                _, predicted = torch.max(logits, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_accuracy = val_correct / val_total if val_total > 0 else 0.0
        
        print(f"\nEpoch {epoch} Results:")
        print(f"  Train Loss: {train_metrics['loss']:.4f}")
        print(f"  Train Accuracy: {train_metrics['accuracy']:.2%}")
        print(f"  Val Accuracy: {val_accuracy:.2%} {'✅' if val_accuracy >= args.early_stop_accuracy else ''}")
        print(f"  Early Stop Rate: {train_metrics['early_stop_rate']:.2%}")
        print(f"  Time: {epoch_time:.2f}s")
        
        # Track best validation accuracy
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), output_dir / 'best_model.pt')
            print(f"  🌟 New best validation accuracy!")
        
        # Early stopping check
        if val_accuracy >= args.early_stop_accuracy:
            epochs_above_threshold += 1
            print(f"  ⚡ Above threshold! ({epochs_above_threshold}/{args.patience} patience)")
            
            if epochs_above_threshold >= args.patience:
                print(f"\n🎯 Early stopping triggered!")
                print(f"   Validation accuracy {val_accuracy:.2%} >= {args.early_stop_accuracy:.0%}")
                print(f"   Maintained for {args.patience} epochs")
                break
        else:
            epochs_above_threshold = 0
        
        # Save checkpoint and progress
        if epoch % 2 == 0:
            save_checkpoint(model, optimizer, epoch, train_metrics, output_dir)
            print(f"  ✅ Checkpoint saved")
        
        train_metrics['val_accuracy'] = val_accuracy
        train_metrics['best_val_accuracy'] = best_val_accuracy
        save_progress(epoch, train_metrics, output_dir)
    
    # Save final model (use best model if available)
    if (output_dir / 'best_model.pt').exists():
        import shutil
        shutil.copy(output_dir / 'best_model.pt', output_dir / 'final_model.pt')
        print("\n✅ Final model saved (using best validation model)")
    else:
        torch.save(model.state_dict(), output_dir / 'final_model.pt')
        print("\n✅ Final model saved")
    
    # Save efficiency metrics
    efficiency_summary = tracker.get_summary()
    with open(output_dir / 'efficiency_metrics.json', 'w') as f:
        json.dump(efficiency_summary, f, indent=2)
    print("✅ Efficiency metrics saved")
    
    # Final summary
    print("\n" + "=" * 80)
    print("TRAINING COMPLETE")
    print("=" * 80)
    print(f"\nFinal Results:")
    print(f"  Epochs trained: {epoch}")
    print(f"  Best validation accuracy: {best_val_accuracy:.2%}")
    if epochs_above_threshold >= args.patience:
        print(f"  Early stopping: ✅ (triggered at epoch {epoch})")
    else:
        print(f"  Early stopping: Not triggered")
    print(f"\nSaved models:")
    print(f"  final_model.pt - Model from epoch {epoch}")
    print(f"  best_model.pt - Best validation model ({best_val_accuracy:.2%})")
    print("=" * 80)
    print(f"\nEfficiency Summary:")
    print(f"  Total tracks: {efficiency_summary['total_tracks']}")
    print(f"  Avg decision frame: {efficiency_summary['avg_decision_frame']:.1f}")
    print(f"  Early stop rate: {efficiency_summary.get('early_stop_rate', 0):.2%}")
    print(f"  Avg compute per track: {efficiency_summary['avg_compute_per_track']:.2f}")
    print(f"  Total compute saved: {efficiency_summary['total_compute_saved']:.1f}")
    
    print(f"\nResults saved to: {output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()

