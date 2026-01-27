#!/usr/bin/env python3
"""
Worker 1: Phase 2 Training with MagVit Features

Trains track persistence model using MagVit visual features.
Compares performance with Phase 1 baseline (statistical features).
"""

import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
from pathlib import Path
import json
from datetime import datetime
import logging
from typing import Dict, Tuple, Optional
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Imports will be available on EC2
try:
    from experiments.track_persistence.models.modular_system import (
        ModularTrackingPipeline,
        SimpleTransformerSequence,
        PersistenceClassificationHead
    )
    from experiments.track_persistence.models.magvit_feature_extractor import (
        MagVitFeatureExtractor,
        HybridFeatureExtractor
    )
except ImportError as e:
    print(f"⚠️  Import error (expected if not on EC2): {e}")
    print("This script should be run on EC2 where all dependencies are available.")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TrackDataset(Dataset):
    """Dataset for track persistence classification."""
    
    def __init__(self, videos: np.ndarray, metadata: list):
        """
        Args:
            videos: (N, T, H, W, C) video array
            metadata: List of metadata dictionaries
        """
        self.videos = torch.from_numpy(videos).float()
        
        # Create binary labels: persistent (1) vs non-persistent (0)
        self.labels = []
        for meta in metadata:
            label = meta.get('label')
            if label == 'persistent' or label == 'mixed':
                self.labels.append(1)  # Persistent
            else:
                self.labels.append(0)  # Non-persistent (brief, noise)
        
        self.labels = torch.tensor(self.labels, dtype=torch.long)
        self.metadata = metadata
    
    def __len__(self):
        return len(self.videos)
    
    def __getitem__(self, idx):
        return {
            'video': self.videos[idx],
            'label': self.labels[idx],
            'metadata': self.metadata[idx]
        }


def load_dataset(data_dir: Path) -> Tuple[np.ndarray, list, dict]:
    """Load dataset from directory."""
    logger.info(f"Loading dataset from {data_dir}")
    
    videos = np.load(data_dir / 'videos.npy')
    
    with open(data_dir / 'metadata.json', 'r') as f:
        metadata = json.load(f)
    
    with open(data_dir / 'summary.json', 'r') as f:
        summary = json.load(f)
    
    logger.info(f"✓ Loaded {len(videos)} videos")
    logger.info(f"  Shape: {videos.shape}")
    logger.info(f"  Size: {videos.nbytes / 1024 / 1024:.2f} MB")
    
    return videos, metadata, summary


def create_phase2_model(
    magvit_checkpoint: Path,
    feature_type: str = 'magvit',
    freeze_encoder: bool = True,
    device: str = 'cuda'
) -> ModularTrackingPipeline:
    """
    Create Phase 2 model with MagVit features.
    
    Args:
        magvit_checkpoint: Path to MagVit checkpoint
        feature_type: 'magvit', 'hybrid', or 'statistical' (for comparison)
        freeze_encoder: Whether to freeze MagVit encoder
        device: Device to use
    
    Returns:
        ModularTrackingPipeline
    """
    logger.info(f"Creating Phase 2 model (feature_type={feature_type})")
    
    if feature_type == 'magvit':
        # Pure MagVit features
        feature_extractor = MagVitFeatureExtractor(
            magvit_checkpoint=str(magvit_checkpoint),
            output_dim=256,
            freeze_encoder=freeze_encoder,
            device=device
        )
        logger.info("✓ Using MagVit visual features")
        
    elif feature_type == 'hybrid':
        # Statistical + MagVit features
        feature_extractor = HybridFeatureExtractor(
            magvit_checkpoint=str(magvit_checkpoint),
            stat_dim=16,
            visual_dim=256,
            freeze_encoder=freeze_encoder,
            device=device
        )
        logger.info("✓ Using hybrid (statistical + visual) features")
        
    elif feature_type == 'statistical':
        # Baseline statistical features (for comparison)
        from experiments.track_persistence.models.modular_system import SimpleStatisticalExtractor
        feature_extractor = SimpleStatisticalExtractor(output_dim=16)
        logger.info("✓ Using statistical features (baseline)")
        
    else:
        raise ValueError(f"Unknown feature_type: {feature_type}")
    
    # Sequence model (same as Phase 1)
    sequence_model = SimpleTransformerSequence(
        input_dim=feature_extractor.output_dim,
        hidden_dim=256,
        num_layers=4,
        num_heads=8,
        dropout=0.1
    )
    
    # Task head (same as Phase 1)
    task_head = PersistenceClassificationHead(
        input_dim=256,
        hidden_dim=128,
        num_classes=2
    )
    
    # Create pipeline
    pipeline = ModularTrackingPipeline(
        feature_extractor=feature_extractor,
        sequence_model=sequence_model,
        task_head=task_head
    ).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in pipeline.parameters())
    trainable_params = sum(p.numel() for p in pipeline.parameters() if p.requires_grad)
    
    logger.info(f"Model created:")
    logger.info(f"  Total parameters: {total_params:,}")
    logger.info(f"  Trainable parameters: {trainable_params:,}")
    logger.info(f"  Frozen parameters: {total_params - trainable_params:,}")
    
    return pipeline


def train_epoch(
    model: ModularTrackingPipeline,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: str
) -> Dict[str, float]:
    """Train for one epoch."""
    model.train()
    
    total_loss = 0
    correct = 0
    total = 0
    
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    
    for batch in tqdm(dataloader, desc="Training"):
        videos = batch['video'].to(device)
        labels = batch['label'].to(device)
        
        # Forward
        optimizer.zero_grad()
        logits = model(videos)
        loss = criterion(logits, labels)
        
        # Backward
        loss.backward()
        optimizer.step()
        
        # Metrics
        total_loss += loss.item()
        
        preds = torch.argmax(logits, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        
        # For F1 calculation
        true_positives += ((preds == 1) & (labels == 1)).sum().item()
        false_positives += ((preds == 1) & (labels == 0)).sum().item()
        false_negatives += ((preds == 0) & (labels == 1)).sum().item()
    
    # Calculate metrics
    accuracy = correct / total
    precision = true_positives / (true_positives + false_positives + 1e-10)
    recall = true_positives / (true_positives + false_negatives + 1e-10)
    f1 = 2 * precision * recall / (precision + recall + 1e-10)
    
    return {
        'loss': total_loss / len(dataloader),
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


def validate(
    model: ModularTrackingPipeline,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: str
) -> Dict[str, float]:
    """Validate model."""
    model.eval()
    
    total_loss = 0
    correct = 0
    total = 0
    
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validating"):
            videos = batch['video'].to(device)
            labels = batch['label'].to(device)
            
            # Forward
            logits = model(videos)
            loss = criterion(logits, labels)
            
            # Metrics
            total_loss += loss.item()
            
            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
            true_positives += ((preds == 1) & (labels == 1)).sum().item()
            false_positives += ((preds == 1) & (labels == 0)).sum().item()
            false_negatives += ((preds == 0) & (labels == 1)).sum().item()
    
    accuracy = correct / total
    precision = true_positives / (true_positives + false_positives + 1e-10)
    recall = true_positives / (true_positives + false_negatives + 1e-10)
    f1 = 2 * precision * recall / (precision + recall + 1e-10)
    
    return {
        'loss': total_loss / len(dataloader),
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


def train_model(
    model: ModularTrackingPipeline,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    num_epochs: int,
    learning_rate: float,
    output_dir: Path,
    device: str
) -> Dict:
    """Train Phase 2 model."""
    
    logger.info(f"\n{'='*70}")
    logger.info("Starting Phase 2 Training")
    logger.info(f"{'='*70}")
    logger.info(f"Epochs: {num_epochs}")
    logger.info(f"Learning rate: {learning_rate}")
    logger.info(f"Device: {device}")
    logger.info(f"Output directory: {output_dir}")
    
    # Setup
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training history
    history = {
        'train_history': {'loss': [], 'accuracy': [], 'f1': []},
        'val_history': {'loss': [], 'accuracy': [], 'f1': []}
    }
    
    best_val_loss = float('inf')
    best_epoch = 0
    
    # Training loop
    for epoch in range(1, num_epochs + 1):
        logger.info(f"\n--- Epoch {epoch}/{num_epochs} ---")
        
        # Train
        train_metrics = train_epoch(model, train_loader, criterion, optimizer, device)
        logger.info(f"Train - Loss: {train_metrics['loss']:.4f}, "
                   f"Acc: {train_metrics['accuracy']:.4f}, "
                   f"F1: {train_metrics['f1']:.4f}")
        
        # Validate
        val_metrics = validate(model, val_loader, criterion, device)
        logger.info(f"Val   - Loss: {val_metrics['loss']:.4f}, "
                   f"Acc: {val_metrics['accuracy']:.4f}, "
                   f"F1: {val_metrics['f1']:.4f}")
        
        # Save history
        for key in ['loss', 'accuracy', 'f1']:
            history['train_history'][key].append(train_metrics[key])
            history['val_history'][key].append(val_metrics[key])
        
        # Save best model
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            best_epoch = epoch
            
            checkpoint_path = output_dir / 'checkpoints' / 'best_model.pth'
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_metrics['loss'],
                'val_accuracy': val_metrics['accuracy']
            }, checkpoint_path)
            logger.info(f"✓ Saved best model (val_loss: {val_metrics['loss']:.4f})")
        
        # Periodic checkpoints
        if epoch % 5 == 0:
            checkpoint_path = output_dir / 'checkpoints' / f'checkpoint_epoch{epoch}.pth'
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
            }, checkpoint_path)
    
    # Final evaluation on test set
    logger.info(f"\n{'='*70}")
    logger.info("Final Evaluation on Test Set")
    logger.info(f"{'='*70}")
    
    # Load best model
    checkpoint = torch.load(output_dir / 'checkpoints' / 'best_model.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    test_metrics = validate(model, test_loader, criterion, device)
    logger.info(f"Test - Loss: {test_metrics['loss']:.4f}, "
               f"Acc: {test_metrics['accuracy']:.4f}, "
               f"F1: {test_metrics['f1']:.4f}, "
               f"Precision: {test_metrics['precision']:.4f}, "
               f"Recall: {test_metrics['recall']:.4f}")
    
    # Save results
    results = {
        'num_epochs': num_epochs,
        'best_epoch': best_epoch,
        'best_val_loss': best_val_loss,
        'train_history': history['train_history'],
        'val_history': history['val_history'],
        'test_results': test_metrics,
        'timestamp': datetime.now().isoformat()
    }
    
    results_path = output_dir / 'training_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"\n✓ Results saved to {results_path}")
    logger.info(f"\n{'='*70}")
    logger.info("Phase 2 Training Complete!")
    logger.info(f"{'='*70}")
    logger.info(f"Best Epoch: {best_epoch}")
    logger.info(f"Test Accuracy: {test_metrics['accuracy']:.4f} ({test_metrics['accuracy']*100:.2f}%)")
    logger.info(f"Test F1 Score: {test_metrics['f1']:.4f}")
    
    return results


def main():
    """Main training script."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Phase 2: MagVit Training')
    parser.add_argument('--data', type=str, required=True,
                       help='Path to dataset directory')
    parser.add_argument('--magvit-checkpoint', type=str, required=True,
                       help='Path to MagVit checkpoint')
    parser.add_argument('--feature-type', type=str, default='magvit',
                       choices=['magvit', 'hybrid', 'statistical'],
                       help='Type of features to use')
    parser.add_argument('--epochs', type=int, default=30,
                       help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--freeze-encoder', action='store_true', default=True,
                       help='Freeze MagVit encoder')
    parser.add_argument('--output', type=str, default=None,
                       help='Output directory')
    
    args = parser.parse_args()
    
    # Setup
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {device}")
    
    data_dir = Path(args.data)
    magvit_checkpoint = Path(args.magvit_checkpoint)
    
    if args.output is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(f"output/train_phase2_{args.feature_type}_{timestamp}")
    else:
        output_dir = Path(args.output)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load dataset
    videos, metadata, summary = load_dataset(data_dir)
    
    # Create dataset
    full_dataset = TrackDataset(videos, metadata)
    
    # Split dataset (70% train, 15% val, 15% test)
    total_size = len(full_dataset)
    train_size = int(0.70 * total_size)
    val_size = int(0.15 * total_size)
    test_size = total_size - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    logger.info(f"Dataset split:")
    logger.info(f"  Train: {len(train_dataset)}")
    logger.info(f"  Val: {len(val_dataset)}")
    logger.info(f"  Test: {len(test_dataset)}")
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, 
                              shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                            shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size,
                             shuffle=False, num_workers=4)
    
    # Create model
    model = create_phase2_model(
        magvit_checkpoint=magvit_checkpoint,
        feature_type=args.feature_type,
        freeze_encoder=args.freeze_encoder,
        device=device
    )
    
    # Train
    results = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        output_dir=output_dir,
        device=device
    )
    
    logger.info(f"\n✅ PHASE 2 TRAINING COMPLETE")
    logger.info(f"All outputs saved to: {output_dir}")


if __name__ == '__main__':
    main()

