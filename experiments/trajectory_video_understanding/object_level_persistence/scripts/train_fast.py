"""
Fast training script for quick attention validation.

Train minimal transformer on small dataset subset to quickly verify:
1. Transformer learns to classify persistent vs transient objects
2. Attention weights differentiate between object types
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
import sys
import json
from tqdm import tqdm
import argparse

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.fast_dataset import FastObjectDataset, collate_fn
from src.fast_object_transformer import FastObjectTransformer


def train_epoch(model, dataloader, optimizer, criterion, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for tokens, labels, masks, metadata in tqdm(dataloader, desc="Training"):
        tokens = tokens.to(device)
        labels = labels.to(device)
        masks = masks.to(device)
        
        # Skip if no valid tokens
        if masks.sum().item() == 0:
            continue
        
        # Forward
        optimizer.zero_grad()
        logits, _ = model(tokens, src_key_padding_mask=~masks, return_attention=False)
        
        # Compute loss only on valid tokens
        loss = criterion(logits[masks], labels[masks])
        
        # Check for NaN and skip if found
        if torch.isnan(loss):
            print(f"Warning: NaN loss detected, skipping batch")
            continue
        
        # Backward
        loss.backward()
        
        # Gradient clipping to prevent NaN
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # Stats
        total_loss += loss.item()
        predictions = torch.argmax(logits, dim=-1)
        correct += ((predictions == labels) & masks).sum().item()
        total += masks.sum().item()
    
    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total if total > 0 else 0
    
    return avg_loss, accuracy


def validate(model, dataloader, criterion, device):
    """Validate model."""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    persistent_correct = 0
    persistent_total = 0
    transient_correct = 0
    transient_total = 0
    
    with torch.no_grad():
        for tokens, labels, masks, metadata in tqdm(dataloader, desc="Validation"):
            tokens = tokens.to(device)
            labels = labels.to(device)
            masks = masks.to(device)
            
            # Skip if no valid tokens
            if masks.sum().item() == 0:
                continue
            
            # Forward
            logits, _ = model(tokens, src_key_padding_mask=~masks, return_attention=False)
            
            # Loss
            loss = criterion(logits[masks], labels[masks])
            
            # Skip NaN losses
            if torch.isnan(loss):
                continue
                
            total_loss += loss.item()
            
            # Accuracy
            predictions = torch.argmax(logits, dim=-1)
            correct += ((predictions == labels) & masks).sum().item()
            total += masks.sum().item()
            
            # Per-class accuracy
            persistent_mask = (labels == 0) & masks
            transient_mask = (labels == 1) & masks
            
            persistent_correct += ((predictions == labels) & persistent_mask).sum().item()
            persistent_total += persistent_mask.sum().item()
            
            transient_correct += ((predictions == labels) & transient_mask).sum().item()
            transient_total += transient_mask.sum().item()
    
    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total if total > 0 else 0
    persistent_acc = persistent_correct / persistent_total if persistent_total > 0 else 0
    transient_acc = transient_correct / transient_total if transient_total > 0 else 0
    
    return avg_loss, accuracy, persistent_acc, transient_acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, required=True, help='Path to dataset')
    parser.add_argument('--output_dir', type=str, default='results/fast_training', help='Output directory')
    parser.add_argument('--max_samples', type=int, default=500, help='Max training samples')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    
    print(f"Training on {args.device}")
    print(f"Max samples: {args.max_samples}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create datasets
    print("Loading datasets...")
    train_dataset = FastObjectDataset(
        data_root=args.data_root,
        max_samples=args.max_samples,
        split='train'
    )
    
    val_dataset = FastObjectDataset(
        data_root=args.data_root,
        max_samples=args.max_samples // 5,  # Smaller validation set
        split='val'
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0  # Avoid multiprocessing issues
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0
    )
    
    # Create model
    print("Creating model...")
    model = FastObjectTransformer(
        feature_dim=256,
        num_heads=8,
        num_layers=2,
        dropout=0.1
    ).to(args.device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # Training loop
    print("Starting training...")
    best_val_acc = 0
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'val_persistent_acc': [],
        'val_transient_acc': []
    }
    
    for epoch in range(args.epochs):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch+1}/{args.epochs}")
        print(f"{'='*60}")
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, args.device)
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        
        # Validate
        val_loss, val_acc, persistent_acc, transient_acc = validate(model, val_loader, criterion, args.device)
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
        print(f"  Persistent Acc: {persistent_acc:.4f}")
        print(f"  Transient Acc: {transient_acc:.4f}")
        
        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_persistent_acc'].append(persistent_acc)
        history['val_transient_acc'].append(transient_acc)
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'persistent_acc': persistent_acc,
                'transient_acc': transient_acc
            }, output_dir / 'best_model.pt')
            print(f"✅ Saved best model (val_acc={val_acc:.4f})")
    
    # Save final model and history
    torch.save({
        'epoch': args.epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, output_dir / 'final_model.pt')
    
    with open(output_dir / 'training_history.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"Training complete!")
    print(f"Best validation accuracy: {best_val_acc:.4f}")
    print(f"Models saved to {output_dir}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()

