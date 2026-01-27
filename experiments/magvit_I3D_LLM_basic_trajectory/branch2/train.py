#!/usr/bin/env python3
"""Branch 2 Training: SlowFast-like"""
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import json
import numpy as np
from datetime import datetime

# Import model
sys.path.insert(0, str(Path(__file__).parent))
from simple_model import SimplifiedSlowFast

sys.path.insert(0, str(Path(__file__).parent.parent))
from evaluation_metrics import classification_accuracy, forecasting_mae

def train_branch2():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Branch 2 - Using device: {device}")
    
    # Load dataset
    data_path = Path(__file__).parent.parent / "results" / "20260121_0436_full_dataset.npz"
    data = np.load(data_path)
    
    videos = torch.from_numpy(data['videos']).float()
    labels = torch.from_numpy(data['labels']).long()
    trajectories = torch.from_numpy(data['trajectory_3d']).float()
    
    n_train = 960
    train_videos, val_videos = videos[:n_train], videos[n_train:]
    train_labels, val_labels = labels[:n_train], labels[n_train:]
    train_traj, val_traj = trajectories[:n_train], trajectories[n_train:]
    
    model = SimplifiedSlowFast(num_classes=4, forecast_frames=4).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)
    
    class_criterion = nn.CrossEntropyLoss()
    forecast_criterion = nn.MSELoss()
    
    best_val_acc = 0
    for epoch in range(30):
        model.train()
        train_loss = 0
        correct = 0
        
        batch_size = 8  # Reduced for memory
        for i in range(0, n_train, batch_size):
            batch_videos = train_videos[i:i+batch_size].to(device)
            batch_labels = train_labels[i:i+batch_size].to(device)
            batch_traj = train_traj[i:i+batch_size, 12:16].to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_videos)
            
            loss_class = class_criterion(outputs['classification'], batch_labels)
            loss_forecast = forecast_criterion(outputs['forecasting'], batch_traj)
            loss = loss_class + 0.5 * loss_forecast
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs['classification'].max(1)
            correct += predicted.eq(batch_labels).sum().item()
        
        train_acc = correct / n_train
        scheduler.step()
        
        model.eval()
        with torch.no_grad():
            val_videos_gpu = val_videos.to(device)
            outputs = model(val_videos_gpu)
            val_acc = classification_accuracy(
                outputs['classification'].argmax(1).cpu(),
                val_labels
            )
            val_mae = forecasting_mae(
                outputs['forecasting'].cpu().numpy(),
                val_traj[:, 12:16].numpy()
            )
        
        print(f"Epoch {epoch+1}/30 | Train Acc: {train_acc:.3f} | Val Acc: {val_acc:.3f} | MAE: {val_mae:.3f}")
        
        status = {
            "branch": "slowfast-magvit-gpt4",
            "epoch": epoch + 1,
            "train_acc": float(train_acc),
            "val_acc": float(val_acc),
            "val_mae": float(val_mae),
            "timestamp": datetime.now().isoformat()
        }
        status_path = Path(__file__).parent / "status" / "status.json"
        status_path.parent.mkdir(exist_ok=True)
        with open(status_path, 'w') as f:
            json.dump(status, f, indent=2)
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 
                      Path(__file__).parent / "results" / "best_model.pth")
    
    print(f"Branch 2 Training complete! Best Val Acc: {best_val_acc:.3f}")

if __name__ == "__main__":
    train_branch2()
