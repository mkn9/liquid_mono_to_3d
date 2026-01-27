"""Branch 1: Simplified I3D-like 3D CNN + Classification + Forecasting"""
import torch
import torch.nn as nn

class SimplifiedI3D(nn.Module):
    """Simplified 3D CNN mimicking I3D structure."""
    
    def __init__(self, num_classes=4, forecast_frames=4):
        super().__init__()
        self.num_classes = num_classes
        self.forecast_frames = forecast_frames
        
        # 3D CNN backbone
        self.features = nn.Sequential(
            nn.Conv3d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.MaxPool3d((1, 2, 2)),
            
            nn.Conv3d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(),
            nn.MaxPool3d((2, 2, 2)),
            
            nn.Conv3d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm3d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool3d((1, 1, 1))
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )
        
        # Forecasting head (predict 3D positions)
        self.forecaster = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, forecast_frames * 3)
        )
    
    def forward(self, x):
        # x: (B, T, C, H, W) → (B, C, T, H, W)
        x = x.permute(0, 2, 1, 3, 4)
        
        # Extract features
        feat = self.features(x)
        feat = feat.view(feat.size(0), -1)
        
        # Classification
        class_logits = self.classifier(feat)
        
        # Forecasting
        forecast_flat = self.forecaster(feat)
        forecast = forecast_flat.view(-1, self.forecast_frames, 3)
        
        return {'classification': class_logits, 'forecasting': forecast}

