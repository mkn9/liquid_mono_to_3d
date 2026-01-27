"""Branch 2: SlowFast-like dual-pathway CNN"""
import torch
import torch.nn as nn

class SimplifiedSlowFast(nn.Module):
    def __init__(self, num_classes=4, forecast_frames=4):
        super().__init__()
        # Slow pathway (all frames)
        self.slow = nn.Sequential(
            nn.Conv3d(3, 64, 3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.MaxPool3d((1, 2, 2)),
            nn.Conv3d(64, 128, 3, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool3d((1, 1, 1))
        )
        # Fast pathway (fewer frames, lighter)
        self.fast = nn.Sequential(
            nn.Conv3d(3, 32, 3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.MaxPool3d((1, 2, 2)),
            nn.Conv3d(32, 64, 3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool3d((1, 1, 1))
        )
        self.classifier = nn.Sequential(
            nn.Linear(192, 128),  # 128 + 64 = 192
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )
        self.forecaster = nn.Sequential(
            nn.Linear(192, 256),
            nn.ReLU(),
            nn.Linear(256, forecast_frames * 3)
        )
    
    def forward(self, x):
        x = x.permute(0, 2, 1, 3, 4)  # (B,T,C,H,W) → (B,C,T,H,W)
        slow_feat = self.slow(x).view(x.size(0), -1)
        fast_feat = self.fast(x[:, :, ::4]).view(x.size(0), -1)  # Every 4th frame
        feat = torch.cat([slow_feat, fast_feat], dim=1)
        return {
            'classification': self.classifier(feat),
            'forecasting': self.forecaster(feat).view(-1, 4, 3)
        }

