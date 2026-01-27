"""
Simple 3D CNN Baseline for Trajectory Classification

HONEST IMPLEMENTATION:
- Basic 3D convolutional neural network (NOT I3D, NOT SlowFast)
- No MAGVIT (no VQ-VAE, no codebook)
- No CLIP
- Template-based text generation (NOT real LLM)

This is a proof-of-concept baseline to establish performance floor.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Simple3DCNNClassifier(nn.Module):
    """Simple 3D CNN for video classification.
    
    This is an HONEST simple baseline:
    - Uses basic Conv3d layers (not pretrained I3D/SlowFast)
    - No fancy video transformations
    - Straightforward architecture for trajectory classification
    """
    
    def __init__(self, num_classes=4, input_channels=3):
        super().__init__()
        
        # Simple 3D CNN architecture
        self.conv1 = nn.Conv3d(input_channels, 32, kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1))
        self.bn1 = nn.BatchNorm3d(32)
        
        self.conv2 = nn.Conv3d(32, 64, kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1))
        self.bn2 = nn.BatchNorm3d(64)
        
        self.conv3 = nn.Conv3d(64, 128, kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1))
        self.bn3 = nn.BatchNorm3d(128)
        
        self.conv4 = nn.Conv3d(128, 256, kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1))
        self.bn4 = nn.BatchNorm3d(256)
        
        # Global average pooling + classifier
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Linear(256, num_classes)
        
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        """Forward pass.
        
        Args:
            x: Video tensor of shape (B, T, C, H, W)
        
        Returns:
            logits: Class logits of shape (B, num_classes)
        """
        # Reshape from (B, T, C, H, W) to (B, C, T, H, W) for Conv3d
        x = x.permute(0, 2, 1, 3, 4)
        
        # Conv blocks
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        
        # Global pooling
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        
        # Classifier
        x = self.dropout(x)
        logits = self.fc(x)
        
        return logits


def generate_description_from_template(trajectory_class, params):
    """Generate trajectory description using TEMPLATE (not real LLM).
    
    IMPORTANT: This is a placeholder template-based function.
    This is NOT using GPT-4, Mistral, Phi-2, or any real LLM.
    Real LLM integration would require API keys and actual model calls.
    
    Args:
        trajectory_class: Integer class (0=linear, 1=circular, 2=helical, 3=parabolic)
        params: Dictionary of trajectory parameters
    
    Returns:
        description: Template-based human-readable description
    """
    templates = {
        0: "A linear trajectory moving in a straight line from {start} to {end}.",
        1: "A circular trajectory with center at {center} and radius {radius} units.",
        2: "A helical trajectory spiraling around axis at {center} with radius {radius} units, ascending {height} units vertically.",
        3: "A parabolic trajectory following a curved path with initial velocity {velocity} and acceleration {acceleration}."
    }
    
    # Get template
    template = templates.get(trajectory_class, "Unknown trajectory type.")
    
    # Fill in parameters (basic string formatting, not LLM)
    try:
        description = template.format(**params)
    except KeyError:
        description = template  # Return template if params don't match
    
    return description


def generate_equation_from_template(trajectory_class, params):
    """Generate symbolic equation using TEMPLATE (not real LLM).
    
    IMPORTANT: This is a placeholder template-based function.
    This is NOT using GPT-4, Mistral, or any real LLM.
    Real LLM integration would require API keys and actual model calls.
    
    Args:
        trajectory_class: Integer class (0=linear, 1=circular, 2=helical, 3=parabolic)
        params: Dictionary of trajectory parameters
    
    Returns:
        equation: Template-based symbolic equation string
    """
    templates = {
        0: "r(t) = ({start_x} + {vx}*t, {start_y} + {vy}*t, {start_z} + {vz}*t)",
        1: "r(t) = ({cx} + {r}*cos(ωt), {cy} + {r}*sin(ωt), {cz})",
        2: "r(t) = ({cx} + {r}*cos(ωt), {cy} + {r}*sin(ωt), {cz} + {vz}*t)",
        3: "r(t) = ({x0} + {vx0}*t, {y0} + {vy0}*t, {z0} + {vz0}*t - 0.5*g*t²)"
    }
    
    # Get template
    template = templates.get(trajectory_class, "r(t) = unknown")
    
    # Fill in parameters (basic string formatting, not LLM)
    try:
        equation = template.format(**params)
    except KeyError:
        equation = template  # Return template if params don't match
    
    return equation


if __name__ == "__main__":
    # Quick sanity check
    model = Simple3DCNNClassifier(num_classes=4)
    x = torch.randn(2, 16, 3, 64, 64)  # B, T, C, H, W
    y = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test template functions
    desc = generate_description_from_template(0, {"start": [0, 0, 1], "end": [1, 1, 2]})
    print(f"Description: {desc}")
    
    eq = generate_equation_from_template(1, {"cx": 0.1, "cy": 0.1, "cz": 1.0, "r": 0.5})
    print(f"Equation: {eq}")
