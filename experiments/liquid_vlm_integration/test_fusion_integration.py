"""
Worker 3: Test Liquid Fusion with Real 2D+3D Features
"""
import torch
import numpy as np
from pathlib import Path
import sys

# Import Liquid components
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "experiments/trajectory_video_understanding"))
from vision_language_integration.liquid_e2e_pipeline import LiquidE2EPipeline

# Import real data generators
sys.path.insert(0, str(Path(__file__).parent))
from extract_2d_features import load_trajectory_video, extract_2d_features_batch

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from simple_3d_tracker import generate_synthetic_tracks, triangulate_tracks, set_up_cameras


def test_real_data_fusion():
    """Test fusion with real MagVIT 2D + real triangulated 3D."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Get real 3D trajectory
    np.random.seed(42)
    sensor1_track, sensor2_track, points_3d_true = generate_synthetic_tracks()
    P1, P2, _, _ = set_up_cameras()
    points_3d_reconstructed = triangulate_tracks(sensor1_track, sensor2_track, P1, P2)
    
    # Add realistic noise
    noise = np.random.randn(*points_3d_reconstructed.shape) * 0.01
    noisy_3d = torch.from_numpy(points_3d_reconstructed + noise).float().unsqueeze(0).to(device)
    
    # Get real 2D features from MagVIT (already on correct device)
    features_2d = extract_2d_features_batch(num_samples=1)  # (1, 512)
    features_2d = features_2d.to(device)
    features_2d.requires_grad_(True)
    
    # Test Liquid E2E pipeline
    pipeline = LiquidE2EPipeline().to(device)
    llm_embeddings = pipeline(noisy_3d, features_2d)
    
    return {
        "llm_embeddings": llm_embeddings,
        "features_2d": features_2d,
        "features_3d": noisy_3d
    }


def compare_fusion_methods():
    """Compare Liquid Fusion vs static linear fusion."""
    from vision_language_integration.dual_visual_adapter import LiquidDualModalFusion
    import torch.nn as nn
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Get real data
    features_2d = extract_2d_features_batch(num_samples=1).to(device)
    features_3d = torch.randn(1, 256).to(device)  # Simulated 3D features
    
    # Liquid Fusion
    liquid_fusion = LiquidDualModalFusion().to(device)
    liquid_output = liquid_fusion(features_2d, features_3d, reset_state=True)
    
    # Static baseline (simple concatenation + linear)
    static_fusion = nn.Sequential(
        nn.Linear(512 + 256, 4096)
    ).to(device)
    static_input = torch.cat([features_2d, features_3d], dim=-1)
    static_output = static_fusion(static_input)
    
    return {
        "liquid_output": liquid_output,
        "static_output": static_output
    }
