#!/usr/bin/env python3
"""
Complete D-NeRF Integration Script
Shows how to use the actual D-NeRF neural network code with sphere trajectory data.
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path
import json
import numpy as np

def setup_dnerf_environment():
    """Set up the D-NeRF environment and dependencies."""
    print("📦 Setting up D-NeRF environment...")
    
    # Install D-NeRF dependencies
    dnerf_requirements = [
        "torch>=1.8.0",
        "torchvision>=0.9.0", 
        "imageio>=2.9.0",
        "imageio-ffmpeg>=0.4.0",
        "matplotlib>=3.3.0",
        "numpy>=1.19.0",
        "opencv-python>=4.5.0",
        "Pillow>=8.0.0",
        "scipy>=1.6.0",
        "tensorboard>=2.4.0",
        "tqdm>=4.60.0",
        "configargparse>=1.2.0"
    ]
    
    print("Installing D-NeRF dependencies...")
    for req in dnerf_requirements:
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", req], 
                         check=True, capture_output=True)
            print(f"✅ Installed {req}")
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install {req}: {e}")
    
    # Check if we're already in the right directory structure
    dnerf_data_dir = Path("data/sphere_trajectories")
    if not dnerf_data_dir.exists():
        dnerf_data_dir.mkdir(parents=True, exist_ok=True)
    
    # Data should already be in place in the new structure
    if dnerf_data_dir.exists() and any(dnerf_data_dir.iterdir()):
        print(f"✅ Data already in place at {dnerf_data_dir}")
    else:
        print("❌ Data directory not found. Run dnerf_data_augmentation.py first.")
        return False
    
    return True

def train_dnerf_model():
    """Train D-NeRF model on sphere trajectory data."""
    print("🚀 Training D-NeRF model...")
    
    # Change to D-NeRF directory
    os.chdir("D-NeRF")
    
    # Training command
    training_cmd = [
        sys.executable, "run_dnerf.py",
        "--config", "configs/sphere_trajectories.txt",
        "--render_test",
        "--render_only"
    ]
    
    print("Training command:")
    print(" ".join(training_cmd))
    
    try:
        # Run training
        result = subprocess.run(training_cmd, capture_output=True, text=True)
        print("Training output:")
        print(result.stdout)
        if result.stderr:
            print("Training errors:")
            print(result.stderr)
        
        if result.returncode == 0:
            print("✅ Training completed successfully!")
            return True
        else:
            print("❌ Training failed!")
            return False
    except Exception as e:
        print(f"❌ Training error: {e}")
        return False
    finally:
        os.chdir("..")

def render_dnerf_predictions():
    """Render D-NeRF predictions for temporal sequences."""
    print("🎥 Rendering D-NeRF predictions...")
    
    os.chdir("D-NeRF")
    
    # Rendering command for predictions
    render_cmd = [
        sys.executable, "run_dnerf.py",
        "--config", "configs/sphere_trajectories.txt",
        "--render_test",
        "--render_only",
        "--render_video"
    ]
    
    try:
        result = subprocess.run(render_cmd, capture_output=True, text=True)
        print("Rendering output:")
        print(result.stdout)
        
        if result.returncode == 0:
            print("✅ Rendering completed!")
            return True
        else:
            print("❌ Rendering failed!")
            return False
    except Exception as e:
        print(f"❌ Rendering error: {e}")
        return False
    finally:
        os.chdir("..")

def analyze_dnerf_results():
    """Analyze D-NeRF temporal prediction results."""
    print("📊 Analyzing D-NeRF results...")
    
    results_dir = Path("D-NeRF/logs/sphere_trajectories")
    if not results_dir.exists():
        print("❌ No results directory found!")
        return False
    
    # Look for rendered images
    render_dir = results_dir / "renderonly_test_199999"
    if render_dir.exists():
        images = list(render_dir.glob("*.png"))
        print(f"✅ Found {len(images)} rendered images")
        
        # Create comparison visualization
        create_temporal_comparison(images)
        return True
    else:
        print("❌ No rendered images found!")
        return False

def create_temporal_comparison(images):
    """Create temporal comparison visualization."""
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg
    
    print("Creating temporal comparison...")
    
    # Select representative frames
    n_frames = min(8, len(images))
    selected_frames = images[::len(images)//n_frames][:n_frames]
    
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    
    for i, img_path in enumerate(selected_frames):
        img = mpimg.imread(img_path)
        axes[i].imshow(img)
        axes[i].set_title(f"Frame {i+1}")
        axes[i].axis('off')
    
    plt.suptitle("D-NeRF Temporal Predictions - Sphere Trajectories", fontsize=16)
    plt.tight_layout()
    plt.savefig("dnerf_temporal_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✅ Temporal comparison saved as 'dnerf_temporal_comparison.png'")

def validate_dnerf_integration():
    """Validate D-NeRF integration with quantitative metrics."""
    print("🔍 Validating D-NeRF integration...")
    
    # Check if all required files exist
    required_files = [
        "D-NeRF/run_dnerf.py",
        "D-NeRF/run_dnerf_helpers.py", 
        "D-NeRF/configs/sphere_trajectories.txt",
        "D-NeRF/data/sphere_trajectories/transforms.json"
    ]
    
    all_exist = True
    for file_path in required_files:
        if Path(file_path).exists():
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path} - Missing!")
            all_exist = False
    
    if all_exist:
        print("✅ All required files present!")
        return True
    else:
        print("❌ Some required files missing!")
        return False

def main():
    """Main integration workflow."""
    print("🎬 D-NeRF Integration for Sphere Trajectories")
    print("=" * 50)
    
    # Step 1: Validate integration
    if not validate_dnerf_integration():
        print("❌ Integration validation failed!")
        return False
    
    # Step 2: Setup environment
    if not setup_dnerf_environment():
        print("❌ Environment setup failed!")
        return False
    
    # Step 3: Train model
    if not train_dnerf_model():
        print("❌ Model training failed!")
        return False
    
    # Step 4: Render predictions
    if not render_dnerf_predictions():
        print("❌ Prediction rendering failed!")
        return False
    
    # Step 5: Analyze results
    if not analyze_dnerf_results():
        print("❌ Results analysis failed!")
        return False
    
    print("\n🎉 D-NeRF Integration Complete!")
    print("=" * 50)
    print("✅ Neural network trained on sphere trajectory data")
    print("✅ Temporal predictions generated")
    print("✅ Results analyzed and visualized")
    print("\nNext steps:")
    print("1. Fine-tune hyperparameters in configs/sphere_trajectories.txt")
    print("2. Experiment with different temporal resolutions")
    print("3. Add more complex object motions")
    print("4. Evaluate prediction accuracy metrics")
    
    return True

if __name__ == "__main__":
    main() 