#!/bin/bash

# NeRF Helper Script
# This script activates the correct conda environment and runs NeRF commands

echo "🚀 NeRF Helper Script"
echo "====================="

# Source conda
source ~/miniconda3/etc/profile.d/conda.sh

# Activate NeRF environment
echo "Activating NeRF environment..."
conda activate nerf_env

# Change to NeRF directory
cd /home/ubuntu/mono_to_3d/neural_radiance_fields/open_source_implementations/nerf-pytorch

echo "Current directory: $(pwd)"
echo "Python version: $(python --version)"
echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo ""

# Check if arguments were provided
if [ $# -eq 0 ]; then
    echo "Usage examples:"
    echo "  ./run_nerf.sh quick          # Quick test (10 iterations)"
    echo "  ./run_nerf.sh train          # Full training (200k iterations)"
    echo "  ./run_nerf.sh render         # Render test views"
    echo "  ./run_nerf.sh test           # Test data loading only"
    echo ""
    echo "Available datasets: lego, fern, chair, drums, hotdog, materials, mic, ship"
    echo ""
    echo "Note: NeRF training runs for 200,000 iterations by default (4-8 hours)"
    exit 1
fi

# Handle different command types
case "$1" in
    "quick")
        echo "🔥 Running quick NeRF test (10 iterations using modified script)..."
        python /home/ubuntu/mono_to_3d/neural_radiance_fields/quick_test_nerf.py
        ;;
    "train")
        echo "🔥 Running full NeRF training (200,000 iterations)..."
        echo "⚠️  This will take 4-8 hours on GPU!"
        read -p "Continue? (y/N): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            python run_nerf.py --config configs/lego.txt
        else
            echo "Training cancelled."
        fi
        ;;
    "render")
        echo "🔥 Rendering test views (requires pre-trained model)..."
        python run_nerf.py --config configs/lego.txt --render_only --render_test
        ;;
    "test")
        echo "🔥 Testing data loading and basic setup..."
        python -c "
import sys
sys.path.append('.')
from load_blender import load_blender_data
print('Loading data...')
images, poses, render_poses, hwf, i_split = load_blender_data('./data/nerf_synthetic/lego', half_res=True, testskip=1)
print(f'✅ Data loaded successfully!')
print(f'Images shape: {images.shape}')
print(f'Poses shape: {poses.shape}')
print(f'HWF: {hwf}')
print(f'Splits: train={len(i_split[0])}, val={len(i_split[1])}, test={len(i_split[2])}')
"
        ;;
    *)
        echo "❌ Unknown command: $1"
        echo "Use: quick, train, render, or test"
        exit 1
        ;;
esac

echo ""
echo "✅ NeRF command completed!" 