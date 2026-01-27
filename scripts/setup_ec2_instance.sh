#!/bin/bash
# Quick setup script for new EC2 instance

set -e

echo "🚀 Setting up liquid_mono_to_3d on EC2..."

# Update system
echo "📦 Updating system packages..."
sudo apt-get update -qq

# Install Python dependencies
echo "🐍 Installing Python environment..."
python3 -m pip install --upgrade pip

# Install PyTorch with CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install other dependencies
pip install transformers accelerate sentencepiece
pip install opencv-python matplotlib seaborn pandas numpy scipy
pip install pytest pytest-cov

# Verify CUDA
echo ""
echo "🔍 Verifying CUDA setup..."
python3 -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}')"

echo ""
echo "✅ Setup complete!"
echo ""
echo "Next steps:"
echo "1. Copy data and models (see DATA_SETUP.md)"
echo "2. Run tests: cd experiments/trajectory_video_understanding && pytest"
echo "3. Start training or inference"
