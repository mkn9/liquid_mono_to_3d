#!/bin/bash
# Download Open-MAGVIT2 Pretrained Checkpoint
# Following honesty principle: using real pretrained model

set -e

echo "📥 Downloading Open-MAGVIT2 Video Tokenizer..."
echo "=============================================="
echo ""

# Create weights directory
mkdir -p ~/magvit_weights
cd ~/magvit_weights

# Check if already downloaded
if [ -f "video_128_262144.ckpt" ]; then
    SIZE=$(stat -f%z "video_128_262144.ckpt" 2>/dev/null || stat -c%s "video_128_262144.ckpt")
    if [ "$SIZE" -gt 1000000000 ]; then
        echo "✅ Checkpoint already exists ($(echo "scale=2; $SIZE/1000000000" | bc) GB)"
        echo "   Skipping download"
        exit 0
    fi
fi

# Download from HuggingFace
echo "Downloading from HuggingFace..."
echo "URL: https://huggingface.co/TencentARC/Open-MAGVIT2-Tokenizer-262144-Video"
echo ""

# Use wget with progress
wget -c https://huggingface.co/TencentARC/Open-MAGVIT2-Tokenizer-262144-Video/resolve/main/model.ckpt \
  -O video_128_262144.ckpt

echo ""
echo "✅ Download complete!"
ls -lh video_128_262144.ckpt
