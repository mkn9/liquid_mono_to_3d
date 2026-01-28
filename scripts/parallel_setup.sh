#!/bin/bash
# Parallel Development Setup Script for Liquid NN Integration
# Creates 3 worker branches and sets up monitoring

set -e

echo "🚀 Setting up parallel development infrastructure..."

# Navigate to project root
cd ~/liquid_mono_to_3d

# Create integration branch
echo "📌 Creating integration branch..."
git checkout -b liquid-nn-integration 2>/dev/null || git checkout liquid-nn-integration

# Create worker branches
echo "👷 Creating worker branches..."

# Worker 1: Liquid Fusion Layer
git checkout liquid-nn-integration
git checkout -b worker/liquid-worker-1-fusion 2>/dev/null || git checkout worker/liquid-worker-1-fusion
echo "✅ Worker 1 branch created: liquid-worker-1-fusion"

# Worker 2: Liquid 3D Reconstruction
git checkout liquid-nn-integration
git checkout -b worker/liquid-worker-2-3d 2>/dev/null || git checkout worker/liquid-worker-2-3d
echo "✅ Worker 2 branch created: liquid-worker-2-3d"

# Worker 3: Integration & Evaluation
git checkout liquid-nn-integration
git checkout -b worker/liquid-worker-3-integration 2>/dev/null || git checkout worker/liquid-worker-3-integration
echo "✅ Worker 3 branch created: liquid-worker-3-integration"

# Push all branches
echo "📤 Pushing branches to remote..."
git push -u origin liquid-nn-integration 2>/dev/null || echo "Integration branch already pushed"
git checkout worker/liquid-worker-1-fusion
git push -u origin worker/liquid-worker-1-fusion 2>/dev/null || echo "Worker 1 already pushed"
git checkout worker/liquid-worker-2-3d
git push -u origin worker/liquid-worker-2-3d 2>/dev/null || echo "Worker 2 already pushed"
git checkout worker/liquid-worker-3-integration
git push -u origin worker/liquid-worker-3-integration 2>/dev/null || echo "Worker 3 already pushed"

echo ""
echo "✅ All branches created and pushed!"
echo ""
echo "Branch structure:"
echo "  main"
echo "   └─ liquid-nn-integration"
echo "       ├─ worker/liquid-worker-1-fusion"
echo "       ├─ worker/liquid-worker-2-3d"
echo "       └─ worker/liquid-worker-3-integration"
echo ""
echo "Next: Run scripts/start_parallel_workers.sh to begin development"

