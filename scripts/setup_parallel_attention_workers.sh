#!/bin/bash
# Setup parallel git worktrees for attention improvement workers
# Worker 1: Attention-supervised loss
# Worker 2: Pre-trained ResNet features

set -e

echo "================================================================"
echo "Setting up Parallel Attention Workers"
echo "================================================================"

# Base directory
BASE_DIR=~/mono_to_3d
cd $BASE_DIR

# Ensure we're on the right branch
git fetch origin
git checkout early-persistence/magvit
git pull origin early-persistence/magvit

echo ""
echo "================================================================"
echo "Creating Worker 1: Attention-Supervised (worker1_attention)"
echo "================================================================"

# Create branch if doesn't exist
git branch early-persistence/attention-supervised 2>/dev/null || true
git worktree add ~/worker1_attention early-persistence/attention-supervised 2>/dev/null || echo "Worktree already exists"

# Copy base implementation to worker 1
cd ~/worker1_attention
mkdir -p experiments/trajectory_video_understanding/parallel_workers/worker1_attention/{src,tests,scripts,results}

echo "✅ Worker 1 worktree created at ~/worker1_attention"

echo ""
echo "================================================================"
echo "Creating Worker 2: Pre-trained Features (worker2_pretrained)"
echo "================================================================"

# Create branch if doesn't exist
cd $BASE_DIR
git branch early-persistence/pretrained-features 2>/dev/null || true
git worktree add ~/worker2_pretrained early-persistence/pretrained-features 2>/dev/null || echo "Worktree already exists"

# Copy base implementation to worker 2
cd ~/worker2_pretrained
mkdir -p experiments/trajectory_video_understanding/parallel_workers/worker2_pretrained/{src,tests,scripts,results}

echo "✅ Worker 2 worktree created at ~/worker2_pretrained"

echo ""
echo "================================================================"
echo "Worktree Setup Complete"
echo "================================================================"
echo ""
echo "Worker directories:"
echo "  Worker 1: ~/worker1_attention (early-persistence/attention-supervised)"
echo "  Worker 2: ~/worker2_pretrained (early-persistence/pretrained-features)"
echo ""
echo "Next steps:"
echo "  1. Implement Worker 1 (TDD)"
echo "  2. Implement Worker 2 (TDD)"
echo "  3. Start parallel training"
echo "  4. Monitor for early stopping"
echo ""

