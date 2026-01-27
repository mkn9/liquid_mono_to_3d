#!/bin/bash
# Setup Git Worktrees for Parallel Object-Level Development

set -e

echo "================================================================"
echo "SETTING UP PARALLEL DEVELOPMENT WORKTREES"
echo "================================================================"

# Ensure we're in the main repo
MAIN_REPO=~/mono_to_3d
cd $MAIN_REPO

# Create branches
echo ""
echo "Creating git branches..."
git checkout -b object-level/detection-tracking 2>/dev/null || git checkout object-level/detection-tracking
git push -u origin object-level/detection-tracking

git checkout -b object-level/transformer 2>/dev/null || git checkout object-level/transformer
git push -u origin object-level/transformer

git checkout early-persistence/magvit  # Return to main branch

echo "✅ Branches created"

# Create worktrees
echo ""
echo "Creating git worktrees..."

WORKER1_DIR=~/mono_to_3d_worker1
WORKER2_DIR=~/mono_to_3d_worker2

# Remove existing worktrees if they exist
if [ -d "$WORKER1_DIR" ]; then
    echo "Removing existing worker1 worktree..."
    git worktree remove $WORKER1_DIR --force 2>/dev/null || rm -rf $WORKER1_DIR
fi

if [ -d "$WORKER2_DIR" ]; then
    echo "Removing existing worker2 worktree..."
    git worktree remove $WORKER2_DIR --force 2>/dev/null || rm -rf $WORKER2_DIR
fi

# Add new worktrees
git worktree add $WORKER1_DIR object-level/detection-tracking
git worktree add $WORKER2_DIR object-level/transformer

echo "✅ Worktrees created:"
echo "   Worker 1: $WORKER1_DIR (detection-tracking)"
echo "   Worker 2: $WORKER2_DIR (transformer)"

# Setup directory structure in each worktree
echo ""
echo "Setting up directory structures..."

for WORKER_DIR in $WORKER1_DIR $WORKER2_DIR; do
    cd $WORKER_DIR
    
    # Create directory structure
    mkdir -p experiments/trajectory_video_understanding/object_level_persistence/{src,tests,scripts,results,artifacts}
    mkdir -p experiments/trajectory_video_understanding/object_level_persistence/results/{checkpoints,logs,visualizations}
    
    # Create initial files
    touch experiments/trajectory_video_understanding/object_level_persistence/results/HEARTBEAT.txt
    touch experiments/trajectory_video_understanding/object_level_persistence/results/PROGRESS.txt
    
    # Link venv from main repo
    if [ ! -d "venv" ] && [ -d "$MAIN_REPO/venv" ]; then
        ln -s $MAIN_REPO/venv venv
    fi
    
    echo "✅ Setup $WORKER_DIR"
done

# Return to main repo
cd $MAIN_REPO

echo ""
echo "================================================================"
echo "WORKTREE SETUP COMPLETE"
echo "================================================================"
echo ""
echo "Worker directories:"
echo "  Worker 1: $WORKER1_DIR"
echo "  Worker 2: $WORKER2_DIR"
echo ""
echo "To work in a worktree:"
echo "  cd $WORKER1_DIR"
echo "  source venv/bin/activate"
echo "  # Make changes, commit, push"
echo ""
echo "To remove worktrees when done:"
echo "  git worktree remove $WORKER1_DIR"
echo "  git worktree remove $WORKER2_DIR"
echo ""
echo "================================================================"

