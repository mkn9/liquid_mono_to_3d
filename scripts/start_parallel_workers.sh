#!/bin/bash
# Start All 3 Workers in Parallel using tmux
# Each worker runs in its own tmux session

set -e

PROJECT_DIR="$HOME/liquid_mono_to_3d"
cd "$PROJECT_DIR"

echo "🚀 Starting parallel workers in tmux sessions..."
echo ""

# Kill existing sessions if they exist
tmux kill-session -t worker1 2>/dev/null || true
tmux kill-session -t worker2 2>/dev/null || true
tmux kill-session -t worker3 2>/dev/null || true
tmux kill-session -t monitoring 2>/dev/null || true

# Create Worker 1 session (Liquid Fusion)
echo "👷 Starting Worker 1 (Liquid Fusion Layer)..."
tmux new-session -d -s worker1 -c "$PROJECT_DIR"
tmux send-keys -t worker1 "git checkout worker/liquid-worker-1-fusion" C-m
tmux send-keys -t worker1 "echo '🔵 Worker 1: Liquid Fusion Layer'" C-m
tmux send-keys -t worker1 "echo 'Branch: worker/liquid-worker-1-fusion'" C-m
tmux send-keys -t worker1 "echo 'Focus: Dual-modal fusion with Liquid dynamics'" C-m
tmux send-keys -t worker1 "echo ''" C-m
tmux send-keys -t worker1 "echo 'Ready for commands. Start with:'" C-m
tmux send-keys -t worker1 "echo '  bash scripts/worker1_tasks.sh'" C-m

# Create Worker 2 session (Liquid 3D)
echo "👷 Starting Worker 2 (Liquid 3D Reconstruction)..."
tmux new-session -d -s worker2 -c "$PROJECT_DIR"
tmux send-keys -t worker2 "git checkout worker/liquid-worker-2-3d" C-m
tmux send-keys -t worker2 "echo '🟢 Worker 2: Liquid 3D Reconstruction'" C-m
tmux send-keys -t worker2 "echo 'Branch: worker/liquid-worker-2-3d'" C-m
tmux send-keys -t worker2 "echo 'Focus: Temporally-smooth 3D trajectories'" C-m
tmux send-keys -t worker2 "echo ''" C-m
tmux send-keys -t worker2 "echo 'Ready for commands. Start with:'" C-m
tmux send-keys -t worker2 "echo '  bash scripts/worker2_tasks.sh'" C-m

# Create Worker 3 session (Integration)
echo "👷 Starting Worker 3 (Integration & Evaluation)..."
tmux new-session -d -s worker3 -c "$PROJECT_DIR"
tmux send-keys -t worker3 "git checkout worker/liquid-worker-3-integration" C-m
tmux send-keys -t worker3 "echo '🟡 Worker 3: Integration & Evaluation'" C-m
tmux send-keys -t worker3 "echo 'Branch: worker/liquid-worker-3-integration'" C-m
tmux send-keys -t worker3 "echo 'Focus: End-to-end testing and comparison'" C-m
tmux send-keys -t worker3 "echo ''" C-m
tmux send-keys -t worker3 "echo 'Ready for commands. Start with:'" C-m
tmux send-keys -t worker3 "echo '  bash scripts/worker3_tasks.sh'" C-m

# Create Monitoring session
echo "📊 Starting monitoring session..."
tmux new-session -d -s monitoring -c "$PROJECT_DIR"
tmux send-keys -t monitoring "echo '📊 Monitoring Dashboard'" C-m
tmux send-keys -t monitoring "echo 'Starting heartbeat monitor...'" C-m
tmux send-keys -t monitoring "bash scripts/heartbeat_monitor.sh" C-m

# Split monitoring window for sync
tmux split-window -t monitoring -v
tmux send-keys -t monitoring.1 "echo 'Starting periodic sync to MacBook...'" C-m
tmux send-keys -t monitoring.1 "while true; do bash scripts/sync_to_macbook.sh; sleep 300; done" C-m

echo ""
echo "✅ All workers started!"
echo ""
echo "Access workers with:"
echo "  tmux attach -t worker1    # Liquid Fusion"
echo "  tmux attach -t worker2    # Liquid 3D"
echo "  tmux attach -t worker3    # Integration"
echo "  tmux attach -t monitoring # Monitoring & Sync"
echo ""
echo "Detach from tmux: Ctrl-b then d"
echo "Switch windows: Ctrl-b then arrow keys"
echo ""
echo "List all sessions:"
echo "  tmux ls"
echo ""
echo "View monitoring dashboard:"
echo "  cat monitoring/status.txt"
echo ""
echo "🎯 Start working:"
echo "  tmux attach -t worker1"
echo "  bash scripts/worker1_tasks.sh  # Run Day 1 tasks"

