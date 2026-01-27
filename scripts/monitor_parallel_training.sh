#!/bin/bash
# Monitor parallel training on EC2
# Usage: bash scripts/monitor_parallel_training.sh

set -e

EC2_HOST="ubuntu@34.196.155.11"
SSH_KEY="/Users/mike/keys/AutoGenKeyPair.pem"
TIMEOUT=15  # seconds

echo "======================================"
echo "Parallel Training Monitor"
echo "======================================"
echo ""

# Function to run SSH with timeout
ssh_cmd() {
    timeout $TIMEOUT ssh -o ConnectTimeout=10 -i "$SSH_KEY" "$EC2_HOST" "$1" 2>/dev/null
    return $?
}

# Check if EC2 is reachable
echo "🔍 Checking EC2 connection..."
if ssh_cmd "echo 'Connected'" > /dev/null; then
    echo "✅ EC2 reachable"
else
    echo "⚠️  EC2 connection timeout (likely under heavy load)"
    echo "   This is normal during training. Try again in 5-10 minutes."
    exit 0
fi

echo ""
echo "======================================"
echo "Process Status"
echo "======================================"

ssh_cmd 'cd ~/mono_to_3d/parallel_training/logs && \
for pid_file in *.pid; do \
  worker=$(basename "$pid_file" .pid); \
  pid=$(cat "$pid_file"); \
  if ps -p "$pid" > /dev/null 2>&1; then \
    echo "✅ $worker (PID: $pid) - RUNNING"; \
  else \
    echo "❌ $worker (PID: $pid) - STOPPED"; \
  fi; \
done' || echo "⚠️  Could not check process status"

echo ""
echo "======================================"
echo "Log File Sizes (indicators of progress)"
echo "======================================"

ssh_cmd 'ls -lh ~/mono_to_3d/parallel_training/logs/worker_*_20260125_1944.log 2>/dev/null | \
awk "{printf \"%-30s %10s\n\", \$9, \$5}"' || echo "⚠️  Could not check log sizes"

echo ""
echo "======================================"
echo "Recent Log Activity (last 10 lines each)"
echo "======================================"

for worker in i3d slowfast transformer magvit; do
    echo ""
    echo "--- Worker: $worker ---"
    ssh_cmd "tail -10 ~/mono_to_3d/parallel_training/logs/worker_${worker}_20260125_1944.log 2>/dev/null" || \
        echo "⚠️  Could not read log (timeout or file empty)"
done

echo ""
echo "======================================"
echo "Checkpoint Status"
echo "======================================"

ssh_cmd 'for worker in ~/mono_to_3d/parallel_training/worker_*/experiments/trajectory_video_understanding/*/results/validation/checkpoints; do \
    worker_name=$(basename $(dirname $(dirname $(dirname $(dirname $worker))))); \
    checkpoint_count=$(ls "$worker" 2>/dev/null | wc -l | tr -d " "); \
    if [ "$checkpoint_count" -gt "0" ]; then \
        echo "✅ $worker_name: $checkpoint_count checkpoints"; \
    else \
        echo "⏳ $worker_name: No checkpoints yet"; \
    fi; \
done' || echo "⚠️  Could not check checkpoints"

echo ""
echo "======================================"
echo "GPU Utilization"
echo "======================================"

ssh_cmd 'nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits | \
head -1 | awk -F, "{printf \"GPU: %s%% | Memory: %s / %s MB\n\", \$1, \$2, \$3}"' || \
    echo "⚠️  Could not check GPU status"

echo ""
echo "======================================"
echo "Disk Space"
echo "======================================"

ssh_cmd 'df -h / | tail -1 | awk "{printf \"Used: %s / %s (%s used)\n\", \$3, \$2, \$5}"' || \
    echo "⚠️  Could not check disk space"

echo ""
echo "======================================"
echo "Quick Actions"
echo "======================================"
echo ""
echo "View live logs:"
echo "  ssh -i $SSH_KEY $EC2_HOST 'tail -f ~/mono_to_3d/parallel_training/logs/worker_i3d_20260125_1944.log'"
echo ""
echo "Kill all training (if needed):"
echo "  ssh -i $SSH_KEY $EC2_HOST 'cd ~/mono_to_3d/parallel_training/logs && for pid in *.pid; do kill \$(cat \$pid); done'"
echo ""
echo "Copy results after training:"
echo "  scp -i $SSH_KEY -r $EC2_HOST:~/mono_to_3d/parallel_training/worker_*/experiments/*/results/ ./results/"
echo ""
echo "======================================"
echo "Monitor complete. Run again to refresh."
echo "======================================"

