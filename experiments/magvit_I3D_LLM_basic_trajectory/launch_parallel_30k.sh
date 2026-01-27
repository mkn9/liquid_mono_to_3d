#!/bin/bash
# Launch parallel 30K dataset generation in background with logging
#
# Usage: bash launch_parallel_30k.sh

set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Activate venv
source ../../venv/bin/activate

# Create logs directory
mkdir -p logs

# Timestamp for log file
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="logs/${TIMESTAMP}_parallel_30k_generation.log"

echo "=================================="
echo "PARALLEL 30K DATASET GENERATION"
echo "=================================="
echo "Start time: $(date)"
echo "Log file: $LOG_FILE"
echo "PID file: logs/parallel_30k.pid"
echo ""
echo "Running in background..."
echo "Monitor with:"
echo "  tail -f $LOG_FILE"
echo "  ps aux | grep generate_parallel_30k"
echo ""

# Run in background, save PID
nohup python generate_parallel_30k.py > "$LOG_FILE" 2>&1 &
PID=$!
echo $PID > logs/parallel_30k.pid

echo "✓ Started with PID: $PID"
echo ""
echo "Check status:"
echo "  tail -f $LOG_FILE"
echo "  kill -0 $PID  # Check if still running"
echo ""
echo "Estimated time: 15-20 minutes"
echo "=================================="

