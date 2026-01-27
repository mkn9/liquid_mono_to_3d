#!/bin/bash
# Check progress of parallel 30K generation
# Usage: bash check_progress.sh

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo "=================================="
echo "PARALLEL GENERATION STATUS CHECK"
echo "=================================="
echo "Time: $(date)"
echo ""

# Check if PID file exists
if [ -f "logs/parallel_30k.pid" ]; then
    PID=$(cat logs/parallel_30k.pid)
    echo "PID: $PID"
    
    # Check if process is running
    if kill -0 $PID 2>/dev/null; then
        echo "Status: ✓ RUNNING"
        
        # Show CPU/memory usage
        echo ""
        echo "Resource Usage:"
        ps aux | grep $PID | grep -v grep | awk '{printf "  CPU: %s%%  MEM: %s%%  TIME: %s\n", $3, $4, $10}'
        
        # Count worker processes
        WORKERS=$(ps aux | grep python | grep generate_parallel_30k | grep -v grep | wc -l)
        echo "  Workers: $WORKERS"
    else
        echo "Status: ✗ NOT RUNNING (completed or failed)"
    fi
else
    echo "Status: No PID file found"
    echo "  (Generation not started or PID file removed)"
fi

echo ""
echo "Recent Log Output:"
echo "----------------------------------"
# Find most recent log file
LATEST_LOG=$(ls -t logs/*_parallel_30k_generation.log 2>/dev/null | head -1)
if [ -n "$LATEST_LOG" ]; then
    echo "From: $LATEST_LOG"
    echo ""
    tail -20 "$LATEST_LOG"
else
    echo "No log file found"
fi

echo ""
echo "=================================="

