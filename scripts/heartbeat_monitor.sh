#!/bin/bash
# Heartbeat Monitoring Script
# Provides periodic progress updates during long-running tasks
# Usage: bash scripts/heartbeat_monitor.sh [process_name] [interval_seconds]

PROCESS_NAME=${1:-"python3"}
INTERVAL=${2:-60}  # Default: 1 minute

echo "=== Heartbeat Monitor Started ==="
echo "Monitoring: $PROCESS_NAME"
echo "Interval: ${INTERVAL}s"
echo "Time: $(date)"
echo ""

while true; do
    # Check if process is running
    if pgrep -f "$PROCESS_NAME" > /dev/null; then
        echo "⏰ [$(date '+%Y-%m-%d %H:%M:%S')] ✅ $PROCESS_NAME is running"
        
        # Show GPU usage if nvidia-smi available
        if command -v nvidia-smi &> /dev/null; then
            echo "   GPU Memory: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1) MB"
        fi
        
        # Show CPU usage
        if command -v top &> /dev/null; then
            CPU_USAGE=$(top -bn1 | grep "Cpu(s)" | sed "s/.*, *\([0-9.]*\)%* id.*/\1/" | awk '{print 100 - $1"%"}')
            echo "   CPU Usage: $CPU_USAGE"
        fi
        
        # Show disk usage for results directory
        if [ -d "experiments/liquid_vlm_integration/results" ]; then
            RESULTS_SIZE=$(du -sh experiments/liquid_vlm_integration/results 2>/dev/null | cut -f1)
            echo "   Results Size: $RESULTS_SIZE"
        fi
        
    else
        echo "⏰ [$(date '+%Y-%m-%d %H:%M:%S')] ⚠️  $PROCESS_NAME not found - may have completed"
    fi
    
    echo ""
    sleep $INTERVAL
done
