#!/bin/bash
# Heartbeat Monitor for Parallel Workers
# Monitors all 3 workers and reports status every 30 seconds

MONITOR_DIR="$HOME/liquid_mono_to_3d/monitoring"
STATUS_DIR="$HOME/liquid_mono_to_3d/status"
LOG_FILE="$HOME/liquid_mono_to_3d/logs/heartbeat.log"

mkdir -p "$MONITOR_DIR" "$STATUS_DIR" "$HOME/liquid_mono_to_3d/logs"

echo "💓 Heartbeat monitor started at $(date)" | tee -a "$LOG_FILE"
echo "Monitoring workers: 1 (fusion), 2 (3d), 3 (integration)" | tee -a "$LOG_FILE"
echo "---" | tee -a "$LOG_FILE"

while true; do
    TIMESTAMP=$(date +"%Y-%m-%d %H:%M:%S")
    
    # Create heartbeat file
    cat > "$MONITOR_DIR/heartbeat.json" << EOF
{
  "timestamp": "$TIMESTAMP",
  "workers": {
EOF
    
    # Check Worker 1
    WORKER1_PID=$(pgrep -f "worker1.*python" | head -1)
    if [ -n "$WORKER1_PID" ]; then
        WORKER1_STATUS="running"
        WORKER1_CPU=$(ps -p $WORKER1_PID -o %cpu= | tr -d ' ')
        WORKER1_MEM=$(ps -p $WORKER1_PID -o %mem= | tr -d ' ')
    else
        WORKER1_STATUS="idle"
        WORKER1_CPU="0.0"
        WORKER1_MEM="0.0"
    fi
    
    # Check Worker 2
    WORKER2_PID=$(pgrep -f "worker2.*python" | head -1)
    if [ -n "$WORKER2_PID" ]; then
        WORKER2_STATUS="running"
        WORKER2_CPU=$(ps -p $WORKER2_PID -o %cpu= | tr -d ' ')
        WORKER2_MEM=$(ps -p $WORKER2_PID -o %mem= | tr -d ' ')
    else
        WORKER2_STATUS="idle"
        WORKER2_CPU="0.0"
        WORKER2_MEM="0.0"
    fi
    
    # Check Worker 3
    WORKER3_PID=$(pgrep -f "worker3.*python" | head -1)
    if [ -n "$WORKER3_PID" ]; then
        WORKER3_STATUS="running"
        WORKER3_CPU=$(ps -p $WORKER3_PID -o %cpu= | tr -d ' ')
        WORKER3_MEM=$(ps -p $WORKER3_PID -o %mem= | tr -d ' ')
    else
        WORKER3_STATUS="idle"
        WORKER3_CPU="0.0"
        WORKER3_MEM="0.0"
    fi
    
    # Write JSON
    cat >> "$MONITOR_DIR/heartbeat.json" << EOF
    "worker1": {
      "status": "$WORKER1_STATUS",
      "pid": "${WORKER1_PID:-null}",
      "cpu_percent": $WORKER1_CPU,
      "mem_percent": $WORKER1_MEM
    },
    "worker2": {
      "status": "$WORKER2_STATUS",
      "pid": "${WORKER2_PID:-null}",
      "cpu_percent": $WORKER2_CPU,
      "mem_percent": $WORKER2_MEM
    },
    "worker3": {
      "status": "$WORKER3_STATUS",
      "pid": "${WORKER3_PID:-null}",
      "cpu_percent": $WORKER3_CPU,
      "mem_percent": $WORKER3_MEM
    }
  },
  "system": {
    "gpu_available": $(nvidia-smi > /dev/null 2>&1 && echo "true" || echo "false"),
    "disk_usage": "$(df -h /home/ubuntu | awk 'NR==2 {print $5}')"
  }
}
EOF
    
    # Log heartbeat
    echo "[$TIMESTAMP] W1:$WORKER1_STATUS W2:$WORKER2_STATUS W3:$WORKER3_STATUS" >> "$LOG_FILE"
    
    # Create human-readable status
    cat > "$MONITOR_DIR/status.txt" << EOF
=== Parallel Workers Heartbeat ===
Time: $TIMESTAMP

Worker 1 (Fusion):      $WORKER1_STATUS (CPU: ${WORKER1_CPU}%, MEM: ${WORKER1_MEM}%)
Worker 2 (3D Recon):    $WORKER2_STATUS (CPU: ${WORKER2_CPU}%, MEM: ${WORKER2_MEM}%)
Worker 3 (Integration): $WORKER3_STATUS (CPU: ${WORKER3_CPU}%, MEM: ${WORKER3_MEM}%)

Last update: $TIMESTAMP
EOF
    
    # Sleep for 30 seconds
    sleep 30
done

