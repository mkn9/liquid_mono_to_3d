#!/bin/bash
# AI Keep-Alive / Watchdog Monitor
# Provides periodic heartbeat to prevent Cursor AI from hanging during long operations
# Usage: ./ai_keepalive.sh [task_name] [interval_seconds]

TASK_NAME="${1:-AI_Task}"
INTERVAL="${2:-30}"  # Default: 30 seconds
LOG_FILE="/tmp/ai_keepalive_${TASK_NAME}_$(date +%Y%m%d_%H%M%S).log"
HEARTBEAT_FILE="/tmp/ai_heartbeat.txt"

echo "🔴 AI Keep-Alive Monitor Started"
echo "Task: $TASK_NAME"
echo "Interval: ${INTERVAL}s"
echo "Log: $LOG_FILE"
echo "========================================="
echo ""

# Initialize log
cat > "$LOG_FILE" << EOF
AI Keep-Alive Monitor
Task: $TASK_NAME
Started: $(date)
Interval: ${INTERVAL}s
PID: $$
========================================

EOF

COUNTER=0
START_TIME=$(date +%s)

while true; do
    COUNTER=$((COUNTER + 1))
    CURRENT_TIME=$(date +%s)
    ELAPSED=$((CURRENT_TIME - START_TIME))
    ELAPSED_MIN=$((ELAPSED / 60))
    ELAPSED_SEC=$((ELAPSED % 60))
    
    TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')
    
    # Heartbeat message
    MESSAGE="⏱️  HEARTBEAT #${COUNTER} [${ELAPSED_MIN}m ${ELAPSED_SEC}s] - ${TIMESTAMP} - ${TASK_NAME} - System Active ✅"
    
    # Output to terminal (visible to user)
    echo "$MESSAGE"
    
    # Log to file
    echo "$MESSAGE" >> "$LOG_FILE"
    
    # Update heartbeat file (can be checked by other processes)
    echo "$MESSAGE" > "$HEARTBEAT_FILE"
    
    # Flush output (critical for keep-alive)
    sync
    
    # Wait for next heartbeat
    sleep "$INTERVAL"
done

