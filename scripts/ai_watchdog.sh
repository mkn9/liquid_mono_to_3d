#!/bin/bash
# AI Watchdog - Monitors for AI hangs and provides diagnostics
# Checks if AI is making progress and alerts if stuck
# Usage: ./ai_watchdog.sh [check_interval] [hang_threshold]

CHECK_INTERVAL="${1:-60}"      # Check every 60 seconds
HANG_THRESHOLD="${2:-300}"     # Alert if no progress for 5 minutes
HEARTBEAT_FILE="/tmp/ai_heartbeat.txt"
LAST_ACTIVITY_FILE="/tmp/ai_last_activity.txt"

echo "🔍 AI Watchdog Monitor Started"
echo "Check Interval: ${CHECK_INTERVAL}s"
echo "Hang Threshold: ${HANG_THRESHOLD}s"
echo "========================================="
echo ""

ALERT_COUNT=0

while true; do
    CURRENT_TIME=$(date +%s)
    TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')
    
    # Check if heartbeat file exists and is recent
    if [ -f "$HEARTBEAT_FILE" ]; then
        HEARTBEAT_TIME=$(stat -f %m "$HEARTBEAT_FILE" 2>/dev/null || stat -c %Y "$HEARTBEAT_FILE" 2>/dev/null)
        TIME_SINCE_HEARTBEAT=$((CURRENT_TIME - HEARTBEAT_TIME))
        
        if [ "$TIME_SINCE_HEARTBEAT" -gt "$HANG_THRESHOLD" ]; then
            ALERT_COUNT=$((ALERT_COUNT + 1))
            echo ""
            echo "🚨 [$TIMESTAMP] ALERT #${ALERT_COUNT}: AI May Be Hung!"
            echo "   Last heartbeat: ${TIME_SINCE_HEARTBEAT}s ago (threshold: ${HANG_THRESHOLD}s)"
            echo "   Heartbeat file: $HEARTBEAT_FILE"
            echo ""
            echo "   Diagnostic Actions:"
            echo "   1. Check terminal output above for last activity"
            echo "   2. Look for error messages"
            echo "   3. Consider interrupting (Ctrl+C) and restarting"
            echo "   4. Check: tail -20 $HEARTBEAT_FILE"
            echo ""
        else
            echo "✅ [$TIMESTAMP] AI Active - Last heartbeat: ${TIME_SINCE_HEARTBEAT}s ago"
        fi
    else
        echo "⚠️  [$TIMESTAMP] No heartbeat file found - Keep-alive may not be running"
        echo "   Start keep-alive: ./scripts/ai_keepalive.sh <task_name> 30"
    fi
    
    # Check for Cursor/Python processes
    AI_PROCESSES=$(ps aux | grep -E "(python|cursor|node)" | grep -v grep | wc -l)
    echo "   Active AI-related processes: $AI_PROCESSES"
    
    # Check system load
    if command -v uptime &> /dev/null; then
        LOAD=$(uptime | awk -F'load average:' '{print $2}')
        echo "   System load:$LOAD"
    fi
    
    echo ""
    
    sleep "$CHECK_INTERVAL"
done

