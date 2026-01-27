#!/bin/bash
# AI Progress Tracker - Monitors multiple AI tasks simultaneously
# Provides dashboard view of all active AI operations
# Usage: ./ai_progress_tracker.sh

REFRESH_INTERVAL=10  # Refresh dashboard every 10 seconds

echo "🎯 AI Progress Tracker - Multi-Task Dashboard"
echo "========================================="
echo ""

while true; do
    clear
    
    echo "╔════════════════════════════════════════════════════════════════╗"
    echo "║         AI PROGRESS TRACKER - MULTI-TASK DASHBOARD            ║"
    echo "║                $(date '+%Y-%m-%d %H:%M:%S')                        ║"
    echo "╚════════════════════════════════════════════════════════════════╝"
    echo ""
    
    # Check for active keep-alive monitors
    KEEPALIVE_PIDS=$(pgrep -f "ai_keepalive.sh" || echo "")
    if [ -n "$KEEPALIVE_PIDS" ]; then
        echo "🟢 ACTIVE KEEP-ALIVE MONITORS:"
        echo "   PIDs: $KEEPALIVE_PIDS"
        for PID in $KEEPALIVE_PIDS; do
            CMD=$(ps -p "$PID" -o args= 2>/dev/null)
            echo "   └─ $PID: $CMD"
        done
    else
        echo "⚪ No active keep-alive monitors"
    fi
    echo ""
    
    # Check heartbeat status
    echo "💓 HEARTBEAT STATUS:"
    if [ -f "/tmp/ai_heartbeat.txt" ]; then
        echo "   Last Heartbeat:"
        tail -1 /tmp/ai_heartbeat.txt | sed 's/^/   │  /'
        
        # Calculate time since last heartbeat
        HEARTBEAT_TIME=$(stat -f %m /tmp/ai_heartbeat.txt 2>/dev/null || stat -c %Y /tmp/ai_heartbeat.txt 2>/dev/null)
        CURRENT_TIME=$(date +%s)
        TIME_SINCE=$((CURRENT_TIME - HEARTBEAT_TIME))
        
        if [ "$TIME_SINCE" -lt 60 ]; then
            echo "   └─ Status: ✅ Active (${TIME_SINCE}s ago)"
        elif [ "$TIME_SINCE" -lt 300 ]; then
            echo "   └─ Status: ⚠️  Slow (${TIME_SINCE}s ago)"
        else
            echo "   └─ Status: 🚨 Possible Hang (${TIME_SINCE}s ago)"
        fi
    else
        echo "   └─ Status: ⚪ No heartbeat detected"
    fi
    echo ""
    
    # Check for EC2 processes
    echo "🖥️  EC2 PROCESSES (via SSH):"
    EC2_CHECK=$(ssh -i /Users/mike/keys/AutoGenKeyPair.pem ubuntu@34.196.155.11 \
        'ps aux | grep -E "(python|pytest|train)" | grep -v grep | head -5' 2>/dev/null || echo "Unable to connect")
    
    if [ "$EC2_CHECK" == "Unable to connect" ]; then
        echo "   └─ Status: ⚠️  Cannot connect to EC2"
    elif [ -z "$EC2_CHECK" ]; then
        echo "   └─ Status: ⚪ No active Python processes"
    else
        echo "$EC2_CHECK" | while read line; do
            echo "   │  $line"
        done
    fi
    echo ""
    
    # Check system resources
    echo "📊 LOCAL SYSTEM STATUS:"
    echo "   CPU Usage: $(top -l 1 | grep "CPU usage" | awk '{print $3}' 2>/dev/null || echo "N/A")"
    echo "   Memory: $(vm_stat | grep "Pages active" | awk '{print $3}' 2>/dev/null || echo "N/A")"
    echo "   Disk: $(df -h / | tail -1 | awk '{print $5 " used"}' 2>/dev/null || echo "N/A")"
    echo ""
    
    # Check for log files
    echo "📝 RECENT LOGS:"
    LOG_COUNT=$(ls -1 /tmp/ai_keepalive_*.log 2>/dev/null | wc -l)
    if [ "$LOG_COUNT" -gt 0 ]; then
        echo "   Found $LOG_COUNT log file(s):"
        ls -lht /tmp/ai_keepalive_*.log 2>/dev/null | head -3 | while read line; do
            echo "   │  $line"
        done
    else
        echo "   └─ No log files found"
    fi
    echo ""
    
    echo "════════════════════════════════════════════════════════════════"
    echo "Press Ctrl+C to stop monitoring | Refresh: ${REFRESH_INTERVAL}s"
    echo ""
    
    sleep "$REFRESH_INTERVAL"
done

