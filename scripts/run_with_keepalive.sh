#!/bin/bash
# Run command with automatic AI keep-alive monitoring
# Auto-stops keep-alive when command completes (success or failure)
#
# Usage: 
#   ./scripts/run_with_keepalive.sh <command>
#   ./scripts/run_with_keepalive.sh "python train.py"
#   ./scripts/run_with_keepalive.sh "bash scripts/prove.sh"

# Check if command provided
if [ $# -eq 0 ]; then
    echo "❌ Error: No command provided"
    echo "Usage: ./scripts/run_with_keepalive.sh <command>"
    echo "Example: ./scripts/run_with_keepalive.sh 'python train.py'"
    exit 1
fi

COMMAND="$@"
TASK_NAME=$(echo "$COMMAND" | sed 's/[^a-zA-Z0-9_-]/_/g' | cut -c1-50)
KEEPALIVE_INTERVAL=30
KEEPALIVE_LOG="/tmp/ai_keepalive_${TASK_NAME}_$(date +%Y%m%d_%H%M%S).log"
KEEPALIVE_PID=""

# Cleanup function - ALWAYS called on exit
cleanup() {
    local EXIT_CODE=$?
    
    echo ""
    echo "🧹 Cleanup: Stopping keep-alive monitor..."
    
    if [ -n "$KEEPALIVE_PID" ]; then
        kill $KEEPALIVE_PID 2>/dev/null
        wait $KEEPALIVE_PID 2>/dev/null
        echo "✅ Keep-alive stopped (PID: $KEEPALIVE_PID)"
    else
        # Fallback: kill by name
        pkill -f "ai_keepalive.sh.*$TASK_NAME" 2>/dev/null
        echo "✅ Keep-alive processes stopped"
    fi
    
    echo "📋 Keep-alive log: $KEEPALIVE_LOG"
    
    if [ $EXIT_CODE -eq 0 ]; then
        echo "✅ Command completed successfully"
    else
        echo "⚠️  Command exited with code: $EXIT_CODE"
    fi
    
    exit $EXIT_CODE
}

# Register cleanup to run on ANY exit (success, error, Ctrl+C)
trap cleanup EXIT INT TERM

# Start keep-alive in background
echo "🚀 Starting AI keep-alive monitor..."
echo "   Task: $TASK_NAME"
echo "   Interval: ${KEEPALIVE_INTERVAL}s"
echo "   Log: $KEEPALIVE_LOG"
echo ""

# Start keep-alive and capture its PID
./scripts/ai_keepalive.sh "$TASK_NAME" "$KEEPALIVE_INTERVAL" > "$KEEPALIVE_LOG" 2>&1 &
KEEPALIVE_PID=$!

echo "✅ Keep-alive started (PID: $KEEPALIVE_PID)"
echo "📊 Watch progress: tail -f $KEEPALIVE_LOG"
echo ""
echo "▶️  Running: $COMMAND"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Run the actual command
# This will block until command completes
eval "$COMMAND"
COMMAND_EXIT_CODE=$?

# Note: cleanup() is automatically called due to trap
# We just need to exit with the command's exit code
exit $COMMAND_EXIT_CODE

