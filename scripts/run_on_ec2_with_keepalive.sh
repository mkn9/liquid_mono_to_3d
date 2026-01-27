#!/bin/bash
# Run command on EC2 with automatic AI keep-alive monitoring
# Auto-stops keep-alive when EC2 command completes
#
# Usage: 
#   ./scripts/run_on_ec2_with_keepalive.sh <ec2_command>
#   ./scripts/run_on_ec2_with_keepalive.sh "cd mono_to_3d && python train.py"
#   ./scripts/run_on_ec2_with_keepalive.sh "cd mono_to_3d && bash scripts/prove.sh"

# Configuration
EC2_HOST="ubuntu@34.196.155.11"
EC2_KEY="/Users/mike/keys/AutoGenKeyPair.pem"

# Check if command provided
if [ $# -eq 0 ]; then
    echo "❌ Error: No command provided"
    echo "Usage: ./scripts/run_on_ec2_with_keepalive.sh <ec2_command>"
    echo "Example: ./scripts/run_on_ec2_with_keepalive.sh 'cd mono_to_3d && python train.py'"
    exit 1
fi

EC2_COMMAND="$@"
TASK_NAME="EC2_$(echo "$EC2_COMMAND" | sed 's/[^a-zA-Z0-9_-]/_/g' | cut -c1-40)"
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
        echo "✅ EC2 command completed successfully"
    else
        echo "⚠️  EC2 command exited with code: $EXIT_CODE"
    fi
    
    exit $EXIT_CODE
}

# Register cleanup to run on ANY exit (success, error, Ctrl+C)
trap cleanup EXIT INT TERM

# Start keep-alive in background
echo "🚀 Starting AI keep-alive monitor (Local)..."
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
echo "🖥️  Connecting to EC2..."
echo "▶️  Running: $EC2_COMMAND"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Run command on EC2 via SSH
# This will block until SSH command completes
ssh -i "$EC2_KEY" "$EC2_HOST" "$EC2_COMMAND"
SSH_EXIT_CODE=$?

# Note: cleanup() is automatically called due to trap
# We just need to exit with the SSH command's exit code
exit $SSH_EXIT_CODE

