#!/bin/bash
# Monitor Parallel Execution
# Checks progress every 30 seconds and shows status

TASKS="clutter_transient_objects videogpt_3d_implementation magvit_pretrained_models"
CHECK_INTERVAL=30
MAX_CHECKS=120  # 1 hour max

echo "============================================================"
echo "Monitoring Parallel Task Execution"
echo "============================================================"
echo "Tasks: $TASKS"
echo "Check interval: ${CHECK_INTERVAL}s"
echo ""

check_count=0

while [ $check_count -lt $MAX_CHECKS ]; do
    check_count=$((check_count + 1))
    timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    
    echo "[$timestamp] Check #$check_count/$MAX_CHECKS"
    echo "------------------------------------------------------------"
    
    # Check if processes are running
    running=$(ps aux | grep -E "run_parallel_tasks|ssh.*ec2.*task_" | grep -v grep | wc -l | tr -d ' ')
    if [ "$running" -gt 0 ]; then
        echo "🔄 Execution processes running: $running"
    else
        echo "⏸️  No execution processes found"
    fi
    
    # Check for output files on EC2
    for task in $TASKS; do
        # Check if task has produced output
        output_check=$(ssh -i /Users/mike/keys/AutoGenKeyPair.pem -o StrictHostKeyChecking=no ubuntu@34.196.155.11 \
            "ls -t ~/mono_to_3d/experiments/*/output/*.json 2>/dev/null | grep -i $task | head -1" 2>/dev/null)
        
        if [ -n "$output_check" ]; then
            echo "✅ $task: Output file detected"
        else
            echo "⏳ $task: Waiting for output..."
        fi
    done
    
    echo ""
    
    # Check if execution completed
    if [ "$running" -eq 0 ] && [ $check_count -gt 5 ]; then
        echo "✅ Execution appears to have completed"
        break
    fi
    
    if [ $check_count -lt $MAX_CHECKS ]; then
        sleep $CHECK_INTERVAL
    fi
done

echo ""
echo "============================================================"
echo "Monitoring Complete"
echo "============================================================"

