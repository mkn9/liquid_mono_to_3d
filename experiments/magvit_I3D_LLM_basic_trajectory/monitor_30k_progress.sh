#!/bin/bash
# Monitor 30K generation progress by watching shared memory usage

echo "Monitoring 30K Dataset Generation"
echo "=================================="
echo ""

while true; do
    clear
    echo "30K DATASET GENERATION MONITOR"
    echo "=============================="
    date
    echo ""
    
    # Check if process still running
    ssh -i /Users/mike/keys/AutoGenKeyPair.pem ubuntu@34.196.155.11 "ps -p 18132 -o pid,etime,%cpu,rss 2>/dev/null" > /tmp/process_status.txt
    
    if [ $? -eq 0 ]; then
        echo "✅ Process RUNNING:"
        cat /tmp/process_status.txt
        echo ""
        
        # Check worker status
        echo "Worker Status:"
        ssh -i /Users/mike/keys/AutoGenKeyPair.pem ubuntu@34.196.155.11 "ps -p 18142,18143 -o pid,%cpu,rss,stat 2>/dev/null"
        echo ""
        
        # Check shared memory usage
        echo "Shared Memory (Generated Data):"
        ssh -i /Users/mike/keys/AutoGenKeyPair.pem ubuntu@34.196.155.11 "lsof -p 18132 2>/dev/null | grep '/dev/shm/torch' | awk '{sum+=\$7} END {print \"  Total: \" sum/1024/1024/1024 \" GB\"}'"
        echo ""
        
        # Check for output file
        OUTPUT_EXISTS=$(ssh -i /Users/mike/keys/AutoGenKeyPair.pem ubuntu@34.196.155.11 "ls ~/mono_to_3d/experiments/magvit_I3D_LLM_basic_trajectory/results/*30k*.npz 2>/dev/null | wc -l")
        
        if [ "$OUTPUT_EXISTS" -gt 0 ]; then
            echo "✅ OUTPUT FILE CREATED!"
            ssh -i /Users/mike/keys/AutoGenKeyPair.pem ubuntu@34.196.155.11 "ls -lh ~/mono_to_3d/experiments/magvit_I3D_LLM_basic_trajectory/results/*30k*.npz"
            echo ""
            echo "Generation complete! Run sync_30k_results.sh to download."
            break
        else
            echo "⏳ No output file yet (will save at completion)"
        fi
    else
        echo "✅ PROCESS COMPLETED!"
        echo ""
        echo "Checking for output file..."
        ssh -i /Users/mike/keys/AutoGenKeyPair.pem ubuntu@34.196.155.11 "ls -lh ~/mono_to_3d/experiments/magvit_I3D_LLM_basic_trajectory/results/*30k*.npz 2>/dev/null"
        echo ""
        echo "Run sync_30k_results.sh to download results."
        break
    fi
    
    echo ""
    echo "Next update in 30 seconds... (Ctrl+C to stop monitoring)"
    sleep 30
done

