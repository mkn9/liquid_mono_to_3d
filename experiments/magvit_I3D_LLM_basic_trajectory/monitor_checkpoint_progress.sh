#!/bin/bash
# Monitor checkpoint version progress (updates visible on MacBook!)

while true; do
    clear
    echo "30K CHECKPOINT GENERATION MONITOR"
    echo "=================================="
    date
    echo ""
    
    # Copy progress file to MacBook
    scp -i /Users/mike/keys/AutoGenKeyPair.pem \
        ubuntu@34.196.155.11:~/mono_to_3d/experiments/magvit_I3D_LLM_basic_trajectory/results/PROGRESS.txt \
        /tmp/30k_progress.txt 2>/dev/null
    
    if [ -f /tmp/30k_progress.txt ]; then
        cat /tmp/30k_progress.txt
        echo ""
        
        # Check if complete
        if grep -q "COMPLETE" /tmp/30k_progress.txt 2>/dev/null; then
            echo "✅ GENERATION COMPLETE!"
            echo ""
            echo "Syncing results..."
            scp -i /Users/mike/keys/AutoGenKeyPair.pem \
                ubuntu@34.196.155.11:~/mono_to_3d/experiments/magvit_I3D_LLM_basic_trajectory/results/*30k*.npz \
                ./results/
            echo "✅ Results synced to ./results/"
            break
        fi
    else
        echo "⏳ Waiting for first checkpoint..."
        echo "(Generation just started)"
    fi
    
    echo ""
    echo "Next update in 30 seconds... (Ctrl+C to stop)"
    sleep 30
done

