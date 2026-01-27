#!/bin/bash
# Periodic sync of training results from EC2 to MacBook
# This ensures results are visible locally as training progresses
# Usage: bash scripts/sync_training_results.sh [interval_seconds]

set -e

EC2_HOST="ubuntu@34.196.155.11"
SSH_KEY="/Users/mike/keys/AutoGenKeyPair.pem"
LOCAL_BASE="/Users/mike/Dropbox/Documents/Machine_Learning/LLM_and_Multimodal_Models/repos/mono_to_3d/experiments/trajectory_video_understanding"
SYNC_INTERVAL=${1:-60}  # Default: sync every 60 seconds

echo "======================================"
echo "Training Results Auto-Sync (EC2 → MacBook)"
echo "======================================"
echo "Sync interval: ${SYNC_INTERVAL} seconds"
echo "Local base: ${LOCAL_BASE}"
echo "Press Ctrl+C to stop"
echo "======================================"
echo ""

SYNC_COUNT=0

while true; do
    SYNC_COUNT=$((SYNC_COUNT + 1))
    TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')
    
    echo "[${TIMESTAMP}] Sync #${SYNC_COUNT} starting..."
    
    # Create local results directories
    mkdir -p "${LOCAL_BASE}/training_results_live"/{logs,checkpoints,metrics}
    
    # Sync training logs
    echo "  📄 Syncing training logs..."
    scp -i "${SSH_KEY}" -o ConnectTimeout=10 -q \
        "${EC2_HOST}:~/mono_to_3d/parallel_training/logs/worker_*_20260125_1944.log" \
        "${LOCAL_BASE}/training_results_live/logs/" 2>/dev/null || echo "    ⚠️  Could not sync logs (EC2 busy)"
    
    # Sync process status
    echo "  📊 Syncing process status..."
    ssh -i "${SSH_KEY}" -o ConnectTimeout=10 "${EC2_HOST}" \
        'cd ~/mono_to_3d/parallel_training/logs && \
         for pid_file in *.pid; do \
           worker=$(basename "$pid_file" .pid); \
           pid=$(cat "$pid_file"); \
           if ps -p "$pid" > /dev/null 2>&1; then \
             echo "✅ $worker (PID: $pid) RUNNING"; \
           else \
             echo "❌ $worker (PID: $pid) STOPPED"; \
           fi; \
         done' > "${LOCAL_BASE}/training_results_live/process_status.txt" 2>/dev/null || \
         echo "    ⚠️  Could not sync process status"
    
    # Sync checkpoints (if they exist)
    echo "  💾 Syncing checkpoints..."
    for worker in i3d slowfast transformer magvit; do
        scp -i "${SSH_KEY}" -o ConnectTimeout=10 -q -r \
            "${EC2_HOST}:~/mono_to_3d/parallel_training/worker_${worker}/experiments/trajectory_video_understanding/*/results/validation/checkpoints/*.pt" \
            "${LOCAL_BASE}/training_results_live/checkpoints/${worker}/" 2>/dev/null || \
            echo "    ⏳ No checkpoints yet for ${worker}"
    done
    
    # Sync metrics (if they exist)
    echo "  📈 Syncing metrics..."
    for worker in i3d slowfast transformer magvit; do
        scp -i "${SSH_KEY}" -o ConnectTimeout=10 -q \
            "${EC2_HOST}:~/mono_to_3d/parallel_training/worker_${worker}/experiments/trajectory_video_understanding/*/results/validation/metrics.json" \
            "${LOCAL_BASE}/training_results_live/metrics/${worker}_metrics.json" 2>/dev/null || \
            echo "    ⏳ No metrics yet for ${worker}"
    done
    
    # Create timestamp file
    echo "${TIMESTAMP} - Sync #${SYNC_COUNT}" > "${LOCAL_BASE}/training_results_live/LAST_SYNC.txt"
    
    # Generate summary
    echo "  📋 Generating summary..."
    cat > "${LOCAL_BASE}/training_results_live/LIVE_SUMMARY.txt" << EOF
Training Results Live Summary
Generated: ${TIMESTAMP}
Sync Count: ${SYNC_COUNT}

====================================
PROCESS STATUS
====================================
$(cat "${LOCAL_BASE}/training_results_live/process_status.txt" 2>/dev/null || echo "Not available")

====================================
LOG FILE SIZES (Progress Indicator)
====================================
$(ls -lh "${LOCAL_BASE}/training_results_live/logs/"*.log 2>/dev/null | awk '{printf "%-40s %10s\n", $9, $5}' || echo "No logs synced yet")

====================================
CHECKPOINT COUNTS
====================================
$(for worker in i3d slowfast transformer magvit; do \
    count=$(ls "${LOCAL_BASE}/training_results_live/checkpoints/${worker}/"*.pt 2>/dev/null | wc -l | tr -d ' '); \
    if [ "$count" -gt "0" ]; then \
        echo "✅ ${worker}: ${count} checkpoints"; \
    else \
        echo "⏳ ${worker}: No checkpoints yet"; \
    fi; \
done)

====================================
LATEST LOG ACTIVITY (last 5 lines each)
====================================
EOF
    
    for worker in i3d slowfast transformer magvit; do
        echo "" >> "${LOCAL_BASE}/training_results_live/LIVE_SUMMARY.txt"
        echo "--- ${worker} ---" >> "${LOCAL_BASE}/training_results_live/LIVE_SUMMARY.txt"
        tail -5 "${LOCAL_BASE}/training_results_live/logs/worker_${worker}_20260125_1944.log" 2>/dev/null >> "${LOCAL_BASE}/training_results_live/LIVE_SUMMARY.txt" || \
            echo "(No log data)" >> "${LOCAL_BASE}/training_results_live/LIVE_SUMMARY.txt"
    done
    
    echo ""
    echo "✅ Sync #${SYNC_COUNT} complete at ${TIMESTAMP}"
    echo "   View results: ${LOCAL_BASE}/training_results_live/"
    echo "   View summary: cat ${LOCAL_BASE}/training_results_live/LIVE_SUMMARY.txt"
    echo ""
    
    # Wait for next sync interval
    sleep ${SYNC_INTERVAL}
done

