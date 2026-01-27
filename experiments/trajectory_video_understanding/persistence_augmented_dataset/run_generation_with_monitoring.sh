#!/bin/bash
#
# Run persistence augmented dataset generation with monitoring
#
# This script:
# 1. Starts heartbeat monitoring
# 2. Runs the batch augmentation on EC2
# 3. Automatically syncs results to MacBook
# 4. Stops monitoring when complete
#

set -e

echo "========================================================================"
echo "PERSISTENCE-AUGMENTED DATASET GENERATION WITH MONITORING"
echo "========================================================================"

# Configuration
EC2_HOST="ubuntu@34.196.155.11"
SSH_KEY="/Users/mike/keys/AutoGenKeyPair.pem"
LOCAL_RESULTS_DIR="/Users/mike/Dropbox/Documents/Machine_Learning/LLM_and_Multimodal_Models/repos/mono_to_3d/experiments/trajectory_video_understanding/persistence_augmented_dataset/results"
EC2_OUTPUT_DIR="~/mono_to_3d/experiments/trajectory_video_understanding/persistence_augmented_dataset/output"

# Create local results directory
mkdir -p "$LOCAL_RESULTS_DIR"

echo ""
echo "Configuration:"
echo "  EC2 Host: $EC2_HOST"
echo "  Local results: $LOCAL_RESULTS_DIR"
echo "  EC2 output: $EC2_OUTPUT_DIR"
echo ""

# Function to sync results from EC2
sync_from_ec2() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Syncing results from EC2..."
    scp -i "$SSH_KEY" -r "$EC2_HOST:$EC2_OUTPUT_DIR/"*.{txt,json} "$LOCAL_RESULTS_DIR/" 2>/dev/null || true
    scp -i "$SSH_KEY" "$EC2_HOST:$EC2_OUTPUT_DIR/PROGRESS.txt" "$LOCAL_RESULTS_DIR/" 2>/dev/null || true
    scp -i "$SSH_KEY" "$EC2_HOST:$EC2_OUTPUT_DIR/checkpoint.json" "$LOCAL_RESULTS_DIR/" 2>/dev/null || true
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Sync complete"
}

# Initial sync to clear any old files
sync_from_ec2

# Copy scripts to EC2
echo ""
echo "Copying code to EC2..."
scp -i "$SSH_KEY" -r ../persistence_augmented_dataset "$EC2_HOST:~/mono_to_3d/experiments/trajectory_video_understanding/"
echo "✅ Code copied to EC2"

# Start background sync loop
echo ""
echo "Starting background sync loop..."
(
    while true; do
        sleep 30  # Sync every 30 seconds
        sync_from_ec2
    done
) &
SYNC_PID=$!
echo "✅ Background sync started (PID: $SYNC_PID)"

# Cleanup function
cleanup() {
    echo ""
    echo "Stopping background sync..."
    kill $SYNC_PID 2>/dev/null || true
    echo "✅ Background sync stopped"
    
    # Final sync
    echo "Performing final sync..."
    sync_from_ec2
    echo "✅ Final sync complete"
}

# Register cleanup on exit
trap cleanup EXIT INT TERM

# Run the generation on EC2
echo ""
echo "========================================================================"
echo "STARTING DATASET GENERATION ON EC2"
echo "========================================================================"
echo ""

ssh -i "$SSH_KEY" "$EC2_HOST" << 'ENDSSH'
cd ~/mono_to_3d
source venv/bin/activate

# Run the batch augmentation
python experiments/trajectory_video_understanding/persistence_augmented_dataset/batch_augment_dataset.py \
    --source_dir ~/mono_to_3d/experiments/trajectory_video_understanding/data/trajectory_dataset_10k \
    --output_dir ~/mono_to_3d/experiments/trajectory_video_understanding/persistence_augmented_dataset/output \
    --num_samples 10000 \
    --transients_per_video 3 \
    --checkpoint_interval 100 \
    --sync_interval 500

echo ""
echo "✅ Dataset generation complete on EC2"
ENDSSH

echo ""
echo "========================================================================"
echo "GENERATION COMPLETE"
echo "========================================================================"
echo ""
echo "Results are in: $LOCAL_RESULTS_DIR"
echo ""

# Show final summary
if [ -f "$LOCAL_RESULTS_DIR/GENERATION_SUMMARY.json" ]; then
    echo "Summary:"
    cat "$LOCAL_RESULTS_DIR/GENERATION_SUMMARY.json"
fi

echo ""
echo "✅ All done!"

