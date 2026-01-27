#!/bin/bash
# Sync fast validation results from EC2 to MacBook

set -e

EC2_HOST="ubuntu@34.196.155.11"
EC2_KEY="$HOME/keys/AutoGenKeyPair.pem"
EC2_RESULTS="/home/ubuntu/mono_to_3d/experiments/trajectory_video_understanding/object_level_persistence/results/fast_validation"
LOCAL_RESULTS="/Users/mike/Dropbox/Documents/Machine_Learning/LLM_and_Multimodal_Models/repos/mono_to_3d/experiments/trajectory_video_understanding/object_level_persistence/results/fast_validation"

echo "Syncing fast validation results from EC2 to MacBook..."

# Create local directory
mkdir -p "$LOCAL_RESULTS"

# Sync results
rsync -avz -e "ssh -i $EC2_KEY" \
  --include="*.png" \
  --include="*.json" \
  --include="*.log" \
  --include="*.txt" \
  --include="*.md" \
  --include="*.pt" \
  --exclude="*" \
  $EC2_HOST:$EC2_RESULTS/ \
  $LOCAL_RESULTS/

echo "✅ Sync complete: $LOCAL_RESULTS"

