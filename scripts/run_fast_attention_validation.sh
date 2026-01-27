#!/bin/bash
# Fast-track attention validation pipeline
# Train minimal transformer and generate attention visualizations

set -e

echo "================================================================"
echo "FAST ATTENTION VALIDATION PIPELINE"
echo "================================================================"

# Configuration
WORK_DIR=~/mono_to_3d
DATA_ROOT=$WORK_DIR/experiments/trajectory_video_understanding/persistence_augmented_dataset/output
OUTPUT_ROOT=$WORK_DIR/experiments/trajectory_video_understanding/object_level_persistence/results/fast_validation
SCRIPT_DIR=$WORK_DIR/experiments/trajectory_video_understanding/object_level_persistence/scripts

# Create output directory
mkdir -p $OUTPUT_ROOT

# Activate environment
cd $WORK_DIR
source venv/bin/activate

echo ""
echo "Configuration:"
echo "  Data: $DATA_ROOT"
echo "  Output: $OUTPUT_ROOT"
echo "  Device: $(python -c 'import torch; print("cuda" if torch.cuda.is_available() else "cpu")')"
echo ""

# Update heartbeat
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting fast attention validation" >> $OUTPUT_ROOT/HEARTBEAT.txt

# Step 1: Train model
echo "================================================================"
echo "Step 1: Training minimal transformer (10 epochs, 500 samples)"
echo "================================================================"

python $SCRIPT_DIR/train_fast.py \
    --data_root $DATA_ROOT \
    --output_dir $OUTPUT_ROOT/training \
    --max_samples 500 \
    --epochs 10 \
    --batch_size 8 \
    --lr 1e-3 \
    2>&1 | tee $OUTPUT_ROOT/training.log

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Training complete" >> $OUTPUT_ROOT/HEARTBEAT.txt

# Step 2: Generate attention visualizations
echo ""
echo "================================================================"
echo "Step 2: Generating attention visualizations (20 samples)"
echo "================================================================"

python $SCRIPT_DIR/visualize_attention_fast.py \
    --data_root $DATA_ROOT \
    --checkpoint $OUTPUT_ROOT/training/best_model.pt \
    --output_dir $OUTPUT_ROOT/attention_visualizations \
    --num_samples 20 \
    2>&1 | tee $OUTPUT_ROOT/visualization.log

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Visualization complete" >> $OUTPUT_ROOT/HEARTBEAT.txt

# Step 3: Create summary report
echo ""
echo "================================================================"
echo "Step 3: Creating summary report"
echo "================================================================"

python $SCRIPT_DIR/generate_report.py --output_root $OUTPUT_ROOT

echo ""
echo "================================================================"
echo "PIPELINE COMPLETE"
echo "================================================================"
echo ""
echo "Results saved to: $OUTPUT_ROOT"
echo "  - Training: $OUTPUT_ROOT/training/"
echo "  - Visualizations: $OUTPUT_ROOT/attention_visualizations/"
echo "  - Report: $OUTPUT_ROOT/VALIDATION_REPORT.md"
echo ""
echo "Syncing to MacBook..."

# Note: Sync will be handled by periodic sync script
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Pipeline complete" >> $OUTPUT_ROOT/HEARTBEAT.txt

echo ""
echo "✅ Fast attention validation complete!"
echo "================================================================"

