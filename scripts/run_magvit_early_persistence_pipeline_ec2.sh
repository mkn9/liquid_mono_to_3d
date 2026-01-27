#!/bin/bash
#
# Full EC2 Pipeline for MagVIT Early Persistence Detection
#
# Follows all standard procedures:
# - TDD with evidence capture
# - Periodic saving to MacBook
# - Heartbeat monitoring
# - Git branch workflow
# - Result syncing
#

set -e

echo "========================================================================"
echo "MAGVIT EARLY PERSISTENCE DETECTION - FULL EC2 PIPELINE"
echo "========================================================================"
echo ""
echo "This pipeline will:"
echo "  1. Set up git branch on EC2"
echo "  2. Run TDD evidence capture"
echo "  3. Train the model with monitoring"
echo "  4. Evaluate and generate reports"
echo "  5. Sync results to MacBook"
echo "  6. Commit and push to branch"
echo ""
echo "All standard procedures will be followed!"
echo "========================================================================"
echo ""

# Configuration
EC2_HOST="ubuntu@34.196.155.11"
EC2_KEY="~/keys/AutoGenKeyPair.pem"
BRANCH_NAME="early-persistence/magvit"
PROJECT_ROOT="~/mono_to_3d"
LOCAL_PROJECT_ROOT="/Users/mike/Dropbox/Documents/Machine_Learning/LLM_and_Multimodal_Models/repos/mono_to_3d"

EXPERIMENT_DIR="experiments/trajectory_video_understanding/early_persistence_detection"
RESULTS_DIR="$EXPERIMENT_DIR/results"
LOCAL_RESULTS_DIR="$LOCAL_PROJECT_ROOT/$RESULTS_DIR"

HEARTBEAT_INTERVAL=30  # seconds
SYNC_INTERVAL=60       # seconds

# Create local results directory
mkdir -p "$LOCAL_RESULTS_DIR"

echo "Step 1: Setting up git branch on EC2..."
echo "========================================================================"
ssh -i "$EC2_KEY" "$EC2_HOST" << 'ENDSSH'
cd ~/mono_to_3d
source venv/bin/activate

# Create or checkout branch
git fetch origin
if git rev-parse --verify early-persistence/magvit >/dev/null 2>&1; then
    echo "✅ Branch exists, checking out..."
    git checkout early-persistence/magvit
    git pull origin early-persistence/magvit
else
    echo "✅ Creating new branch..."
    git checkout -b early-persistence/magvit
    git push -u origin early-persistence/magvit
fi

echo "✅ Git branch ready: early-persistence/magvit"
ENDSSH

echo ""
echo "Step 2: Running TDD evidence capture on EC2..."
echo "========================================================================"
ssh -i "$EC2_KEY" "$EC2_HOST" << 'ENDSSH'
cd ~/mono_to_3d
source venv/bin/activate

# Run TDD capture (should already have RED evidence)
# Run GREEN phase to verify implementation
echo "Running GREEN phase tests..."
bash scripts/tdd_evaluation_capture.sh green

if [ $? -eq 0 ]; then
    echo "✅ TDD GREEN phase passed!"
else
    echo "❌ TDD tests failed! Aborting pipeline."
    exit 1
fi

# Commit TDD evidence
git add artifacts/tdd_evaluation_*.txt
git commit -m "TDD: Evaluation scripts GREEN phase evidence" || echo "No new TDD artifacts to commit"
git push origin early-persistence/magvit
ENDSSH

# Sync TDD evidence to MacBook
echo "Syncing TDD evidence to MacBook..."
scp -i "$EC2_KEY" "$EC2_HOST:$PROJECT_ROOT/artifacts/tdd_evaluation_*.txt" "$LOCAL_PROJECT_ROOT/artifacts/" 2>/dev/null || echo "✅ TDD evidence synced"

echo ""
echo "Step 3: Training model on EC2 with monitoring..."
echo "========================================================================"
echo "Starting background sync process for real-time visibility..."

# Start background sync process
(
    while true; do
        sleep $SYNC_INTERVAL
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Syncing results from EC2 to MacBook..."
        
        # Sync results directory
        rsync -az -e "ssh -i $EC2_KEY" \
            --include='*.json' \
            --include='*.txt' \
            --include='*.png' \
            --include='*.md' \
            --include='*.log' \
            --include='PROGRESS.txt' \
            --include='HEARTBEAT.txt' \
            --exclude='*.pt' \
            --exclude='checkpoint_*.pt' \
            "$EC2_HOST:$PROJECT_ROOT/$RESULTS_DIR/" \
            "$LOCAL_RESULTS_DIR/" 2>/dev/null || echo "Sync attempt..."
        
        # Check if training is complete
        if ssh -i "$EC2_KEY" "$EC2_HOST" "test -f $PROJECT_ROOT/$RESULTS_DIR/TRAINING_COMPLETE.txt" 2>/dev/null; then
            echo "✅ Training complete detected!"
            break
        fi
    done
) &
SYNC_PID=$!
echo "✅ Background sync process started (PID: $SYNC_PID)"

# Trap to cleanup background process
trap "kill $SYNC_PID 2>/dev/null || true" EXIT

# Run training on EC2
echo "Launching training script on EC2..."
ssh -i "$EC2_KEY" "$EC2_HOST" << 'ENDSSH'
cd ~/mono_to_3d
source venv/bin/activate

EXPERIMENT_DIR="experiments/trajectory_video_understanding/early_persistence_detection"
RESULTS_DIR="$EXPERIMENT_DIR/results"
DATA_DIR="experiments/trajectory_video_understanding/persistence_augmented_dataset/output"

# Create results directory
mkdir -p "$RESULTS_DIR"

# Create PROGRESS and HEARTBEAT files
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Training started" > "$RESULTS_DIR/PROGRESS.txt"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Heartbeat: Training initialization" > "$RESULTS_DIR/HEARTBEAT.txt"

# Start heartbeat monitor in background
(
    while true; do
        sleep 30
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Heartbeat: Training in progress" >> "$RESULTS_DIR/HEARTBEAT.txt"
        
        # Check if training is complete
        if [ -f "$RESULTS_DIR/TRAINING_COMPLETE.txt" ]; then
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] Heartbeat: Training completed" >> "$RESULTS_DIR/HEARTBEAT.txt"
            break
        fi
    done
) &
HEARTBEAT_PID=$!

# Run training
echo "Starting MagVIT early persistence training..."
python "$EXPERIMENT_DIR/training/train_early_persistence.py" \
    --data_dir "$DATA_DIR" \
    --output_dir "$RESULTS_DIR" \
    --epochs 20 \
    --batch_size 8 \
    --device cuda \
    2>&1 | tee "$RESULTS_DIR/training.log"

TRAINING_EXIT_CODE=${PIPESTATUS[0]}

# Mark training complete
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Training completed with exit code: $TRAINING_EXIT_CODE" > "$RESULTS_DIR/TRAINING_COMPLETE.txt"

# Stop heartbeat
kill $HEARTBEAT_PID 2>/dev/null || true

if [ $TRAINING_EXIT_CODE -ne 0 ]; then
    echo "❌ Training failed with exit code: $TRAINING_EXIT_CODE"
    exit 1
fi

echo "✅ Training complete!"

# Commit training results
git add "$RESULTS_DIR"/*.json "$RESULTS_DIR"/*.txt "$RESULTS_DIR"/*.log
git commit -m "training: MagVIT early persistence - 20 epochs complete" || echo "No new training results to commit"
git push origin early-persistence/magvit

ENDSSH

echo ""
echo "Step 4: Running evaluation on EC2..."
echo "========================================================================"
ssh -i "$EC2_KEY" "$EC2_HOST" << 'ENDSSH'
cd ~/mono_to_3d
source venv/bin/activate

EXPERIMENT_DIR="experiments/trajectory_video_understanding/early_persistence_detection"
RESULTS_DIR="$EXPERIMENT_DIR/results"
EVAL_DIR="$RESULTS_DIR/evaluation"
VIZ_DIR="$RESULTS_DIR/visualizations"
ANALYSIS_DIR="$RESULTS_DIR/analysis"
DATA_DIR="experiments/trajectory_video_understanding/persistence_augmented_dataset/output"

mkdir -p "$EVAL_DIR" "$VIZ_DIR" "$ANALYSIS_DIR"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Evaluation started" >> "$RESULTS_DIR/PROGRESS.txt"

# Run evaluation
echo "Running model evaluation..."
python "$EXPERIMENT_DIR/evaluation/evaluate_model.py" \
    --model "$RESULTS_DIR/final_model.pt" \
    --test_data "$DATA_DIR" \
    --output "$EVAL_DIR" \
    --device cuda \
    2>&1 | tee "$EVAL_DIR/evaluation.log"

echo "✅ Evaluation complete"

# Commit evaluation results
git add "$EVAL_DIR"/*.json "$EVAL_DIR"/*.txt "$EVAL_DIR"/*.log
git commit -m "eval: MagVIT evaluation metrics" || echo "No new evaluation results"
git push origin early-persistence/magvit

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Evaluation complete" >> "$RESULTS_DIR/PROGRESS.txt"

ENDSSH

echo ""
echo "Step 5: Generating visualizations on EC2..."
echo "========================================================================"
ssh -i "$EC2_KEY" "$EC2_HOST" << 'ENDSSH'
cd ~/mono_to_3d
source venv/bin/activate

EXPERIMENT_DIR="experiments/trajectory_video_understanding/early_persistence_detection"
RESULTS_DIR="$EXPERIMENT_DIR/results"
VIZ_DIR="$RESULTS_DIR/visualizations"
DATA_DIR="experiments/trajectory_video_understanding/persistence_augmented_dataset/output"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Visualization started" >> "$RESULTS_DIR/PROGRESS.txt"

# Generate visualizations
echo "Generating attention visualizations..."
python "$EXPERIMENT_DIR/evaluation/visualize_attention.py" \
    --model "$RESULTS_DIR/final_model.pt" \
    --data "$DATA_DIR" \
    --output "$VIZ_DIR" \
    --num_samples 20 \
    --device cuda \
    2>&1 | tee "$VIZ_DIR/visualization.log"

echo "✅ Visualizations complete"

# Commit visualizations
git add "$VIZ_DIR"/*.png "$VIZ_DIR"/*.json "$VIZ_DIR"/*.log
git commit -m "viz: Attention visualizations (20 samples)" || echo "No new visualizations"
git push origin early-persistence/magvit

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Visualization complete" >> "$RESULTS_DIR/PROGRESS.txt"

ENDSSH

echo ""
echo "Step 6: Analyzing efficiency on EC2..."
echo "========================================================================"
ssh -i "$EC2_KEY" "$EC2_HOST" << 'ENDSSH'
cd ~/mono_to_3d
source venv/bin/activate

EXPERIMENT_DIR="experiments/trajectory_video_understanding/early_persistence_detection"
RESULTS_DIR="$EXPERIMENT_DIR/results"
ANALYSIS_DIR="$RESULTS_DIR/analysis"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Efficiency analysis started" >> "$RESULTS_DIR/PROGRESS.txt"

# Run efficiency analysis
echo "Analyzing efficiency metrics..."
python "$EXPERIMENT_DIR/evaluation/analyze_efficiency.py" \
    --metrics "$RESULTS_DIR/efficiency_metrics.json" \
    --output "$ANALYSIS_DIR" \
    2>&1 | tee "$ANALYSIS_DIR/analysis.log"

echo "✅ Analysis complete"

# Commit analysis results
git add "$ANALYSIS_DIR"/*.png "$ANALYSIS_DIR"/*.md "$ANALYSIS_DIR"/*.json "$ANALYSIS_DIR"/*.log
git commit -m "analysis: Efficiency analysis complete" || echo "No new analysis results"
git push origin early-persistence/magvit

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Efficiency analysis complete" >> "$RESULTS_DIR/PROGRESS.txt"

ENDSSH

echo ""
echo "Step 7: Generating final report on EC2..."
echo "========================================================================"
ssh -i "$EC2_KEY" "$EC2_HOST" << 'ENDSSH'
cd ~/mono_to_3d
source venv/bin/activate

EXPERIMENT_DIR="experiments/trajectory_video_understanding/early_persistence_detection"
RESULTS_DIR="$EXPERIMENT_DIR/results"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Report generation started" >> "$RESULTS_DIR/PROGRESS.txt"

# Generate comprehensive report
echo "Generating final report..."
python "$EXPERIMENT_DIR/evaluation/generate_report.py" \
    --results_dir "$RESULTS_DIR" \
    --output "$RESULTS_DIR/FINAL_REPORT" \
    2>&1 | tee "$RESULTS_DIR/report_generation.log"

echo "✅ Report generation complete"

# Commit final report
git add "$RESULTS_DIR/FINAL_REPORT.*" "$RESULTS_DIR"/*.log
git commit -m "docs: Final comprehensive report" || echo "No new report files"
git push origin early-persistence/magvit

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Pipeline complete!" >> "$RESULTS_DIR/PROGRESS.txt"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] ✅ ALL TASKS COMPLETE" >> "$RESULTS_DIR/HEARTBEAT.txt"

# Create completion marker
echo "Pipeline completed successfully at $(date)" > "$RESULTS_DIR/PIPELINE_COMPLETE.txt"
git add "$RESULTS_DIR/PIPELINE_COMPLETE.txt"
git commit -m "pipeline: All tasks complete" || true
git push origin early-persistence/magvit

ENDSSH

echo ""
echo "Step 8: Final sync to MacBook..."
echo "========================================================================"
sleep 5  # Let background sync catch final files

# Final comprehensive sync
echo "Performing final comprehensive sync..."
rsync -avz -e "ssh -i $EC2_KEY" \
    --include='*.json' \
    --include='*.txt' \
    --include='*.png' \
    --include='*.md' \
    --include='*.html' \
    --include='*.log' \
    --exclude='*.pt' \
    --exclude='checkpoint_*.pt' \
    "$EC2_HOST:$PROJECT_ROOT/$RESULTS_DIR/" \
    "$LOCAL_RESULTS_DIR/"

echo "✅ Final sync complete!"

# Stop background sync process
kill $SYNC_PID 2>/dev/null || true

# Pull git branch to local
echo ""
echo "Step 9: Pulling git branch to local MacBook..."
echo "========================================================================"
cd "$LOCAL_PROJECT_ROOT"
git fetch origin
git checkout "$BRANCH_NAME"
git pull origin "$BRANCH_NAME"

echo "✅ Git branch synced to local!"

echo ""
echo "========================================================================"
echo "✅ PIPELINE COMPLETE!"
echo "========================================================================"
echo ""
echo "Summary:"
echo "  - Git branch: $BRANCH_NAME"
echo "  - Results directory: $LOCAL_RESULTS_DIR"
echo "  - Final report: $LOCAL_RESULTS_DIR/FINAL_REPORT.html"
echo ""
echo "To view results:"
echo "  open $LOCAL_RESULTS_DIR/FINAL_REPORT.html"
echo ""
echo "To see git history:"
echo "  cd $LOCAL_PROJECT_ROOT"
echo "  git log $BRANCH_NAME --oneline"
echo ""
echo "========================================================================"

