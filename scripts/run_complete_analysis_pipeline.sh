#!/bin/bash
# Complete Analysis Pipeline for Early Persistence Detection
# Runs all 4 components: Evaluation, Attention, Efficiency, Report

set -e

REPO_ROOT=$(git rev-parse --show-toplevel)
EXPERIMENT_DIR="$REPO_ROOT/experiments/trajectory_video_understanding/early_persistence_detection"
RESULTS_DIR="$EXPERIMENT_DIR/results"
EVAL_DIR="$EXPERIMENT_DIR/evaluation"
DATA_DIR="$REPO_ROOT/experiments/trajectory_video_understanding/persistence_augmented_dataset/output"

MODEL_PATH="$RESULTS_DIR/best_model.pt"
OUTPUT_BASE="$RESULTS_DIR/complete_analysis_$(date +%Y%m%d_%H%M)"

echo "========================================================================"
echo "COMPLETE ANALYSIS PIPELINE - EARLY PERSISTENCE DETECTION"
echo "========================================================================"
echo "Timestamp: $(date '+%Y-%m-%d %H:%M:%S')"
echo "Model: $MODEL_PATH"
echo "Data: $DATA_DIR"
echo "Output: $OUTPUT_BASE"
echo "========================================================================"

cd "$REPO_ROOT"

# Activate virtual environment
if [ -d "$REPO_ROOT/venv" ]; then
    source "$REPO_ROOT/venv/bin/activate"
    echo "✅ Activated Python virtual environment"
fi

# Create heartbeat function
update_heartbeat() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" >> "$OUTPUT_BASE/PIPELINE_HEARTBEAT.txt"
}

# Initialize
mkdir -p "$OUTPUT_BASE"
update_heartbeat "Pipeline started"

echo ""
echo "========================================================================"
echo "TASK 1: MODEL EVALUATION WITH ATTENTION EXTRACTION"
echo "========================================================================"
update_heartbeat "Starting Task 1: Model Evaluation"

python3 "$EVAL_DIR/evaluate_model_complete.py" \
    --model "$MODEL_PATH" \
    --data "$DATA_DIR" \
    --output "$OUTPUT_BASE/evaluation" \
    --device cuda \
    --num_samples 500

update_heartbeat "✅ Task 1 Complete: Model Evaluation"
echo "✅ Task 1 Complete: Evaluation metrics saved"

echo ""
echo "========================================================================"
echo "TASK 2: ATTENTION VISUALIZATION"
echo "========================================================================"
update_heartbeat "Starting Task 2: Attention Visualization"

python3 "$EVAL_DIR/visualize_attention_complete.py" \
    --attention_data "$OUTPUT_BASE/evaluation/attention_weights.pt" \
    --output "$OUTPUT_BASE/attention_visualizations" \
    --num_samples 20

update_heartbeat "✅ Task 2 Complete: Attention Visualizations"
echo "✅ Task 2 Complete: Attention heatmaps generated"

echo ""
echo "========================================================================"
echo "TASK 3: EFFICIENCY ANALYSIS"
echo "========================================================================"
update_heartbeat "Starting Task 3: Efficiency Analysis"

python3 "$EVAL_DIR/analyze_efficiency_complete.py" \
    --metrics "$OUTPUT_BASE/evaluation/efficiency_metrics.json" \
    --output "$OUTPUT_BASE/efficiency_analysis"

update_heartbeat "✅ Task 3 Complete: Efficiency Analysis"
echo "✅ Task 3 Complete: Efficiency plots generated"

echo ""
echo "========================================================================"
echo "TASK 4: COMPREHENSIVE REPORT GENERATION"
echo "========================================================================"
update_heartbeat "Starting Task 4: Report Generation"

python3 "$EVAL_DIR/generate_comprehensive_report.py" \
    --eval_metrics "$OUTPUT_BASE/evaluation/evaluation_metrics.json" \
    --efficiency_stats "$OUTPUT_BASE/efficiency_analysis/efficiency_statistics.json" \
    --attention_analysis "$OUTPUT_BASE/attention_visualizations/attention_efficiency_analysis.json" \
    --output "$OUTPUT_BASE/final_report"

update_heartbeat "✅ Task 4 Complete: Comprehensive Report"
echo "✅ Task 4 Complete: Report generated"

# Create summary file
echo "" > "$OUTPUT_BASE/ANALYSIS_COMPLETE.txt"
echo "========================================================================"  >> "$OUTPUT_BASE/ANALYSIS_COMPLETE.txt"
echo "COMPLETE ANALYSIS PIPELINE - FINISHED"  >> "$OUTPUT_BASE/ANALYSIS_COMPLETE.txt"
echo "========================================================================"  >> "$OUTPUT_BASE/ANALYSIS_COMPLETE.txt"
echo "Completed: $(date '+%Y-%m-%d %H:%M:%S')"  >> "$OUTPUT_BASE/ANALYSIS_COMPLETE.txt"
echo ""  >> "$OUTPUT_BASE/ANALYSIS_COMPLETE.txt"
echo "Generated Files:"  >> "$OUTPUT_BASE/ANALYSIS_COMPLETE.txt"
echo "  • evaluation/evaluation_metrics.json"  >> "$OUTPUT_BASE/ANALYSIS_COMPLETE.txt"
echo "  • evaluation/efficiency_metrics.json"  >> "$OUTPUT_BASE/ANALYSIS_COMPLETE.txt"
echo "  • evaluation/attention_weights.pt"  >> "$OUTPUT_BASE/ANALYSIS_COMPLETE.txt"
echo "  • attention_visualizations/*.png (20 heatmaps)"  >> "$OUTPUT_BASE/ANALYSIS_COMPLETE.txt"
echo "  • attention_visualizations/attention_distribution_aggregate.png"  >> "$OUTPUT_BASE/ANALYSIS_COMPLETE.txt"
echo "  • attention_visualizations/attention_efficiency_analysis.json"  >> "$OUTPUT_BASE/ANALYSIS_COMPLETE.txt"
echo "  • efficiency_analysis/decision_frame_histogram.png"  >> "$OUTPUT_BASE/ANALYSIS_COMPLETE.txt"
echo "  • efficiency_analysis/compute_savings_analysis.png"  >> "$OUTPUT_BASE/ANALYSIS_COMPLETE.txt"
echo "  • efficiency_analysis/inference_time_analysis.png"  >> "$OUTPUT_BASE/ANALYSIS_COMPLETE.txt"
echo "  • efficiency_analysis/efficiency_statistics.json"  >> "$OUTPUT_BASE/ANALYSIS_COMPLETE.txt"
echo "  • final_report/COMPREHENSIVE_EVALUATION_REPORT.md"  >> "$OUTPUT_BASE/ANALYSIS_COMPLETE.txt"
echo ""  >> "$OUTPUT_BASE/ANALYSIS_COMPLETE.txt"
echo "========================================================================"  >> "$OUTPUT_BASE/ANALYSIS_COMPLETE.txt"

update_heartbeat "Pipeline completed successfully"

echo ""
echo "========================================================================"
echo "ALL TASKS COMPLETE"
echo "========================================================================"
echo ""
echo "Results saved to: $OUTPUT_BASE"
echo ""
echo "Key files:"
echo "  📊 Final Report: $OUTPUT_BASE/final_report/COMPREHENSIVE_EVALUATION_REPORT.md"
echo "  📈 Attention Viz: $OUTPUT_BASE/attention_visualizations/"
echo "  ⚡ Efficiency: $OUTPUT_BASE/efficiency_analysis/"
echo "  📋 Summary: $OUTPUT_BASE/ANALYSIS_COMPLETE.txt"
echo ""
echo "========================================================================"
echo "To view results on MacBook, sync with:"
echo "  rsync -avz -e 'ssh -i ~/keys/AutoGenKeyPair.pem' \\"
echo "    ubuntu@34.196.155.11:~/mono_to_3d/$OUTPUT_BASE \\"
echo "    $OUTPUT_BASE"
echo "========================================================================"

cat "$OUTPUT_BASE/ANALYSIS_COMPLETE.txt"

