#!/bin/bash
# Complete Evaluation Pipeline for Early Persistence Detection
# Runs attention visualization, efficiency analysis, and report generation

set -e

REPO_ROOT=$(git rev-parse --show-toplevel)
EXPERIMENT_DIR="$REPO_ROOT/experiments/trajectory_video_understanding/early_persistence_detection"
RESULTS_DIR="$EXPERIMENT_DIR/results"
EVAL_DIR="$EXPERIMENT_DIR/evaluation"
DATA_DIR="$REPO_ROOT/experiments/trajectory_video_understanding/persistence_augmented_dataset/output"

MODEL_PATH="$RESULTS_DIR/best_model.pt"

echo "========================================"
echo "COMPLETE EVALUATION PIPELINE"
echo "========================================"
echo "Model: $MODEL_PATH"
echo "Data: $DATA_DIR"
echo "Output: $RESULTS_DIR"
echo ""

cd "$REPO_ROOT"

# Activate virtual environment if it exists
if [ -d "$REPO_ROOT/venv" ]; then
    source "$REPO_ROOT/venv/bin/activate"
    echo "✅ Activated Python virtual environment"
fi

# 1. Run full model evaluation with efficiency metrics
echo ""
echo "1. Running model evaluation with efficiency metrics..."
echo "========================================" python3 "$EVAL_DIR/evaluate_model.py" \
    --model "$MODEL_PATH" \
    --test_data "$DATA_DIR" \
    --output "$RESULTS_DIR/evaluation" \
    --device cuda \
    --num_samples 500

echo "✅ Model evaluation complete"

# 2. Generate attention visualizations
echo ""
echo "2. Generating attention visualizations..."
echo "========================================"

python3 "$EVAL_DIR/visualize_attention.py" \
    --model "$MODEL_PATH" \
    --data "$DATA_DIR" \
    --output "$RESULTS_DIR/attention_visualizations" \
    --num_samples 20 \
    --device cuda

echo "✅ Attention visualizations complete"

# 3. Analyze efficiency metrics
echo ""
echo "3. Analyzing efficiency metrics..."
echo "========================================"

python3 "$EVAL_DIR/analyze_efficiency.py" \
    --metrics "$RESULTS_DIR/evaluation/efficiency_metrics.json" \
    --output "$RESULTS_DIR/efficiency_analysis"

echo "✅ Efficiency analysis complete"

# 4. Generate comprehensive report
echo ""
echo "4. Generating comprehensive report..."
echo "========================================"

python3 "$EVAL_DIR/generate_report.py" \
    --eval_results "$RESULTS_DIR/evaluation" \
    --output "$RESULTS_DIR/final_report"

echo "✅ Report generation complete"

# Summary
echo ""
echo "========================================"
echo "EVALUATION COMPLETE"
echo "========================================"
echo ""
echo "Results saved to: $RESULTS_DIR"
echo ""
echo "Generated files:"
echo "  - evaluation/evaluation_metrics.json"
echo "  - evaluation/confusion_matrix.png"
echo "  - evaluation/efficiency_metrics.json"
echo "  - attention_visualizations/*.png"
echo "  - efficiency_analysis/*.png"
echo "  - final_report/report.md"
echo "  - final_report/report.html"
echo ""
echo "========================================"

