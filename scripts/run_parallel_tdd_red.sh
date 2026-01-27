#!/bin/bash
# Run TDD RED phase on both workers in parallel

set -e

echo "================================================================"
echo "PARALLEL TDD RED PHASE - Both Workers"
echo "================================================================"

WORKER1_DIR=~/mono_to_3d_worker1
WORKER2_DIR=~/mono_to_3d_worker2

# Function to run TDD RED for a worker
run_tdd_red() {
    local WORKER_DIR=$1
    local WORKER_NAME=$2
    local TEST_FILE=$3
    
    echo ""
    echo "----------------------------------------"
    echo "Running TDD RED for $WORKER_NAME"
    echo "----------------------------------------"
    
    cd $WORKER_DIR
    source venv/bin/activate
    
    # Navigate to test directory
    TEST_DIR=experiments/trajectory_video_understanding/object_level_persistence
    cd $TEST_DIR
    
    # Create artifacts directory
    mkdir -p artifacts
    
    # Run pytest and capture output
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting TDD RED phase for $WORKER_NAME" | tee artifacts/tdd_red.txt
    
    if pytest tests/$TEST_FILE -v 2>&1 | tee -a artifacts/tdd_red.txt; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] ❌ ERROR: Tests passed in RED phase (should fail)!" | tee -a artifacts/tdd_red.txt
        exit 1
    else
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] ✅ Tests FAILED as expected in RED phase" | tee -a artifacts/tdd_red.txt
    fi
    
    # Update heartbeat
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $WORKER_NAME: TDD RED phase complete - tests failed as expected" >> results/HEARTBEAT.txt
    
    # Commit RED phase
    cd $WORKER_DIR
    git add $TEST_DIR/
    git commit -m "test($WORKER_NAME): TDD RED phase complete [FAIL as expected]

Tests implemented: $(grep -c 'def test_' $TEST_DIR/tests/$TEST_FILE)
All tests fail with NotImplementedError - ready for GREEN phase"
    
    git push origin $(git branch --show-current)
    
    echo "✅ $WORKER_NAME TDD RED complete"
}

# Export function for parallel execution
export -f run_tdd_red

# Run both workers in parallel
echo ""
echo "Starting parallel TDD RED execution..."

# Worker 1: Object Detector
(run_tdd_red "$WORKER1_DIR" "Worker1-Detector" "test_object_detector.py" 2>&1 | sed 's/^/[W1] /') &
PID1=$!

# Worker 2: Object Tokenizer
(run_tdd_red "$WORKER2_DIR" "Worker2-Tokenizer" "test_object_tokenizer.py" 2>&1 | sed 's/^/[W2] /') &
PID2=$!

# Wait for both to complete
echo "Waiting for workers to complete..."
wait $PID1
STATUS1=$?
wait $PID2
STATUS2=$?

echo ""
echo "================================================================"
echo "PARALLEL TDD RED PHASE COMPLETE"
echo "================================================================"
echo "Worker 1 (Detector):  $([ $STATUS1 -eq 0 ] && echo '✅ PASS' || echo '❌ FAIL')"
echo "Worker 2 (Tokenizer): $([ $STATUS2 -eq 0 ] && echo '✅ PASS' || echo '❌ FAIL')"
echo ""

if [ $STATUS1 -eq 0 ] && [ $STATUS2 -eq 0 ]; then
    echo "✅ Both workers completed TDD RED phase successfully"
    echo "Next: Implement GREEN phase for both workers"
    exit 0
else
    echo "❌ One or more workers failed"
    exit 1
fi

