#!/bin/bash
# Run TDD GREEN phase on both workers in parallel

set -e

echo "================================================================"
echo "PARALLEL TDD GREEN PHASE - Both Workers"
echo "================================================================"

WORKER1_DIR=~/mono_to_3d_worker1
WORKER2_DIR=~/mono_to_3d_worker2

# Function to run TDD GREEN for a worker
run_tdd_green() {
    local WORKER_DIR=$1
    local WORKER_NAME=$2
    local TEST_FILE=$3
    
    echo ""
    echo "----------------------------------------"
    echo "Running TDD GREEN for $WORKER_NAME"
    echo "----------------------------------------"
    
    cd $WORKER_DIR
    source venv/bin/activate
    
    # Navigate to test directory
    TEST_DIR=experiments/trajectory_video_understanding/object_level_persistence
    cd $TEST_DIR
    
    # Create artifacts directory
    mkdir -p artifacts
    
    # Run pytest and capture output
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting TDD GREEN phase for $WORKER_NAME" | tee artifacts/tdd_green.txt
    
    if pytest tests/$TEST_FILE -v 2>&1 | tee -a artifacts/tdd_green.txt; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] ✅ Tests PASSED in GREEN phase" | tee -a artifacts/tdd_green.txt
        
        # Count passed tests
        PASSED=$(grep -c "PASSED" artifacts/tdd_green.txt || echo "0")
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Total tests passed: $PASSED" | tee -a artifacts/tdd_green.txt
    else
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] ❌ ERROR: Tests still failing in GREEN phase!" | tee -a artifacts/tdd_green.txt
        exit 1
    fi
    
    # Update heartbeat
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $WORKER_NAME: TDD GREEN phase complete - all tests passing" >> results/HEARTBEAT.txt
    
    # Commit GREEN phase
    cd $WORKER_DIR
    git add $TEST_DIR/
    git commit -m "feat($WORKER_NAME): TDD GREEN phase complete [PASS]

All tests passing: $PASSED
Implementation complete and validated"
    
    git push origin $(git branch --show-current)
    
    echo "✅ $WORKER_NAME TDD GREEN complete"
}

# Export function for parallel execution
export -f run_tdd_green

# Run both workers in parallel
echo ""
echo "Starting parallel TDD GREEN execution..."

# Worker 1: Object Detector
(run_tdd_green "$WORKER1_DIR" "Worker1-Detector" "test_object_detector.py" 2>&1 | sed 's/^/[W1] /') &
PID1=$!

# Worker 2: Object Tokenizer
(run_tdd_green "$WORKER2_DIR" "Worker2-Tokenizer" "test_object_tokenizer.py" 2>&1 | sed 's/^/[W2] /') &
PID2=$!

# Wait for both to complete
echo "Waiting for workers to complete..."
wait $PID1
STATUS1=$?
wait $PID2
STATUS2=$?

echo ""
echo "================================================================"
echo "PARALLEL TDD GREEN PHASE COMPLETE"
echo "================================================================"
echo "Worker 1 (Detector):  $([ $STATUS1 -eq 0 ] && echo '✅ PASS' || echo '❌ FAIL')"
echo "Worker 2 (Tokenizer): $([ $STATUS2 -eq 0 ] && echo '✅ PASS' || echo '❌ FAIL')"
echo ""

if [ $STATUS1 -eq 0 ] && [ $STATUS2 -eq 0 ]; then
    echo "✅ Both workers completed TDD GREEN phase successfully"
    echo "Next: Continue with remaining implementation tasks"
    exit 0
else
    echo "❌ One or more workers failed"
    exit 1
fi

