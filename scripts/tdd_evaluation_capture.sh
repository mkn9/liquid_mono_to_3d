#!/bin/bash
# TDD Evidence Capture for Evaluation Scripts
# Captures RED, GREEN, and REFACTOR phases for evaluation code

set -e

# Determine project root (works on both MacBook and EC2)
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"
cd "$PROJECT_ROOT"

# Activate virtual environment
if [ -d "venv" ]; then
    source venv/bin/activate
fi

ARTIFACTS_DIR="$PROJECT_ROOT/artifacts"
mkdir -p "$ARTIFACTS_DIR"

TEST_FILE="experiments/trajectory_video_understanding/early_persistence_detection/evaluation/tests/test_evaluation_scripts.py"

PHASE="${1:-red}"

echo "========================================"
echo "TDD EVIDENCE CAPTURE: Evaluation Scripts"
echo "Phase: $(echo $PHASE | tr '[:lower:]' '[:upper:]')"
echo "========================================"

case "$PHASE" in
    red|RED)
        echo ""
        echo "Running RED phase..."
        set +e
        pytest "$TEST_FILE" -v --tb=short > "$ARTIFACTS_DIR/tdd_evaluation_red.txt" 2>&1
        RED_EXIT_CODE=$?
        set -e

        if [ $RED_EXIT_CODE -eq 0 ]; then
            echo "❌ ERROR: Tests passed in RED phase! They should fail!"
            echo "Tests must fail before implementation."
            exit 1
        else
            echo "✅ RED phase complete: Tests failed as expected"
            echo "Evidence saved to: $ARTIFACTS_DIR/tdd_evaluation_red.txt"
            echo ""
            echo "Next: Implement the evaluation scripts, then run:"
            echo "  bash scripts/tdd_evaluation_capture.sh green"
        fi
        ;;

    green|GREEN)
        echo ""
        echo "Running GREEN phase..."
        set +e
        pytest "$TEST_FILE" -v --tb=short > "$ARTIFACTS_DIR/tdd_evaluation_green.txt" 2>&1
        GREEN_EXIT_CODE=$?
        set -e

        if [ $GREEN_EXIT_CODE -ne 0 ]; then
            echo "❌ ERROR: Tests failed in GREEN phase!"
            echo "Review failures in: $ARTIFACTS_DIR/tdd_evaluation_green.txt"
            echo ""
            echo "Fix implementation and run again:"
            echo "  bash scripts/tdd_evaluation_capture.sh green"
            exit 1
        else
            echo "✅ GREEN phase complete: All tests pass!"
            echo "Evidence saved to: $ARTIFACTS_DIR/tdd_evaluation_green.txt"
            echo ""
            echo "Next: If refactoring is needed, run:"
            echo "  bash scripts/tdd_evaluation_capture.sh refactor"
            echo "Otherwise, proceed to integration!"
        fi
        ;;

    refactor|REFACTOR)
        echo ""
        echo "Running REFACTOR phase..."
        set +e
        pytest "$TEST_FILE" -v --tb=short > "$ARTIFACTS_DIR/tdd_evaluation_refactor.txt" 2>&1
        REFACTOR_EXIT_CODE=$?
        set -e

        if [ $REFACTOR_EXIT_CODE -ne 0 ]; then
            echo "❌ ERROR: Tests failed after refactoring!"
            echo "Review failures in: $ARTIFACTS_DIR/tdd_evaluation_refactor.txt"
            exit 1
        else
            echo "✅ REFACTOR phase complete: All tests still pass!"
            echo "Evidence saved to: $ARTIFACTS_DIR/tdd_evaluation_refactor.txt"
            echo ""
            echo "🎉 TDD cycle complete! Ready for production."
        fi
        ;;

    *)
        echo "❌ ERROR: Unknown phase '$PHASE'"
        echo "Usage: bash scripts/tdd_evaluation_capture.sh [red|green|refactor]"
        exit 1
        ;;
esac

