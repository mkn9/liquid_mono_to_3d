#!/bin/bash
# Heartbeat Monitor for VLM Integration
# Shows live progress of Workers 2-5

EXPERIMENT_DIR="experiments/liquid_vlm_integration"

echo "=== Liquid VLM Integration - Heartbeat Monitor ==="
echo "Timestamp: $(date +'%Y-%m-%d %H:%M:%S')"
echo ""

# Worker Status
echo "📊 Worker Status:"
for worker in 2 3 4 5; do
    GREEN_FILE=$(ls -t ${EXPERIMENT_DIR}/artifacts/*worker${worker}*green.txt 2>/dev/null | head -1)
    if [ -f "$GREEN_FILE" ]; then
        STATUS=$(tail -1 "$GREEN_FILE" | grep -o "[0-9]* passed")
        if [ ! -z "$STATUS" ]; then
            echo "  Worker $worker: ✅ $STATUS"
        else
            echo "  Worker $worker: ⚙️  Running"
        fi
    else
        echo "  Worker $worker: 🔴 Not started"
    fi
done

echo ""

# Latest Results
echo "📁 Latest Results:"
ls -lht ${EXPERIMENT_DIR}/results/ 2>/dev/null | head -5 | tail -4

echo ""

# Test Summary
echo "🧪 Test Summary:"
TOTAL_TESTS=$(grep -r "passed" ${EXPERIMENT_DIR}/artifacts/*green.txt 2>/dev/null | wc -l)
echo "  Total GREEN tests: $TOTAL_TESTS"

echo ""
echo "✅ Monitor complete at $(date +'%H:%M:%S')"
