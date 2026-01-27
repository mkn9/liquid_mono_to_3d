#!/usr/bin/env bash
# Simple Test Evidence Capture
# For situations where you just need to capture test output (not full TDD cycle)
# Usage: bash scripts/test_capture.sh [label]

set -euo pipefail

# Label defaults to timestamp if not provided
LABEL="${1:-$(date +%Y%m%d_%H%M%S)}"
mkdir -p artifacts

OUTPUT_FILE="artifacts/test_${LABEL}.txt"

echo "Capturing test output to: $OUTPUT_FILE"

# Generate provenance header
{
  echo "=== Test Evidence Provenance ==="
  echo "Label: $LABEL"
  echo "Timestamp: $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
  echo "Git Commit: $(git rev-parse HEAD 2>/dev/null || echo 'not in git repo')"
  echo "Git Branch: $(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo 'not in git repo')"
  echo "Git Status: $(git diff-index --quiet HEAD -- && echo 'clean' || echo 'uncommitted changes')"
  echo "Python: $(python --version 2>&1 | head -1)"
  echo "pytest: $(pytest --version 2>&1 | head -1)"
  echo "Working Dir: $(pwd)"
  echo "Hostname: $(hostname)"
  echo "OS: $(uname -s -r)"
  echo "Top Dependencies:"
  pip freeze 2>/dev/null | grep -E '^(numpy|torch|pytest|pandas|matplotlib|opencv-python)' | head -10 || echo "  (pip freeze unavailable)"
  echo "=== End Provenance ==="
  echo ""
} | tee "$OUTPUT_FILE"

echo "=== Test Run: $LABEL ===" | tee -a "$OUTPUT_FILE"
echo "=== Command: pytest -q ===" | tee -a "$OUTPUT_FILE"
echo "" | tee -a "$OUTPUT_FILE"

# Run tests and capture output
set +e
pytest -q 2>&1 | tee -a "$OUTPUT_FILE"
EXIT_CODE=${PIPESTATUS[0]}
set -e

echo "" | tee -a "$OUTPUT_FILE"
echo "=== exit_code: $EXIT_CODE ===" | tee -a "$OUTPUT_FILE"

if [ $EXIT_CODE -eq 0 ]; then
  echo "✅ Tests passed"
else
  echo "❌ Tests failed (exit code: $EXIT_CODE)"
fi

echo ""
echo "Evidence saved to: $OUTPUT_FILE"

exit $EXIT_CODE

