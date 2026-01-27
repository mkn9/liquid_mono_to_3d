#!/usr/bin/env bash
# TDD Evidence Capture Script
# Automatically captures RED → GREEN → REFACTOR test outputs
# Usage: bash scripts/tdd_capture.sh

set -euo pipefail

mkdir -p artifacts

# Generate provenance header for evidence file
generate_provenance() {
  local output_file="$1"
  {
    echo "=== TDD Evidence Provenance ==="
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
  } | tee "$output_file"
}

run_and_capture () {
  local label="$1"
  shift
  local output_file="artifacts/tdd_${label}.txt"
  
  # Generate provenance header
  generate_provenance "$output_file"
  
  # Capture test run
  echo "=== $label: $* ===" | tee -a "$output_file"
  # Run command, capture all output, preserve exit code
  set +e
  "$@" 2>&1 | tee -a "$output_file"
  status=${PIPESTATUS[0]}
  set -e
  echo "=== exit_code: $status ===" | tee -a "$output_file"
  return $status
}

echo "Starting TDD workflow with evidence capture..."
echo ""

# "RED" should fail (exit code != 0). If it passes, that's a workflow issue.
echo "Phase 1: RED (tests should fail before implementation)"
if run_and_capture red pytest -q; then
  echo "❌ RED unexpectedly passed. This is not TDD. Failing." | tee -a artifacts/tdd_red.txt
  echo ""
  echo "TDD VIOLATION: Tests passed before implementation exists."
  echo "This means either:"
  echo "  1. Implementation already exists (delete it first)"
  echo "  2. Tests are not properly checking functionality"
  echo ""
  exit 1
fi
echo "✅ RED phase complete: Tests failed as expected"
echo ""

# GREEN must pass
echo "Phase 2: GREEN (tests should pass after implementation)"
if ! run_and_capture green pytest -q; then
  echo "❌ GREEN phase failed. Implementation does not pass tests." | tee -a artifacts/tdd_green.txt
  echo ""
  echo "Fix implementation until tests pass, then run:"
  echo "  pytest -q 2>&1 | tee artifacts/tdd_green.txt"
  echo ""
  exit 1
fi
echo "✅ GREEN phase complete: Tests pass"
echo ""

# REFACTOR is just another pass run after cleanup/formatting
echo "Phase 3: REFACTOR (tests should still pass after refactoring)"
if ! run_and_capture refactor pytest -q; then
  echo "❌ REFACTOR phase failed. Refactoring broke tests." | tee -a artifacts/tdd_refactor.txt
  echo ""
  echo "Revert refactoring or fix issues, then run:"
  echo "  pytest -q 2>&1 | tee artifacts/tdd_refactor.txt"
  echo ""
  exit 1
fi
echo "✅ REFACTOR phase complete: Tests still pass"
echo ""

echo "========================================="
echo "✅ TDD workflow complete!"
echo "Evidence saved under artifacts/tdd_*.txt"
echo ""
echo "Next steps:"
echo "  1. Review artifacts/tdd_*.txt files"
echo "  2. Commit code changes WITH artifacts/"
echo "  3. Reference artifacts in documentation"
echo "========================================="

