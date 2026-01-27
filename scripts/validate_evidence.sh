#!/usr/bin/env bash
# Evidence Validator Script
# Checks that TDD evidence files exist and are valid
# Usage: bash scripts/validate_evidence.sh [--strict]
#
# Exit codes:
#   0 = Evidence valid
#   1 = Evidence missing or invalid
#   2 = Usage error

set -euo pipefail

STRICT_MODE=false
if [[ "${1:-}" == "--strict" ]]; then
  STRICT_MODE=true
fi

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

error() {
  echo -e "${RED}❌ ERROR: $*${NC}" >&2
}

warning() {
  echo -e "${YELLOW}⚠️  WARNING: $*${NC}" >&2
}

success() {
  echo -e "${GREEN}✅ $*${NC}"
}

info() {
  echo "ℹ️  $*"
}

# Check if running in git repo
if ! git rev-parse --git-dir > /dev/null 2>&1; then
  error "Not in a git repository"
  exit 2
fi

# Get list of changed files in last commit
CHANGED_FILES=$(git diff --name-only HEAD~1 HEAD 2>/dev/null || echo "")

# Check if any source or test files changed
SRC_CHANGED=false
TEST_CHANGED=false

while IFS= read -r file; do
  if [[ -z "$file" ]]; then
    continue
  fi
  
  # Check for source code changes (excluding artifacts/)
  if [[ "$file" =~ \.(py|ipynb)$ ]] && [[ ! "$file" =~ ^artifacts/ ]]; then
    if [[ "$file" =~ ^tests/ ]] || [[ "$file" =~ test_ ]]; then
      TEST_CHANGED=true
    else
      SRC_CHANGED=true
    fi
  fi
done <<< "$CHANGED_FILES"

# If no relevant changes, validation passes
if [[ "$SRC_CHANGED" == "false" ]] && [[ "$TEST_CHANGED" == "false" ]]; then
  success "No source or test files changed - validation not required"
  exit 0
fi

info "Detected changes:"
[[ "$SRC_CHANGED" == "true" ]] && info "  - Source code changed"
[[ "$TEST_CHANGED" == "true" ]] && info "  - Tests changed"
echo ""

# Check for evidence files
EVIDENCE_DIR="artifacts"
VALIDATION_PASSED=true

# Required evidence files (at minimum, need green or refactor)
REQUIRED_FILES=()
if [[ "$STRICT_MODE" == "true" ]]; then
  # Strict mode: require all TDD phases
  REQUIRED_FILES=("tdd_red.txt" "tdd_green.txt" "tdd_refactor.txt")
else
  # Lenient mode: require at least green or refactor evidence
  REQUIRED_FILES=("tdd_green.txt" "tdd_refactor.txt")
fi

# Check if artifacts directory exists
if [[ ! -d "$EVIDENCE_DIR" ]]; then
  error "Evidence directory '$EVIDENCE_DIR/' does not exist"
  VALIDATION_PASSED=false
else
  info "Checking evidence files in $EVIDENCE_DIR/..."
  echo ""
  
  # In lenient mode, check for ANY valid evidence file
  if [[ "$STRICT_MODE" == "false" ]]; then
    FOUND_EVIDENCE=false
    
    # Check for TDD evidence files
    if [[ -f "$EVIDENCE_DIR/tdd_green.txt" ]] || [[ -f "$EVIDENCE_DIR/tdd_refactor.txt" ]]; then
      FOUND_EVIDENCE=true
    fi
    
    # Check for test_*.txt files as alternative evidence
    if ls "$EVIDENCE_DIR"/test_*.txt 1> /dev/null 2>&1; then
      FOUND_EVIDENCE=true
    fi
    
    if [[ "$FOUND_EVIDENCE" == "false" ]]; then
      error "No evidence files found"
      info "Required: At least one of:"
      info "  - artifacts/tdd_green.txt"
      info "  - artifacts/tdd_refactor.txt"
      info "  - artifacts/test_*.txt"
      VALIDATION_PASSED=false
    fi
  fi
  
  # Check each required file (strict mode) or TDD files if they exist (lenient mode)
  for file in "${REQUIRED_FILES[@]}"; do
    filepath="$EVIDENCE_DIR/$file"
    
    if [[ ! -f "$filepath" ]]; then
      if [[ "$STRICT_MODE" == "true" ]]; then
        error "Missing: $filepath"
        VALIDATION_PASSED=false
      else
        warning "Optional file missing: $filepath (not required in lenient mode)"
      fi
      continue
    fi
    
    info "Validating: $filepath"
    
    # Check file is not empty
    if [[ ! -s "$filepath" ]]; then
      error "$filepath is empty"
      VALIDATION_PASSED=false
      continue
    fi
    
    # Check for provenance section
    if ! grep -q "=== TDD Evidence Provenance ===" "$filepath" && \
       ! grep -q "=== Test Evidence Provenance ===" "$filepath"; then
      warning "$filepath missing provenance section (old format?)"
    fi
    
    # Check exit code is recorded
    if ! grep -q "=== exit_code:" "$filepath"; then
      error "$filepath missing exit code"
      VALIDATION_PASSED=false
      continue
    fi
    
    # Extract exit code
    EXIT_CODE=$(grep "=== exit_code:" "$filepath" | tail -1 | awk '{print $3}')
    
    # Validate exit code based on phase
    if [[ "$file" == "tdd_red.txt" ]]; then
      # RED phase should have non-zero exit code (tests should fail)
      if [[ "$EXIT_CODE" == "0" ]]; then
        error "$filepath: RED phase should fail (exit code should be non-zero, got $EXIT_CODE)"
        error "This suggests tests passed before implementation (not proper TDD)"
        VALIDATION_PASSED=false
      else
        success "$filepath: RED phase failed as expected (exit code: $EXIT_CODE)"
      fi
    elif [[ "$file" == "tdd_green.txt" ]] || [[ "$file" == "tdd_refactor.txt" ]]; then
      # GREEN and REFACTOR phases should have zero exit code (tests should pass)
      if [[ "$EXIT_CODE" != "0" ]]; then
        error "$filepath: Tests should pass (exit code should be 0, got $EXIT_CODE)"
        VALIDATION_PASSED=false
      else
        success "$filepath: Tests passed (exit code: $EXIT_CODE)"
      fi
    fi
  done
fi

echo ""
echo "========================================="

if [[ "$VALIDATION_PASSED" == "true" ]]; then
  success "Evidence validation PASSED"
  echo ""
  info "Evidence files are valid and support TDD claims"
  exit 0
else
  error "Evidence validation FAILED"
  echo ""
  info "To fix:"
  info "  1. Run: bash scripts/tdd_capture.sh"
  info "  2. Or: bash scripts/test_capture.sh [label]"
  info "  3. Commit artifacts/ with your changes"
  echo ""
  info "To skip validation (exploratory work):"
  info "  SKIP_EVIDENCE=1 git push"
  exit 1
fi

