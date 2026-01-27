#!/bin/bash
# Master Verification Script
# Run this before claiming ANY work is complete

set -e  # Exit immediately if any command fails

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

echo -e "${BLUE}======================================================================${NC}"
echo -e "${BLUE}          VERIFICATION BEFORE CLAIMING COMPLETE${NC}"
echo -e "${BLUE}======================================================================${NC}"
echo
echo "This script verifies 7 types of lies/misrepresentations:"
echo "  1. Component Existence"
echo "  2. Architecture Fidelity"
echo "  3. Integration Reality"
echo "  4. TDD Compliance"
echo "  5. Execution Method"
echo "  6. Visual Evidence"
echo "  7. Test Sufficiency"
echo
echo "If ANY check fails, you cannot claim 'complete'."
echo
read -p "Press Enter to start verification..."
echo

# Track failures
FAILED_CHECKS=()

# ============================================================================
# CHECK 1: Component Existence
# ============================================================================

echo -e "${BLUE}[1/7] Checking Component Existence...${NC}"
echo "      (Verifying claimed components actually exist)"
echo

if [ -f "scripts/verify_component_claims.py" ] && [ -f ".component_spec.yaml" ]; then
    if python3 scripts/verify_component_claims.py \
        --spec .component_spec.yaml \
        --branch-spec .branch_specifications.json 2>&1 | tee /tmp/component_check.log; then
        echo -e "      ${GREEN}✓ Component existence verified${NC}"
    else
        echo -e "      ${RED}✗ FAILED: Components claimed but not found${NC}"
        cat /tmp/component_check.log | tail -20
        FAILED_CHECKS+=("Component Existence")
    fi
else
    echo -e "      ${YELLOW}⚠ Skipped: No component spec found${NC}"
    echo -e "      ${YELLOW}  Create .component_spec.yaml to enable this check${NC}"
fi
echo

# ============================================================================
# CHECK 2: Architecture Fidelity
# ============================================================================

echo -e "${BLUE}[2/7] Checking Architecture Fidelity...${NC}"
echo "      (Verifying models match their claimed architectures)"
echo

if [ -f "tests/test_honesty_enforcement.py" ]; then
    if pytest tests/test_honesty_enforcement.py::test_i3d_architecture_is_not_simplified \
           tests/test_honesty_enforcement.py::test_slowfast_architecture_is_not_simplified \
           tests/test_honesty_enforcement.py::test_model_has_inception_modules_if_claims_i3d \
           -v 2>&1 | tee /tmp/arch_check.log; then
        echo -e "      ${GREEN}✓ Architecture fidelity verified${NC}"
    else
        echo -e "      ${RED}✗ FAILED: Models don't match claimed architectures${NC}"
        echo -e "      ${RED}  Likely cause: 'Simplified' models named as full architectures${NC}"
        FAILED_CHECKS+=("Architecture Fidelity")
    fi
else
    echo -e "      ${YELLOW}⚠ Skipped: No honesty enforcement tests found${NC}"
fi
echo

# ============================================================================
# CHECK 3: Integration Reality
# ============================================================================

echo -e "${BLUE}[3/7] Checking Integration Reality...${NC}"
echo "      (Verifying APIs/models are actually called, not just templates)"
echo

if [ -f "tests/test_honesty_enforcement.py" ]; then
    if pytest tests/test_honesty_enforcement.py::test_gpt4_integration_makes_actual_api_call \
           tests/test_honesty_enforcement.py::test_magvit_encodes_and_decodes_video \
           tests/test_honesty_enforcement.py::test_clip_encodes_images_and_text \
           -v 2>&1 | tee /tmp/integration_check.log; then
        echo -e "      ${GREEN}✓ Integration reality verified${NC}"
    else
        echo -e "      ${RED}✗ FAILED: Integrations are templates, not real${NC}"
        echo -e "      ${RED}  Likely cause: Template code named as 'integration'${NC}"
        FAILED_CHECKS+=("Integration Reality")
    fi
else
    echo -e "      ${YELLOW}⚠ Skipped: No integration tests found${NC}"
fi
echo

# ============================================================================
# CHECK 4: TDD Compliance
# ============================================================================

echo -e "${BLUE}[4/7] Checking TDD Compliance...${NC}"
echo "      (Verifying tests written before implementation)"
echo

TDD_FAILED=0

# Check for artifacts directories
for branch in branch1 branch2 branch3 branch4; do
    BRANCH_DIR="experiments/magvit_I3D_LLM_basic_trajectory/$branch"
    if [ -d "$BRANCH_DIR" ]; then
        if [ ! -d "$BRANCH_DIR/artifacts" ]; then
            echo -e "      ${RED}✗ $branch: No artifacts directory${NC}"
            TDD_FAILED=1
        else
            # Check for TDD evidence files
            if ! ls "$BRANCH_DIR/artifacts"/tdd_*_red.txt >/dev/null 2>&1; then
                echo -e "      ${RED}✗ $branch: No TDD RED evidence${NC}"
                TDD_FAILED=1
            elif ! ls "$BRANCH_DIR/artifacts"/tdd_*_green.txt >/dev/null 2>&1; then
                echo -e "      ${RED}✗ $branch: No TDD GREEN evidence${NC}"
                TDD_FAILED=1
            else
                echo -e "      ${GREEN}✓ $branch: TDD evidence present${NC}"
            fi
        fi
    fi
done

if [ $TDD_FAILED -eq 1 ]; then
    FAILED_CHECKS+=("TDD Compliance")
    echo -e "      ${RED}✗ FAILED: TDD not followed for all components${NC}"
else
    echo -e "      ${GREEN}✓ TDD compliance verified${NC}"
fi
echo

# ============================================================================
# CHECK 5: Execution Method
# ============================================================================

echo -e "${BLUE}[5/7] Checking Execution Method...${NC}"
echo "      (Verifying parallel claims match actual execution)"
echo

if [ -f "tests/test_honesty_enforcement.py" ]; then
    if pytest tests/test_honesty_enforcement.py::test_parallel_execution_has_overlapping_timestamps \
           -v 2>&1 | tee /tmp/parallel_check.log; then
        echo -e "      ${GREEN}✓ Execution method verified${NC}"
    else
        echo -e "      ${RED}✗ FAILED: Claimed parallel but executed sequentially${NC}"
        echo -e "      ${RED}  Logs show no overlapping timestamps${NC}"
        FAILED_CHECKS+=("Execution Method")
    fi
else
    echo -e "      ${YELLOW}⚠ Skipped: No execution tests found${NC}"
fi
echo

# ============================================================================
# CHECK 6: Visual Evidence
# ============================================================================

echo -e "${BLUE}[6/7] Checking Visual Evidence...${NC}"
echo "      (Verifying visualizations exist for all results)"
echo

VISUAL_FAILED=0

for branch in branch1 branch2 branch3 branch4; do
    RESULTS_DIR="experiments/magvit_I3D_LLM_basic_trajectory/$branch/results"
    if [ -d "$RESULTS_DIR" ]; then
        # Count image files
        IMG_COUNT=$(find "$RESULTS_DIR" -name "*.png" -o -name "*.jpg" | wc -l)
        
        if [ "$IMG_COUNT" -lt 3 ]; then
            echo -e "      ${RED}✗ $branch: Only $IMG_COUNT images (need ≥3)${NC}"
            VISUAL_FAILED=1
        else
            echo -e "      ${GREEN}✓ $branch: $IMG_COUNT visualizations found${NC}"
        fi
    fi
done

if [ $VISUAL_FAILED -eq 1 ]; then
    FAILED_CHECKS+=("Visual Evidence")
    echo -e "      ${RED}✗ FAILED: Insufficient visualizations${NC}"
    echo -e "      ${RED}  Need: confusion matrix, training curves, sample predictions${NC}"
else
    echo -e "      ${GREEN}✓ Visual evidence verified${NC}"
fi
echo

# ============================================================================
# CHECK 7: Test Sufficiency
# ============================================================================

echo -e "${BLUE}[7/7] Checking Test Sufficiency...${NC}"
echo "      (Verifying test coverage >80% and tests verify functionality)"
echo

if command -v pytest-cov >/dev/null 2>&1 || pytest --cov 2>/dev/null; then
    if pytest --cov=experiments/magvit_I3D_LLM_basic_trajectory \
              --cov-report=term \
              --cov-report=json \
              --cov-fail-under=80 2>&1 | tee /tmp/coverage_check.log; then
        echo -e "      ${GREEN}✓ Test coverage ≥80%${NC}"
    else
        echo -e "      ${RED}✗ FAILED: Test coverage <80%${NC}"
        FAILED_CHECKS+=("Test Sufficiency")
    fi
else
    echo -e "      ${YELLOW}⚠ Skipped: pytest-cov not installed${NC}"
    echo -e "      ${YELLOW}  Install with: pip install pytest-cov${NC}"
fi

# Also check test quality
if [ -f "tests/test_honesty_enforcement.py" ]; then
    if pytest tests/test_honesty_enforcement.py::test_tests_actually_test_functionality_not_just_imports \
           -v 2>&1 | tee /tmp/test_quality.log; then
        echo -e "      ${GREEN}✓ Test quality verified${NC}"
    else
        echo -e "      ${RED}✗ FAILED: Tests just check imports, not functionality${NC}"
        FAILED_CHECKS+=("Test Quality")
    fi
fi
echo

# ============================================================================
# FINAL REPORT
# ============================================================================

echo
echo -e "${BLUE}======================================================================${NC}"

if [ ${#FAILED_CHECKS[@]} -eq 0 ]; then
    echo -e "${GREEN}                    ✓ ALL CHECKS PASSED${NC}"
    echo -e "${GREEN}======================================================================${NC}"
    echo
    echo "All verifications passed. It is SAFE to claim this work is complete."
    echo
    echo "Evidence package location:"
    echo "  - TDD artifacts: experiments/*/artifacts/"
    echo "  - Visual results: experiments/*/results/*.png"
    echo "  - Test coverage: coverage.json"
    echo
    echo "Next steps:"
    echo "  1. Commit all evidence files"
    echo "  2. Update README with verified capabilities"
    echo "  3. Push to remote repository"
    echo
    exit 0
else
    echo -e "${RED}                    ✗ VERIFICATION FAILED${NC}"
    echo -e "${RED}======================================================================${NC}"
    echo
    echo -e "${RED}The following checks FAILED:${NC}"
    for check in "${FAILED_CHECKS[@]}"; do
        echo -e "${RED}  ✗ $check${NC}"
    done
    echo
    echo -e "${YELLOW}You CANNOT claim this work is complete until all checks pass.${NC}"
    echo
    echo "What to do:"
    echo "  1. Review failed checks above"
    echo "  2. Either FIX the implementation or RENAME to be honest"
    echo "  3. Re-run this script"
    echo
    echo "Common fixes:"
    echo "  - SimplifiedI3D → Basic3DCNN"
    echo "  - llm_integration_gpt4.py → template_generator.py"
    echo "  - train_parallel.sh → train_sequential.sh"
    echo "  - Add missing TDD evidence"
    echo "  - Generate required visualizations"
    echo
    exit 1
fi

