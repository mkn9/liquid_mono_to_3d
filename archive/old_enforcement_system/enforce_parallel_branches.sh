#!/bin/bash
# Enforcement script for parallel git branch development
# Ensures branches are truly independent and properly isolated

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

echo "========================================================================"
echo "PARALLEL BRANCH ENFORCEMENT SCRIPT"
echo "========================================================================"
echo

# Configuration
REQUIRED_BRANCHES_FILE="$PROJECT_ROOT/.required_branches"
BRANCH_SPEC_FILE="$PROJECT_ROOT/.branch_specifications.json"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if required branches file exists
if [ ! -f "$REQUIRED_BRANCHES_FILE" ]; then
    echo -e "${RED}ERROR: .required_branches file not found${NC}"
    echo "Create this file with one branch name per line"
    exit 1
fi

# Read required branches
mapfile -t REQUIRED_BRANCHES < "$REQUIRED_BRANCHES_FILE"

echo "Required branches:"
for branch in "${REQUIRED_BRANCHES[@]}"; do
    echo "  - $branch"
done
echo

# 1. Check all required branches exist
echo "1. Checking branch existence..."
MISSING_BRANCHES=()
for branch in "${REQUIRED_BRANCHES[@]}"; do
    if ! git rev-parse --verify "$branch" >/dev/null 2>&1; then
        MISSING_BRANCHES+=("$branch")
    fi
done

if [ ${#MISSING_BRANCHES[@]} -gt 0 ]; then
    echo -e "${RED}✗ FAIL: Missing branches:${NC}"
    for branch in "${MISSING_BRANCHES[@]}"; do
        echo "    - $branch"
    done
    exit 1
fi
echo -e "${GREEN}✓ All required branches exist${NC}"
echo

# 2. Check branches have diverged (not identical)
echo "2. Checking branch divergence..."
BASE_BRANCH="${REQUIRED_BRANCHES[0]}"
IDENTICAL_BRANCHES=()

for branch in "${REQUIRED_BRANCHES[@]:1}"; do
    # Compare commit histories
    BASE_COMMIT=$(git rev-parse "$BASE_BRANCH")
    BRANCH_COMMIT=$(git rev-parse "$branch")
    
    if [ "$BASE_COMMIT" == "$BRANCH_COMMIT" ]; then
        IDENTICAL_BRANCHES+=("$branch (identical to $BASE_BRANCH)")
    fi
done

if [ ${#IDENTICAL_BRANCHES[@]} -gt 0 ]; then
    echo -e "${RED}✗ FAIL: Branches have not diverged:${NC}"
    for branch in "${IDENTICAL_BRANCHES[@]}"; do
        echo "    - $branch"
    done
    echo -e "${YELLOW}Each branch must have unique commits for parallel work${NC}"
    exit 1
fi
echo -e "${GREEN}✓ All branches have diverged${NC}"
echo

# 3. Check for branch-specific files
echo "3. Checking branch-specific implementations..."
BRANCH_SPECIFIC_FAILURES=()

for branch in "${REQUIRED_BRANCHES[@]}"; do
    # Get branch-specific directory (extract from branch name)
    BRANCH_DIR=$(echo "$branch" | sed 's|.*/||' | tr '-' '_')
    
    # Check if branch has unique files in its directory
    if ! git ls-tree -r "$branch" --name-only | grep -q "$BRANCH_DIR"; then
        BRANCH_SPECIFIC_FAILURES+=("$branch: No branch-specific directory found")
    fi
    
    # Check for model implementation
    if ! git ls-tree -r "$branch" --name-only | grep -q "model.py\|architecture.py"; then
        BRANCH_SPECIFIC_FAILURES+=("$branch: No model implementation found")
    fi
done

if [ ${#BRANCH_SPECIFIC_FAILURES[@]} -gt 0 ]; then
    echo -e "${RED}✗ FAIL: Branch-specific implementations missing:${NC}"
    for failure in "${BRANCH_SPECIFIC_FAILURES[@]}"; do
        echo "    - $failure"
    done
    exit 1
fi
echo -e "${GREEN}✓ All branches have specific implementations${NC}"
echo

# 4. Check for result files in each branch
echo "4. Checking for branch-specific results..."
MISSING_RESULTS=()

for branch in "${REQUIRED_BRANCHES[@]}"; do
    # Check for results directory
    if ! git ls-tree -r "$branch" --name-only | grep -q "results/.*\\.json\|results/.*\\.md"; then
        MISSING_RESULTS+=("$branch")
    fi
done

if [ ${#MISSING_RESULTS[@]} -gt 0 ]; then
    echo -e "${YELLOW}⚠ WARNING: Some branches missing result files:${NC}"
    for branch in "${MISSING_RESULTS[@]}"; do
        echo "    - $branch"
    done
    echo -e "${YELLOW}Results should be committed after training${NC}"
    echo
else
    echo -e "${GREEN}✓ All branches have results${NC}"
    echo
fi

# 5. Check for unique architecture components
echo "5. Verifying architectural differences..."
ARCH_SIMILARITY=()

# Extract key model components from each branch and compare
for i in "${!REQUIRED_BRANCHES[@]}"; do
    for j in "${!REQUIRED_BRANCHES[@]}"; do
        if [ $i -lt $j ]; then
            branch1="${REQUIRED_BRANCHES[$i]}"
            branch2="${REQUIRED_BRANCHES[$j]}"
            
            # Get model files from both branches
            files1=$(git ls-tree -r "$branch1" --name-only | grep -E "model.py|architecture.py" | head -1)
            files2=$(git ls-tree -r "$branch2" --name-only | grep -E "model.py|architecture.py" | head -1)
            
            if [ -n "$files1" ] && [ -n "$files2" ]; then
                # Compare file contents
                content1=$(git show "$branch1:$files1" 2>/dev/null || echo "")
                content2=$(git show "$branch2:$files2" 2>/dev/null || echo "")
                
                # Simple similarity check (if contents are identical)
                if [ "$content1" == "$content2" ]; then
                    ARCH_SIMILARITY+=("$branch1 and $branch2 have identical model files")
                fi
            fi
        fi
    done
done

if [ ${#ARCH_SIMILARITY[@]} -gt 0 ]; then
    echo -e "${RED}✗ FAIL: Architectures are not unique:${NC}"
    for similarity in "${ARCH_SIMILARITY[@]}"; do
        echo "    - $similarity"
    done
    echo -e "${YELLOW}Each branch must implement a DIFFERENT architecture${NC}"
    exit 1
fi
echo -e "${GREEN}✓ All branches have unique architectures${NC}"
echo

# 6. Generate branch comparison report
echo "6. Generating comparison report..."
REPORT_FILE="$PROJECT_ROOT/BRANCH_ENFORCEMENT_REPORT.md"

cat > "$REPORT_FILE" << EOF
# Branch Enforcement Report

**Generated:** $(date -u +"%Y-%m-%d %H:%M:%S UTC")

## Summary

✓ All required branches exist
✓ Branches have diverged (not identical)
✓ Branch-specific implementations present
✓ Architectures are unique

## Branch Details

EOF

for branch in "${REQUIRED_BRANCHES[@]}"; do
    echo "### $branch" >> "$REPORT_FILE"
    echo "" >> "$REPORT_FILE"
    
    # Count commits unique to this branch
    UNIQUE_COMMITS=$(git rev-list --count "$branch" ^"${REQUIRED_BRANCHES[0]}" 2>/dev/null || echo "0")
    echo "- **Unique commits:** $UNIQUE_COMMITS" >> "$REPORT_FILE"
    
    # List key files
    echo "- **Key files:**" >> "$REPORT_FILE"
    git ls-tree -r "$branch" --name-only | grep -E "model.py|train.py|config" | head -5 | sed 's/^/  - /' >> "$REPORT_FILE"
    
    # Check for results
    RESULT_COUNT=$(git ls-tree -r "$branch" --name-only | grep -c "results/" 2>/dev/null || echo "0")
    echo "- **Result files:** $RESULT_COUNT" >> "$REPORT_FILE"
    
    echo "" >> "$REPORT_FILE"
done

echo -e "${GREEN}✓ Report saved to: $REPORT_FILE${NC}"
echo

echo "========================================================================"
echo -e "${GREEN}✓ PARALLEL BRANCH ENFORCEMENT: PASSED${NC}"
echo "========================================================================"
echo
echo "All branches are properly isolated and contain unique implementations."
echo

