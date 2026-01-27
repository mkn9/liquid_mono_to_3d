# Generic Cursor AI Project Template
**Version:** 1.0  
**Created:** January 26, 2026  
**Purpose:** Production-ready template for software development projects using Cursor AI

---

## Overview

This template provides a proven structure for AI-assisted software development with Cursor AI, incorporating:
- **Test-Driven Development** (TDD) with evidence capture
- **Scientific integrity** protocols for honest documentation
- **Git workflow** with pre-push validation
- **Documentation standards** for maintainability
- **Proof bundle system** for verifiable completion

**Suitable for:** Python projects, data science, ML/AI, web applications, general software development

---

## Directory Structure

```
project-root/
├── .git/                           # Git repository
│   └── hooks/                      # Git hooks for validation
│       └── pre-push               # Validates TDD evidence before push
│
├── cursorrules                     # PRIMARY: AI assistant directives
├── requirements.md                 # SECONDARY: Detailed methodology and standards
├── README.md                       # Project overview for humans
├── requirements.txt                # Python dependencies
├── pytest.ini                      # PyTest configuration
│
├── src/                            # Source code
│   ├── __init__.py
│   ├── core/                       # Core functionality
│   │   ├── __init__.py
│   │   └── ...
│   └── utils/                      # Utility functions
│       ├── __init__.py
│       └── ...
│
├── tests/                          # Test files (test_*.py)
│   ├── __init__.py
│   ├── test_core/
│   │   └── test_*.py
│   └── test_utils/
│       └── test_*.py
│
├── scripts/                        # Automation scripts
│   ├── prove.sh                    # Full test suite + proof bundle
│   ├── tdd_capture.sh              # TDD phase evidence capture
│   └── setup_project.sh            # Initial project setup
│
├── artifacts/                      # Evidence & proof bundles
│   ├── tdd_red.txt                 # RED phase (tests fail)
│   ├── tdd_green.txt               # GREEN phase (tests pass)
│   ├── tdd_refactor.txt            # REFACTOR phase (tests still pass)
│   ├── tdd_structural.txt          # Structural tests (after code)
│   └── proof/                      # Proof bundles by git SHA
│       └── <git_sha>/
│           ├── prove.log
│           ├── meta.txt
│           └── manifest.txt
│
├── docs/                           # Documentation
│   ├── architecture.md
│   ├── api.md
│   └── development.md
│
├── results/                        # Output files (timestamped)
│   └── YYYYMMDD_HHMM_*.ext
│
├── data/                           # Data files (if applicable)
│   ├── raw/
│   ├── processed/
│   └── README.md
│
├── notebooks/                      # Jupyter notebooks (exploratory)
│   └── YYYYMMDD_exploration_*.ipynb
│
├── .gitignore                      # Git ignore patterns
├── .cursorignore                   # Cursor ignore patterns (optional)
│
└── CHAT_HISTORY/                   # Session documentation
    └── YYYYMMDD_session_name.md
```

---

## Core Files

### 1. `cursorrules`

**Purpose:** Primary directives for AI assistant (Cursor AI agent)  
**Contents:** Concise, actionable rules and gates

**Template:**

```
You are an expert <language> developer working in <domain>.

🚨 PROOF-BUNDLE RULE (PRIMARY GATE - MANDATORY) 🚨

You may NOT claim any task is "done", "complete", or "verified" unless:
1. scripts/prove.sh exited successfully (exit code 0) for the current git commit
2. artifacts/proof/<git_sha>/ exists and contains:
   - prove.log (full test output)
   - meta.txt (git commit, timestamp, environment)
   - manifest.txt (file checksums)

If you cannot run scripts/prove.sh, you MUST state:
"NOT VERIFIED. Run: bash scripts/prove.sh"

NO exceptions. NO partial credit. NO "mostly done".

The proof bundle ties ALL evidence to a specific git commit.
This is the ONLY definition of "done".


🚨 ABSOLUTE INTEGRITY REQUIREMENT 🚨
NEVER PRESENT ASPIRATIONAL OR PLANNED WORK AS COMPLETED WORK

Documentation Integrity Rules (MANDATORY):
- VERIFY ALL CLAIMS: Check actual files, data, and results before documenting
- EVIDENCE-BASED ONLY: Only document what can be proven with concrete evidence
- FILE EXISTENCE: Always verify files exist (use ls, find, read_file) before claiming
- DATA VALIDATION: Load data and check actual shapes/counts before documenting
- NO ASPIRATIONAL CLAIMS: Never document plans/intentions as completed work
- DISTINGUISH: "code written" vs "code executed" vs "results verified"

Verification Protocol (REQUIRED):
1. CHECK: ls/find to verify files exist
2. VALIDATE: Load data, check shape[0] for counts
3. CONFIRM: Run tests, show output as proof
4. DISTINGUISH: State which of 3 stages work is in

Prohibited (INTEGRITY VIOLATIONS):
- ❌ Claiming files exist without ls/find verification
- ❌ Documenting counts without loading data
- ❌ Stating "100% success" without test output
- ❌ Using past tense for unexecuted work

If Verification Cannot Be Done:
- State: "Code exists but has not been executed"
- Use: "Designed to generate", "Intended to produce", "NOT YET EXECUTED"

REFERENCE: See @requirements.md Section 3.1 for full protocol with examples


🚨 DETERMINISTIC TDD REQUIREMENT (NON-NEGOTIABLE) 🚨

⚠️ BEFORE WRITING ANY IMPLEMENTATION CODE, AI MUST:
1. State: "Following TDD per cursorrules..."
2. Write tests FIRST (test_*.py file)
3. Run: bash scripts/tdd_capture.sh
4. Show RED phase evidence (artifacts/tdd_red.txt with FAILURES)
5. ONLY THEN implement code
6. Show GREEN phase evidence (artifacts/tdd_green.txt with PASSES)
7. Show REFACTOR phase evidence (artifacts/tdd_refactor.txt with PASSES)

IF AI WRITES IMPLEMENTATION BEFORE TESTS = PROTOCOL VIOLATION

Test-Driven Development (MANDATORY):
- ALWAYS follow: Red → Green → Refactor
- NEVER write implementation before tests
- TESTS MUST be deterministic (fixed seeds, explicit tolerances)
- MUST capture and commit evidence of all test phases

Evidence Requirements:
- Behavioral tests BEFORE code → artifacts/tdd_red.txt, tdd_green.txt
- Structural tests AFTER code → artifacts/tdd_structural.txt
- Refactoring → artifacts/tdd_refactor.txt
- ALWAYS commit artifacts/ directory with code changes

REFERENCE: See @requirements.md Section 3.4 for comprehensive TDD methodology


🚨 OUTPUT FILE NAMING REQUIREMENT 🚨

ALL output files in results/ directories MUST use datetime prefix:
- Format: YYYYMMDD_HHMM_descriptive_name.ext
- Example: 20260120_1430_analysis_results.png
- Ensures chronological ordering when listed

Python helper:
```python
from datetime import datetime
timestamp = datetime.now().strftime("%Y%m%d_%H%M")
filename = f"{timestamp}_descriptive_name.ext"
```

REFERENCE: See @requirements.md Section 5.4 for complete naming convention


Key Principles:
- Write concise, technical responses with accurate examples
- Prioritize readability and reproducibility
- Use descriptive variable names
- Follow language-specific style guidelines (PEP 8 for Python)
- ALWAYS create comprehensive unit tests for all new functionality

Dependencies:
[List project-specific dependencies here]

REFERENCE: See @requirements.md for comprehensive guidelines, detailed examples, and testing implementation guide
```

---

### 2. `requirements.md`

**Purpose:** Comprehensive methodology, standards, and detailed examples  
**Contents:** Everything `cursorrules` references via `@requirements.md`

**Structure:**

```markdown
# Project Requirements

**Last Updated:** [Date]

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Functional Requirements](#2-functional-requirements)
3. [Governance & Integrity Standards](#3-governance--integrity-standards)
   - 3.1 [Documentation Integrity Protocol](#31-documentation-integrity-protocol)
   - 3.2 [Proof Bundle System - Definition of "Done"](#32-proof-bundle-system---definition-of-done)
   - 3.3 [Scientific Integrity Protocol](#33-scientific-integrity-protocol)
   - 3.4 [Testing Standards (TDD)](#34-testing-standards-tdd)
   - 3.5 [Chat History Protocol](#35-chat-history-protocol)
4. [Development Environment & Setup](#4-development-environment--setup)
5. [Technical Requirements](#5-technical-requirements)
6. [Testing Implementation Guide](#6-testing-implementation-guide)

---

## 1. Project Overview

[Describe your project: what it does, key capabilities, architecture]

---

## 2. Functional Requirements

[List functional requirements organized by feature/component]

---

## 3. Governance & Integrity Standards

### 3.1 Documentation Integrity Protocol

**Principle:** Only document what can be proven with concrete evidence.

**Three Stages of Work:**
1. **Code Written:** Files exist, not yet executed
2. **Code Executed:** Ran successfully, saw output
3. **Results Verified:** Tests passed, proof bundle created

**Verification Steps:**
1. CHECK: Use ls/find to verify files exist
2. VALIDATE: Load data, check actual counts/shapes
3. CONFIRM: Run tests, capture output
4. DISTINGUISH: State which stage work is in

**Examples:**

✅ **Correct:**
- "Code written: generate_dataset.py creates 1000 samples"
- "Executed: Generated 847 samples (verified: ls output/ | wc -l)"
- "Verified: All 847 samples validated (test output: artifacts/tdd_green.txt)"

❌ **Incorrect:**
- "Generated 1000 samples" (without verification)
- "Dataset complete" (aspirational, not evidence-based)

---

### 3.2 Proof Bundle System - Definition of "Done"

**Rule:** A task is "done" ONLY when `scripts/prove.sh` exits successfully and creates a proof bundle.

**Proof Bundle Contents:**
```
artifacts/proof/<git_sha>/
├── prove.log          # Full test output
├── meta.txt           # Git commit, timestamp, environment
└── manifest.txt       # File checksums (SHA256)
```

**Workflow:**
1. Complete implementation + tests
2. Run: `bash scripts/prove.sh`
3. If exit code 0 → proof bundle created → task is DONE
4. If exit code != 0 → task is NOT done
5. Commit proof bundle with code

**No Exceptions:** "Mostly done," "almost working," "just needs..." = NOT DONE

---

### 3.3 Scientific Integrity Protocol

**Principle:** Never use synthetic/mock data as real data.

**Rules:**
- ✅ Use mock data for TESTING (unit tests, integration tests)
- ❌ NEVER present mock data as real results
- ✅ Use real trained models for RESULTS
- ❌ NEVER use untrained/random models for results
- ✅ Clearly label "mock," "stub," "synthetic" when used

**Prohibited:**
- Presenting random predictions as model results
- Using fake data to demonstrate capabilities
- Hallucinating metrics without evidence

---

### 3.4 Testing Standards (TDD)

#### 3.4.1 Classic TDD (Red-Green-Refactor)

**Workflow:**
1. **RED:** Write tests FIRST → Run → Tests FAIL → Capture evidence
2. **GREEN:** Write minimal code → Run → Tests PASS → Capture evidence
3. **REFACTOR:** Improve code → Run → Tests STILL PASS → Capture evidence

**Evidence Files:**
- `artifacts/tdd_red.txt` - Must show FAILURES
- `artifacts/tdd_green.txt` - Must show PASSES
- `artifacts/tdd_refactor.txt` - Must show PASSES (after refactoring)

**Script:** `bash scripts/tdd_capture.sh [phase]`

---

#### 3.4.2 Two-Stage Testing (Behavioral vs. Structural)

**Behavioral Tests (Black Box):**
- Written BEFORE code
- Test specifications/requirements
- No knowledge of internals needed
- Example: "Function returns sorted list"

**Structural Tests (White Box):**
- Written AFTER code
- Test internal implementation details
- Requires knowledge of internals
- Example: "Function uses quicksort algorithm"

**Evidence:** `artifacts/tdd_structural.txt`

---

#### 3.4.3 Deterministic Tests

**Requirements:**
- Fixed seeds for RNG: `random.seed(42)`, `np.random.seed(42)`, `torch.manual_seed(42)`
- Explicit tolerances: `np.allclose(a, b, atol=1e-6, rtol=1e-5)`
- NO float equality: Never use `assert x == y` for floats

**Why:** Tests must produce same results on every run.

---

### 3.5 Chat History Protocol

**Purpose:** Document each work session for future reference.

**Format:**
```
CHAT_HISTORY/YYYYMMDD_session_name.md

Contents:
- Session Summary
- Timeline of Events
- Technical Details
- Key Decisions
- Files Created/Modified
- Open Questions
- Next Steps
```

**When to Create:**
- At end of each significant work session
- Before context window refresh
- When switching between major tasks

---

## 4. Development Environment & Setup

[Describe environment setup, dependencies, installation instructions]

**Example:**
```bash
# Clone repository
git clone <repo_url>
cd <project_dir>

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run tests
bash scripts/prove.sh
```

---

## 5. Technical Requirements

[Language-specific standards, coding conventions, architecture patterns]

**Example Sections:**
- Code Style (PEP 8 for Python)
- Error Handling
- Logging
- Performance Requirements
- Security Considerations

---

## 6. Testing Implementation Guide

[Detailed testing patterns, examples, best practices]

**Example:**
```python
import pytest
import numpy as np

def test_example():
    \"\"\"Test description.\"\"\"
    # Arrange
    np.random.seed(42)
    input_data = np.random.rand(100)
    
    # Act
    result = function_under_test(input_data)
    
    # Assert
    expected = calculate_expected(input_data)
    np.testing.assert_allclose(result, expected, atol=1e-6)
```

---
```

---

### 3. `scripts/prove.sh`

**Purpose:** Run all tests and create proof bundle

```bash
#!/bin/bash
# scripts/prove.sh - Run tests and create proof bundle

set -e  # Exit on any error

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

echo "🧪 Running full test suite..."

# Get current git commit
GIT_SHA=$(git rev-parse --short HEAD 2>/dev/null || echo "no-git")
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# Create proof directory
PROOF_DIR="artifacts/proof/${GIT_SHA}"
mkdir -p "$PROOF_DIR"

# Run tests and capture output
pytest tests/ -v --tb=short > "${PROOF_DIR}/prove.log" 2>&1

EXIT_CODE=$?

# Create meta.txt
cat > "${PROOF_DIR}/meta.txt" << EOF
Git Commit: ${GIT_SHA}
Timestamp: ${TIMESTAMP}
Hostname: $(hostname)
Python: $(python --version 2>&1)
Pytest: $(pytest --version 2>&1)
Exit Code: ${EXIT_CODE}
EOF

# Create manifest.txt (checksums of key files)
echo "File Checksums:" > "${PROOF_DIR}/manifest.txt"
find src/ tests/ -type f -name "*.py" -exec shasum -a 256 {} \; >> "${PROOF_DIR}/manifest.txt" 2>/dev/null || true

if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ All tests passed!"
    echo "✅ Proof bundle created: ${PROOF_DIR}/"
    exit 0
else
    echo "❌ Tests failed (exit code: ${EXIT_CODE})"
    echo "📋 See: ${PROOF_DIR}/prove.log"
    exit $EXIT_CODE
fi
```

---

### 4. `scripts/tdd_capture.sh`

**Purpose:** Capture TDD phase evidence

```bash
#!/bin/bash
# scripts/tdd_capture.sh - Capture TDD phase evidence

PHASE="${1:-red}"  # red, green, refactor, structural

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

mkdir -p artifacts

ARTIFACT_FILE="artifacts/tdd_${PHASE}.txt"

echo "🧪 Running tests for TDD phase: ${PHASE}"
echo "📝 Capturing output to: ${ARTIFACT_FILE}"

# Run tests and capture output
{
    echo "=== TDD Phase: ${PHASE} ==="
    echo "=== Timestamp: $(date) ==="
    echo "=== Git Commit: $(git rev-parse --short HEAD 2>/dev/null || echo 'no-git') ==="
    echo ""
    pytest tests/ -v --tb=short
    echo ""
    echo "=== Exit Code: $? ==="
} > "$ARTIFACT_FILE" 2>&1

EXIT_CODE=$?

echo ""
if [ "$PHASE" = "red" ]; then
    if [ $EXIT_CODE -ne 0 ]; then
        echo "✅ RED phase: Tests failed as expected"
    else
        echo "⚠️  WARNING: RED phase should have failing tests!"
    fi
else
    if [ $EXIT_CODE -eq 0 ]; then
        echo "✅ ${PHASE^^} phase: Tests passed"
    else
        echo "❌ ${PHASE^^} phase: Tests failed (should pass)"
    fi
fi

echo "📋 Evidence saved: ${ARTIFACT_FILE}"
exit 0
```

---

### 5. `.git/hooks/pre-push`

**Purpose:** Validate TDD evidence before pushing

```bash
#!/bin/bash
# .git/hooks/pre-push - Validate TDD evidence before push

echo "🔍 Validating TDD evidence before push..."

# Get list of changed files
CHANGED_FILES=$(git diff --name-only @{u}.. 2>/dev/null || git diff --name-only HEAD~1..HEAD)

# Check if any source or test files changed
SOURCE_CHANGED=$(echo "$CHANGED_FILES" | grep -E '(src/.*\.py|tests/.*\.py)' || true)

if [ -z "$SOURCE_CHANGED" ]; then
    echo "✅ No source or test files changed - validation not required"
    exit 0
fi

echo "📝 Source/test files changed:"
echo "$SOURCE_CHANGED"

# Check for TDD evidence
if [ ! -f "artifacts/tdd_red.txt" ] || [ ! -f "artifacts/tdd_green.txt" ]; then
    echo "❌ TDD evidence missing!"
    echo "   Required: artifacts/tdd_red.txt and artifacts/tdd_green.txt"
    echo "   Run: bash scripts/tdd_capture.sh red"
    echo "   Run: bash scripts/tdd_capture.sh green"
    exit 1
fi

echo "✅ TDD evidence found"
echo "✅ Pre-push validation passed - pushing to remote"
exit 0
```

**Install:**
```bash
chmod +x .git/hooks/pre-push
```

---

## Quick Start

### 1. Clone This Template

```bash
# Option A: Use as GitHub template
# (Click "Use this template" on GitHub)

# Option B: Clone and remove git history
git clone <template-repo-url> my-new-project
cd my-new-project
rm -rf .git
git init
git add .
git commit -m "Initial commit from template"
```

### 2. Customize for Your Project

1. **Edit `cursorrules`:**
   - Change language/domain description
   - Update dependencies list

2. **Edit `requirements.md`:**
   - Section 1: Project Overview (describe YOUR project)
   - Section 2: Functional Requirements (YOUR features)
   - Keep Sections 3-6 (governance/standards) as-is

3. **Update `README.md`:**
   - Project name
   - Description
   - Installation instructions
   - Usage examples

4. **Configure `requirements.txt`:**
   - Add your Python dependencies

5. **Set up git hooks:**
   ```bash
   chmod +x scripts/*.sh
   cp .git/hooks/pre-push.sample .git/hooks/pre-push
   chmod +x .git/hooks/pre-push
   ```

### 3. Verify Setup

```bash
# Run tests (should pass even with empty test suite)
bash scripts/prove.sh

# Check proof bundle created
ls artifacts/proof/

# Verify TDD script works
bash scripts/tdd_capture.sh red
cat artifacts/tdd_red.txt
```

### 4. Start Development

1. **Create a feature:**
   ```bash
   git checkout -b feature/my-feature
   ```

2. **Follow TDD:**
   - Write tests first: `tests/test_my_feature.py`
   - Capture RED phase: `bash scripts/tdd_capture.sh red`
   - Write implementation: `src/my_feature.py`
   - Capture GREEN phase: `bash scripts/tdd_capture.sh green`
   - Refactor if needed: `bash scripts/tdd_capture.sh refactor`

3. **Create proof bundle:**
   ```bash
   bash scripts/prove.sh
   ```

4. **Commit everything:**
   ```bash
   git add .
   git commit -m "Add my-feature with TDD evidence"
   ```

5. **Push (git hook validates TDD evidence):**
   ```bash
   git push origin feature/my-feature
   ```

---

## Key Principles

### 1. Evidence-Based Development
- Never claim "done" without proof bundle
- Always capture TDD evidence
- Document only verifiable facts

### 2. Scientific Integrity
- No mock/synthetic data as real results
- Always distinguish: written vs executed vs verified
- Be honest about limitations

### 3. Test-Driven Development
- Tests first, always
- Deterministic tests (fixed seeds, tolerances)
- Evidence for RED, GREEN, REFACTOR phases

### 4. Documentation Standards
- Chat history for each session
- Timestamped output files
- Clear, verifiable claims

### 5. Git Workflow
- Pre-push validation ensures quality
- Proof bundles tied to commits
- Feature branches for development

---

## Customization Guide

### For Different Languages

**JavaScript/TypeScript:**
- Replace pytest with Jest
- Update `scripts/prove.sh` to run `npm test`
- Adapt TDD capture script for Jest output

**Java:**
- Replace pytest with JUnit
- Update `scripts/prove.sh` to run Maven/Gradle tests
- Adapt TDD capture script for JUnit output

**Go:**
- Replace pytest with `go test`
- Update `scripts/prove.sh` accordingly
- Adapt TDD capture script for Go test output

### For Different Project Types

**Web Application:**
- Add `frontend/` and `backend/` directories
- Include integration tests
- Add deployment scripts

**Data Science:**
- Add `notebooks/` for Jupyter notebooks
- Add `data/` for datasets
- Include data validation tests

**Machine Learning:**
- Add `models/` for trained models
- Add `experiments/` for ML experiments
- Include model evaluation tests

---

## FAQ

**Q: Do I need to use all of this for a small project?**
A: Core requirements: cursorrules, TDD evidence, proof bundles. Everything else is optional but recommended.

**Q: Can I skip TDD for exploratory work?**
A: Exploratory work (notebooks, prototypes) doesn't require TDD. Production code does.

**Q: What if my CI/CD already runs tests?**
A: Great! The proof bundle system complements CI/CD by tying evidence to git commits locally.

**Q: How do I handle large data files?**
A: Use `.gitignore` for data. Document data sources in `data/README.md`. Use checksums for validation.

**Q: Can I use this with other AI assistants (not Cursor)?**
A: Yes! The principles apply to any AI-assisted development. Adjust `cursorrules` format as needed.

---

## Resources

- **TDD Guide:** See `requirements.md` Section 3.4
- **Documentation Standards:** See `requirements.md` Section 3.1
- **Testing Patterns:** See `requirements.md` Section 6
- **Git Workflow:** Pre-push hook validates before push

---

## License

[Choose appropriate license for your project]

**Template itself:** MIT License (free to use, modify, distribute)

---

## Credits

**Based on:** Production practices from mono_to_3d project (January 2026)  
**Key Principles:** TDD, Evidence-Based Development, Scientific Integrity  
**Designed for:** Cursor AI but applicable to any AI-assisted development

---

**Template Version:** 1.0  
**Last Updated:** January 26, 2026  
**Maintained by:** [Your name/organization]

