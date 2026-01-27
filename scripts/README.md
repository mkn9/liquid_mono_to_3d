# Test Evidence Capture Scripts

This directory contains scripts for capturing test evidence to ensure TDD integrity.

## Overview

These scripts implement the **Evidence-or-Abstain** protocol: never claim tests were run without captured proof.

**Key Features:**
- ✅ Automatic evidence capture with provenance (timestamp, git commit, environment)
- ✅ Validation that RED phase actually fails (prevents fake TDD)
- ✅ Pre-push hook prevents accidental bypass
- ✅ Explicit override for exploratory work (`SKIP_EVIDENCE=1`)

## Quick Start

```bash
# 1. Install pre-push hook (one-time setup)
bash scripts/install_hooks.sh

# 2. Use for TDD workflow
bash scripts/tdd_capture.sh

# 3. Evidence automatically validated on push
git push  # Hook runs validator automatically
```

## Scripts

### 1. `install_hooks.sh` - One-Time Setup

**Purpose:** Install Git pre-push hook to enforce evidence validation.

**Usage:**
```bash
bash scripts/install_hooks.sh
```

**What it does:**
- Installs pre-push hook in `.git/hooks/`
- Hook runs before every `git push`
- Validates evidence files exist for changed code
- Blocks push if evidence missing (unless `SKIP_EVIDENCE=1`)

**Run once after cloning repository or setting up new workspace.**

---

### 2. `tdd_capture.sh` - Full TDD Workflow

**Purpose:** Capture complete RED → GREEN → REFACTOR cycle with validation.

**Usage:**
```bash
bash scripts/tdd_capture.sh
```

**What it does:**
1. **RED Phase:** Runs tests (expects failures before implementation exists)
   - Validates tests actually fail (prevents fake TDD)
   - Saves output to `artifacts/tdd_red.txt`
   - Exits with error if tests unexpectedly pass

2. **GREEN Phase:** Runs tests (expects passes after implementation)
   - Saves output to `artifacts/tdd_green.txt`
   - Exits with error if tests fail

3. **REFACTOR Phase:** Runs tests (expects passes after refactoring)
   - Saves output to `artifacts/tdd_refactor.txt`
   - Exits with error if tests fail

**Output Files:**
```
artifacts/
├── tdd_red.txt       # RED phase failures (proves tests existed first)
├── tdd_green.txt     # GREEN phase passes (proves implementation works)
└── tdd_refactor.txt  # REFACTOR phase passes (proves refactoring safe)
```

**Each file includes provenance:**
```
=== TDD Evidence Provenance ===
Timestamp: 2026-01-19 16:45:23 UTC
Git Commit: af296d3c2a8b4d5e6f7a8b9c0d1e2f3a4b5c6d7e
Git Branch: classification/magvit-trajectories
Python: 3.11.5
pytest: 7.4.3
Environment: EC2 (ubuntu-22.04)
Dependencies: numpy==1.24.3, torch==2.1.0, ...
=== End Provenance ===
```

This makes evidence harder to fake and provides full reproducibility context.

**Example Workflow:**
```bash
# 1. Write tests first
vim tests/test_new_feature.py

# 2. Run TDD capture (will fail at RED if implementation exists)
bash scripts/tdd_capture.sh

# 3. Write minimal implementation
vim src/new_feature.py

# 4. Re-run TDD capture (RED → GREEN)
bash scripts/tdd_capture.sh

# 5. Refactor code
vim src/new_feature.py

# 6. Re-run TDD capture (GREEN → REFACTOR)
bash scripts/tdd_capture.sh

# 7. Commit code AND evidence
git add src/ tests/ artifacts/
git commit -m "Add new feature with TDD evidence"
```

---

### 3. `test_capture.sh` - Simple Test Capture

**Purpose:** Capture single test run without full TDD workflow.

**Usage:**
```bash
bash scripts/test_capture.sh [label]
```

**Parameters:**
- `label` (optional): Name for the output file. Defaults to timestamp.

**What it does:**
- Runs `pytest -q`
- Captures all output
- Saves to `artifacts/test_[label].txt`
- Returns pytest exit code

**Output File:**
```
artifacts/test_[label].txt
```

**Example Usage:**
```bash
# Capture test run with custom label
bash scripts/test_capture.sh verification

# Capture test run with timestamp (default)
bash scripts/test_capture.sh

# Use for bug fix verification
git checkout bugfix-branch
bash scripts/test_capture.sh bugfix_verification
```

---

### 4. `validate_evidence.sh` - Evidence Validator

**Purpose:** Validate that evidence files exist and are valid.

**Usage:**
```bash
# Lenient mode (allows GREEN-only, doesn't require RED)
bash scripts/validate_evidence.sh

# Strict mode (requires RED, GREEN, and REFACTOR)
bash scripts/validate_evidence.sh --strict
```

**What it checks:**
- Evidence files exist for changed source/test files
- Files contain provenance metadata
- Exit codes are correct:
  - RED: non-zero (tests should fail before implementation)
  - GREEN: zero (tests should pass after implementation)
  - REFACTOR: zero (tests should still pass after refactoring)

**Exit codes:**
- `0` = Validation passed
- `1` = Validation failed
- `2` = Usage error

**Note:** This is automatically run by the pre-push hook.

---

### 5. `pre-push.hook` - Git Pre-Push Hook

**Purpose:** Automatically validate evidence before pushing to remote.

**Installation:**
```bash
bash scripts/install_hooks.sh
```

**What it does:**
- Runs automatically before every `git push`
- Calls `validate_evidence.sh` to check evidence files
- Blocks push if validation fails
- Allows explicit bypass with `SKIP_EVIDENCE=1`

**To bypass (for exploratory work):**
```bash
SKIP_EVIDENCE=1 git push
```

This creates an audit trail showing you explicitly bypassed validation.

---

### 6. Manual Capture - Direct Command

**Purpose:** Maximum control over test execution and capture.

**Usage:**
```bash
pytest -q 2>&1 | tee artifacts/test_description.txt
```

**When to use:**
- Specific pytest arguments needed (`-k`, `-m`, `--maxfail`)
- Want to see output while capturing
- Need custom output file name

**Examples:**
```bash
# Capture only unit tests
pytest -q -m unit 2>&1 | tee artifacts/test_unit_only.txt

# Capture specific test file
pytest -q tests/test_camera.py 2>&1 | tee artifacts/test_camera_verification.txt

# Capture with verbose output
pytest -v tests/ 2>&1 | tee artifacts/test_verbose.txt
```

---

## Evidence Requirements

### What Must Be Committed

**ALWAYS commit the `artifacts/` directory with code changes:**

```bash
git add src/ tests/ artifacts/
git commit -m "Descriptive commit message with TDD evidence"
```

### What to Include in Documentation

**BAD (no evidence):**
```markdown
## Results

All tests passed. TDD workflow followed correctly.
```

**GOOD (with evidence references):**
```markdown
## Results

**TDD Evidence:**

RED phase output: `artifacts/tdd_red.txt`
- 5 tests failed as expected (function not implemented)

GREEN phase output: `artifacts/tdd_green.txt`
- All 5 tests pass after implementation

REFACTOR phase output: `artifacts/tdd_refactor.txt`
- All 5 tests still pass after refactoring
```

### Integration with Documentation Integrity

These scripts enforce Section 3.1 (Documentation Integrity) and Section 3.3 (TDD Standards) of `requirements.md`:

> **VERIFY EVERY CLAIM BEFORE DOCUMENTING IT**

> **NEVER claim tests were run without captured evidence**

---

## Troubleshooting

### "RED unexpectedly passed"

**Cause:** Implementation already exists before RED phase.

**Solution:**
```bash
# Either delete/rename implementation temporarily
mv src/module.py src/module.py.backup
bash scripts/tdd_capture.sh  # RED will fail correctly

# Or skip RED and just capture GREEN/REFACTOR
bash scripts/test_capture.sh green_phase
```

### "GREEN phase failed"

**Cause:** Implementation doesn't pass tests yet.

**Solution:**
```bash
# Fix implementation, then manually capture GREEN
pytest -q 2>&1 | tee artifacts/tdd_green.txt

# Continue with REFACTOR when ready
pytest -q 2>&1 | tee artifacts/tdd_refactor.txt
```

### Need to re-run specific phase

**You don't have to use the full script every time:**

```bash
# Just capture RED phase manually
pytest -q 2>&1 | tee artifacts/tdd_red.txt

# Just capture GREEN phase manually
pytest -q 2>&1 | tee artifacts/tdd_green.txt

# Just capture REFACTOR phase manually
pytest -q 2>&1 | tee artifacts/tdd_refactor.txt
```

---

## References

- **cursorrules:** Evidence-or-Abstain Protocol section
- **requirements.md Section 3.1:** Documentation Integrity Protocol
- **requirements.md Section 3.3:** Testing Standards (TDD)

---

**Key Principle:**

> If you can't provide evidence, don't claim it happened.

