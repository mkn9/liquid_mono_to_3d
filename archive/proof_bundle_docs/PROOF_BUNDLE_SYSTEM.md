# Proof Bundle System

**The ONLY definition of "done" for any task.**

---

## Core Principle

**A task is complete only if `scripts/prove.sh` succeeds and produces a proof bundle for the current git commit.**

That's it. One command. One gate. Binary result: pass or fail.

---

## How It Works

### 1. Run One Command

```bash
bash scripts/prove.sh
```

### 2. Get Binary Result

- **Exit code 0** → Task is complete, proof bundle created
- **Exit code != 0** → Task incomplete, check `artifacts/proof/<commit>/prove.log`

### 3. Proof Bundle Created

```
artifacts/proof/<git_sha>/
├── prove.log       # Full test output
├── meta.txt        # Git commit, timestamp, environment
├── manifest.txt    # File checksums (ties evidence to commit)
├── pip_freeze.txt  # Python dependencies
└── contracts.log   # Optional: contract outputs
```

---

## What `prove.sh` Does

1. **Captures environment** (git commit, python version, platform)
2. **Runs all tests** (`pytest -q` by default)
3. **Runs component contracts** (if `contracts/*.yaml` exist)
4. **Creates manifest** (checksums all files)
5. **Ties everything to git commit** (proof is commit-specific)

---

## Component Contracts (Optional)

Define requirements for claimed components as YAML files:

### Example: `contracts/magvit.yaml`

```yaml
name: "magvit_integration"

# Must be able to import
imports:
  - "from magvit2 import MAGVIT_VQ_VAE"

# Must run successfully
commands:
  - name: "encode_decode"
    cmd: "python scripts/test_magvit.py"
    outputs:
      - "artifacts/proof_outputs/magvit_reconstruction.png"

# Tests must pass
tests:
  - name: "magvit_tests"
    cmd: "pytest tests/test_magvit.py -v"
```

### Why Contracts?

- **Machine-checkable**: No debate about whether MAGVIT exists
- **Scalable**: Add `contracts/new_component.yaml` for each claim
- **Structured**: Same schema for all components
- **Evidence**: Contracts can require specific output files

---

## Comparison to Old System

| Aspect | Old System (3-Gate) | Proof Bundle | Winner |
|--------|---------------------|--------------|--------|
| **Simplicity** | 7 checks, 3 gates, multiple scripts | 1 command | Proof Bundle |
| **Clarity** | "Did I follow all rules?" | "Did prove.sh pass?" | Proof Bundle |
| **Evidence** | Scattered files | Single commit-tied bundle | Proof Bundle |
| **Debuggability** | Which of 7 checks failed? | One log file | Proof Bundle |
| **Agent Clarity** | Complex checklist | Binary pass/fail | Proof Bundle |

---

## Usage Examples

### Claiming Work is Done

```bash
# Run proof
bash scripts/prove.sh

# If succeeds:
git add artifacts/proof/
git commit -m "Complete: Feature X with proof bundle"
git push
```

### Checking Historical Work

```bash
# What was proven at commit abc123?
ls artifacts/proof/abc123/

# Read the proof
cat artifacts/proof/abc123/prove.log
cat artifacts/proof/abc123/meta.txt
```

### CI/CD Integration

```yaml
# .github/workflows/verify.yml
- name: Run proof bundle
  run: bash scripts/prove.sh
  
- name: Upload proof
  uses: actions/upload-artifact@v2
  with:
    name: proof-bundle
    path: artifacts/proof/
```

---

## Rules

### Rule 1: prove.sh Defines "Done"

Cannot claim complete without `prove.sh` succeeding.

### Rule 2: Tests Must Be Deterministic

```python
# ✅ GOOD: Deterministic
torch.manual_seed(42)
x = torch.randn(10)
result = model(x)
assert torch.allclose(result, expected, atol=1e-5)

# ❌ BAD: Non-deterministic
x = torch.randn(10)  # Random every time
assert result > 0  # Vague assertion
```

### Rule 3: Proof Bundle is Committed

Don't `.gitignore` the `artifacts/proof/` directory. Commit it so proof is preserved.

### Rule 4: If Can't Run, Say So

If you can't run `prove.sh`, you MUST state:

> "NOT VERIFIED. Run: bash scripts/prove.sh"

---

## FAQ

### Q: What if tests take too long?

**A:** Split into fast unit tests (for `prove.sh`) and slow integration tests (separate).

```bash
# Fast tests (for prove.sh)
pytest tests/unit/ -q

# Slow tests (run separately)
pytest tests/integration/ -q
```

### Q: What if I don't have contracts?

**A:** Contracts are optional. `prove.sh` will just run `pytest` and skip contracts.

### Q: Can I run prove.sh on EC2?

**A:** Yes, designed for it:

```bash
# On EC2
cd ~/project
bash scripts/prove.sh

# Copy proof bundle back to MacBook
scp -r ubuntu@ec2:~/project/artifacts/proof/$SHA artifacts/proof/
```

### Q: What if prove.sh fails?

**A:** Read `artifacts/proof/<commit>/prove.log` to see what failed. Fix and re-run.

---

## Migration from Old System

### What Changed

| Old | New | Why |
|-----|-----|-----|
| 7 verification scripts | 1 script (`prove.sh`) | Simplicity |
| Complex checklists | Binary pass/fail | Clarity |
| Scattered evidence | Commit-tied bundle | Traceability |
| "Did I follow TDD?" | Tests pass or don't | No debate |

### What to Keep

- ✅ TDD workflow (still required, but proven by tests passing)
- ✅ Evidence capture (now automatic via `prove.sh`)
- ✅ Git commit linkage (even stronger now)

### What to Discard

- ❌ 3-gate system
- ❌ 7 separate check types
- ❌ Multiple verification scripts
- ❌ Complex enforcement checklists

---

## Summary

### Before (Complex)
```
Did I follow TDD? Check...
Did I create visualizations? Check...
Did I run parallel? Check...
Did I capture evidence? Check...
Did I... (5 more checks)
```

### After (Simple)
```bash
bash scripts/prove.sh
# Pass → Done
# Fail → Not done
```

---

**One command. One definition of "done". No debate.**

