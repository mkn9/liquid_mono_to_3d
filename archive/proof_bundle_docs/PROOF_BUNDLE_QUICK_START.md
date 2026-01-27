# Proof Bundle System - Quick Start

## What Is It?

**One command that defines "done" for any task.**

```bash
bash scripts/prove.sh
```

- **Exit 0** → Task complete with proof
- **Exit != 0** → Task incomplete

---

## The Problem It Solves

### Before (Unreliable)

```
Developer: "MAGVIT integration complete!"
Reviewer: "Where's the import?"
Developer: "Well... I meant I planned to..."
```

**Result:** Wasted time, lost trust, unclear status.

### After (Reliable)

```
Developer: "Task complete"
Reviewer: "Did prove.sh pass?"
Developer: "Yes, here's the proof bundle"
Reviewer: [Checks artifacts/proof/<commit>/] "Verified."
```

**Result:** Binary verification, zero debate.

---

## How to Use

### 1. Work Normally

Write code, write tests, follow TDD...

### 2. Claim "Done"

```bash
bash scripts/prove.sh
```

### 3. Check Result

```bash
# If exit 0:
✓ PROOF BUNDLE CREATED
Location: artifacts/proof/<git_sha>/

# Commit it:
git add artifacts/proof/
git commit -m "Complete: Feature X with proof"
```

### 4. If It Fails

```bash
# Read the log:
cat artifacts/proof/<git_sha>/prove.log

# Fix issues, re-run:
bash scripts/prove.sh
```

---

## What's in a Proof Bundle?

```
artifacts/proof/<git_sha>/
├── prove.log       # Full test output (did tests pass?)
├── meta.txt        # Git SHA, timestamp, environment
├── manifest.txt    # File checksums (tamper evidence)
├── pip_freeze.txt  # Exact dependencies
└── contracts.log   # Component contracts (optional)
```

**Key insight:** Everything is tied to a specific git commit. No ambiguity.

---

## Component Contracts (Optional)

### The Problem

**Claim:** "I integrated MAGVIT"  
**Reality:** No imports, no tests, no usage

### The Solution

Define what "integrated MAGVIT" means:

```yaml
# contracts/magvit.yaml
name: "magvit_integration"

imports:
  - "from magvit2 import MAGVIT_VQ_VAE"

commands:
  - cmd: "python scripts/test_magvit_encode.py"
    outputs:
      - "artifacts/proof_outputs/magvit_reconstruction.png"

tests:
  - cmd: "pytest tests/test_magvit.py -v"
```

When you run `scripts/prove.sh`, it checks all `contracts/*.yaml`:

- ✅ If all pass → Proof bundle created
- ❌ If any fail → No proof bundle, see `contracts.log`

---

## Rules

### Rule 1: prove.sh Defines "Done"

You cannot claim "done" without a proof bundle.

### Rule 2: Tests Must Be Deterministic

```python
# ✅ GOOD
torch.manual_seed(42)
result = model(input)
assert torch.allclose(result, expected, atol=1e-5)

# ❌ BAD
result = model(random_input())  # Different every time
assert result > 0  # Vague
```

### Rule 3: Commit Proof Bundles

Don't `.gitignore` them. They're the evidence.

### Rule 4: If Can't Run, Say So

If you can't run `prove.sh` (e.g., no EC2 access), state:

> "NOT VERIFIED. Run: bash scripts/prove.sh"

---

## Examples

### Example 1: Feature Development

```bash
# Develop feature
git checkout -b feature/trajectory-classification

# Write code + tests
# ...

# Verify complete
bash scripts/prove.sh

# If passes:
git add artifacts/proof/
git commit -m "Add trajectory classification with proof"
git push
```

### Example 2: Bug Fix

```bash
# Create failing test (RED)
# Fix bug (GREEN)
# Refactor (REFACTOR)

# Verify all tests pass
bash scripts/prove.sh

# Proof bundle shows test history
git add artifacts/proof/
git commit -m "Fix trajectory clipping bug with proof"
```

### Example 3: Historical Verification

```bash
# What was proven at commit abc123?
cat artifacts/proof/abc123/prove.log

# When was it proven?
cat artifacts/proof/abc123/meta.txt

# What dependencies were used?
cat artifacts/proof/abc123/pip_freeze.txt
```

---

## Comparison to Old System

| **Question** | **Old System** | **Proof Bundle** |
|--------------|----------------|------------------|
| Is task done? | "I think so?" | prove.sh exit 0 |
| How to verify? | Run 7 scripts | Run 1 script |
| Where's proof? | Scattered files | One directory |
| When was it proven? | Unclear | meta.txt timestamp |
| Can reproduce? | Maybe | Yes (pip_freeze.txt) |

---

## FAQ

### Q: Do I still need TDD?

**A:** Yes, but it's now proven automatically. If your tests pass in `prove.sh`, you have TDD evidence.

### Q: What if I claim MAGVIT but don't have it?

**A:** If you create `contracts/magvit.yaml`, prove.sh will fail until the imports work.

### Q: Can I run on EC2?

**A:** Yes:

```bash
# On EC2
ssh ubuntu@ec2
cd ~/project
bash scripts/prove.sh

# Copy proof back
scp -r ubuntu@ec2:~/project/artifacts/proof/<sha> artifacts/proof/
```

### Q: What if tests are slow?

**A:** Split into fast/slow:

```bash
# Fast (for prove.sh)
pytest tests/unit/ -q

# Slow (run separately)
pytest tests/integration/ --slow
```

---

## Summary

### The Old Way (Complex)
- 7 different checks
- 3 verification gates
- Multiple scripts
- Unclear when "done"

### The New Way (Simple)
```bash
bash scripts/prove.sh
# Exit 0 = done
# Exit != 0 = not done
```

**One command. Zero debate. Binary result.**

---

## Next Steps

1. **Try it now:**
   ```bash
   bash scripts/prove.sh
   ```

2. **Check proof bundle:**
   ```bash
   ls artifacts/proof/$(git rev-parse HEAD)/
   ```

3. **Add a contract (optional):**
   ```bash
   cp contracts/magvit_integration.yaml.DISABLED contracts/my_component.yaml
   # Edit contracts/my_component.yaml
   bash scripts/prove.sh
   ```

4. **Commit proof:**
   ```bash
   git add artifacts/proof/
   git commit -m "Proof bundle for current state"
   ```

---

**Ready? Run it now:**

```bash
bash scripts/prove.sh
```

