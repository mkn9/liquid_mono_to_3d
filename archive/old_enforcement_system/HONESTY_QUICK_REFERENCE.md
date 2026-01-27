# Honesty Quick Reference
**One-Page Guide to Prevent Lies & Misrepresentations**

---

## Before ANY Code: The Contract

```bash
# Create COMPONENT_CONTRACT.md listing:
# 1. What you claim (e.g., "MAGVIT integration")
# 2. How to verify (e.g., "import magvit2.MAGVIT_VQ_VAE works")
# 3. Test that proves it (e.g., "test_magvit_encodes_video()")
# 4. Evidence format (e.g., "reconstructed_frames.png")
```

**Rule:** No coding until contract written and approved.

---

## During Development: The 7 Checks

| # | Lie Type | Check | Evidence |
|---|----------|-------|----------|
| 1 | **Component Missing** | `import component` works | Import succeeds |
| 2 | **Architecture Fake** | Param count matches | test_param_count() |
| 3 | **Integration Fake** | API call succeeds | api_response.json |
| 4 | **TDD Skipped** | Test commit before impl | git log timestamps |
| 5 | **Sequential not Parallel** | Overlapping timestamps | process_logs/ |
| 6 | **No Visual Evidence** | 5+ PNG files exist | results/*.png |
| 7 | **Tests Insufficient** | >80% coverage + functional | pytest --cov |

**Rule:** Run checks BEFORE claiming component complete.

---

## The Test Sufficiency Rule

### ❌ BAD Test (Just Checks Existence)
```python
def test_magvit_exists():
    from magvit import MAGVIT  # Just checks import
    assert True
```

### ✅ GOOD Test (Verifies Functionality)
```python
def test_magvit_encodes_and_decodes():
    from magvit import MAGVIT
    model = MAGVIT()
    
    # Real data
    video = torch.randn(1, 16, 3, 64, 64)
    
    # Actual operations
    codes = model.encode(video)
    reconstructed = model.decode(codes)
    
    # Meaningful assertions
    assert codes.dtype == torch.long  # Quantized
    assert reconstructed.shape == video.shape
    assert torch.allclose(video, reconstructed, atol=0.2)
    
    # Save evidence
    save_comparison(video, reconstructed, 'artifacts/magvit_proof.png')
```

**Rule:** Every claimed component needs a GOOD test, not just BAD test.

---

## The Naming Rule

| If You Have | Name It | NOT |
|-------------|---------|-----|
| Basic Conv3d | `Basic3DCNN` | ~~SimplifiedI3D~~ |
| Templates | `template_generator.py` | ~~llm_integration.py~~ |
| No quantization | `basic_encoder.py` | ~~magvit_encoder.py~~ |
| Sequential exec | `train_sequential.sh` | ~~train_parallel.sh~~ |

**Rule:** Name reflects reality, not aspiration.

---

## Before Claiming "Complete"

```bash
# Run this script - if it fails, you're NOT complete
bash scripts/verify_all_before_complete.sh

# Checks:
# ✅ All claimed components exist
# ✅ Architecture matches name
# ✅ APIs actually called
# ✅ TDD evidence present
# ✅ Execution method verified
# ✅ Visual evidence saved
# ✅ Test coverage >80%

# If ALL pass: You can claim complete
# If ANY fail: Document what's missing
```

**Rule:** Script must pass before "complete" claim.

---

## If You Catch Yourself Lying

### Immediate Actions:
1. **Stop** - Don't push the lie
2. **Rename** - `SimplifiedI3D` → `Basic3DCNN`
3. **Document** - Create `HONEST_ASSESSMENT.md`
4. **Fix or Disclose** - Either implement properly OR say "simplified version"

### Example:
```bash
# Honest commit
git mv magvit_encoder.py basic_video_encoder.py
echo "NOTE: This is a basic Conv3d encoder, NOT MAGVIT" >> README.md
git commit -m "HONEST: Rename to reflect actual implementation"
```

**Rule:** Honesty failure is recoverable. Hiding it is not.

---

## The Core Principle

### **If you can't prove it, you can't claim it.**

**Proof requires:**
- ✅ Working import/API call
- ✅ Test that verifies functionality
- ✅ Evidence file (image/log/response)
- ✅ Git history (for TDD/parallel)

**No proof = No claim.**

---

## Installation

```bash
# 1. Copy to project root
cp docs/HONESTY_QUICK_REFERENCE.md .

# 2. Print and post near monitor
lpr HONESTY_QUICK_REFERENCE.md

# 3. Add to pre-push hook
cat >> .git/hooks/pre-push << 'EOF'
bash scripts/verify_all_before_complete.sh || exit 1
EOF
chmod +x .git/hooks/pre-push
```

---

## Daily Checklist

```
Morning:
□ Review component contract
□ Check what can be PROVEN today

During Work:
□ Write test BEFORE implementation
□ Verify test actually tests the claim
□ Generate evidence files

Before Push:
□ Run verify_all_before_complete.sh
□ All 7 checks pass
□ Evidence files committed

Before Claiming Complete:
□ Manual code review
□ Rename anything simplified
□ README matches reality
```

---

**Print this. Post it. Follow it. Every time.**

**No exceptions. No shortcuts.**

