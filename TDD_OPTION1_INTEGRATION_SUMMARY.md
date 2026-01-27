# TDD Integration with Option 1: Complete Implementation

**Created:** January 18, 2026  
**Status:** Ready to implement

---

## Integration Overview

The **DETERMINISTIC SIM TDD RULE** integrates perfectly with **Option 1 (Minimal Consolidation)** by:

1. **Preventing integrity failures** - Tests prove claims
2. **Creating audit trails** - Tests document actual behavior  
3. **Ensuring reproducibility** - Deterministic tests are verifiable
4. **Enforcing verification-first** - Write tests before code

---

## Complete File Structure (Option 1 + TDD)

```
mono_to_3d/
├── README.md                                    # ✅ Keep: Project overview + doc map
├── PROJECT_REQUIREMENTS.md                      # 🆕 From requirements.md split
├── DEVELOPMENT_WORKFLOW.md                      # 🆕 EC2, git, day-to-day
├── TESTING_GUIDE.md                             # 🆕 Comprehensive testing (with TDD)
├── COORDINATE_SYSTEM_DOCUMENTATION.md           # ✅ Keep: Technical reference
│
├── cursorrules                                  # 🔄 Refactored: AI-focused (~180 lines)
│   ├── ABSOLUTE INTEGRITY REQUIREMENT           #     (existing, ~60 lines)
│   ├── DETERMINISTIC TDD REQUIREMENT            # 🆕 (new, ~50 lines)
│   ├── CRITICAL COMPUTATION RULE                #     (existing, ~20 lines)
│   ├── CRITICAL ERROR PREVENTION RULES          #     (existing, ~40 lines)
│   └── Key Principles (condensed)               #     (~10 lines + references)
│
├── pytest.ini                                   # 🆕 Pytest configuration
│
├── config.yaml                                  # 🆕 Project configuration
├── main_macbook.py                              # 🆕 MacBook orchestration script
│
├── docs/
│   ├── governance/
│   │   ├── DOCUMENTATION_INTEGRITY.md           # 🔄 Renamed from root
│   │   └── SCIENTIFIC_INTEGRITY.md              # 🔄 Moved from root
│   └── setup/
│       ├── EC2_SETUP_GUIDE.md                   # 🔄 Consolidated EC2 docs
│       └── LOCAL_ENVIRONMENT.md                 # 🔄 From ENVIRONMENT_SETUP.md
│
├── tests/
│   ├── __init__.py
│   ├── conftest.py                              # 🆕 Shared fixtures
│   ├── test_invariants.py                       # 🆕 Cross-module invariants
│   ├── test_golden.py                           # 🆕 Canonical golden tests
│   └── unit/                                    # ✅ Organize existing tests
│       ├── test_camera.py
│       ├── test_triangulation.py
│       └── ...
│
└── [source code and other files]
```

**Files DELETED:**
- ❌ `requirements.md` (split into multiple focused docs)
- ❌ `COMPUTATION_RULES.md` (merged into DEVELOPMENT_WORKFLOW.md)
- ❌ `ENVIRONMENT_SETUP.md` (moved to docs/setup/)
- ❌ `UNIT_TESTING_IMPROVEMENTS.md` (merged into TESTING_GUIDE.md)
- ❌ `DOCUMENTATION_INTEGRITY_PROTOCOL.md` (moved to docs/governance/)
- ❌ `SCIENTIFIC_INTEGRITY_PROTOCOL.md` (moved to docs/governance/)

---

## Updated cursorrules Structure

### Complete cursorrules (≈180 lines)

```
cursorrules
├── ABSOLUTE INTEGRITY REQUIREMENT (~60 lines)
│   ├── Documentation Integrity Rules (MANDATORY)
│   ├── Verification Protocol (REQUIRED)
│   ├── Prohibited Behaviors
│   ├── Required Evidence
│   └── Reference: @docs/governance/DOCUMENTATION_INTEGRITY.md
│
├── DETERMINISTIC TDD REQUIREMENT (~50 lines) 🆕
│   ├── Test-Driven Development Protocol
│   ├── Red → Green → Refactor cycle
│   ├── Deterministic testing requirements
│   ├── Numeric comparison rules
│   ├── Reproducibility requirements
│   └── Reference: @TESTING_GUIDE.md for examples
│
├── CRITICAL COMPUTATION RULE (~20 lines)
│   ├── ALL computation on EC2 only
│   ├── Package installation restrictions
│   ├── Verification protocol
│   └── Reference: @DEVELOPMENT_WORKFLOW.md
│
├── CRITICAL ERROR PREVENTION RULES (~40 lines)
│   ├── Jupyter Notebook Syntax Rules
│   ├── PyTorch Device Consistency Rules
│   ├── Class Initialization Rules
│   └── Code Quality Validation
│
└── Key Principles (condensed) (~10 lines)
    ├── Python best practices (PEP 8)
    ├── Data analysis (pandas, vectorization)
    ├── Visualization (matplotlib, seaborn)
    └── References to detailed guides
```

### Excerpt: New TDD Section in cursorrules

```markdown
🚨 DETERMINISTIC TDD REQUIREMENT (NON-NEGOTIABLE) 🚨

Test-Driven Development Protocol (MANDATORY):
- ALWAYS follow Red → Green → Refactor cycle
- NEVER write implementation before tests
- Tests MUST be deterministic (fixed seeds, explicit tolerances)

Red (Tests First) - MANDATORY SEQUENCE:
1. Restate spec: inputs/outputs, units, shapes, assumptions, tolerances
2. Write tests FIRST covering:
   a) Invariant tests (no NaNs/Infs, bounds, conservation, symmetry)
   b) Golden/regression test with canonical scenario
3. Run: pytest -q
4. CONFIRM tests fail for right reason before implementing

Green (Minimum Implementation):
- Implement ONLY to pass tests
- DO NOT modify tests during Green
- Run: pytest -q until all pass

Refactor (Only After Green):
- Refactor ONLY after tests pass
- Re-run: pytest -q after refactors
- Justify tolerance changes explicitly

Numeric Comparisons (REQUIRED):
- NEVER use == for floats
- USE: np.allclose(actual, expected, atol=X, rtol=Y)
- USE: pytest.approx(expected, abs=X, rel=Y)
- Document tolerance choices in test docstrings

Reproducibility (REQUIRED):
- Use numpy.random.Generator with explicit seeds
- Document seed values in test docstrings
- NO global np.random usage

Test Organization (REQUIRED):
- tests/test_invariants.py: Cross-module invariant tests
- tests/test_golden.py: Canonical golden/regression tests
- tests/unit/: Module-specific unit tests

REFERENCE: See @TESTING_GUIDE.md for detailed examples and templates
```

---

## Integration Points

### 1. cursorrules ⟷ TESTING_GUIDE.md

**cursorrules:**
- Enforces TDD workflow (Red → Green → Refactor)
- Requires deterministic tests
- Mandates numeric comparison standards
- ~50 lines of rules

**TESTING_GUIDE.md:**
- Detailed examples of each TDD phase
- Test templates and patterns
- Tolerance selection guidance
- Numeric comparison best practices
- ~200+ lines of reference material

**Relationship:** cursorrules enforces the "what", TESTING_GUIDE.md explains the "how"

### 2. TDD ⟷ Documentation Integrity

**Synergy:**

| Documentation Integrity Rule | TDD Enforcement |
|------------------------------|-----------------|
| "Verify sample counts before documenting" | `test_dataset_generation_count()` proves count |
| "Check files exist before claiming" | `test_output_files_created()` verifies existence |
| "Never claim 100% success without proof" | Test suite IS the proof |
| "Distinguish code written vs executed" | Tests demonstrate execution |

**Example:**

```python
# TDD creates evidence for documentation integrity

def test_magvit_3d_generates_claimed_samples():
    """INTEGRITY TEST: Verify documentation claims about MAGVIT 3D.
    
    Documentation claims:
    - "50 samples generated"
    - "Shape: (50, 16, 3)"
    - "Multi-view videos created"
    
    This test PROVES these claims or they must be corrected.
    
    Seed: 42
    """
    rng = np.random.default_rng(42)
    
    dataset = generate_magvit_3d_dataset(num_samples=50, rng=rng)
    
    # PROVE: "50 samples generated"
    assert len(dataset['trajectories_3d']) == 50
    
    # PROVE: Shape claim
    assert dataset['trajectories_3d'].shape == (50, 16, 3)
    
    # PROVE: Multi-view videos exist
    assert 'multi_view_videos' in dataset
    assert dataset['multi_view_videos'].shape[0] == 50
```

**Documentation can now say:**
```markdown
## MAGVIT 3D Results

**VERIFIED:** ✅ All claims verified by test suite

Evidence:
```bash
$ pytest tests/test_magvit_3d.py::test_magvit_3d_generates_claimed_samples -v
PASSED
```

- 50 samples: ✅ Confirmed by test
- Shape (50, 16, 3): ✅ Confirmed by test  
- Multi-view videos: ✅ Confirmed by test

See test output for verification.
```

### 3. TDD ⟷ Scientific Integrity

**Synergy:**

| Scientific Integrity Concern | TDD Solution |
|------------------------------|--------------|
| "Is this real or synthetic data?" | Tests document data source in docstrings |
| "Can results be reproduced?" | Deterministic tests with fixed seeds |
| "What are the actual values?" | Golden tests store expected values |
| "How do we know it works?" | Tests prove it works |

---

## Practical Workflow Example

### Scenario: Implement MAGVIT 3D Dataset Generation

#### Phase 1: RED (Tests First)

**Step 1:** Agent restates spec in test docstring:

```python
def test_magvit_3d_dataset_generation():
    """Generate MAGVIT 3D trajectory dataset.
    
    SPECIFICATION:
    - Inputs: num_samples (int), rng (Generator)
    - Outputs: dict with keys ['trajectories_3d', 'multi_view_videos', 'labels']
    - Shapes: 
      - trajectories_3d: (num_samples, 16, 3)
      - multi_view_videos: (num_samples, 3, 16, 128, 128, 3)
      - labels: (num_samples,)
    - Units: trajectories in meters, videos in uint8 pixels
    - Assumptions: 3 cameras, 16 frames, 128x128 resolution
    - Deterministic: Fixed seed required
    - Tolerance: Exact shapes (no numeric tolerance)
    
    Seed: 42
    """
    rng = np.random.default_rng(42)
    
    # Invariant test: Check shapes
    dataset = generate_magvit_3d_dataset(num_samples=3, rng=rng)
    
    assert dataset['trajectories_3d'].shape == (3, 16, 3)
    assert dataset['multi_view_videos'].shape == (3, 3, 16, 128, 128, 3)
    assert dataset['labels'].shape == (3,)
    
    # Golden test: Check first trajectory is deterministic
    expected_first_point = np.array([0.0, 0.0, 0.5])  # Computed once
    np.testing.assert_allclose(
        dataset['trajectories_3d'][0, 0, :],
        expected_first_point,
        rtol=1e-10
    )
```

**Step 2:** Run tests, confirm failure:

```bash
$ pytest -q tests/test_magvit_3d.py
F
NameError: name 'generate_magvit_3d_dataset' is not defined
```

✅ **Good failure** - function doesn't exist yet

#### Phase 2: GREEN (Minimum Implementation)

**Step 3:** Implement minimum code to pass:

```python
def generate_magvit_3d_dataset(num_samples, rng):
    """Generate MAGVIT 3D dataset."""
    trajectories_3d = rng.random((num_samples, 16, 3))
    videos = rng.integers(0, 255, (num_samples, 3, 16, 128, 128, 3), dtype=np.uint8)
    labels = rng.integers(0, 3, num_samples)
    
    return {
        'trajectories_3d': trajectories_3d,
        'multi_view_videos': videos,
        'labels': labels
    }
```

**Step 4:** Run tests:

```bash
$ pytest -q tests/test_magvit_3d.py
.
1 passed in 0.15s
```

✅ **Tests pass** - proceed to refactor

#### Phase 3: REFACTOR (Improve Quality)

**Step 5:** Refactor for clarity and proper geometry:

```python
def generate_magvit_3d_dataset(num_samples, rng=None):
    """Generate MAGVIT 3D trajectory dataset with multi-camera views.
    
    Args:
        num_samples: Number of trajectory samples to generate
        rng: numpy.random.Generator for reproducibility (default: seed 42)
    
    Returns:
        dict containing:
        - 'trajectories_3d': (num_samples, 16, 3) 3D trajectories in meters
        - 'multi_view_videos': (num_samples, 3, 16, 128, 128, 3) RGB videos
        - 'labels': (num_samples,) shape labels (0=cube, 1=cylinder, 2=cone)
    """
    if rng is None:
        rng = np.random.default_rng(42)
    
    # Generate 3D trajectories for different shapes
    trajectories_3d = np.zeros((num_samples, 16, 3))
    labels = np.zeros(num_samples, dtype=int)
    
    for i in range(num_samples):
        shape_type = i % 3  # Cycle through shapes
        labels[i] = shape_type
        
        # Generate appropriate trajectory for shape
        trajectories_3d[i] = generate_shape_trajectory(shape_type, rng)
    
    # Render multi-camera views
    multi_view_videos = render_multi_view_videos(trajectories_3d, rng)
    
    return {
        'trajectories_3d': trajectories_3d,
        'multi_view_videos': multi_view_videos,
        'labels': labels
    }
```

**Step 6:** Re-run tests after refactor:

```bash
$ pytest -q tests/test_magvit_3d.py
.
1 passed in 0.18s
```

✅ **Tests still pass** - refactor successful!

#### Phase 4: DOCUMENT (With Evidence)

**Now documentation can say (WITH PROOF):**

```markdown
## MAGVIT 3D Implementation

**Status: IMPLEMENTED AND TESTED** ✅

Implementation completed following TDD protocol:
1. ✅ Tests written first (Red phase)
2. ✅ Minimum implementation (Green phase)  
3. ✅ Refactored for quality (Refactor phase)

Test coverage:
```bash
$ pytest tests/test_magvit_3d.py -v
tests/test_magvit_3d.py::test_magvit_3d_dataset_generation PASSED
tests/test_magvit_3d.py::test_dataset_deterministic PASSED
tests/test_magvit_3d.py::test_trajectory_shapes PASSED
```

All tests passing. Implementation verified.
```

---

## Benefits Summary

### 1. **Prevents Integrity Failures**

| Old Problem | TDD Solution |
|-------------|--------------|
| Claimed "50 samples" without checking | Test verifies exact count |
| Claimed files exist without verifying | Test asserts file creation |
| Claimed "works" without proof | Test suite IS the proof |
| Results not reproducible | Deterministic tests with seeds |

### 2. **Creates Audit Trail**

- Every claim in documentation references test that proves it
- Tests serve as executable specification
- Git history shows test → implementation progression
- Code review can verify TDD process was followed

### 3. **Natural Workflow Integration**

```
Integrity Rule: "VERIFY before documenting"
         ↓
TDD Requirement: "TESTS before implementation"
         ↓
Result: Can't document without tests proving it works
         ↓
Integrity naturally enforced!
```

### 4. **Complements Existing Safeguards**

| Safeguard Layer | Purpose |
|-----------------|---------|
| Documentation Integrity Protocol | Rules for WHAT to verify |
| TDD Requirement | Process for HOW to verify |
| Scientific Integrity Protocol | Rules for data authenticity |
| Test Suite | Evidence that rules were followed |

---

## Implementation Checklist

### Immediate (Created):
- [x] `pytest.ini` - Pytest configuration
- [x] `tests/test_invariants.py` - Invariant test template
- [x] `tests/test_golden.py` - Golden test template
- [x] `TDD_INTEGRATION_PROPOSAL.md` - Detailed guide
- [x] `TDD_OPTION1_INTEGRATION_SUMMARY.md` - This document

### Next Steps:
- [ ] Update `cursorrules` with TDD section (~50 lines)
- [ ] Create `TESTING_GUIDE.md` from consolidation
- [ ] Create `tests/conftest.py` with shared fixtures
- [ ] Reorganize existing tests into `tests/unit/`
- [ ] Implement Option 1 file consolidation
- [ ] Create `config.yaml` and `main_macbook.py`

### Validation:
- [ ] All existing tests still pass
- [ ] New test templates are functional
- [ ] cursorrules enforces TDD workflow
- [ ] Documentation references are correct

---

## Quick Reference Card

### For AI Agents:

```
BEFORE writing ANY new function:

1. ❓ What's the spec? (inputs, outputs, units, shapes, assumptions)
2. ✍️  Write tests FIRST (invariants + golden case)
3. ▶️  Run: pytest -q (confirm tests fail correctly)
4. 💻 Implement minimum code to pass
5. ▶️  Run: pytest -q (confirm tests pass)
6. 🔧 Refactor for quality (if needed)
7. ▶️  Run: pytest -q (confirm still pass)
8. 📝 Document with test references as proof

NEVER:
- ❌ Write implementation before tests
- ❌ Use == for float comparisons
- ❌ Use global np.random
- ❌ Document without test evidence
```

### For Humans:

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_magvit_3d.py

# Run with coverage
pytest --cov=. --cov-report=html

# Run only invariant tests
pytest -m invariant

# Run only golden tests  
pytest -m golden
```

---

## Conclusion

**TDD + Option 1 = Comprehensive Integrity System**

The DETERMINISTIC SIM TDD RULE integrates seamlessly with Option 1 consolidation:

1. **cursorrules:** Enforces TDD workflow (~50 lines)
2. **TESTING_GUIDE.md:** Detailed examples and patterns
3. **pytest.ini:** Standardizes test execution
4. **test_invariants.py & test_golden.py:** Templates for all tests

**Result:**
- Integrity failures impossible (tests prove claims)
- Reproducible results (deterministic tests)
- Audit trail (git history + test suite)
- Professional workflow (industry standard TDD)

**This system ensures: "If it's not tested, it doesn't work. If it's tested, we have proof."**

---

*Ready to implement. Awaiting user approval.*

