# Two-File Strategy: cursorrules + requirements.md

**Purpose:** Consolidate all guidance into just 2 files instead of 4+  
**Created:** January 18, 2026

---

## Philosophy

**Two files are sufficient:**
1. **cursorrules** - Concise behavior rules for AI agents (~200 lines)
2. **requirements.md** - Comprehensive reference for AI + humans (~1000 lines)

**Why this works:**
- cursorrules is ALWAYS in context (enforces behavior)
- requirements.md is @-referenced as needed (provides details)
- No need for separate TESTING_GUIDE.md, DEVELOPMENT_WORKFLOW.md, etc.

---

## cursorrules Structure (~200 lines)

### Template:

```markdown
You are an expert Python developer working in data analysis, development, and AIML.

🚨 ABSOLUTE INTEGRITY REQUIREMENT 🚨

Documentation Integrity Rules (MANDATORY):
- VERIFY ALL CLAIMS: Check files, data, results before documenting
- EVIDENCE-BASED ONLY: Document what can be proven with evidence
- FILE EXISTENCE: Use ls/find to verify before claiming files exist
- DATA VALIDATION: Load data, check shapes/counts before documenting
- NO ASPIRATIONAL CLAIMS: Never present plans as completed work

Verification Protocol (REQUIRED):
1. CHECK: ls/find to verify files exist
2. VALIDATE: Load data, check actual shapes/counts
3. CONFIRM: Run tests, capture output as proof
4. DISTINGUISH: "Code written" vs "Code executed" vs "Results verified"

Prohibited (INTEGRITY VIOLATIONS):
- ❌ Claiming files exist without ls/find verification
- ❌ Documenting sample counts without loading data
- ❌ Stating "100% success" without showing test output
- ❌ Using past tense for unexecuted work

REFERENCE: See @requirements.md Section 3.1 for full protocol and examples


🚨 DETERMINISTIC TDD REQUIREMENT (NON-NEGOTIABLE) 🚨

Test-Driven Development (MANDATORY):
- ALWAYS: Red → Green → Refactor
- NEVER: Write implementation before tests
- TESTS MUST: Be deterministic (fixed seeds, explicit tolerances)

Red Phase (Tests First):
1. Restate spec: inputs, outputs, units, shapes, assumptions, tolerances
2. Write tests FIRST:
   - Invariant tests (no NaNs/Infs, bounds, conservation, symmetry)
   - Golden test (canonical scenario with known output)
3. Run: pytest -q
4. CONFIRM tests fail for right reason

Green Phase (Minimum Implementation):
- Implement ONLY to pass tests
- DO NOT modify tests during Green
- Run: pytest -q until pass

Refactor Phase (Only After Green):
- Refactor ONLY after tests pass
- Re-run: pytest -q after refactors
- Justify any tolerance changes

Numeric Comparisons (REQUIRED):
- NEVER: float == float
- USE: np.allclose(actual, expected, atol=X, rtol=Y)
- USE: pytest.approx(expected, abs=X, rel=Y)

Reproducibility (REQUIRED):
- Use numpy.random.Generator with explicit seeds
- Document seed in test docstring
- NO global np.random

REFERENCE: See @requirements.md Section 3.3 for TDD examples and templates


🚨 CRITICAL COMPUTATION RULE 🚨

ALL COMPUTATION ON EC2 ONLY:
- MacBook: File editing, documentation, SSH ONLY
- EC2: ALL Python execution, testing, package installation
- NEVER install: pytorch, numpy, pytest, ML packages on MacBook
- ALWAYS verify: hostname before running Python

REFERENCE: See @requirements.md Section 4 for EC2 setup and workflow


🚨 CRITICAL ERROR PREVENTION RULES 🚨

Jupyter Notebook Syntax:
- ALWAYS validate f-string syntax before saving
- NEVER break f-strings across lines in JSON
- Use ast.parse() to validate Python syntax

PyTorch Device Consistency:
- ALWAYS specify device explicitly for ALL tensors
- Define device at start: device = torch.device('cuda' if ...)
- Move ALL tensors to same device

Class Initialization:
- ALWAYS implement __init__ with parameters
- Store initialization parameters as instance attributes
- Add comprehensive docstrings


Key Principles:
- Write concise, technical responses with accurate Python examples
- Follow PEP 8, use descriptive variable names
- Prefer vectorized operations over loops
- Use pandas method chaining, numpy for numerical ops
- Create informative plots with proper labels
- Handle missing data appropriately
- Implement data quality checks

REFERENCE: See @requirements.md for detailed guidelines and examples
```

**Total: ~180-200 lines**

---

## requirements.md Structure (~1000 lines)

### Template:

```markdown
# Project Requirements

**Last Updated:** January 18, 2026

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Functional Requirements](#2-functional-requirements)
3. [Governance & Integrity Standards](#3-governance--integrity-standards)
   - 3.1 [Documentation Integrity Protocol](#31-documentation-integrity-protocol)
   - 3.2 [Scientific Integrity Protocol](#32-scientific-integrity-protocol)
   - 3.3 [Testing Standards (TDD)](#33-testing-standards-tdd)
4. [Development Environment & Setup](#4-development-environment--setup)
5. [Technical Requirements](#5-technical-requirements)
6. [Testing Implementation Guide](#6-testing-implementation-guide)

---

## 1. Project Overview

This project implements 3D track reconstruction from multiple 2D monocular camera tracks using stereo vision and triangulation.

### Key Components
- Camera calibration and projection
- 2D object tracking
- 3D triangulation
- Trajectory reconstruction
- Visualization

---

## 2. Functional Requirements

### Camera System
- Support for 2+ synchronized cameras
- Camera calibration (intrinsic and extrinsic parameters)
- Projection and back-projection
- Multi-view geometry

### Tracking & Reconstruction
- 2D track import from multiple sources
- Temporal synchronization
- 3D triangulation (linear and non-linear)
- Track filtering and smoothing
- Occlusion handling

### Visualization
- 3D interactive plotting
- Multi-view display
- Time-based playback
- Camera position/boresight visualization

### Output
- Export 3D tracks (CSV, JSON)
- Visualization images (PNG)
- Analysis reports

---

## 3. Governance & Integrity Standards

### 3.1 Documentation Integrity Protocol

#### Core Principle
> **VERIFY EVERY CLAIM BEFORE DOCUMENTING IT**

#### Mandatory Verification Steps

**1. File Existence Claims**

BEFORE documenting files exist:
```bash
# Verify files exist
ls path/to/file.png
# OR
find . -name "file.png" -type f
```

Include verification output in documentation:
```markdown
**Verification:**
```bash
$ ls neural_video_experiments/magvit/results/
magvit_3d_dataset.npz
```
File confirmed to exist.
```

**2. Data Count Claims**

BEFORE documenting sample counts:
```python
import numpy as np
data = np.load('dataset.npz')
print(f"Actual samples: {len(data['trajectories_3d'])}")
print(f"Shape: {data['trajectories_3d'].shape}")
```

Include verification in documentation:
```markdown
**Verification:**
```python
>>> data = np.load('magvit_3d_dataset.npz')
>>> len(data['trajectories_3d'])
3
>>> data['trajectories_3d'].shape
(3, 16, 3)
```

**Finding:** Only 3 samples exist (not 50 as initially planned).
```

**3. Success/Completion Claims**

BEFORE claiming tests passed:
```bash
pytest tests/test_module.py -v
```

Include actual test output:
```markdown
**Verification:**
```bash
$ pytest tests/test_magvit_3d.py -v
tests/test_magvit_3d.py::test_dataset_generation PASSED
tests/test_magvit_3d.py::test_shapes PASSED
2 passed in 0.15s
```

Tests confirm functionality.
```

#### Language Standards

**✅ For VERIFIED work:**
- "Verification shows N samples exist" + [evidence]
- "File listing confirms file.png created" + [ls output]
- "Test execution demonstrates all pass" + [pytest output]
- "Actual data shape is (3, 16, 3)" + [load output]

**✅ For UNVERIFIED/PLANNED work:**
- "Code has been written to..."
- "Designed to produce..."
- "**NOT YET EXECUTED**"
- "**PLAN:** Will generate..."

**❌ PROHIBITED without verification:**
- "Successfully generated" (requires proof)
- "Results show" (must show actual results)
- "100% success" (must measure and prove)
- Past tense for unexecuted work

#### Three-State Model

ALL work exists in one of three states:

1. **CODE WRITTEN**: Files exist in repo, functions implemented
   - Document: "Code has been written to..."

2. **CODE EXECUTED**: Code has been run, outputs generated
   - Document: "Execution produced..." + [show evidence]

3. **RESULTS VERIFIED**: Outputs confirmed, data validated
   - Document: "Verification confirms..." + [show proof]

**Never conflate State 1 with States 2 or 3.**

#### Example: WRONG vs RIGHT

**❌ WRONG (Integrity Violation):**
```markdown
## Results

Successfully generated 50 3D trajectory samples.

Visualizations:
- magvit_3d_trajectories.png
- magvit_3d_cameras.png

100% success rate.
```

**Problem:** No verification, files don't exist, count is wrong.

**✅ RIGHT (Evidence-Based):**
```markdown
## Results

**STATUS: Minimal execution (proof-of-concept only)**

Code exists to generate 3D trajectories.

**Actual execution verification:**
```bash
$ ls -lh neural_video_experiments/magvit/results/*.npz
magvit_3d_dataset.npz  5.5K
```

**Actual data count:**
```python
>>> data = np.load('magvit_3d_dataset.npz')
>>> len(data['trajectories_3d'])
3
```

**Finding:** Only 3 samples were generated (not the planned 50).

**Visualization check:**
```bash
$ find . -name "*magvit_3d*.png"
[no results]
```

**Finding:** No visualization files exist.

**Status:** Code written and minimally tested (3 samples), but full execution (50 samples + visualizations) NOT completed.
```

---

### 3.2 Scientific Integrity Protocol

#### Purpose
Prevent synthetic/fake data from being presented as real experimental results.

#### Data Source Verification

**REQUIRED checks before visualization:**
- [ ] Data origin: Real experiments or synthetic?
- [ ] Training evidence: Logs, weights, convergence curves?
- [ ] Ground truth: Physical measurements or simulated?

#### Synthetic Data Labeling

If data is synthetic, MUST be:
- [ ] Clearly labeled: "SYNTHETIC" in filename, title, caption
- [ ] Purpose stated: Why synthetic? (testing, validation, demo)
- [ ] Limitations noted: How does it differ from real data?

#### File Naming Convention

- Real data: `dnerf_predictions_real.png`
- Synthetic data: `dnerf_predictions_SYNTHETIC.png`
- Mixed: `dnerf_predictions_real_vs_synthetic.png`

#### Plot Titles

- Real: "D-NeRF Predictions from Trained Model"
- Synthetic: "SYNTHETIC D-NeRF Predictions (Demonstration Only)"

#### Prohibited Actions

❌ **STRICTLY FORBIDDEN:**
- Creating synthetic data and presenting as real
- Using placeholder data without labeling
- Generating fake neural network outputs
- Mixing synthetic/real without distinction

#### Acceptable Synthetic Uses

✅ **ACCEPTABLE with proper labeling:**
- Algorithm testing with known inputs
- Method validation with ground truth
- Demonstration of visualization code
- Edge case testing

---

### 3.3 Testing Standards (TDD)

#### The Red → Green → Refactor Cycle

**This is mandatory for all new functionality.**

##### RED Phase: Tests First

**Step 1: Restate Specification**

Before writing tests, document:
- Inputs and outputs
- Units and shapes
- Assumptions
- Tolerance strategy

Example:
```python
def test_triangulation_golden():
    """Test 3D triangulation with known camera setup.
    
    SPECIFICATION:
    - Inputs: 2D points from 2 cameras, projection matrices
    - Outputs: 3D points in world coordinates
    - Units: Pixels (input), meters (output)
    - Shapes: Input (N, 2), Output (N, 3)
    - Assumptions: Cameras calibrated, no noise
    - Tolerance: atol=0.01 (1cm), rtol=1e-5 (0.001%)
    
    Seed: None (deterministic geometry)
    """
```

**Step 2: Write Tests First**

**a) Invariant Tests** (properties that MUST hold):

```python
def test_triangulation_no_nan_inf():
    """Triangulation must never produce NaN or Inf."""
    points_2d_cam1 = np.array([[100, 100], [200, 150]])
    points_2d_cam2 = np.array([[150, 105], [180, 145]])
    
    points_3d = triangulate(points_2d_cam1, points_2d_cam2, P1, P2)
    
    assert np.all(np.isfinite(points_3d)), "3D points contain NaN/Inf"

def test_rotation_preserves_norm():
    """Rotation matrices must preserve vector lengths."""
    R = create_rotation_matrix(theta=np.pi/4, axis='z')
    v = np.array([1.0, 2.0, 3.0])
    
    v_rotated = R @ v
    
    np.testing.assert_allclose(
        np.linalg.norm(v_rotated),
        np.linalg.norm(v),
        rtol=1e-10,
        err_msg="Rotation changed vector norm"
    )
```

**b) Golden Tests** (canonical scenario with known output):

```python
def test_triangulation_canonical():
    """Test triangulation with known stereo setup.
    
    Scenario:
    - Cameras 0.65m apart horizontally
    - Both looking at (0, 0, 2) in world coordinates
    - Standard camera parameters
    
    Expected:
    - Recovers (0, 0, 2) within 1cm tolerance
    
    Seed: None (deterministic)
    Tolerance: atol=0.01 (1cm), rtol=1e-5
    
    Justification:
    - 1cm accounts for sub-pixel discretization
    - No noise in this pure geometry test
    """
    # Setup cameras (deterministic)
    P1, P2 = setup_stereo_cameras(baseline=0.65, focal_length=800)
    
    # Known 3D point
    point_3d_true = np.array([0.0, 0.0, 2.0])
    
    # Project to 2D
    point_2d_cam1 = project_3d_to_2d(point_3d_true, P1)
    point_2d_cam2 = project_3d_to_2d(point_3d_true, P2)
    
    # Triangulate back
    point_3d_reconstructed = triangulate(point_2d_cam1, point_2d_cam2, P1, P2)
    
    # Verify
    np.testing.assert_allclose(
        point_3d_reconstructed,
        point_3d_true,
        atol=0.01,  # 1cm
        rtol=1e-5,
        err_msg="Failed to recover known 3D point"
    )
```

**Step 3: Run Tests, Confirm Failure**

```bash
pytest -q tests/test_triangulation.py
```

Expected output:
```
F
NameError: name 'triangulate' is not defined
```

✅ **Good failure** - function doesn't exist yet

##### GREEN Phase: Minimum Implementation

Implement ONLY what's needed to pass tests:

```python
def triangulate(points_2d_cam1, points_2d_cam2, P1, P2):
    """Triangulate 3D points from stereo correspondences."""
    points_3d = cv2.triangulatePoints(P1, P2, 
                                       points_2d_cam1.T, 
                                       points_2d_cam2.T)
    points_3d = points_3d[:3] / points_3d[3]  # Homogeneous to Cartesian
    return points_3d.T
```

Run tests:
```bash
pytest -q tests/test_triangulation.py
```

Expected:
```
..
2 passed in 0.03s
```

✅ **Tests pass** - proceed to Refactor

**CRITICAL: Do NOT modify tests during Green phase**

##### REFACTOR Phase: Improve Quality

Only AFTER tests pass:

```python
def triangulate(points_2d_cam1, points_2d_cam2, P1, P2):
    """Triangulate 3D points from stereo 2D correspondences.
    
    Uses Direct Linear Transform (DLT) via OpenCV.
    
    Args:
        points_2d_cam1: Nx2 array, camera 1 pixel coordinates
        points_2d_cam2: Nx2 array, camera 2 pixel coordinates
        P1: 3x4 projection matrix for camera 1
        P2: 3x4 projection matrix for camera 2
    
    Returns:
        points_3d: Nx3 array, 3D points in world coordinates
    
    Raises:
        ValueError: If input shapes don't match
    """
    # Validate
    assert points_2d_cam1.shape == points_2d_cam2.shape
    assert points_2d_cam1.shape[1] == 2
    
    # Triangulate (OpenCV expects transposed input)
    points_4d = cv2.triangulatePoints(
        P1, P2,
        points_2d_cam1.T.astype(np.float32),
        points_2d_cam2.T.astype(np.float32)
    )
    
    # Convert homogeneous → Cartesian
    points_3d = points_4d[:3] / points_4d[3]
    
    return points_3d.T
```

Re-run tests:
```bash
pytest -q tests/test_triangulation.py
```

✅ **Tests still pass** - refactor successful

#### Deterministic Testing Requirements

##### Random Number Generation

**❌ WRONG: Global random state**
```python
import numpy as np
np.random.seed(42)  # Global state - BAD
trajectory = np.random.randn(100, 3)
```

**✅ CORRECT: Explicit Generator**
```python
def generate_trajectory(rng=None):
    if rng is None:
        rng = np.random.default_rng(42)
    return rng.standard_normal(size=(100, 3))

def test_trajectory_deterministic():
    """Test trajectory generation is deterministic.
    
    Seed: 42
    Expected: Specific first point (golden value)
    """
    rng = np.random.default_rng(42)
    
    trajectory = generate_trajectory(rng)
    
    # Golden value (computed once, stored)
    expected_first = np.array([0.49671415, -0.1382643, 0.64768854])
    
    np.testing.assert_allclose(
        trajectory[0],
        expected_first,
        rtol=1e-10,
        err_msg="Not deterministic"
    )
```

##### Seed Documentation

Every test using randomness MUST document:
```python
def test_noise_robustness():
    """Test triangulation with measurement noise.
    
    Seed: 12345
    Why: Produces moderate noise (not too easy/hard)
    Tolerance: atol=0.05 (5cm) for 2-pixel Gaussian noise
    
    Verifies robustness to realistic sensor noise.
    """
    rng = np.random.default_rng(12345)
    # ... test implementation
```

#### Numeric Comparison Best Practices

##### Never Use == for Floats

**❌ WRONG:**
```python
assert computed == 1.0  # Fails due to floating point
```

**✅ CORRECT:**
```python
np.testing.assert_allclose(computed, 1.0, rtol=1e-10, atol=1e-12)
# OR
assert computed == pytest.approx(1.0, rel=1e-10, abs=1e-12)
```

##### Choosing Tolerances

| Tolerance | Use Case |
|-----------|----------|
| `rtol=1e-10, atol=1e-12` | Pure math, no noise |
| `rtol=1e-5, atol=1e-6` | Standard algorithms |
| `rtol=1e-3, atol=0.01` | Real sensor data |
| `rtol=0.1, atol=1.0` | Approximate methods |

**Always justify in comments:**
```python
np.testing.assert_allclose(
    reconstructed_3d,
    ground_truth_3d,
    atol=0.02,  # 2cm: accounts for 1-pixel reprojection error
    rtol=1e-5,
    err_msg="3D reconstruction outside acceptable error"
)
```

##### Norm-Based Checks for Large Arrays

```python
def assert_arrays_close_norm(actual, expected, tol=1e-6, eps=1e-10):
    """Relative norm comparison for large arrays.
    
    Checks: ||actual - expected|| / max(||expected||, eps) < tol
    """
    diff_norm = np.linalg.norm(actual - expected)
    expected_norm = np.linalg.norm(expected)
    relative_error = diff_norm / max(expected_norm, eps)
    
    assert relative_error < tol, (
        f"Relative error {relative_error:.2e} > {tol:.2e}\n"
        f"||diff||={diff_norm:.2e}, ||expected||={expected_norm:.2e}"
    )
```

#### Test Organization

```
tests/
├── __init__.py
├── conftest.py           # Shared fixtures
├── test_invariants.py    # Cross-module invariants
├── test_golden.py        # Canonical golden tests
└── unit/
    ├── test_camera.py
    ├── test_triangulation.py
    └── test_tracking.py
```

#### Integration with Integrity Protocols

**TDD prevents documentation integrity failures:**

```python
def test_dataset_count_matches_documentation():
    """INTEGRITY TEST: Verify documented sample count.
    
    Documentation claims: "50 samples generated"
    This test PROVES the claim or forces correction.
    
    Seed: 42
    """
    rng = np.random.default_rng(42)
    dataset = generate_dataset(num_samples=50, rng=rng)
    
    # PROVE documentation claim
    assert len(dataset['data']) == 50, (
        f"Documentation claims 50, but {len(dataset['data'])} generated"
    )
```

**Documentation can then reference test:**
```markdown
## Dataset Results

**VERIFIED:** ✅ 50 samples generated

Evidence:
```bash
$ pytest tests/test_dataset.py::test_dataset_count_matches_documentation
PASSED
```

Test confirms claim. See above for verification.
```

---

## 4. Development Environment & Setup

### EC2 Computation Rule

**ALL computation must be performed on EC2:**

- **MacBook**: File editing, documentation, git, SSH ONLY
- **EC2**: ALL Python execution, testing, package installation

### Connection

```bash
# Standard SSH
ssh -i /Users/mike/keys/AutoGenKeyPair.pem ubuntu@34.196.155.11

# With port forwarding for Jupyter
ssh -i /Users/mike/keys/AutoGenKeyPair.pem -L 8888:localhost:8888 ubuntu@34.196.155.11
```

### Verification

Always verify you're on EC2 before running code:
```bash
hostname  # Should show EC2 instance (e.g., ip-172-31-32-83)
pwd       # Should be /home/ubuntu/mono_to_3d
```

### Workflow

1. **MacBook**: Edit files, commit changes
2. **MacBook**: SSH to EC2
3. **EC2**: `git pull` latest changes
4. **EC2**: Run all computation, tests, training
5. **EC2**: Commit results if needed
6. **MacBook**: Pull results for documentation

### Package Installation

**NEVER install on MacBook:**
- pytorch, torch
- numpy
- pytest
- scikit-learn
- opencv-python
- Any ML/scientific packages

**EC2 only:**
```bash
# On EC2
cd ~/mono_to_3d
source venv/bin/activate
pip install -r requirements.txt
```

### Jupyter Lab on EC2

```bash
# Start Jupyter on EC2
ssh -i /Users/mike/keys/AutoGenKeyPair.pem ubuntu@34.196.155.11 \
  "cd mono_to_3d && source venv/bin/activate && jupyter lab --ip=0.0.0.0 --port=8888 --no-browser"

# In separate terminal: port forward
ssh -i /Users/mike/keys/AutoGenKeyPair.pem -L 8888:localhost:8888 -N ubuntu@34.196.155.11

# Access: http://localhost:8888/lab?token=[TOKEN]
```

---

## 5. Technical Requirements

### Performance
- Target: 30 FPS for live processing
- Latency: <100ms end-to-end
- Memory: <2GB RAM

### Accuracy
- 3D position error: <5cm at 2m distance
- Stereo matching: >95% correct correspondences

### Dependencies

Core:
- Python 3.8+
- OpenCV >= 4.5.0
- NumPy >= 1.20.0
- Matplotlib >= 3.4.0

Testing:
- pytest >= 7.0.0
- pytest-cov >= 4.0.0

ML (EC2 only):
- PyTorch >= 2.0.0
- torchvision

---

## 6. Testing Implementation Guide

### pytest Configuration

Create `pytest.ini`:
```ini
[pytest]
addopts = -q --strict-markers
testpaths = tests
markers =
    unit: Unit tests
    integration: Integration tests
    golden: Golden/regression tests
    invariant: Invariant tests
```

### Shared Fixtures

Create `tests/conftest.py`:
```python
import pytest
import numpy as np

@pytest.fixture
def rng():
    """Provide deterministic RNG for tests."""
    return np.random.default_rng(42)

@pytest.fixture
def stereo_cameras():
    """Standard stereo camera setup for tests."""
    P1, P2 = setup_cameras(baseline=0.65, focal_length=800)
    return P1, P2
```

### Test Execution

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_triangulation.py

# Run with coverage
pytest --cov=. --cov-report=html

# Run only golden tests
pytest -m golden

# Verbose output
pytest -v
```

---

## References

- PEP 8: https://pep8.org/
- NumPy Testing: https://numpy.org/doc/stable/reference/routines.testing.html
- pytest Documentation: https://docs.pytest.org/
- OpenCV Triangulation: https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html

---

**This document is the single source of truth for project requirements and standards.**
```

**Total: ~900-1000 lines (comprehensive)**

---

## Answer to Your Questions

### Q: "What should be in cursorrules vs requirements.md?"

**cursorrules:**
- Short, enforceable rules (~200 lines)
- What AI MUST do/not do
- References to requirements.md for details

**requirements.md:**
- Complete reference (~1000 lines)
- Full protocols with examples
- Detailed explanations
- Code templates

### Q: "Is testing guide necessary outside requirements.md/cursorrules?"

**NO - Include testing in requirements.md Section 3.3**

The testing guide I created can be folded into requirements.md as a comprehensive section. cursorrules just enforces the workflow, requirements.md provides all the details and examples.

### Q: "Can all guidance be in just cursorrules + requirements.md?"

**YES - This is optimal!**

**Delete:**
- ❌ TESTING_GUIDE.md (merge into requirements.md Section 3.3)
- ❌ DOCUMENTATION_INTEGRITY_PROTOCOL.md (merge into requirements.md Section 3.1)
- ❌ SCIENTIFIC_INTEGRITY_PROTOCOL.md (merge into requirements.md Section 3.2)
- ❌ DEVELOPMENT_WORKFLOW.md (merge into requirements.md Section 4)

**Keep:**
- ✅ cursorrules (~200 lines, AI behavior rules)
- ✅ requirements.md (~1000 lines, complete reference)
- ✅ README.md (project overview, quick start)
- ✅ pytest.ini (pytest configuration)

---

## Recommended Action

**I should create:**

1. **Updated cursorrules** (~200 lines)
   - Integrity rules (~40 lines)
   - TDD rules (~50 lines)
   - Computation rules (~20 lines)
   - Error prevention (~40 lines)
   - Key principles (~20 lines)
   - References to requirements.md

2. **Consolidated requirements.md** (~1000 lines)
   - Section 1: Project Overview
   - Section 2: Functional Requirements
   - Section 3: Governance & Integrity
     - 3.1: Documentation Integrity (full protocol)
     - 3.2: Scientific Integrity (data authenticity)
     - 3.3: Testing Standards (comprehensive TDD guide)
   - Section 4: Development Environment
   - Section 5: Technical Requirements
   - Section 6: Testing Implementation

**Analysis documents can be deleted after implementation** (they served their purpose for decision-making).

**Want me to create these two consolidated files now?**

