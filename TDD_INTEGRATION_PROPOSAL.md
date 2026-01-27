# TDD Integration with Option 1 Documentation Structure

**Purpose:** Integrate DETERMINISTIC SIM TDD RULE into Option 1 consolidation strategy  
**Created:** January 18, 2026

---

## Integration Strategy Overview

The **DETERMINISTIC SIM TDD RULE** complements the integrity safeguards by:
- Preventing false claims about code working (tests prove it works)
- Ensuring reproducible results (deterministic = verifiable)
- Creating evidence trail (tests are documentation of behavior)
- Enforcing verification-first mindset (tests before code)

**Key Insight:** TDD naturally prevents integrity failures because you can't claim "it works" without tests passing.

---

## Where Each Component Goes (Option 1 Structure)

### **1. cursorrules - Enforce TDD Workflow**

Add to cursorrules as **mandatory AI behavior**:

```markdown
🚨 DETERMINISTIC TDD REQUIREMENT (NON-NEGOTIABLE) 🚨

Test-Driven Development Protocol:
- ALWAYS follow Red → Green → Refactor cycle
- NEVER write implementation before tests
- Tests MUST be deterministic (fixed seeds, explicit tolerances)

Red (Tests First) - MANDATORY SEQUENCE:
1. Restate spec: inputs/outputs, units, shapes, assumptions, tolerances
2. Write tests FIRST covering:
   a) Invariant tests (no NaNs/Infs, bounds, conservation, symmetry)
   b) Golden/regression test with canonical scenario
3. Run: pytest -q
4. CONFIRM tests fail for right reason (name failing test, expected behavior)
5. DO NOT proceed to implementation until step 4 complete

Green (Minimum Implementation):
- Implement ONLY what's needed to pass tests
- DO NOT modify tests during Green phase
- Run: pytest -q until all tests pass
- CONFIRM tests pass before proceeding

Refactor (Only After Green):
- Refactor ONLY after tests pass
- Re-run: pytest -q after each refactor
- Justify any tolerance changes with explicit reasoning

Numeric Comparisons (REQUIRED):
- NEVER use == for floats
- USE: np.allclose(actual, expected, atol=X, rtol=Y) with explicit tolerances
- USE: pytest.approx(expected, abs=X, rel=Y)
- For large arrays: ||y - y_ref|| / max(||y_ref||, eps) < tol

Reproducibility (REQUIRED):
- Use numpy.random.Generator (NO global np.random)
- Set explicit seeds in ALL tests using randomness
- Document seed values and RNG state in test docstrings

Violation Response:
- If uncertain about requirements: ASK before writing code
- If tests are slow/flaky: propose smaller unit/integration split
- If tolerance fails: explain why, never silently loosen

REFERENCE: See @docs/development/TESTING_GUIDE.md for examples and details
```

**Why in cursorrules:**
- AI must follow this for EVERY new function/module
- Non-negotiable workflow enforcement
- Complements integrity rules (can't fake passing tests)

---

### **2. docs/development/TESTING_GUIDE.md - Detailed Reference**

Comprehensive guide with examples, created from consolidation:

```markdown
# Testing Guide

## Table of Contents
1. Test-Driven Development Workflow
2. Deterministic Simulation Testing
3. Test Organization
4. Numeric Comparison Best Practices
5. Examples & Templates

## 1. Test-Driven Development Workflow

### The Red → Green → Refactor Cycle

**This is mandatory for all new functionality.**

#### RED Phase: Tests First

**Before writing any implementation code:**

1. **Restate the specification clearly:**
   - What are the inputs and outputs?
   - What are the units and shapes?
   - What assumptions are we making?
   - What tolerance strategy will we use?
   
2. **Write tests first** covering:
   
   **a) Invariant Tests** (properties that MUST always hold):
   ```python
   def test_triangulation_produces_finite_values():
       """Triangulation must never produce NaN or Inf values."""
       points_2d_cam1 = np.array([[100, 100], [200, 150]])
       points_2d_cam2 = np.array([[150, 105], [180, 145]])
       
       points_3d = triangulate(points_2d_cam1, points_2d_cam2, P1, P2)
       
       assert np.all(np.isfinite(points_3d)), "3D points contain NaN/Inf"
   
   def test_distance_is_non_negative():
       """Euclidean distance must be >= 0."""
       p1 = np.array([1.0, 2.0, 3.0])
       p2 = np.array([4.0, 5.0, 6.0])
       
       dist = euclidean_distance(p1, p2)
       
       assert dist >= 0, f"Distance is negative: {dist}"
   
   def test_rotation_preserves_vector_norms():
       """Rotation matrix must preserve vector lengths."""
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
   
   **b) Golden/Regression Tests** (canonical scenario with known output):
   ```python
   def test_triangulation_golden_case():
       """Test triangulation with canonical example.
       
       Scenario: Two cameras 0.65m apart, both looking at point (0, 0, 2) in 3D.
       Expected: Triangulation should recover (0, 0, 2) within tolerance.
       
       Seed: None (deterministic geometry)
       Tolerance: atol=0.01 (1cm), rtol=1e-5
       """
       # Camera setup (deterministic)
       cam1_pos = np.array([0.0, 0.0, 0.0])
       cam2_pos = np.array([0.65, 0.0, 0.0])
       P1, P2 = setup_stereo_cameras(cam1_pos, cam2_pos, focal_length=800)
       
       # Known 3D point
       point_3d_true = np.array([0.0, 0.0, 2.0])
       
       # Project to 2D (deterministic)
       point_2d_cam1 = project_3d_to_2d(point_3d_true, P1)
       point_2d_cam2 = project_3d_to_2d(point_3d_true, P2)
       
       # Triangulate back
       point_3d_reconstructed = triangulate(point_2d_cam1, point_2d_cam2, P1, P2)
       
       # Verify with explicit tolerances
       np.testing.assert_allclose(
           point_3d_reconstructed,
           point_3d_true,
           atol=0.01,  # 1cm absolute tolerance
           rtol=1e-5,  # 0.001% relative tolerance
           err_msg="Triangulation failed to recover known 3D point"
       )
   ```

3. **Run tests and confirm they fail:**
   ```bash
   pytest -q tests/test_triangulation.py
   ```
   
   **Expected output:**
   ```
   F                                                                    [100%]
   ================================ FAILURES =================================
   _______________________ test_triangulation_golden_case ____________________
   
   NameError: name 'triangulate' is not defined
   ```
   
   **✅ Good failure:** Function doesn't exist yet (expected!)
   
   **❌ Bad failure:** Test has syntax error, wrong imports, bad test logic

#### GREEN Phase: Minimum Implementation

**Now implement the minimum code to pass tests:**

```python
def triangulate(points_2d_cam1, points_2d_cam2, P1, P2):
    """Triangulate 3D points from stereo 2D correspondences.
    
    Args:
        points_2d_cam1: Nx2 array of 2D points in camera 1
        points_2d_cam2: Nx2 array of 2D points in camera 2
        P1: 3x4 camera projection matrix for camera 1
        P2: 3x4 camera projection matrix for camera 2
    
    Returns:
        points_3d: Nx3 array of triangulated 3D points
    """
    # Implement minimum logic to pass tests
    points_3d = cv2.triangulatePoints(P1, P2, 
                                       points_2d_cam1.T, 
                                       points_2d_cam2.T)
    points_3d = points_3d[:3] / points_3d[3]  # Homogeneous to Cartesian
    return points_3d.T
```

**Run tests:**
```bash
pytest -q tests/test_triangulation.py
```

**Expected output:**
```
..                                                                    [100%]
2 passed in 0.03s
```

**✅ All tests pass → proceed to Refactor**

**❌ Tests fail → debug implementation, don't modify tests**

#### REFACTOR Phase: Improve Code Quality

**Only after tests pass, refactor for clarity/performance:**

```python
def triangulate(points_2d_cam1, points_2d_cam2, P1, P2):
    """Triangulate 3D points from stereo 2D correspondences using DLT.
    
    Uses Direct Linear Transform (DLT) triangulation via OpenCV.
    
    Args:
        points_2d_cam1: Nx2 array of 2D points in camera 1, shape (N, 2)
        points_2d_cam2: Nx2 array of 2D points in camera 2, shape (N, 2)
        P1: Camera 1 projection matrix, shape (3, 4)
        P2: Camera 2 projection matrix, shape (3, 4)
    
    Returns:
        points_3d: Triangulated 3D points in world coordinates, shape (N, 3)
        
    Notes:
        - Input points should be in pixel coordinates
        - Output is in same units as camera matrices
        - Uses homogeneous coordinates internally
    """
    # Validate inputs
    assert points_2d_cam1.shape == points_2d_cam2.shape, "Point arrays must match"
    assert points_2d_cam1.shape[1] == 2, "Points must be 2D"
    
    # Triangulate using OpenCV (expects transposed input)
    points_4d_homogeneous = cv2.triangulatePoints(
        P1, P2, 
        points_2d_cam1.T.astype(np.float32), 
        points_2d_cam2.T.astype(np.float32)
    )
    
    # Convert from homogeneous (4D) to Cartesian (3D)
    points_3d = points_4d_homogeneous[:3] / points_4d_homogeneous[3]
    
    return points_3d.T  # Return as Nx3
```

**Re-run tests after refactoring:**
```bash
pytest -q tests/test_triangulation.py
```

**✅ Tests still pass → refactor successful**

**❌ Tests fail → revert refactor, debug**

---

## 2. Deterministic Simulation Testing

### Why Deterministic Tests Matter

**Problem:** Non-deterministic tests make it impossible to verify integrity
- "Test passed" vs "Test passed by chance"
- Can't reproduce failures
- Can't verify someone else's claims

**Solution:** All tests must be reproducible with fixed seeds

### Random Number Generation Rules

**❌ WRONG: Global random state**
```python
import numpy as np

def generate_trajectory():
    np.random.seed(42)  # Global state (bad!)
    return np.random.randn(100, 3)

def test_trajectory_generation():
    traj = generate_trajectory()
    assert traj.shape == (100, 3)  # Non-deterministic without seed here!
```

**✅ CORRECT: Explicit Generator with seed**
```python
import numpy as np

def generate_trajectory(rng=None):
    """Generate random trajectory.
    
    Args:
        rng: numpy.random.Generator instance (for reproducibility)
    """
    if rng is None:
        rng = np.random.default_rng(42)  # Default seed for determinism
    
    return rng.standard_normal(size=(100, 3))

def test_trajectory_generation_deterministic():
    """Test trajectory generation is deterministic with fixed seed.
    
    Seed: 42
    Expected: Specific trajectory values (golden test)
    """
    rng = np.random.default_rng(42)  # Explicit seed in test
    
    traj = generate_trajectory(rng)
    
    # Golden values (computed once, stored as expected)
    expected_first_point = np.array([0.49671415, -0.1382643, 0.64768854])
    
    np.testing.assert_allclose(
        traj[0],
        expected_first_point,
        rtol=1e-10,
        err_msg="Trajectory generation is not deterministic"
    )
```

### Seed Documentation Standard

**Every test using randomness MUST document:**
1. The seed value
2. Why that seed was chosen
3. What behavior is being tested

```python
def test_noise_robustness_with_fixed_seed():
    """Test triangulation robustness to measurement noise.
    
    Seed: 12345
    Why: Chosen to produce moderate noise (not too easy, not too hard)
    Tolerance: atol=0.05 (5cm) due to 2-pixel Gaussian noise
    
    This test verifies triangulation works with realistic sensor noise.
    """
    rng = np.random.default_rng(12345)
    
    # Add noise to 2D points
    noise_std = 2.0  # pixels
    points_2d_noisy = points_2d_true + rng.normal(0, noise_std, points_2d_true.shape)
    
    # Triangulate
    points_3d = triangulate(points_2d_noisy, points_2d_cam2, P1, P2)
    
    # Verify within noise-appropriate tolerance
    np.testing.assert_allclose(
        points_3d,
        points_3d_true,
        atol=0.05,  # 5cm tolerance for 2-pixel noise
        err_msg="Triangulation not robust to measurement noise"
    )
```

---

## 3. Test Organization

### Directory Structure

```
mono_to_3d/
├── tests/
│   ├── __init__.py
│   ├── conftest.py                    # Shared fixtures
│   ├── test_invariants.py             # Cross-module invariant tests
│   ├── test_golden.py                 # Canonical golden tests
│   ├── unit/
│   │   ├── test_camera.py
│   │   ├── test_triangulation.py
│   │   ├── test_tracking.py
│   │   └── test_visualization.py
│   ├── integration/
│   │   ├── test_stereo_pipeline.py
│   │   └── test_end_to_end.py
│   └── performance/
│       └── test_benchmarks.py
├── pytest.ini                          # Pytest configuration
└── [source code]
```

### Test File Template

```python
"""
Test module for [functionality].

All tests follow Red → Green → Refactor TDD cycle.
All random tests use explicit seeds for determinism.
All numeric comparisons use explicit tolerances.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from module_under_test import function_to_test


class TestFunctionInvariants:
    """Invariant tests: properties that must always hold."""
    
    def test_output_is_finite(self):
        """Output must never contain NaN or Inf."""
        # Test implementation
        pass
    
    def test_output_shape_matches_input(self):
        """Output shape must match documented behavior."""
        # Test implementation
        pass


class TestFunctionGolden:
    """Golden/regression tests: canonical scenarios with known outputs."""
    
    def test_canonical_scenario_1(self):
        """Test [specific scenario].
        
        Scenario: [describe setup]
        Expected: [expected outcome]
        Seed: [if random, specify seed]
        Tolerance: [atol/rtol with justification]
        """
        # Test implementation
        pass


class TestFunctionEdgeCases:
    """Edge case tests: boundary conditions, error handling."""
    
    def test_empty_input_raises_error(self):
        """Empty input should raise ValueError."""
        with pytest.raises(ValueError, match="Input cannot be empty"):
            function_to_test(np.array([]))
```

---

## 4. Numeric Comparison Best Practices

### Never Use == for Floats

**❌ WRONG:**
```python
assert computed_value == 1.0  # Fails due to floating point errors
```

**✅ CORRECT:**
```python
np.testing.assert_allclose(computed_value, 1.0, rtol=1e-10, atol=1e-12)
# or
assert computed_value == pytest.approx(1.0, rel=1e-10, abs=1e-12)
```

### Choosing Tolerances

**General guidelines:**

| Tolerance | Use Case | Example |
|-----------|----------|---------|
| `rtol=1e-10, atol=1e-12` | Pure math, no noise | Rotation matrix properties |
| `rtol=1e-5, atol=1e-6` | Standard numerical algorithms | Triangulation with perfect data |
| `rtol=1e-3, atol=0.01` | Real sensor data | 3D reconstruction with noise |
| `rtol=0.1, atol=1.0` | Approximate methods | Neural network predictions |

**Always justify tolerance in comments:**
```python
np.testing.assert_allclose(
    reconstructed_3d,
    ground_truth_3d,
    atol=0.02,  # 2cm tolerance: accounts for 1-pixel reprojection error
    rtol=1e-5,
    err_msg="3D reconstruction outside acceptable error"
)
```

### Norm-Based Checks for Large Arrays

**For large arrays, use relative norm:**

```python
def assert_arrays_close_norm(actual, expected, tol=1e-6, eps=1e-10):
    """Assert arrays are close using relative norm.
    
    Checks: ||actual - expected|| / max(||expected||, eps) < tol
    
    This is more robust for large arrays than element-wise comparisons.
    """
    diff_norm = np.linalg.norm(actual - expected)
    expected_norm = np.linalg.norm(expected)
    relative_error = diff_norm / max(expected_norm, eps)
    
    assert relative_error < tol, (
        f"Relative error {relative_error:.2e} exceeds tolerance {tol:.2e}\n"
        f"||diff|| = {diff_norm:.2e}, ||expected|| = {expected_norm:.2e}"
    )

# Usage
def test_trajectory_reconstruction():
    """Test full trajectory reconstruction."""
    trajectory_3d = reconstruct_trajectory(frames_2d, cameras)
    
    assert_arrays_close_norm(
        trajectory_3d,
        ground_truth_trajectory,
        tol=1e-4,  # 0.01% relative error
        eps=1e-10
    )
```

---

## 5. Examples & Templates

### Example 1: Testing Camera Projection

**Red Phase (tests first):**

```python
# tests/unit/test_camera.py

def test_projection_produces_finite_2d_points():
    """Camera projection must produce finite 2D coordinates."""
    P = create_projection_matrix(focal_length=800, center=(320, 240))
    point_3d = np.array([0.0, 0.0, 2.0])
    
    point_2d = project_3d_to_2d(point_3d, P)
    
    assert np.all(np.isfinite(point_2d)), "2D projection contains NaN/Inf"
    assert point_2d.shape == (2,), f"Expected shape (2,), got {point_2d.shape}"


def test_projection_known_point_golden():
    """Test projection with known geometric setup.
    
    Scenario: Point at (0, 0, 2) with camera at origin, focal=800, center=(320, 240)
    Expected: Projects to image center (320, 240)
    Tolerance: atol=0.1 (sub-pixel accuracy)
    """
    P = create_projection_matrix(focal_length=800, center=(320, 240))
    point_3d = np.array([0.0, 0.0, 2.0])
    
    point_2d = project_3d_to_2d(point_3d, P)
    
    expected_2d = np.array([320.0, 240.0])
    np.testing.assert_allclose(
        point_2d,
        expected_2d,
        atol=0.1,
        err_msg="Projection of point on optical axis failed"
    )
```

**Green Phase (implementation):**

```python
# camera.py

def project_3d_to_2d(point_3d, P):
    """Project 3D point to 2D using camera matrix."""
    point_3d_homog = np.append(point_3d, 1.0)
    point_2d_homog = P @ point_3d_homog
    point_2d = point_2d_homog[:2] / point_2d_homog[2]
    return point_2d
```

**Refactor Phase (improved):**

```python
def project_3d_to_2d(point_3d, P):
    """Project 3D point to 2D image coordinates using projection matrix.
    
    Args:
        point_3d: 3D point in world/camera coordinates, shape (3,)
        P: 3x4 projection matrix
    
    Returns:
        point_2d: 2D pixel coordinates, shape (2,)
    
    Raises:
        ValueError: If point projects behind camera (z <= 0)
    """
    # Input validation
    assert point_3d.shape == (3,), f"Expected shape (3,), got {point_3d.shape}"
    assert P.shape == (3, 4), f"Projection matrix must be 3x4, got {P.shape}"
    
    # Convert to homogeneous coordinates
    point_3d_homog = np.append(point_3d, 1.0)
    
    # Project
    point_2d_homog = P @ point_3d_homog
    
    # Check depth
    if point_2d_homog[2] <= 0:
        raise ValueError(f"Point projects behind camera: z={point_2d_homog[2]}")
    
    # Convert from homogeneous to Cartesian
    point_2d = point_2d_homog[:2] / point_2d_homog[2]
    
    return point_2d
```

---

## Integration with Integrity Protocols

### How TDD Enforces Documentation Integrity

**Problem we had:** Documentation claimed "50 samples generated" without verification

**TDD solution:** Can't claim it works without tests proving it

```python
def test_dataset_generation_count():
    """Verify dataset contains claimed number of samples.
    
    This test prevents documentation integrity failures by verifying
    actual sample counts match documentation claims.
    
    Seed: 42
    Expected: Exactly 50 samples as documented
    """
    rng = np.random.default_rng(42)
    
    dataset = generate_magvit_3d_dataset(num_samples=50, rng=rng)
    
    # VERIFY CLAIM: "50 samples generated"
    assert len(dataset['trajectories_3d']) == 50, (
        f"Documentation claims 50 samples, but {len(dataset['trajectories_3d'])} were generated"
    )
    
    # VERIFY CLAIM: "3D trajectory shape (N, 16, 3)"
    assert dataset['trajectories_3d'].shape == (50, 16, 3), (
        f"Expected shape (50, 16, 3), got {dataset['trajectories_3d'].shape}"
    )
```

**Result:** Documentation can reference test results as proof:

```markdown
## MAGVIT 3D Results

**Status: VERIFIED**

Dataset generation confirmed by test suite:
- `test_dataset_generation_count` ✅ PASSED
- `test_trajectory_shapes_match_spec` ✅ PASSED
- `test_visualization_files_created` ✅ PASSED

Verification command:
```bash
$ pytest tests/test_magvit_3d.py -v
tests/test_magvit_3d.py::test_dataset_generation_count PASSED
tests/test_magvit_3d.py::test_trajectory_shapes_match_spec PASSED
tests/test_magvit_3d.py::test_visualization_files_created PASSED
```

Evidence: 50 samples confirmed in test output above.
```

---

## Summary: TDD as Integrity Insurance

| Integrity Risk | TDD Prevention |
|----------------|----------------|
| Claiming sample counts without verification | Golden test verifies exact count |
| Claiming files exist without checking | Test asserts file existence |
| Claiming "100% success" without proof | Test suite IS the proof |
| Claiming algorithm works | Tests demonstrate it works |
| Non-reproducible results | Deterministic tests with seeds |

**Key Principle:** 
> **If it's not tested, it doesn't exist. If it's tested, we have proof.**

TDD creates an audit trail that prevents integrity failures by making claims verifiable.

