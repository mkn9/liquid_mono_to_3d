# Project Requirements

**3D Track Reconstruction from Multiple 2D Mono Tracks**

**Last Updated:** January 31, 2026

---

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

This application reconstructs 3D tracks of objects using 2D tracking information from two or more monocular camera sensors. By triangulating position data from multiple viewpoints, the system creates accurate 3D trajectory representations.

### Key Capabilities
- Camera calibration and projection
- 2D object tracking integration
- 3D triangulation and reconstruction
- Multi-view trajectory analysis
- Interactive 3D visualization
- Track filtering and persistence analysis

### System Architecture
- **Stereo/Multi-Camera Configuration**: Two or more cameras with known parameters
- **Calibration**: Intrinsic and extrinsic camera parameters
- **Synchronization**: Frame-level temporal alignment
- **Reconstruction**: Linear and non-linear triangulation methods
- **Filtering**: Track persistence and noise reduction

---

## 2. Functional Requirements

### 2.1 Camera System

#### Camera Calibration
- Support for standard calibration patterns (checkerboard, charuco)
- Storage and retrieval of camera parameters
- Extrinsic calibration between multiple cameras
- Camera matrix, distortion coefficients

#### Camera Visualization
- Display camera positions in 3D space with distinct markers
- Visualize camera boresight (line of sight) for each camera:
  - Show principal axis vector indicating viewing direction
  - Use different colors for each camera's boresight line
  - Include field of view indicators (viewing frustum)
  - Display boresight intersection points when applicable
- Interactive controls to adjust camera viewing angles
  - Toggle boresight line visibility
- Distance and angle measurements between cameras

### 2.2 Tracking Input

#### Input Format Requirements
- Support for multiple input formats (CSV, JSON, custom formats)
- Each 2D track must contain:
  - Unique track ID
  - Timestamp
  - x,y coordinates (pixel or normalized)
  - Optional confidence value
  - Optional bounding box dimensions
- Minimum of two synchronized camera views required

#### Track Preprocessing
- Temporal synchronization across cameras
- Track ID association between views
- Gap filling and interpolation
- Outlier detection and removal

### 2.3 3D Reconstruction

#### Triangulation Methods
- Linear triangulation (Direct Linear Transform)
- Non-linear optimization (bundle adjustment)
- Handling of occlusions and track interruptions
- Noise filtering and smoothing algorithms
- Kalman filtering for trajectory refinement

#### Coordinate Systems
- Camera coordinates (individual camera reference frames)
- World coordinates (global 3D coordinate system)
- Homogeneous transformation matrices
- **Non-standard coordinate system**: Cameras looking in +Y direction

### 2.4 Track Persistence Filtering

#### Persistence Classification
- Distinguish persistent tracks from transient detections
- Filter out brief, noisy, or artifactual detections
- Retain long-duration object tracks
- Configurable persistence thresholds

#### Modular Architecture
- Interchangeable feature extractors
- Flexible sequence models (Transformer, LSTM, etc.)
- Configurable task heads for classification

### 2.5 Visualization

#### 3D Visualization
- Interactive 3D plotting (matplotlib, plotly)
- Time-based playback of tracks
- Multi-view display (2D views alongside 3D reconstruction)
- Camera position and trajectory visualization
- Coordinate system visualization

#### Output Formats
- 3D track data export (CSV, JSON, NPZ) with datetime prefix
- Visualization images (PNG) with datetime prefix for chronological ordering
- Analysis reports (Markdown, HTML) with datetime prefix
- Error metrics and statistics (JSON, CSV) with datetime prefix
- **See Section 5.4 for mandatory filename format: `YYYYMMDD_HHMM_descriptive_name.ext`**

---

## 3. Governance & Integrity Standards

### 3.1 Documentation Integrity Protocol

#### Core Principle
> **VERIFY EVERY CLAIM BEFORE DOCUMENTING IT**

If you cannot verify a claim with concrete evidence, do not present it as fact.

#### Mandatory Verification Steps

**1. File Existence Claims**

BEFORE writing: "File X exists" or "Visualization Y was generated"

REQUIRED VERIFICATION:
```bash
# Check file exists
ls path/to/file.png
# OR
find . -name "file.png" -type f
# OR use read_file tool
```

Show in documentation:
```markdown
**Verification:**
```bash
$ ls neural_video_experiments/magvit/results/
magvit_3d_dataset.npz  5.5K
```
File confirmed to exist.
```

**NEVER claim files exist without this verification.**

**2. Data Count Claims**

BEFORE writing: "N samples generated" or "Dataset contains X items"

REQUIRED VERIFICATION:
```python
import numpy as np
data = np.load('dataset.npz')
print(f"Actual count: {len(data['key'])}")
print(f"Actual shape: {data['key'].shape}")
```

Show in documentation:
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

**NEVER claim sample counts without loading and checking the data.**

**3. Success/Completion Claims**

BEFORE writing: "100% success" or "All tests passed"

REQUIRED VERIFICATION:
```bash
# Run and show actual output
python -m pytest -v
# OR
python script.py > output.log 2>&1
cat output.log
```

Show in documentation:
```markdown
**Verification:**
```bash
$ pytest tests/test_magvit_3d.py -v
tests/test_magvit_3d.py::test_generation PASSED
tests/test_magvit_3d.py::test_shapes PASSED
2 passed in 0.15s
```
Tests confirm functionality.
```

**NEVER claim success without showing actual results.**

**4. Visualization/Plot Claims**

BEFORE writing: "Figure X shows..." or "Plots demonstrate..."

REQUIRED VERIFICATION:
```bash
# List all visualization files
ls -lh output/*.png
# OR
find . -name "*.png" -type f
```

**NEVER reference visualizations without confirming they exist.**

**5. Performance/Metric Claims**

BEFORE writing: "Achieved X accuracy" or "Runtime of Y seconds"

REQUIRED VERIFICATION:
- Show actual measurement output
- Include timestamps
- Provide calculation details

**NEVER claim metrics without showing how they were measured.**

#### Language Standards

**✅ ACCEPTABLE Language (Verified Work)**

When work HAS been completed and verified:
- "Verification confirms N samples exist" + [show evidence]
- "File listing shows output.png was created" + [show ls output]
- "Test execution demonstrates 36/36 tests passed" + [show test output]
- "Actual data shape is (3, 16, 3)" + [show data loading output]

**✅ ACCEPTABLE Language (Unverified/Planned Work)**

When work has NOT been completed or cannot be verified:
- "Code has been written to generate N samples"
- "Designed to produce visualizations"
- "Intended to create the following outputs"
- "**NOT YET EXECUTED**"
- "**PLAN:** Will generate..."
- "**DESIGN:** Intended to produce..."

**❌ PROHIBITED Language (Integrity Violations)**

NEVER use these without verification:
- "Successfully generated N samples" ← requires verification
- "Results show..." ← must show actual results
- "File X contains..." ← must verify file exists
- "100% success rate" ← must measure and prove
- "All tests passed" ← must show test output
- "Visualizations demonstrate..." ← must verify visualizations exist
- Past tense for unexecuted work ← must clarify not yet done

#### The Three-State Model

ALL work falls into one of three states. Be explicit about which state:

**State 1: CODE WRITTEN**
- Code files exist in repository
- Functions/classes are implemented
- Tests are written
- **Documentation:** "Code has been written to..."

**State 2: CODE EXECUTED**
- Code has been run
- Output files were generated
- Logs/results are available
- **Documentation:** "Execution produced..." + [show evidence]

**State 3: RESULTS VERIFIED**
- Output files confirmed to exist
- Data counts validated
- Metrics measured and documented
- **Documentation:** "Verification confirms..." + [show proof]

**NEVER conflate State 1 with State 2 or 3.**

#### Verification Checklist

Before committing documentation, answer ALL these questions:

**File Claims:**
- [ ] Did I verify every claimed file exists? (Show ls/find output)
- [ ] Did I check file sizes/timestamps? (Show ls -lh output)
- [ ] Did I actually read file contents if referencing them?

**Data Claims:**
- [ ] Did I load data files to verify counts? (Show shape output)
- [ ] Did I check actual array/dataframe dimensions?
- [ ] Did I examine sample data contents?

**Success Claims:**
- [ ] Did I run tests and capture output? (Show pytest output)
- [ ] Did I measure actual metrics? (Show calculations)
- [ ] Did I verify success criteria were met?

**Visualization Claims:**
- [ ] Did I verify image files exist? (Show ls *.png)
- [ ] Did I check image file sizes are non-zero?
- [ ] Did I view or inspect the images?

**Language Check:**
- [ ] Did I distinguish "code written" from "code executed"?
- [ ] Did I avoid past tense for unexecuted work?
- [ ] Did I label plans/designs clearly as such?
- [ ] Did I provide evidence for every factual claim?

#### Red Flags - Stop and Verify

If your documentation contains ANY of these, STOP and verify:

🚩 Claiming specific sample counts (e.g., "50 samples")  
🚩 Referencing file names (e.g., "output.png shows...")  
🚩 Stating "100%" or "all" (e.g., "all tests passed")  
🚩 Using past tense about execution (e.g., "generated", "created")  
🚩 Claiming success (e.g., "successfully completed")  
🚩 Describing visualizations (e.g., "plots demonstrate")  
🚩 Providing metrics (e.g., "achieved 98% accuracy")  
🚩 Listing deliverables (e.g., "produced the following files")

**For EVERY red flag: Provide concrete verification evidence.**

---

### 3.2 Proof Bundle System - Definition of "Done"

**PRIMARY GATE: A task is complete ONLY if `bash scripts/prove.sh` exits 0.**

#### Core Principle

One command defines "done" for any task:

```bash
bash scripts/prove.sh
```

- **Exit 0** → Task complete with proof bundle
- **Exit != 0** → Task incomplete

No exceptions. No partial credit. No "mostly done".

#### What prove.sh Does

1. Captures environment (git SHA, timestamp, python version)
2. Runs all tests (`pytest -q`)
3. Optionally runs component contracts (`contracts/*.yaml`)
4. Creates file manifest with checksums
5. Saves everything to `artifacts/proof/<git_sha>/`

#### Proof Bundle Structure

```
artifacts/proof/<git_sha>/
├── prove.log       # Full test output
├── meta.txt        # Git commit, timestamp, environment
├── manifest.txt    # File checksums (tamper-evident)
├── pip_freeze.txt  # Python dependencies
└── contracts.log   # Optional contract results
```

**Key Feature:** Everything tied to a specific git commit. No ambiguity.

#### Component Contracts (Optional)

Define machine-checkable requirements for claimed components as YAML files:

**Example: `contracts/magvit.yaml`**

```yaml
name: "magvit_integration"

# Must be able to import
imports:
  - "from magvit2 import MAGVIT_VQ_VAE"

# Must run successfully
commands:
  - cmd: "python scripts/test_magvit_encode.py"
    outputs:
      - "artifacts/proof_outputs/magvit_reconstruction.png"

# Tests must pass
tests:
  - cmd: "pytest tests/test_magvit.py -v"
```

**Why Contracts:**
- Machine-checkable (no debate about whether component exists)
- Scalable (add one YAML per component)
- Structured (same schema for all components)
- Evidence-generating (can require specific outputs)

#### Rules

**Rule 1:** Cannot claim "done" without proof bundle  
**Rule 2:** Tests must be deterministic (fixed seeds, explicit tolerances)  
**Rule 3:** Commit proof bundles (don't .gitignore them)  
**Rule 4:** If cannot run prove.sh, state: "NOT VERIFIED. Run: bash scripts/prove.sh"

#### Deterministic Test Requirements

```python
# ✅ GOOD: Deterministic
torch.manual_seed(42)
np.random.seed(42)
x = torch.randn(10)
result = model(x)
assert torch.allclose(result, expected, atol=1e-5, rtol=1e-5)

# ❌ BAD: Non-deterministic
x = torch.randn(10)  # Random every time, no seed
result = model(x)
assert result > 0  # Vague assertion
assert result == expected  # Float equality without tolerance
```

#### Usage Examples

**Verify current state:**
```bash
bash scripts/prove.sh
echo $?  # 0 = success

# View proof
ls artifacts/proof/$(git rev-parse HEAD)/
cat artifacts/proof/$(git rev-parse HEAD)/prove.log
```

**Verify historical state:**
```bash
# What was proven at commit abc123?
cat artifacts/proof/abc123/prove.log
cat artifacts/proof/abc123/meta.txt
```

**Enable component contract:**
```bash
# Will expose lies if component doesn't exist
mv contracts/magvit.yaml.DISABLED contracts/magvit.yaml
bash scripts/prove.sh
# Will fail until component actually implemented
```

#### Integration with TDD

The proof bundle system complements (not replaces) TDD:
- TDD evidence is automatically captured in prove.log
- Red/Green/Refactor phases proven by test pass/fail
- No separate evidence capture needed
- prove.sh validates all TDD requirements

---

### 3.3 Scientific Integrity Protocol

#### Purpose
Prevent synthetic/fake data from being presented as real experimental results. This ensures scientific integrity and prevents hallucination in research outputs.

#### Data Source Verification

**BEFORE** creating any visualization or analysis, MUST verify:

- [ ] **Data Origin**: Is this from actual trained networks or real experiments?
- [ ] **Training Evidence**: Are there training logs, model weights, convergence curves?
- [ ] **Ground Truth**: Physical measurements or neural network outputs?

#### Synthetic Data Identification

If data is synthetic or simulated, it MUST be:

- [ ] **Clearly Labeled**: All plots, filenames, descriptions include "SYNTHETIC" or "SIMULATED"
- [ ] **Purpose Stated**: Why is synthetic data used (testing, demonstration, validation)?
- [ ] **Limitations Noted**: How does it differ from real data?

#### Prohibited Actions

STRICTLY FORBIDDEN:
- ❌ Creating synthetic data and presenting it as real
- ❌ Using placeholder data without clear labeling
- ❌ Generating fake neural network outputs
- ❌ Mixing synthetic and real data without distinction
- ❌ Using "demonstration" data as if it were experimental results

#### File Naming Convention

- **Real Data**: `dnerf_predictions_real.png`
- **Synthetic Data**: `dnerf_predictions_SYNTHETIC.png`
- **Mixed Data**: `dnerf_predictions_real_vs_synthetic.png`

#### Plot Titles and Labels

- **Real**: "D-NeRF Predictions from Trained Model"
- **Synthetic**: "SYNTHETIC D-NeRF Predictions (Demonstration Only)"
- **Simulation**: "Simulated D-NeRF Behavior for Testing"

#### Documentation Requirements

Every data file must include:
- **Source**: Where did this data come from?
- **Method**: How was it generated or collected?
- **Validation**: What evidence supports its authenticity?
- **Limitations**: What are the known issues or constraints?

#### Verification Checklist

**For D-NeRF or Neural Network Results:**
- [ ] Model weights exist (.pth, .pt, .ckpt files)?
- [ ] Training logs present (loss curves, iteration logs, convergence data)?
- [ ] Data pipeline verified (trace from raw input to final output)?
- [ ] Computational evidence (GPU/CPU logs showing actual training)?

**For Experimental Data:**
- [ ] Sensor readings (actual measurements or camera captures)?
- [ ] Timestamp verification (consistent with real data collection)?
- [ ] Calibration data (parameters from real equipment)?
- [ ] Environmental context (realistic conditions)?

#### Approved Synthetic Data Uses

**ACCEPTABLE with proper labeling:**
- Algorithm testing with known synthetic inputs
- Method validation with ground truth synthetic data
- Demonstration of visualization code
- Edge case testing with extreme synthetic scenarios

**Required labeling:**
- "SYNTHETIC" or "SIMULATED" in filename
- Clear labeling in plot titles and captions
- Purpose statement explaining why synthetic
- Distinction from real experimental results

---

### 3.4 Testing Standards (TDD)

#### Evidence-or-Abstain Requirement

**MANDATORY: All TDD claims must include captured evidence.**

##### Core Principle

> **NEVER claim tests were run without captured evidence.**

If you cannot provide evidence, you must explicitly state: "Tests not run; here is command to run."

##### Acceptable Evidence

**ONLY these forms of proof are acceptable:**

1. **Terminal output pasted in documentation/chat:**
   ```bash
   $ pytest -q
   .............                                [100%]
   13 passed in 0.42s
   ```

2. **Committed artifact files:**
   - `artifacts/tdd_red.txt` - RED phase failures
   - `artifacts/tdd_green.txt` - GREEN phase passes
   - `artifacts/tdd_refactor.txt` - REFACTOR phase passes
   - `artifacts/test_*.txt` - Additional verification runs

**Documentation references to these files are required**, not just descriptions of outputs.

##### Evidence Capture Tools

**Tool 1: Full TDD Workflow** (Recommended for new features)

```bash
# Run complete RED → GREEN → REFACTOR cycle with automatic evidence capture
bash scripts/tdd_capture.sh
```

This script:
- Runs tests before implementation (RED phase)
- Validates tests actually fail (prevents fake TDD)
- Saves `artifacts/tdd_red.txt`
- Runs tests after implementation (GREEN phase)
- Saves `artifacts/tdd_green.txt`
- Runs tests after refactoring (REFACTOR phase)
- Saves `artifacts/tdd_refactor.txt`

**Tool 2: Single Test Capture** (For verification or iterative work)

```bash
# Capture single test run with optional label
bash scripts/test_capture.sh [label]
# Creates artifacts/test_[label].txt
```

**Tool 3: Manual Capture** (When you need control)

```bash
# Manually capture test output
pytest -q 2>&1 | tee artifacts/test_description.txt
```

##### Commitment Requirements

**ALWAYS commit the artifacts/ directory with code changes.**

Example commit:
```
git add src/module.py tests/test_module.py artifacts/
git commit -m "Add feature X with TDD evidence"
```

#### The Red → Green → Refactor Cycle

**This is mandatory for all new functionality.**

##### Modified RED Requirements

**RED phase required for:**
- New features (new functions, classes, modules)
- Bug fixes with reproducer tests

**GREEN-only acceptable for:**
- Adding tests to existing, working code
- Refactoring with existing test coverage

**REFACTOR evidence ALWAYS required** - Every change must have final passing test proof.

##### RED Phase: Tests First

**Step 1: Restate Specification**

Before writing any tests or code, document:
- What are the inputs and outputs?
- What are the units and shapes?
- What assumptions are we making?
- What tolerance strategy will we use?

Example:
```python
def test_triangulation_golden():
    """Test 3D triangulation with known camera setup.
    
    SPECIFICATION:
    - Inputs: 2D points from 2 cameras (Nx2), projection matrices (3x4)
    - Outputs: 3D points in world coordinates (Nx3)
    - Units: Pixels (input), meters (output)
    - Shapes: Input (N, 2) per camera, Output (N, 3)
    - Assumptions: Cameras calibrated, perfect correspondences
    - Tolerance: atol=0.01 (1cm), rtol=1e-5 (0.001%)
    
    Scenario: Standard stereo setup, known 3D point at (0, 0, 2)
    Expected: Recovers (0, 0, 2) within tolerance
    Seed: None (deterministic geometry)
    """
```

**Step 2: Write Tests FIRST**

Write two types of tests:

**a) Invariant Tests** (properties that MUST always hold):

```python
def test_triangulation_produces_finite_values():
    """Triangulation must never produce NaN or Inf values."""
    points_2d_cam1 = np.array([[100, 100], [200, 150]])
    points_2d_cam2 = np.array([[150, 105], [180, 145]])
    
    points_3d = triangulate(points_2d_cam1, points_2d_cam2, P1, P2)
    
    assert np.all(np.isfinite(points_3d)), "3D points contain NaN/Inf"
    assert points_3d.shape == (2, 3), f"Expected (2, 3), got {points_3d.shape}"

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

**b) Golden Tests** (canonical scenario with known output):

```python
def test_triangulation_golden_case():
    """Test triangulation with canonical stereo setup.
    
    Scenario:
    - Two cameras 0.65m apart horizontally
    - Both looking at point (0, 0, 2) in world coordinates
    - Standard focal length (800px), image center (320, 240)
    
    Expected:
    - Triangulation recovers (0, 0, 2) within 1cm tolerance
    
    Seed: None (deterministic geometry)
    Tolerance: atol=0.01 (1cm), rtol=1e-5
    
    Justification:
    - 1cm tolerance accounts for sub-pixel discretization
    - No sensor noise in this pure geometry test
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

**Step 3: Run Tests and Confirm Failure**

```bash
pytest -q tests/test_triangulation.py
```

Expected output:
```
F
NameError: name 'triangulate' is not defined
```

✅ **Good failure** - function doesn't exist yet (expected!)  
❌ **Bad failure** - test has syntax error, wrong imports, bad test logic

##### GREEN Phase: Minimum Implementation

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

Run tests:
   ```bash
pytest -q tests/test_triangulation.py
```

Expected output:
```
..
2 passed in 0.03s
```

✅ **All tests pass** → proceed to Refactor  
❌ **Tests fail** → debug implementation, don't modify tests

**CRITICAL: Do NOT modify tests during Green phase**

##### REFACTOR Phase: Improve Code Quality

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
    
    Raises:
        ValueError: If input shapes don't match
    """
    # Validate inputs
    assert points_2d_cam1.shape == points_2d_cam2.shape, "Point arrays must match"
    assert points_2d_cam1.shape[1] == 2, "Points must be 2D"
    assert P1.shape == (3, 4), f"P1 must be 3x4, got {P1.shape}"
    assert P2.shape == (3, 4), f"P2 must be 3x4, got {P2.shape}"
    
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

Re-run tests after refactoring:
```bash
pytest -q tests/test_triangulation.py
```

✅ **Tests still pass** → refactor successful  
❌ **Tests fail** → revert refactor, debug

#### Deterministic Testing Requirements

##### Random Number Generation Rules

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

##### Seed Documentation Standard

Every test using randomness MUST document:
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

#### Numeric Comparison Best Practices

##### Never Use == for Floats

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

##### Choosing Tolerances

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

##### Norm-Based Checks for Large Arrays

For large arrays, use relative norm:

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
```

#### Long-Running Process Testing Requirements (MANDATORY)

**For processes estimated to run >5 minutes, tests MUST include:**

##### 1. Checkpoint Tests

```python
def test_checkpoints_created_at_intervals():
    """Verify checkpoints are saved at regular intervals.
    
    Requirement: Long-running processes must save checkpoints every 1-5 min
    Test Strategy: Generate with checkpoint_interval, verify files created
    Expected: Checkpoint files exist at correct intervals
    """
    output_dir = Path("test_checkpoints")
    output_dir.mkdir(exist_ok=True)
    
    # Generate with checkpoints (use 10% of production scale for testing)
    dataset = generate_with_checkpoints(
        num_samples=3000,  # 10% of 30K production
        checkpoint_interval=1000,
        output_dir=output_dir
    )
    
    # Verify checkpoint files were created
    checkpoints = sorted(output_dir.glob("checkpoint_*.npz"))
    assert len(checkpoints) == 3, \
        f"Expected 3 checkpoints for 3K samples, found {len(checkpoints)}"
    
    # Verify each has correct sample count
    for i, checkpoint_file in enumerate(checkpoints):
        data = np.load(checkpoint_file)
        expected = min(1000, 3000 - i * 1000)
        assert len(data['videos']) == expected, \
            f"Checkpoint {i} has {len(data['videos'])} samples, expected {expected}"
```

##### 2. Progress File Tests

```python
def test_progress_file_created_and_updated():
    """Verify PROGRESS.txt is created and contains required information.
    
    Requirement: Progress must be visible on MacBook without SSH
    Test Strategy: Generate with checkpoints, verify progress file
    Expected: File exists with completion percentage, ETA, timestamp
    """
    output_dir = Path("test_progress")
    output_dir.mkdir(exist_ok=True)
    
    dataset = generate_with_checkpoints(
        num_samples=2000,
        checkpoint_interval=1000,
        output_dir=output_dir
    )
    
    # Verify progress file exists
    progress_file = output_dir / "PROGRESS.txt"
    assert progress_file.exists(), "PROGRESS.txt not created"
    
    # Verify it contains required information
    content = progress_file.read_text()
    assert "Completed:" in content, "Missing completion status"
    assert "/" in content, "Missing X/Y format"
    assert "%" in content, "Missing percentage"
    assert "ETA:" in content, "Missing ETA"
    assert "Last update:" in content, "Missing timestamp"
```

##### 3. Resume Capability Tests

```python
def test_can_resume_from_last_checkpoint():
    """Verify generation can resume from last checkpoint if interrupted.
    
    Requirement: If interrupted, must be able to resume without data loss
    Test Strategy: Generate partial, stop, resume to completion
    Expected: Final dataset has all samples, no duplicates
    """
    output_dir = Path("test_resume")
    output_dir.mkdir(exist_ok=True)
    
    # First run: generate 2000 samples
    dataset1 = generate_with_checkpoints(
        num_samples=2000,
        checkpoint_interval=1000,
        output_dir=output_dir
    )
    
    # Verify checkpoints exist
    checkpoints = list(output_dir.glob("checkpoint_*.npz"))
    assert len(checkpoints) >= 2, "Need checkpoints to test resume"
    
    # Second run: resume and complete to 4000
    dataset2 = generate_with_checkpoints(
        num_samples=4000,
        checkpoint_interval=1000,
        output_dir=output_dir,
        resume=True  # Should detect existing checkpoints
    )
    
    # Verify final dataset has all samples
    assert len(dataset2['videos']) == 4000, \
        f"Expected 4000 samples, got {len(dataset2['videos'])}"
    
    # Verify no duplicates (check determinism)
    assert len(torch.unique(dataset2['labels'])) > 0
```

##### 4. Integration Test at Scale

```python
@pytest.mark.slow  # Mark as slow test (run separately)
def test_medium_scale_generation_completes():
    """Test generation at ~10% of production scale (~5 min runtime).
    
    Requirement: Must validate long-running behavior before production
    Test Strategy: Generate 3K-5K samples (10% of production 30K)
    Expected: Completes successfully with checkpoints and progress
    
    This is a "smoke test" - validates actual long-running behavior
    """
    output_dir = Path("test_5k_integration")
    output_dir.mkdir(exist_ok=True)
    
    start = time.time()
    
    dataset = generate_with_checkpoints(
        num_samples=5000,
        checkpoint_interval=1000,
        frames_per_video=16,
        image_size=(64, 64),
        output_dir=output_dir
    )
    
    elapsed = time.time() - start
    
    # Verify completion
    assert len(dataset['videos']) == 5000, "Incomplete dataset"
    
    # Verify checkpoints were created
    checkpoints = list(output_dir.glob("checkpoint_*.npz"))
    assert len(checkpoints) >= 5, "Missing checkpoints"
    
    # Verify progress file shows completion
    progress_file = output_dir / "PROGRESS.txt"
    assert progress_file.exists(), "Missing progress file"
    assert "COMPLETE" in progress_file.read_text(), "Not marked complete"
    
    print(f"✅ 5K integration test passed in {elapsed/60:.1f} minutes")
```

##### Pre-Launch Checklist

**BEFORE launching full production run (30K samples), verify:**

```python
# In test suite or pre-launch validation script
def validate_ready_for_production():
    """Verify all requirements met before production launch."""
    
    checks = {
        'checkpoint_tests_pass': False,
        'progress_file_tests_pass': False,
        'resume_tests_pass': False,
        'medium_scale_test_pass': False
    }
    
    # Run all checkpoint-related tests
    result = pytest.main([
        'tests/test_checkpoints.py',
        '-v',
        '-m', 'not slow'  # Exclude slow tests for quick check
    ])
    checks['checkpoint_tests_pass'] = (result == 0)
    
    # Run integration test at scale
    result = pytest.main([
        'tests/test_checkpoints.py::test_medium_scale_generation_completes',
        '-v',
        '-m', 'slow'
    ])
    checks['medium_scale_test_pass'] = (result == 0)
    
    # Verify all passed
    all_passed = all(checks.values())
    
    if not all_passed:
        print("❌ NOT READY FOR PRODUCTION")
        for check, passed in checks.items():
            status = "✅" if passed else "❌"
            print(f"  {status} {check}")
        raise RuntimeError("Pre-launch validation failed")
    
    print("✅ ALL CHECKS PASSED - READY FOR PRODUCTION")
    return True
```

**CRITICAL**: Do NOT launch full production run (30K samples) until:
- [ ] All checkpoint tests pass
- [ ] Medium-scale test (5K samples) completes successfully
- [ ] Progress visibility verified
- [ ] Resume capability verified

#### Integration with Integrity Protocols

TDD naturally prevents documentation integrity failures:

```python
def test_dataset_generation_count():
    """Verify dataset contains claimed number of samples.
    
    This test prevents documentation integrity failures by verifying
    actual sample counts match documentation claims.
    
    Documentation claims: "50 samples generated"
    This test PROVES the claim or forces correction.
    
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

Documentation can then reference test results as proof:

```markdown
## MAGVIT 3D Results

**Status: VERIFIED**

Dataset generation confirmed by test suite:
- `test_dataset_generation_count` ✅ PASSED
- `test_trajectory_shapes_match_spec` ✅ PASSED

Verification command:
```bash
$ pytest tests/test_magvit_3d.py -v
tests/test_magvit_3d.py::test_dataset_generation_count PASSED
tests/test_magvit_3d.py::test_trajectory_shapes_match_spec PASSED
```

Evidence: 50 samples confirmed in test output above.
```

**Key Principle:**  
> **If it's not tested, it doesn't exist. If it's tested, we have proof.**

#### Specification By Example (Guidance Without Gaming)

**Problem:** How to orient tests toward useful behavior without making them easy to game with hardcoded responses?

**Solution:** Use specification by example - define WHAT the system should do (intent), not HOW (implementation).

##### The Balance

**❌ Too Vague (No Guidance):**
```python
def test_answer_question():
    answer = bridge.answer_question(video, "Some question?")
    assert isinstance(answer, str)  # Useless - any string passes
```

**❌ Too Specific (Easy to Game):**
```python
def test_answer_question():
    answer = bridge.answer_question(video, "How many objects?")
    assert answer == "There are 5 objects."  # Can hardcode this exact string
```

**✅ Just Right (Specification By Example):**
```python
def test_answer_counting_questions():
    """
    SPECIFICATION: System should count objects accurately.
    
    INTENT: Given a video with N objects, when asked about count
    in various phrasings, answer should contain the number N and
    relevant context.
    
    EXAMPLES: These demonstrate the specification, not exhaustive cases.
    """
    # Test with MULTIPLE videos (can't hardcode all)
    test_cases = [
        ('video_3_objects.pt', 3),
        ('video_5_objects.pt', 5),
        ('video_7_objects.pt', 7),
    ]
    
    for video_path, true_count in test_cases:
        video, _ = load_video_with_metadata(video_path)
        
        # Test with MULTIPLE question phrasings (can't pattern match)
        questions = [
            "How many objects are visible?",
            "Count the objects in this video.",
            "What is the total number of objects?",
        ]
        
        for question in questions:
            answer = bridge.answer_question(video, question)
            
            # ✅ Test PROPERTY (contains correct count)
            numbers = extract_numbers(answer)
            assert true_count in numbers, \
                f"Expected count {true_count}, got: {answer}"
            
            # ✅ Test QUALITY (not just a bare number)
            assert len(answer.split()) > 2, \
                f"Answer should provide context: {answer}"
            
            # ✅ Test RELEVANCE (uses appropriate terminology)
            assert any(word in answer.lower() for word in 
                      ['object', 'sphere', 'item', 'element', 'visible', 'total']), \
                f"Answer should use relevant terminology: {answer}"
```

##### Four Principles of Specification By Example

**1. State Intent in Docstring:**
```python
"""
SPECIFICATION: Descriptions should mention key visual elements.

INTENT: Users should get accurate overview of video content.

KEY ELEMENTS: Object count, colors, motion patterns, persistence

QUALITY CRITERIA: Substantive (>50 chars), grammatically correct,
no hallucinations.
"""
```

**2. Provide Multiple Examples:**
- Test with multiple different videos (different object counts, motions)
- Test with multiple phrasings of similar questions
- Ensures system generalizes, not just matches patterns

**3. Test Properties, Not Exact Values:**
```python
# ✅ GOOD: Test the property
assert abs(predicted_count - true_count) <= 1, "Count approximately correct"

# ❌ BAD: Test exact value
assert predicted_count == 5
```

**4. Include Negative Tests (Critical!):**
```python
def test_no_hallucinations():
    """System must not describe objects that aren't present."""
    video = load_test_video('white_spheres_only.pt')
    description = bridge.describe_video(video)
    
    # ✅ Verify it doesn't mention things that AREN'T there
    false_elements = ['blue', 'red', 'cube', 'pyramid']
    for element in false_elements:
        assert element not in description.lower(), \
            f"Hallucinated '{element}' which is not in video!"
```

**Why This Works:**
- Multiple scenarios → Can't hardcode all
- Property-based → Allows natural language variation
- Negative tests → Catches gaming and hallucinations
- Ground truth → Validates correctness

#### Two-Stage Testing (Behavioral and Structural)

**Key Insight:** Not all tests can be written before seeing the code. There are two types:

##### Type 1: Behavioral Tests (Black Box) - Written FIRST

**Definition:** Test external behavior/interface without knowing internals.

**When to Write:** BEFORE implementation (classic TDD)

**What to Test:**
- ✅ Public interface contracts
- ✅ Input/output types
- ✅ Basic correctness (with ground truth)
- ✅ Error handling (bad inputs)
- ✅ Integration behavior

**Example:**
```python
def test_describe_video_returns_string():
    """Behavioral test - tests interface contract."""
    video = create_test_video()
    description = bridge.describe_video(video)
    
    # ✅ Tests behavior, not implementation
    assert isinstance(description, str)
    assert len(description) > 0
```

##### Type 2: Structural Tests (White Box) - Written AFTER

**Definition:** Test implementation details, internal states, performance characteristics.

**When to Write:** AFTER seeing the implementation

**What to Test:**
- ✅ Internal mechanisms (which layers, which methods)
- ✅ Performance optimizations (caching, batching)
- ✅ Implementation-specific edge cases
- ✅ Internal data flow

**Example:**
```python
def test_feature_extraction_uses_correct_layer():
    """
    WHITE BOX TEST - Written AFTER implementation.
    
    Requires knowledge that Worker 2 model has specific layer structure
    and features should come from 'layer4' of ResNet.
    """
    bridge = VisionLanguageBridge(...)
    
    # Access internal implementation details
    resnet_features = bridge._get_resnet_features(video)
    
    # ✅ Test internal structure
    assert resnet_features.shape[-1] == 512, \
        "Should extract from correct ResNet layer (512-dim)"
    
    # ✅ Test implementation efficiency
    assert bridge._feature_cache is not None, \
        "Should cache features to avoid redundant computation"
```

##### Revised TDD Workflow

```
Phase 1: Write Behavioral Tests FIRST (Before Code)
┌──────────────────────────────────────┐
│ 1. Write ALL behavioral tests       │
│ 2. RED: Run behavioral tests (fail) │
│ 3. Implement ALL code                │
│ 4. GREEN: Run behavioral tests(pass)│
└──────────────────────────────────────┘
              ↓
Phase 2: Add Structural Tests AFTER Implementation
┌──────────────────────────────────────┐
│ 5. Add structural tests (white box) │
│ 6. Run structural tests (pass)      │
│ 7. REFACTOR: Improve code quality    │
│ 8. Run ALL tests (still pass)       │
└──────────────────────────────────────┘
```

**Evidence Files:**
- `artifacts/tdd_red.txt` - Behavioral tests fail
- `artifacts/tdd_green.txt` - Behavioral tests pass
- `artifacts/tdd_structural.txt` - Structural tests pass
- `artifacts/tdd_refactor.txt` - All tests pass after refactoring

**Benefits:**
- Behavioral tests define requirements and guide implementation
- Structural tests verify optimizations and internal correctness
- Together provide complete coverage (external + internal)
- Flexibility to refactor internals if behavioral tests pass

#### API Key and Secrets Management

**For LLM integrations (OpenAI GPT-4, etc.) and cloud services:**

**✅ Storage Locations**

Store API keys in shell configuration files:
- **MacBook**: `~/.zshrc`
- **EC2**: `~/.bashrc`

```bash
# Add to ~/.zshrc (MacBook) or ~/.bashrc (EC2)
export OPENAI_API_KEY="sk-proj-..."
```

**✅ Setup for New Instances**

```bash
# On MacBook: Get the key
grep OPENAI_API_KEY ~/.zshrc

# Copy the entire export line
# SSH to new instance and paste into ~/.bashrc
ssh -i /Users/mike/keys/AutoGenKeyPair.pem ubuntu@NEW_IP
nano ~/.bashrc  # Paste, save (Ctrl+O), exit (Ctrl+X)
source ~/.bashrc
```

**✅ Verification (Without Exposing Key)**

```bash
# Show only first 20 characters (safe)
echo $OPENAI_API_KEY | head -c 20 && echo "..."
# Expected: sk-proj-Nae9JoShWsxa...

# Check length (should be 164 characters)
echo $OPENAI_API_KEY | wc -c
# Expected: 164
```

**✅ Python Usage**

```python
import os

# Access key from environment
api_key = os.environ.get('OPENAI_API_KEY')

# Verify without printing full key
if api_key:
    print(f"✅ Key found: {api_key[:20]}... (length: {len(api_key)})")
```

**✅ For Testing: Mock Services**

```python
@pytest.fixture
def mock_llm():
    """Mock LLM for testing without API key."""
    class MockLLM:
        def generate(self, prompt):
            return "Deterministic test response"
    return MockLLM()

# Use in tests
def test_llm_integration(mock_llm):
    result = mock_llm.generate("test prompt")
    assert isinstance(result, str)
```

**❌ NEVER:**
- Commit API keys to git
- Hardcode secrets in source code
- Write keys in documentation files
- Include keys in chat logs
- Require real API calls for unit tests to pass

**⚠️ Security Best Practices:**
- [ ] API keys stored in environment variables only (not code, not docs)
- [ ] Keys referenced by location (`~/.zshrc`, `~/.bashrc`), never printed in full
- [ ] Tests use mocks (don't require real API keys)
- [ ] `.env` files in `.gitignore` if used
- [ ] Verify without exposing: `echo $KEY | head -c 20`
- [ ] Rotate keys if exposed in git history

**Key Rotation (if compromised):**

1. Generate new key: https://platform.openai.com/api-keys
2. Update `~/.zshrc` on MacBook
3. Update `~/.bashrc` on EC2
4. Delete old key from OpenAI dashboard
5. Verify: `echo $OPENAI_API_KEY | head -c 20`

#### Quick Reference: TDD Best Practices

| Aspect | Do | Don't |
|--------|-----|-------|
| **Test Timing** | Behavioral tests BEFORE code | Try to write ALL tests before seeing code |
| | Structural tests AFTER code | Skip structural tests |
| **Test Content** | Test properties and invariants | Test exact output strings |
| | Multiple examples per scenario | Single example per test |
| | Include negative tests | Only test positive cases |
| **Guidance** | Specify intent in docstrings | Make tests too vague |
| | Provide example scenarios | Make tests too specific |
| **Gaming Prevention** | Multiple inputs, phrasings | Hardcoded expected outputs |
| | Ground truth validation | Pattern matching only |
| **Evidence** | Capture all phases (RED/GREEN/REFACTOR/STRUCTURAL) | Claim tests ran without evidence |

**Key Principle:** Guide the implementation toward useful behavior without constraining it to hardcoded responses. Test WHAT the system should do (properties, invariants), not HOW it says it (exact text).

---

### 3.5 Chat History Protocol

#### Core Principle

> **PRESERVE ALL DEVELOPMENT CONVERSATIONS AS PROJECT DOCUMENTATION**

Chat history is a critical part of project documentation, providing context for technical decisions, problem-solving approaches, and implementation rationale.

#### Mandatory Requirements

**1. Complete Preservation**
- ❌ **NEVER** delete chat history
- ❌ **NEVER** clean up or archive conversations
- ✅ Keep complete historical record
- ✅ All history committed to version control

**2. No Gitignore for Chat History**
- ✅ ALL chat history tracked in git
- ✅ Both JSON and Markdown formats committed
- ✅ Chat history is project documentation, not temporary files
- ⚠️ Review for sensitive information before committing

**3. Hybrid Format Approach**

Use BOTH formats simultaneously:

| Format | Location | Purpose |
|--------|----------|---------|
| **JSON** | `.chat_history/` | Structured data, queryable, metadata support |
| **Markdown** | `docs/chat_history/` | Human-readable, searchable, git-friendly |

**4. Dual Naming Convention**

**Individual Sessions:**
```
YYYYMMDD_HHMMSS_Descriptive_Topic.{json,md}

Examples:
- 20260120_143022_3D_Tracking_Implementation.json
- 20260120_143022_3D_Tracking_Implementation.md
```

**Aggregated Histories:**
```
chat_history_complete.md              # Main project history
CHAT_HISTORY_[TOPIC].md               # Topic-specific history

Examples:
- chat_history_complete.md
- experiments/magvit-3d/CHAT_HISTORY_MAGVIT.md
```

#### Standard Workflow

**Method 1: Interactive Script (Recommended for Quick Use)**

```bash
python3 scripts/save_chat.py
```

Follow the prompts:
1. Enter topic (descriptive name)
2. Enter tags (comma-separated)
3. Enter messages (role: user/assistant)
4. Type `---` to end each message
5. Type `DONE` when finished
6. Choose whether to append to aggregate history
7. Commit the changes to git

**Method 2: Programmatic (For Automated Workflows)**

```python
from chat_logger import ChatLogger
from pathlib import Path

logger = ChatLogger()

# Save with automatic markdown export
paths = logger.save_conversation(
    messages=[
        {"role": "user", "content": "Question here"},
        {"role": "assistant", "content": "Response here"}
    ],
    topic="Descriptive Topic Name",
    tags=["category1", "category2"],
    metadata={"project": "mono_to_3d"}
)

print(f"Saved to: {paths['json']}")
print(f"Markdown: {paths['markdown']}")

# Optional: Append to aggregate history
json_filename = Path(paths['json']).name
logger.append_to_aggregate(json_filename)

# Update index
logger.create_index()

# Commit to version control
# git add .chat_history/ docs/chat_history/ chat_history_complete.md
# git commit -m "docs: add chat history for [topic]"
```

**Method 3: Batch from File**

Create a text file with format:
```
TOPIC: Topic name here
TAGS: tag1, tag2, tag3

USER:
User message content

ASSISTANT:
Assistant response content

USER:
Another user message
```

Then run:
```bash
python3 scripts/save_chat.py conversation.txt
```

#### Directory Structure

```
mono_to_3d/
├── .chat_history/                    # JSON files (tracked in git)
│   ├── 20260120_143022_*.json
│   └── ...
├── docs/
│   └── chat_history/                 # Markdown exports
│       ├── INDEX.md                  # Searchable index
│       ├── 20260120_143022_*.md
│       └── ...
├── chat_history_complete.md          # Main aggregate
└── experiments/
    └── [name]/
        └── CHAT_HISTORY_[NAME].md    # Experiment-specific
```

#### Content Standards

**MUST Include:**
- Technical decisions and rationale
- Problem-solving discussions
- Design trade-offs
- Implementation approaches
- Bug reproduction and fixes
- Testing strategy
- Research context

**MUST Exclude:**
- API keys, passwords, tokens
- Personally identifiable information
- Proprietary business data
- Off-topic conversations

#### Security Checklist

Before committing, verify:
- [ ] No credentials or API keys
- [ ] No PII
- [ ] No sensitive information
- [ ] Content relevant to project

#### Standard Tags

Use consistent tags for categorization:

**Technical:** `3d-tracking`, `camera-calibration`, `visualization`, `testing`, `bug-fix`, `performance`, `refactoring`

**Phases:** `planning`, `implementation`, `debugging`, `review`, `optimization`

**Experiments:** `magvit`, `dnerf`, `neural-video`, `sensor-analysis`

#### Integration with Documentation Integrity

Chat history provides:
- **Evidence trail**: Trace for technical decisions
- **Context preservation**: Understanding of implementation choices
- **Reproducibility**: Complete record of development process

This protocol works with Section 3.1 (Documentation Integrity) to ensure all claims can be traced to their origin and rationale.

#### Enforcement

**Pre-commit Reminder** - Add to `.git/hooks/pre-commit`:
```bash
if git diff --cached --name-only | grep -q "\.py$"; then
    echo "⚠️  REMINDER: Save chat history for this work"
fi
```

**Weekly Review:**
1. Check `docs/chat_history/INDEX.md`
2. Verify tagging consistency
3. Ensure major conversations captured
4. Commit all history

#### Searching Chat History

**Using Search Script:**
```bash
./scripts/search_chat.sh "search term"
./scripts/search_chat.sh "camera calibration"
```

**Using grep directly:**
```bash
# Search markdown files
grep -r "triangulation" docs/chat_history/

# Search JSON files
grep -r "bug" .chat_history/

# Search main aggregate
grep "performance" chat_history_complete.md

# Case-insensitive with context
grep -ri -C 2 "optimization" docs/chat_history/
```

**View Index:**
```bash
cat docs/chat_history/INDEX.md
```

#### Listing Conversations

```python
from chat_logger import ChatLogger

logger = ChatLogger()

# List all conversations
conversations = logger.list_conversations()
for conv in conversations:
    print(f"- {conv['topic']} ({conv['timestamp']})")
    print(f"  Tags: {', '.join(conv['tags'])}")

# Filter by tags
planning_convs = logger.list_conversations(tags=["planning"])

# Filter by date range
jan_convs = logger.list_conversations(
    start_date="20260101",
    end_date="20260131"
)
```

#### Exporting to Markdown

```python
# Export specific conversation
logger.export_to_markdown("20260120_143022_Topic.json")

# Export to custom location
logger.export_to_markdown(
    "20260120_143022_Topic.json",
    output_path="custom/path/output.md"
)
```

#### Daily Workflow

**At End of Work Session:**

1. **Identify conversations to save**
   - Technical decisions made
   - Problems solved
   - Design discussions
   - Bug fixes implemented

2. **Save using script**
   ```bash
   python3 scripts/save_chat.py
   ```

3. **Review and commit**
   ```bash
   git status
   git add .chat_history/ docs/chat_history/
   git commit -m "docs: add chat history for [session topic]"
   ```

4. **Optional: Update aggregate**
   - For major conversations, append to `chat_history_complete.md`
   - Keeps main history file comprehensive

#### Weekly Maintenance

1. **Review INDEX.md**
   ```bash
   cat docs/chat_history/INDEX.md
   ```

2. **Verify tagging consistency**
   - Check that tags are used correctly
   - Ensure major topics are tagged

3. **Ensure completeness**
   - All significant conversations captured?
   - Any missing discussions?

4. **Commit any updates**
   ```bash
   git add docs/chat_history/INDEX.md
   git commit -m "docs: update chat history index"
   ```

#### Session Summary Template

At end of work session, create summary:

```markdown
## Session Summary: [Date]

**Topics Covered:**
- Topic 1
- Topic 2

**Key Decisions:**
1. Decision with rationale
2. Decision with rationale

**Action Items:**
- [ ] Item 1
- [ ] Item 2

**Chat History Files:**
- `YYYYMMDD_HHMMSS_topic1.md`
- `YYYYMMDD_HHMMSS_topic2.md`
```

#### FAQ

**Q: Should I save every single conversation?**  
A: Save conversations that contain technical decisions, problem-solving discussions, implementation approaches, bug fixes, or design rationale. Skip purely administrative or personal chat.

**Q: What if I forget to save during development?**  
A: Save it as soon as you remember. Use approximate timestamp and note "Retroactive documentation" in metadata.

**Q: How do I search through history?**  
A: Use `grep`, IDE search, or the provided `scripts/search_chat.sh` script.

**Q: Can I edit old chat history?**  
A: Only for corrections (typos, sensitive info removal). Note edits in git commit message.

**Q: What about chat history for experiments?**  
A: Create experiment-specific aggregated files in the experiment directory (e.g., `experiments/magvit-3d/CHAT_HISTORY_MAGVIT.md`).

---

## 4. Development Environment & Setup

### 4.1 EC2 Computation Rule (MANDATORY)

**ALL computation must be performed on the EC2 instance:**

- **MacBook**: File editing, documentation, git operations, SSH connections ONLY
- **EC2 Instance**: ALL Python execution, testing, package installation
- **NO EXCEPTIONS**: Never run Python scripts, tests, or install ML packages on MacBook

### 4.2 EC2 Connection

```bash
# Standard connection
ssh -i /Users/mike/keys/AutoGenKeyPair.pem ubuntu@34.196.155.11

# With port forwarding for Jupyter
ssh -i /Users/mike/keys/AutoGenKeyPair.pem -L 8888:localhost:8888 ubuntu@34.196.155.11

# Check EC2 status
ssh -i /Users/mike/keys/AutoGenKeyPair.pem ubuntu@34.196.155.11 'hostname && pwd'
```

### 4.3 Environment Verification

Always verify you are on EC2 before running code:

```bash
# Check hostname - should show EC2 instance
hostname
# Should output something like: ip-172-31-32-83

# Check current directory
pwd
# Should be: /home/ubuntu/mono_to_3d (or similar EC2 path)

# Verify Python packages
python -c "import numpy, cv2, matplotlib, torch; print('All packages available')"
```

### 4.4 Development Workflow

1. **MacBook**: Edit files, commit to git
2. **MacBook**: SSH to EC2: `ssh -i /Users/mike/keys/AutoGenKeyPair.pem ubuntu@34.196.155.11`
3. **EC2**: Pull latest changes: `git pull`
4. **EC2**: Activate venv: `source venv/bin/activate`
5. **EC2**: Run all computation, tests, and package installation
6. **EC2**: Commit results if needed
7. **MacBook**: Pull results for documentation

### 4.5 Package Installation

**NEVER install on MacBook:**
- pytorch / torch / torchvision
- numpy
- pytest / pytest-cov
- scikit-learn
- tensorflow
- opencv-python
- Any ML/scientific computing packages

**MacBook venv should ONLY contain:**
- Basic development tools (if absolutely necessary)
- No computational packages

**EC2 package installation:**
```bash
# On EC2
cd ~/mono_to_3d
source venv/bin/activate
pip install -r requirements.txt
```

### 4.6 Jupyter Lab Setup on EC2

To run Jupyter Lab interactively on the EC2 instance:

**1. Connect to EC2 and start Jupyter Lab:**
```bash
ssh -i /Users/mike/keys/AutoGenKeyPair.pem ubuntu@34.196.155.11 \
  "cd mono_to_3d && source venv/bin/activate && \
   pip install -U jupyterlab ipywidgets && \
   jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root"
```

Note: Jupyter Lab will automatically find an available port (8888, 8889, 8892, etc.)

**2. Set up port forwarding in a separate terminal:**
```bash
ssh -i /Users/mike/keys/AutoGenKeyPair.pem -L [PORT]:localhost:[PORT] -N ubuntu@34.196.155.11
```

Replace `[PORT]` with the actual port number Jupyter Lab is using (check the server output).

**3. Access Jupyter Lab:**

Open browser to: `http://localhost:[PORT]/lab?token=[TOKEN]`

The exact URL with token will be displayed in the Jupyter Lab startup output.

**Key Points:**
- Use `-N` flag for persistent port forwarding without executing commands
- Run port forwarding in background or separate terminal
- Jupyter Lab automatically finds available ports if 8888 is busy
- Token is required for secure access and changes each session

### 4.7 Emergency Cleanup

If packages are accidentally installed on MacBook:
```bash
# On MacBook - remove computational packages immediately
pip uninstall pytorch torch torchvision numpy pytest scikit-learn tensorflow opencv-python -y
```

### 4.8 Verification Checklist

Before running ANY Python code, verify:
- [ ] I am connected to EC2 instance
- [ ] Current directory is on EC2: `/home/ubuntu/mono_to_3d`
- [ ] Hostname shows EC2 instance name
- [ ] I have NOT installed ML packages on MacBook
- [ ] Virtual environment is activated

### 4.9 EC2 Shutdown and Restart Procedures

#### Shutdown Preparation

**Before shutting down EC2:**

1. **Check for active connections**
   ```bash
   # From MacBook
   ps aux | grep ssh
   ps aux | grep 8888  # Check Jupyter port forwarding
   ```

2. **Commit and push all work**
   ```bash
   # On EC2
   cd ~/mono_to_3d
   git status
   git add .
   git commit -m "Save before EC2 shutdown"
   git push origin master
   ```

3. **Verify local repository is up-to-date**
   ```bash
   # On MacBook
   cd ~/path/to/mono_to_3d
   git pull origin master
   ```

#### Safe Shutdown

**Stop EC2 instance through AWS Console:**
1. Go to AWS EC2 Console
2. Select your instance
3. Actions → Instance State → Stop
4. Confirm shutdown

**⚠️ IMPORTANT**: Instance IP address will likely change after restart.

#### Restart Procedure

**1. Start EC2 Instance**
- Start the instance in AWS Console
- **Note the new public IP address** (will be different from before)

**2. Test SSH Connection**
```bash
# Update IP address as needed
ssh -i /Users/mike/keys/AutoGenKeyPair.pem ubuntu@NEW_IP_ADDRESS
```

**3. Navigate to Project**
```bash
cd mono_to_3d
source venv/bin/activate
```

**4. Sync Latest Changes**
```bash
git status
git pull origin master  # if needed
```

**5. Verify Environment**
```bash
# Check Python packages are available
python -c "import numpy, cv2, matplotlib, torch; print('✅ All packages available')"

# Run quick test
python -m pytest tests/ -q
```

**6. Start Jupyter Lab (if needed)**
```bash
# On EC2
jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root
```

**7. Set Up Port Forwarding (from local machine)**
```bash
# In separate terminal on MacBook
ssh -i /Users/mike/keys/AutoGenKeyPair.pem -L 8888:localhost:8888 -N ubuntu@NEW_IP_ADDRESS
```

Then access: `http://localhost:8888/lab?token=[TOKEN]`

#### Critical Reminders After Restart

**IP Address Check**  
⚠️ EC2 instance will have a new public IP address after restart
- Check AWS Console for new IP
- Update all connection commands
- Update port forwarding commands

**Environment Activation**  
Always activate virtual environment:
```bash
source venv/bin/activate
```

**Git Synchronization**  
Verify project is up-to-date:
```bash
git status
git pull origin master
```

#### Emergency Recovery

**If Git Issues:**
```bash
git remote -v  # verify remote
git fetch origin
git reset --hard origin/master  # if needed (use with caution)
```

**If Environment Issues:**
```bash
# Recreate virtual environment if needed
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

**If Jupyter Issues:**
```bash
pip install -U jupyterlab ipywidgets
jupyter lab --generate-config  # if needed
```

#### EC2 Instance Details

**Current Configuration:**
- **Instance IP**: `34.196.155.11` (Note: Changes after restart)
- **SSH Key**: `/Users/mike/keys/AutoGenKeyPair.pem`
- **Project Directory**: `/home/ubuntu/mono_to_3d/`
- **Virtual Environment**: `/home/ubuntu/mono_to_3d/venv/`

**Standard Connection:**
```bash
ssh -i /Users/mike/keys/AutoGenKeyPair.pem ubuntu@34.196.155.11
```

**With Port Forwarding:**
```bash
ssh -i /Users/mike/keys/AutoGenKeyPair.pem -L 8888:localhost:8888 ubuntu@34.196.155.11
```

---

## 5. Technical Requirements

### 5.1 Performance Requirements

- **Frame Rate**: Target 30 FPS for live processing
- **Latency**: <100ms end-to-end processing delay
- **Memory Usage**: <2GB RAM for standard operation

### 5.2 Accuracy Targets

- **3D Position Error**: <5cm at 2m distance
- **Trajectory Smoothness**: Minimal jitter in object paths
- **Stereo Matching**: >95% correct correspondences

### 5.3 Dependencies

#### Core Dependencies
- Python 3.8+
- OpenCV >= 4.5.0
- NumPy >= 1.20.0
- Pandas >= 1.3.0
- Matplotlib >= 3.4.0
- SciPy >= 1.7.0
- Plotly >= 5.3.0
- PyYAML >= 6.0
- Scikit-learn >= 1.0.0

#### Testing Framework
- pytest >= 7.0.0
- pytest-cov >= 4.0.0
- pytest-mock >= 3.0.0
- unittest-xml-reporting >= 3.2.0

#### ML Framework (EC2 ONLY)
- PyTorch >= 2.0.0
- torchvision >= 0.15.0

### 5.4 Output File Naming Convention

**MANDATORY:** All output files in any `results/` directory must have a datetime prefix for chronological ordering.

#### Standard Format

```
YYYYMMDD_HHMM_descriptive_name.ext
```

**Alternative format with seconds** (preferred for high-frequency outputs):
```
YYYYMMDD_HHMMSS_descriptive_name.ext
```

Where:
- `YYYYMMDD` = Year, Month, Day (e.g., 20260120)
- `HHMM` = Hour, Minute (e.g., 1430) or `HHMMSS` with seconds (e.g., 143025)
- `descriptive_name` = Brief description using underscores
- `ext` = File extension (png, csv, json, npy, npz, md, etc.)

#### Examples

**Visualizations:**
- `20260120_1430_magvit_comprehensive_TDD_VALIDATED.png`
- `20260120_1430_smooth_trajectories_comparison.png`
- `20260120_1445_camera_calibration_results.png`

**Data outputs:**
- `20260120_1430_trajectory_data.csv`
- `20260120_1430_reconstruction_results.npz`
- `20260120_1430_test_metrics.json`

**Documentation:**
- `20260120_1430_EXPERIMENT_SUMMARY.md`
- `20260120_1430_TDD_COMPLIANCE_SUMMARY.md`

#### Results Directory Structure

Each major directory has its own `results/` subdirectory:

```
mono_to_3d/
├── experiments/
│   ├── magvit-3d-trajectories/
│   │   ├── results/              # Timestamped outputs here
│   │   │   ├── 20260120_1430_comprehensive_view.png
│   │   │   └── 20260120_1445_smooth_trajectories.png
│   │   └── artifacts/             # TDD evidence (no timestamp needed)
│   ├── track_persistence/
│   │   └── results/              # Timestamped outputs here
│   └── videogpt-2d-trajectories/
│       └── results/              # Timestamped outputs here
├── neural_radiance_fields/
│   └── results/                  # Timestamped outputs here
└── basic/
    └── output/                   # Timestamped outputs here
```

#### Benefits

- **Chronological ordering**: Files sort naturally by date/time when listed
- **No conflicts**: Timestamp ensures uniqueness across runs
- **Easy tracking**: Quickly identify when results were generated
- **Version control**: Can compare results from different time points
- **Reproducibility**: Clear temporal relationship between related outputs

#### Implementation

Python helper function:
```python
from datetime import datetime

def get_timestamped_filename(base_name: str, extension: str = "png") -> str:
    """Generate timestamped filename for results."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    return f"{timestamp}_{base_name}.{extension}"

# Usage
output_path = f"results/{get_timestamped_filename('trajectory_comparison', 'png')}"
# Results in: results/20260120_1430_trajectory_comparison.png
```

#### Exceptions

Files that should **NOT** have timestamps:
- TDD artifacts in `artifacts/` directory (use `tdd_red.txt`, `tdd_green.txt`, etc.)
- Configuration files (e.g., `config.yaml`)
- Source code (e.g., `test_*.py`, `*.py`)
- README files (e.g., `README.md`)
- Fixed reference files (e.g., `golden_output.npz`)

---

## 6. Testing Implementation Guide

### 6.1 Test Organization

```
mono_to_3d/
├── tests/
│   ├── __init__.py
│   ├── conftest.py                    # Shared fixtures
│   ├── test_invariants.py             # Cross-module invariant tests
│   ├── test_golden.py                 # Canonical golden tests
│   └── unit/
│       ├── test_camera.py
│       ├── test_triangulation.py
│       ├── test_tracking.py
│       └── test_visualization.py
├── pytest.ini                          # Pytest configuration
└── [source code]
```

### 6.2 Pytest Configuration

The `pytest.ini` file standardizes test execution:

```ini
[pytest]
addopts = -q --strict-markers --tb=short
testpaths = tests
markers =
    unit: Unit tests for individual functions/classes
    integration: Integration tests across modules
    golden: Golden/regression tests with canonical scenarios
    invariant: Invariant property tests
    deterministic: Tests requiring fixed seeds
```

### 6.3 Shared Fixtures

Create `tests/conftest.py` for shared test fixtures:

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
    from simple_3d_tracker import setup_stereo_cameras
    P1, P2 = setup_stereo_cameras(baseline=0.65, focal_length=800)
    return P1, P2

@pytest.fixture
def sample_3d_point():
    """Sample 3D point for testing."""
    return np.array([0.0, 0.0, 2.0])
```

### 6.4 Test Execution Commands

```bash
# Run all tests
pytest

# Run in quiet mode (default via pytest.ini)
pytest -q

# Run with verbose output
pytest -v

# Run specific test file
pytest tests/test_triangulation.py

# Run specific test method
pytest tests/test_triangulation.py::TestTriangulation::test_golden_case

# Run with coverage report
pytest --cov=. --cov-report=html --cov-report=term

# Run tests with specific marker
pytest -m golden
pytest -m invariant

# Run tests and generate XML report for CI/CD
pytest --junitxml=test_results.xml
```

### 6.5 Coverage Requirements

- **Critical Functions**: 100% coverage (triangulation, camera calibration)
- **Visualization Functions**: 80% coverage (plotting, display functions)
- **Utility Functions**: 90% coverage (helpers, data processing)
- **Error Handling**: 100% coverage (exception paths)
- **Integration Workflows**: 85% coverage (end-to-end pipelines)

### 6.6 Test File Template

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

## References

- **PEP 8 Style Guide**: https://pep8.org/
- **NumPy Testing**: https://numpy.org/doc/stable/reference/routines.testing.html
- **pytest Documentation**: https://docs.pytest.org/
- **OpenCV Documentation**: https://docs.opencv.org/4.x/
- **PyTorch Documentation**: https://pytorch.org/docs/stable/index.html
- **Pandas Documentation**: https://pandas.pydata.org/docs/

---

**This document is the single source of truth for all project requirements and standards.**
