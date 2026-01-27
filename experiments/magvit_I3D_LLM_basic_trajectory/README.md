# MAGVIT + I3D + LLM Basic Trajectory Experiment

**Created:** January 20, 2026  
**Status:** Week 1 Foundation - Day 1 (RED Phase)  
**Goal:** Vision-Language Model for trajectory classification, forecasting, and symbolic description

---

## ⚠️ CRITICAL: TRUE VISION MODEL REQUIREMENT

**This experiment MUST process IMAGES, not coordinate shortcuts.**

Test enforcing this: `test_renderer_outputs_image_tensor_not_coordinates()`

Any model that takes coordinate arrays instead of image tensors is NOT a vision model and violates the core requirement.

---

## Experiment Overview

### Goals
1. **Classify** 4 trajectory types (linear, circular, helical, parabolic)
2. **Forecast** next 4-8 frames given first 8-12 frames
3. **Generate symbolic equations** (e.g., y = ax² + bx + c)
4. **Generate natural language descriptions** (e.g., "The object follows a parabolic path...")

### Architecture (4 Parallel Branches)

| Branch | Video Model | Enhancement | LLM | Status |
|--------|-------------|-------------|-----|--------|
| 1 | I3D | MAGVIT | GPT-4/5 | Planning |
| 2 | SlowFast | MAGVIT | GPT-4/5 | Planning |
| 3 | I3D | CLIP | Mistral | Planning |
| 4 | SlowFast | None | Phi-2/WizardMath | Planning |

---

## Week 1: Foundation (Current Phase)

### Day 1: Trajectory Renderer (TDD)

**Status:** RED Phase - Tests written, implementation pending

**Tests:** `test_trajectory_renderer.py` (16 tests)
- ✅ Tests enforce image output (not coordinates)
- ✅ Tests validate projection correctness
- ✅ Tests check rendering styles
- ✅ Tests handle edge cases

**Next Steps:**
1. Copy to EC2
2. Run TDD workflow: `bash scripts/tdd_capture.sh`
3. Verify RED phase (tests fail - expected!)
4. Implement `trajectory_renderer.py`
5. Verify GREEN phase (tests pass)
6. Refactor and verify REFACTOR phase

---

## Directory Structure

```
experiments/magvit_I3D_LLM_basic_trajectory/
├── README.md (this file)
├── WEEK1_FOUNDATION_PLAN.md
├── BRANCH_SPECIFICATIONS.md
├── results/
│   ├── branch1/
│   ├── branch2/
│   ├── branch3/
│   └── branch4/
├── artifacts/
│   ├── tdd_renderer_red.txt (to be generated)
│   ├── tdd_renderer_green.txt
│   └── tdd_renderer_refactor.txt
├── test_trajectory_renderer.py (DONE - RED phase)
└── trajectory_renderer.py (TODO - GREEN phase)
```

---

## Running on EC2 (MANDATORY)

**⚠️ ALL COMPUTATION MUST BE ON EC2 ⚠️**

### Setup on EC2

```bash
# Connect to EC2
ssh -i /Users/mike/keys/AutoGenKeyPair.pem ubuntu@34.196.155.11

# Navigate to project
cd ~/mono_to_3d

# Activate environment
source venv/bin/activate

# Verify location
hostname  # Should show EC2 instance
pwd       # Should be /home/ubuntu/mono_to_3d
```

### Pull Latest Code

```bash
git pull origin classification/magvit-trajectories
```

### Run TDD Workflow

```bash
# Navigate to experiment
cd experiments/magvit_I3D_LLM_basic_trajectory

# Run complete TDD cycle (RED → GREEN → REFACTOR)
bash ../../scripts/tdd_capture.sh

# This will:
# 1. Run tests (should FAIL - RED phase)
# 2. Save output to artifacts/tdd_red.txt
# 3. Wait for implementation
# 4. Run tests again (should PASS - GREEN phase)
# 5. Save output to artifacts/tdd_green.txt
# 6. Run tests final time (REFACTOR phase)
# 7. Save output to artifacts/tdd_refactor.txt
```

### Commit Evidence

```bash
git add artifacts/
git commit -m "Add TDD evidence for trajectory renderer (RED-GREEN-REFACTOR)"
git push
```

---

## TDD Evidence Requirements

**MANDATORY per requirements.md Section 3.3:**

All TDD phases must have captured evidence:
- ✅ `artifacts/tdd_renderer_red.txt` - Shows tests FAIL before implementation
- ✅ `artifacts/tdd_renderer_green.txt` - Shows tests PASS after implementation
- ✅ `artifacts/tdd_renderer_refactor.txt` - Shows tests still PASS after refactoring

**Never claim TDD was followed without these files.**

---

## Dataset Requirements

### Size (addressing concerns from planning)
- **Base samples:** 1,200 (300 per trajectory type)
- **With augmentation:** ~216,000 effective samples
- **Rationale:** Conservative estimate to ensure good generalization

### Augmentation Strategy
- Noise levels: 5 variations
- Camera angles: 6 views (3 cameras × 2 rotations)
- Phase offsets: 3 starting positions
- Visual styles: 2 rendering modes
- **Multiplier:** 5 × 6 × 3 × 2 = 180x

### Generation Time
- ~30-40 minutes on EC2
- Can scale to 2,000 if needed

---

## Parallel Branch Execution

### Week 2+ (After Foundation Complete)

**Launch all 4 branches in parallel:**

```bash
# EC2 Terminal 1
git checkout magvit-I3D-LLM/i3d-magvit-gpt4
python train_branch.py --config configs/branch1.yaml &

# EC2 Terminal 2
git checkout magvit-I3D-LLM/slowfast-magvit-gpt4
python train_branch.py --config configs/branch2.yaml &

# EC2 Terminal 3
git checkout magvit-I3D-LLM/i3d-mistral-clip
python train_branch.py --config configs/branch3.yaml &

# EC2 Terminal 4
git checkout magvit-I3D-LLM/slowfast-phi2-wizardmath
python train_branch.py --config configs/branch4.yaml &

# MacBook: Monitor status
python dashboard.py
```

**Status updates every 15 minutes to MacBook via git commits.**

---

## Success Criteria

### Minimum Acceptable
- Classification: >85% accuracy
- Forecasting: <20% MAE (4 frames ahead)
- Equations: >80% syntactically correct
- NL descriptions: >90% coherent

### Excellent Performance
- Classification: >95% accuracy
- Forecasting: <10% MAE
- Equations: >90% correct with parameters
- NL descriptions: >95% high quality

---

## Current Status

**Phase:** Week 1, Day 1 - RED Phase  
**Tests Written:** 16 tests in `test_trajectory_renderer.py`  
**Implementation:** Not started (following TDD)  
**TDD Evidence:** Not captured yet

**Next Action:** Run tests on EC2 to capture RED phase evidence.

---

## Key Contacts & Resources

- **Requirements:** See `../../requirements.md` Section 3.3 (TDD)
- **Cursorrules:** See `../../cursorrules` (TDD mandatory)
- **EC2 Connection:** `ssh -i /Users/mike/keys/AutoGenKeyPair.pem ubuntu@34.196.155.11`
- **Planning Docs:** `WEEK1_FOUNDATION_PLAN.md`, `BRANCH_SPECIFICATIONS.md`

---

**Last Updated:** January 20, 2026, 22:45 PM EST

