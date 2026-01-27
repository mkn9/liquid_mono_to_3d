# Week 1 Foundation Plan - Parallel Branch Setup

**Date:** January 20, 2026  
**Experiment:** magvit_I3D_LLM_basic_trajectory  
**Goal:** Build shared foundation for 4 parallel branches

---

## Critical Requirement: TRUE VISION MODEL

**TDD Test Enforcing Image Input (MANDATORY):**
```python
def test_model_must_accept_image_tensors_not_coordinates():
    """Ensure model processes IMAGES, not coordinate shortcuts."""
    # Valid: Image tensor (B, T, C, H, W)
    valid_input = torch.randn(2, 16, 3, 64, 64)
    output = model(valid_input)
    assert output is not None
    
    # Invalid: Coordinate shortcut (B, T, 2)
    invalid_coords = torch.randn(2, 16, 2)
    with pytest.raises((AssertionError, ValueError, RuntimeError)):
        model(invalid_coords)
```

---

## Dataset Size Analysis

### Initial Concern: Is 600 Samples Enough?

**Conservative Estimate (Addressing User Concern):**
- **Classification:** 200-300 samples/class → 800-1200 total
- **Forecasting:** Need more for temporal dynamics → 1000-1500 total
- **With Augmentation:** Effective dataset = actual × 10-20

**Proposed Dataset Sizes:**

| Approach | Base Samples | With Augmentation | Rationale |
|----------|--------------|-------------------|-----------|
| **Minimum** | 800 (200/class) | ~8,000-16,000 | Conservative baseline |
| **Recommended** | 1,200 (300/class) | ~12,000-24,000 | Good for both tasks |
| **Optimal** | 2,000 (500/class) | ~20,000-40,000 | Best generalization |

**Recommendation: Start with 1,200 base samples**
- Generation time: ~30-40 minutes on EC2
- Enough for solid training
- Can scale to 2,000 if needed

**Augmentation Strategy:**
1. **Noise levels:** 5 variations (σ = 0.005, 0.01, 0.02, 0.03, 0.05)
2. **Camera angles:** 3 cameras × 2 slight rotations = 6 views
3. **Phase offsets:** 3 starting positions per trajectory
4. **Visual styles:** 2 rendering styles (dot, trail)
5. **Effective multiplier:** 5 × 6 × 3 × 2 = **180x augmentation**

**Final Dataset: 1,200 × 180 = 216,000 effective samples**

---

## Week 1 Tasks (TDD-Driven)

### Day 1: Trajectory Renderer (TDD)

**Test File:** `test_trajectory_renderer.py`

**Tests (RED → GREEN → REFACTOR):**
1. `test_renderer_outputs_image_tensor_not_coordinates()`
2. `test_rendered_frames_have_correct_shape()`
3. `test_different_trajectories_produce_different_images()`
4. `test_rendering_respects_camera_parameters()`
5. `test_trajectory_appears_in_correct_image_locations()`

**Implementation:** `trajectory_renderer.py`
```python
class TrajectoryRenderer:
    def render_video(
        self,
        trajectory_3d: np.ndarray,  # (T, 3)
        camera_params: dict,
        image_size: Tuple[int, int],
        style: str = 'dot'
    ) -> torch.Tensor:
        """Render trajectory as video frames.
        
        Returns:
            torch.Tensor: (T, 3, H, W) - RGB video
        """
```

**Output:** `artifacts/tdd_renderer_*.txt` (RED, GREEN, REFACTOR)

---

### Day 2: Dataset Generation (TDD)

**Test File:** `test_dataset_generator.py`

**Tests:**
1. `test_generates_correct_number_of_samples()`
2. `test_balanced_class_distribution()`
3. `test_all_samples_are_images_not_coordinates()`
4. `test_augmentation_creates_variations()`
5. `test_dataset_saves_and_loads_correctly()`

**Implementation:** `dataset_generator.py`
```python
def generate_dataset(
    num_samples: int = 1200,
    frames_per_video: int = 16,
    image_size: Tuple[int, int] = (64, 64),
    augmentation: bool = True,
    seed: int = 42
) -> Dict[str, torch.Tensor]:
    """Generate complete dataset of rendered trajectories.
    
    Returns:
        {
            'videos': (N, T, 3, H, W),  # IMAGES, not coordinates
            'labels': (N,),              # 0=linear, 1=circular, 2=helical, 3=parabolic
            'trajectory_3d': (N, T, 3),  # Ground truth for evaluation
            'equations': List[str],      # Symbolic equations
            'descriptions': List[str]    # Natural language
        }
    """
```

**Dataset Generation:**
- Base: 1,200 samples (300 per trajectory type)
- Augmented: ~216,000 effective samples
- Time estimate: 30-40 minutes on EC2
- Storage: ~2-4 GB

**Output:** `artifacts/tdd_dataset_*.txt`

---

### Day 3: Shared Utilities (TDD)

**Test File:** `test_shared_utils.py`

**Utilities to Build:**
1. `image_preprocessing.py` - Normalization, transforms
2. `evaluation_metrics.py` - Classification accuracy, forecasting MAE
3. `visualization_utils.py` - Plot trajectories, save figures
4. `llm_interface.py` - Abstract interface for GPT-4/Mistral/Phi-2

**All with TDD evidence in `artifacts/`**

---

### Day 4-5: Branch Setup & Parallel Infrastructure

**Task 1: Create 4 Git Branches**
```bash
git checkout -b magvit-I3D-LLM/i3d-magvit-gpt4
git checkout -b magvit-I3D-LLM/slowfast-magvit-gpt4
git checkout -b magvit-I3D-LLM/i3d-mistral-clip
git checkout -b magvit-I3D-LLM/slowfast-phi2-wizardmath
```

**Task 2: Parallel Execution Framework**

**File:** `parallel_trainer.py`
```python
class BranchWorker:
    """Worker for one branch - runs until complete."""
    
    def __init__(self, branch_name: str, config: dict):
        self.branch_name = branch_name
        self.config = config
        self.status_file = f"status/{branch_name}_status.json"
        self.results_dir = f"results/{branch_name}/"
        
    def run_until_complete(self):
        """Execute full pipeline: train → eval → LLM → save.
        
        Periodic updates:
        - Every 5 minutes: Update status JSON
        - Every 15 minutes: Save intermediate results
        - On completion: Save final results + artifacts
        """
        while not self.is_complete():
            self.train_epoch()
            self.update_status()
            if self.should_save_checkpoint():
                self.save_checkpoint()
                self.sync_to_macbook()
        
        self.finalize()
```

**Task 3: Status Monitoring**

**File:** `status_monitor.py`
```python
class StatusMonitor:
    """Monitor all 4 branches in real-time."""
    
    def watch_branches(self):
        """Display status of all 4 branches.
        
        Updates every 30 seconds on MacBook terminal.
        """
        while not all_branches_complete():
            for branch in branches:
                status = load_status(f"status/{branch}_status.json")
                print(f"{branch}: Epoch {status['epoch']}/{status['max_epochs']} "
                      f"- Acc: {status['accuracy']:.2%} - Loss: {status['loss']:.4f}")
            time.sleep(30)
```

**Task 4: Result Synchronization to MacBook**

**Auto-sync setup:**
```bash
# On EC2: Periodic rsync to MacBook (every 15 min)
*/15 * * * * rsync -avz ~/mono_to_3d/experiments/magvit_I3D_LLM_basic_trajectory/results/ \
    user@macbook:~/Dropbox/.../experiments/magvit_I3D_LLM_basic_trajectory/results/

# Or use git commits for results
git add results/
git commit -m "Branch X: Epoch Y results"
git push
```

---

## Parallel Execution Strategy

### Launch Command (Week 2+)

**On EC2:**
```bash
# Terminal 1: Branch 1
cd ~/mono_to_3d
source venv/bin/activate
git checkout magvit-I3D-LLM/i3d-magvit-gpt4
python train_branch.py --config configs/branch1.yaml > logs/branch1.log 2>&1 &

# Terminal 2: Branch 2
git checkout magvit-I3D-LLM/slowfast-magvit-gpt4
python train_branch.py --config configs/branch2.yaml > logs/branch2.log 2>&1 &

# Terminal 3: Branch 3
git checkout magvit-I3D-LLM/i3d-mistral-clip
python train_branch.py --config configs/branch3.yaml > logs/branch3.log 2>&1 &

# Terminal 4: Branch 4
git checkout magvit-I3D-LLM/slowfast-phi2-wizardmath
python train_branch.py --config configs/branch4.yaml > logs/branch4.log 2>&1 &

# Terminal 5: Monitor all branches
python status_monitor.py
```

**On MacBook:**
```bash
# Watch status (auto-updates every 30s)
watch -n 30 cat experiments/magvit_I3D_LLM_basic_trajectory/status/*.json

# Or run status dashboard
python experiments/magvit_I3D_LLM_basic_trajectory/dashboard.py
```

---

## Resource Allocation

**EC2 Instance Requirements:**
- **GPU:** 4x GPUs (1 per branch) OR time-slice on 1-2 GPUs
- **Memory:** 32-64 GB RAM
- **Storage:** 50-100 GB for datasets + models
- **Time:** 3-7 days continuous running

**GPU Allocation:**
```python
# In each branch's config
branch_1_config = {
    'device': 'cuda:0',  # GPU 0
    'batch_size': 16,
}

branch_2_config = {
    'device': 'cuda:1',  # GPU 1
    'batch_size': 16,
}

# Or time-slice on single GPU
branch_1_config = {'device': 'cuda:0'}
branch_2_config = {'device': 'cuda:0'}  # Will alternate
```

---

## Success Criteria (Per Branch)

**Minimum Acceptable Performance:**
- ✅ Classification: >85% accuracy
- ✅ Forecasting: <20% MAE (4 frames ahead)
- ✅ Equations: >80% syntactically correct
- ✅ NL descriptions: >90% coherent

**Excellent Performance:**
- 🎯 Classification: >95% accuracy
- 🎯 Forecasting: <10% MAE
- 🎯 Equations: >90% correct with correct parameters
- 🎯 NL descriptions: >95% high quality

---

## Deliverables (End of Week 1)

1. ✅ `trajectory_renderer.py` with TDD evidence
2. ✅ 1,200 base samples generated (300 per class)
3. ✅ Augmentation pipeline tested
4. ✅ 4 git branches created
5. ✅ Parallel execution framework ready
6. ✅ Status monitoring working on MacBook
7. ✅ All TDD evidence in `artifacts/`

**Status Files (Synced to MacBook):**
```
experiments/magvit_I3D_LLM_basic_trajectory/
├── status/
│   ├── branch1_i3d-magvit-gpt4_status.json
│   ├── branch2_slowfast-magvit-gpt4_status.json
│   ├── branch3_i3d-mistral-clip_status.json
│   └── branch4_slowfast-phi2-wizardmath_status.json
├── results/
│   ├── branch1/
│   │   ├── 20260127_1430_classification_metrics.json
│   │   ├── 20260127_1430_forecasting_curves.png
│   │   └── 20260127_1430_sample_equations.txt
│   ├── branch2/
│   ├── branch3/
│   └── branch4/
└── artifacts/
    ├── tdd_renderer_red.txt
    ├── tdd_renderer_green.txt
    ├── tdd_dataset_red.txt
    └── tdd_dataset_green.txt
```

---

## Week 2+ Timeline

**Week 2:** Train classification + forecasting heads (all branches in parallel)
**Week 3:** Integrate LLMs for equations + NL descriptions
**Week 4:** Final evaluation, select winner or ensemble

**Continuous execution until all 4 branches complete.**

---

**Ready to proceed with Day 1: Trajectory Renderer (TDD)?**

