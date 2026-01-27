# Mono-to-3D Project Template
**Version:** 1.0  
**Created:** January 26, 2026  
**Purpose:** Clean template for continuing mono_to_3d development

---

## Overview

This template provides the **essential structure and code** for continuing mono_to_3d project development, **without historical results and deprecated code** that no longer form a basis for future work.

**What's Included:**
- ✅ Core infrastructure (TDD, git workflow, documentation standards)
- ✅ Active experiments (trajectory video understanding, VLM integration)
- ✅ Reusable utilities and test infrastructure
- ✅ Latest working code and architecture

**What's Excluded:**
- ❌ Historical session documents (20+ CHAT_HISTORY files)
- ❌ Deprecated notebooks (3d_tracker_7_backup.ipynb, etc.)
- ❌ Old results/outputs (frame_comparisons/, old .png files)
- ❌ Redundant status documents (PARALLEL_*, SESSION_*, MAGVIT_*)
- ❌ Archives and logs

---

## Directory Structure

```
mono_to_3d/
│
├── .git/                               # Git repository
│   └── hooks/
│       └── pre-push                   # TDD evidence validation
│
├── cursorrules                         # PRIMARY: AI assistant directives
├── requirements.md                     # SECONDARY: Detailed methodology
├── README.md                           # Project overview
├── requirements.txt                    # Python dependencies
├── pytest.ini                          # PyTest configuration
├── .gitignore                          # Git ignore patterns
│
├── scripts/                            # Automation scripts
│   ├── prove.sh                        # Full test suite + proof bundle
│   ├── tdd_capture.sh                  # TDD phase evidence capture
│   ├── sync_results.sh                 # Sync results from EC2 to MacBook
│   └── setup_environment.sh            # Environment setup
│
├── artifacts/                          # Evidence & proof bundles
│   ├── tdd_red.txt
│   ├── tdd_green.txt
│   ├── tdd_refactor.txt
│   ├── tdd_structural.txt
│   └── proof/
│       └── <git_sha>/
│
├── experiments/                        # Active experiments
│   │
│   ├── trajectory_video_understanding/  # Main trajectory work
│   │   ├── early_persistence_detection/
│   │   │   ├── src/
│   │   │   │   ├── models/             # Vision models
│   │   │   │   │   ├── magvit_model.py
│   │   │   │   │   └── transformer_classifier.py
│   │   │   │   ├── data/               # Data loading
│   │   │   │   │   ├── augmented_dataset.py
│   │   │   │   │   └── trajectory_loader.py
│   │   │   │   ├── training/           # Training utilities
│   │   │   │   │   └── train_persistence.py
│   │   │   │   └── evaluation/         # Evaluation tools
│   │   │   │       └── evaluate_model.py
│   │   │   ├── tests/
│   │   │   │   └── test_*.py
│   │   │   └── results/
│   │   │       └── YYYYMMDD_HHMM_*.{png,json,txt}
│   │   │
│   │   ├── vision_language_integration/  # VLM work (latest)
│   │   │   ├── llm_interface.py
│   │   │   ├── vision_language_bridge.py
│   │   │   ├── trajectory_qa.py
│   │   │   ├── demo_real_magvit.py
│   │   │   ├── tests/
│   │   │   │   └── test_*.py
│   │   │   └── demo_results/
│   │   │       └── YYYYMMDD_HHMM_*.json
│   │   │
│   │   ├── persistence_augmented_dataset/  # Real data
│   │   │   ├── output_samples/         # Real trajectory samples
│   │   │   ├── generate_transient_dataset.py
│   │   │   └── README.md
│   │   │
│   │   └── sequential_results_*/       # Trained models (keep latest)
│   │       └── magvit/
│   │           ├── final_model.pt      # 100% accuracy model
│   │           └── checkpoint_*.pt
│   │
│   └── magvit_I3D_LLM_basic_trajectory/  # Basic trajectory work
│       ├── src/
│       │   ├── trajectory_generator.py
│       │   └── noise_models.py
│       ├── tests/
│       │   └── test_*.py
│       └── results/
│           └── YYYYMMDD_HHMM_*.png
│
├── src/                                # Core source code
│   ├── __init__.py
│   ├── camera/                         # Camera system
│   │   ├── __init__.py
│   │   ├── camera_model.py
│   │   └── projection.py
│   ├── tracking/                       # 3D tracking
│   │   ├── __init__.py
│   │   ├── triangulation.py
│   │   └── trajectory.py
│   └── utils/                          # Shared utilities
│       ├── __init__.py
│       ├── visualization.py
│       └── data_io.py
│
├── tests/                              # Root-level tests
│   ├── __init__.py
│   ├── test_camera_system.py
│   ├── test_projection.py
│   └── test_triangulation.py
│
├── docs/                               # Documentation
│   ├── ARCHITECTURE_PLANNING_LNN.md    # Latest architecture planning
│   ├── VLM_STRATEGIC_ASSESSMENT.md     # VLM strategy
│   ├── REAL_VLM_INTEGRATION_SUCCESS.md # Latest success report
│   ├── COORDINATE_SYSTEM_DOCUMENTATION.md
│   └── API_REFERENCE.md
│
├── data/                               # Data directory structure
│   ├── raw/                            # Raw input data
│   ├── processed/                      # Processed data
│   └── README.md                       # Data documentation
│
├── notebooks/                          # Active notebooks only
│   └── YYYYMMDD_exploration_*.ipynb
│
├── CHAT_HISTORY/                       # Session documentation
│   └── YYYYMMDD_session_name.md        # Keep latest only
│
└── .cursorignore                       # Cursor ignore patterns
```

---

## Essential Files to Keep

### Core Infrastructure
```
✅ cursorrules                          # AI assistant directives
✅ requirements.md                      # Methodology & standards
✅ README.md                            # Project overview
✅ requirements.txt                     # Dependencies
✅ pytest.ini                           # Test configuration
✅ .gitignore                           # Git ignore patterns
```

### Scripts
```
✅ scripts/prove.sh                     # Test suite + proof bundle
✅ scripts/tdd_capture.sh               # TDD evidence capture
✅ scripts/sync_results.sh              # EC2 sync
✅ scripts/setup_environment.sh         # Environment setup
```

### Source Code
```
✅ src/camera/                          # Camera system code
✅ src/tracking/                        # 3D tracking code
✅ src/utils/                           # Shared utilities
✅ tests/                               # Root-level tests
```

### Active Experiments
```
✅ experiments/trajectory_video_understanding/
   ├── early_persistence_detection/    # MagVIT model (100% accuracy)
   ├── vision_language_integration/    # Latest VLM work
   ├── persistence_augmented_dataset/  # Real data
   └── sequential_results_*/           # Trained models (latest only)

✅ experiments/magvit_I3D_LLM_basic_trajectory/  # Basic trajectory work
```

### Documentation
```
✅ docs/ARCHITECTURE_PLANNING_LNN.md    # Latest planning (Jan 26, 2026)
✅ docs/VLM_STRATEGIC_ASSESSMENT.md     # VLM strategy
✅ docs/REAL_VLM_INTEGRATION_SUCCESS.md # Latest results
✅ docs/COORDINATE_SYSTEM_DOCUMENTATION.md
```

### Latest Session History
```
✅ CHAT_HISTORY/20260126_VLM_INTEGRATION.md  # Most recent session
✅ CHAT_HISTORY/20260125_PARALLEL_TRAINING.md  # Previous session
(Keep last 2-3 sessions, archive older ones)
```

---

## Files to Remove

### Deprecated Notebooks
```
❌ 3d_tracker_7_backup.ipynb
❌ 3d_tracker_7_clean.ipynb
❌ 3d_tracker_7_executed.ipynb
❌ 3d_tracker_7.ipynb
❌ 3d_tracker_8.ipynb
❌ 3d_tracker_9.ipynb
❌ 3d_tracker_cone.ipynb
❌ 3d_tracker_cylinder.ipynb
❌ 3d_tracker_interactive_FIXED_FINAL_COMPLETE_WORKING.ipynb
❌ 3d_tracker_visualization.ipynb
❌ 3d_visualization.ipynb
❌ test_plot.ipynb
❌ test_plot_bu.ipynb
(Reason: Historical exploration, not current workflow)
```

### Old Results/Outputs
```
❌ 3d_error_visualization.png
❌ 3d_visualization.png
❌ camera_only_sanity_check.png
❌ dnerf_actual_vs_predicted_comparison.png
❌ dnerf_sphere_trajectory_complete.png
❌ error_comparison.png
❌ horizontal_forward_3d_plot.png
❌ horizontal_forward.csv
❌ trajectory_summary.csv
❌ frame_comparisons/
❌ trajectory_comparison_output/
(Reason: Historical results, superseded by experiments/*/results/)
```

### Historical Session Documents
```
❌ CHAT_HISTORY_20260125.md
❌ CHAT_HISTORY_20260126_PARALLEL_TRAINING.md
❌ chat_history_complete.md
❌ CHAT_HISTORY_SESSION_JAN12_2026.md
❌ CHAT_HISTORY_SESSION_JAN13_2026_FUTURE_PREDICTION.md
❌ SESSION_COMPLETE_REAL_IMPLEMENTATION.md
❌ SESSION_HISTORY_JAN18_2026_EVENING.md
❌ SESSION_HISTORY_JAN18_2026.md
❌ SESSION_STATE_JAN13_2026.md
❌ SESSION_STATUS_JAN16_2026.md
❌ SESSION_STATUS_JAN18_2026.md
(Reason: Keep latest 2-3 sessions, archive rest)
```

### Redundant Status/Planning Documents
```
❌ ACTUAL_WORK_PLAN_TRACK_PERSISTENCE.md
❌ BRANCH_COMPARISON_OPTIONS1_VS_3.md
❌ BUG_FIX_BACKGROUND_CLUTTER.md
❌ CHATGPT_FOLLOWUP_ENHANCEMENTS.md
❌ CHATGPT_RECOMMENDATION_ASSESSMENT.md
❌ CLUTTER_ADDITION_RECOMMENDATIONS.md
❌ CONSOLIDATION_SUMMARY.txt
❌ DOCUMENTATION_CONSOLIDATION_ANALYSIS.md
❌ DOCUMENTATION_INTEGRITY_PROTOCOL.md (moved to requirements.md)
❌ EC2_ANALYSIS_COMPLETE_JAN16_2026.md
❌ EVIDENCE_BASED_TDD_IMPLEMENTATION.md (moved to requirements.md)
❌ GIT_TREE_PROCEDURES_CONFIRMATION.md
❌ GIT_TREE_PROCEDURES_RECOMMENDATIONS.md
❌ GIT_TREE_PROCEDURES_VERIFICATION.md
❌ IMPLEMENTATION_COMPLETE_SUMMARY.md
❌ IMPLEMENTATION_SUMMARY.txt
❌ INCREMENTAL_SAVE_REQUIREMENT.md (moved to requirements.md)
❌ MAGVIT_3D_ACTUAL_RESULTS_STATUS.md
❌ MAGVIT_3D_CUBE_CYLINDER_CONE_STATUS.md
❌ MAGVIT_3D_OPTION1_COMPLETE.md
❌ MAGVIT_3D_RECREATION_COMPLETE.md
❌ MAGVIT_TASK_PROGRESS_SUMMARY.md
❌ MAGVIT_TRAJECTORY_STATUS.md
❌ PARALLEL_EXECUTION_COMPLETE_JAN16_2026.md
❌ PARALLEL_EXECUTION_IN_PROGRESS.md
❌ PARALLEL_EXECUTION_STATUS.md
❌ PARALLEL_EXECUTION_SUMMARY_JAN16_2026.md
❌ PARALLEL_FEATURE_EXTRACTOR_PLAN.md
❌ PARALLEL_OPTIONS_ACG_COMPLETE_JAN16_2026.md
❌ PARALLEL_TASKS_SETUP_SUMMARY.md
❌ PARALLEL_TASKS_STATUS.md
❌ PARALLEL_TDD_IMPLEMENTATION_PLAN.md
❌ PARALLEL_TRAINING_COMPLETE_JAN16_2026.md
❌ PRODUCTION_DEPLOYMENT_SUMMARY.md
❌ QUICK_WINS_EXECUTION_SUMMARY.md
❌ REAL_INTEGRATION_PLAN.md
❌ REAL_TRACK_PERSISTENCE_IMPLEMENTATION.md
❌ SENSOR_IMPACT_ANALYSIS.md
❌ TDD_INTEGRATION_PROPOSAL.md
❌ TDD_OPTION1_INTEGRATION_SUMMARY.md
❌ TESTING_AND_DEBUGGING_SETUP.md
❌ TIMESTAMP_IMPLEMENTATION_SUMMARY.md
❌ TRACK_PERSISTENCE_PHASE1_SUMMARY.md
❌ TWO_FILE_STRATEGY.md
❌ UPDATE_REMAINING_SCRIPTS.md
❌ WORKER1_FIX_OPTIONS.md
(Reason: Historical, consolidated into requirements.md or superseded)
```

### Utility Scripts (Historical)
```
❌ 3d_error_visualization.py
❌ dnerf_real_3d_visualization.py
❌ chat_logger.py
❌ create_notebook.py
❌ create_visualization_notebook.py
❌ debug_data_lengths.py
❌ fix_final_syntax.py
❌ fix_plotting_function.py
❌ generate_sphere_trajectories.py
❌ launch_all_experiments.sh
❌ monitor_execution.sh
❌ monitor_experiments.py
❌ monitor_future_prediction.py
❌ monitor_parallel_execution.py
❌ run_parallel_future_prediction.py
❌ run_parallel_tasks.py
❌ setup_all_experiments.py
❌ simple_3d_tracker.py
❌ web_server.py
(Reason: One-off scripts, not part of core workflow)
```

### Old Test Files (Root Level)
```
❌ test_3d_tracker_comprehensive.py
❌ test_3d_tracker_correct_coordinate_system.py
❌ test_3d_tracker_y1_plane.py
❌ test_camera_system_comprehensive.py
❌ test_cone_tracking_comprehensive.py
❌ test_cylinder_tracking.py
❌ test_dnerf_integration.py
❌ test_final_validation.py
❌ test_output_utils_shared.py
❌ test_projection.py
❌ test_sensor_impact_analysis.py
❌ test_sensor_impact_comprehensive.py
❌ test_sphere_trajectory_generation.py
(Reason: Consolidate into tests/ directory or experiment-specific tests/)
```

### Archives and Logs
```
❌ archive/
❌ __pycache__/
❌ jupyter_output.log
❌ pipeline_execution.log
❌ sync_results.log
❌ training_sync.log
❌ training_sync.pid
❌ experiment_summary.json
❌ config.yaml (unless actively used)
(Reason: Historical, regenerated, or cached)
```

### Deprecated Experiment Directories
```
❌ basic/                               # Old basic examples
❌ D-NeRF/                              # Not actively developed
❌ neural_radiance_fields/              # Not actively developed
❌ openCV/                              # Basic examples
❌ openobj_nerf/                        # Not actively developed
❌ ov_nerf/                             # Not actively developed
❌ semantic_nerf/                       # Not actively developed
❌ vision_language_models/              # Superseded by experiments/*/vision_language_integration/
❌ integrated_3d_systems/               # Historical integration attempts
❌ magvit_options/                      # Old options exploration
❌ neural_video_experiments/            # Superseded by experiments/trajectory_video_understanding/
❌ contracts/                           # Old design contracts
❌ example_code/                        # Examples, not core
(Reason: Not part of current development path)
```

### Duplicate/Old Utilities
```
❌ output_utils_shared.py               # If consolidated elsewhere
❌ main_macbook.py                      # Old launcher (if not used)
❌ activate_mono_to_3d_env.sh           # If using venv/activate
(Reason: Redundant or superseded)
```

---

## Cleanup Script

```bash
#!/bin/bash
# cleanup_for_template.sh - Remove historical files for clean template

set -e

echo "🧹 Cleaning up mono_to_3d project for template..."

# Create backup first
BACKUP_DIR="../mono_to_3d_backup_$(date +%Y%m%d_%H%M%S)"
echo "📦 Creating backup: $BACKUP_DIR"
cp -r . "$BACKUP_DIR"

# Remove deprecated notebooks
echo "📓 Removing deprecated notebooks..."
rm -f 3d_tracker_*.ipynb
rm -f 3d_visualization.ipynb
rm -f test_plot*.ipynb
rm -f sensor_impact_analysis.ipynb
rm -f openobj_nerf_demo.ipynb
rm -f ov_nerf_demo.ipynb
rm -f semantic_nerf_demo.ipynb

# Remove old results/outputs
echo "📊 Removing old results..."
rm -f *.png
rm -f *.csv
rm -rf frame_comparisons/
rm -rf trajectory_comparison_output/

# Remove historical session documents (keep latest 2)
echo "📝 Cleaning up session documents..."
mkdir -p CHAT_HISTORY_ARCHIVE
mv CHAT_HISTORY_*.md CHAT_HISTORY_ARCHIVE/ 2>/dev/null || true
mv SESSION_*.md CHAT_HISTORY_ARCHIVE/ 2>/dev/null || true
# Move latest 2 back
ls -t CHAT_HISTORY_ARCHIVE/CHAT_HISTORY_*.md | head -2 | xargs -I {} mv {} .

# Remove redundant status documents
echo "📋 Removing redundant status documents..."
rm -f *_STATUS.md
rm -f *_SUMMARY.md
rm -f *_SUMMARY.txt
rm -f *_COMPLETE*.md
rm -f *_PLAN*.md
rm -f *_PROCEDURES*.md
rm -f *_ANALYSIS*.md
rm -f *_PROPOSAL*.md
rm -f *_ASSESSMENT*.md
rm -f *_RECOMMENDATIONS*.md
rm -f *_FOLLOWUP*.md
rm -f CHATGPT_*.md
rm -f GIT_TREE_*.md
rm -f BRANCH_*.md
rm -f BUG_FIX_*.md

# Keep these specific docs
git checkout docs/ARCHITECTURE_PLANNING_LNN.md 2>/dev/null || true
git checkout docs/VLM_STRATEGIC_ASSESSMENT.md 2>/dev/null || true
git checkout docs/REAL_VLM_INTEGRATION_SUCCESS.md 2>/dev/null || true
git checkout docs/COORDINATE_SYSTEM_DOCUMENTATION.md 2>/dev/null || true

# Remove utility scripts (historical)
echo "🔧 Removing one-off utility scripts..."
rm -f 3d_error_visualization.py
rm -f dnerf_real_3d_visualization.py
rm -f chat_logger.py
rm -f create_notebook.py
rm -f create_visualization_notebook.py
rm -f debug_data_lengths.py
rm -f fix_*.py
rm -f generate_sphere_trajectories.py
rm -f launch_all_experiments.sh
rm -f monitor_*.py
rm -f monitor_*.sh
rm -f run_parallel_*.py
rm -f setup_all_experiments.py
rm -f simple_3d_tracker.py
rm -f web_server.py

# Remove root-level test files (consolidate into tests/)
echo "🧪 Removing root-level test files..."
rm -f test_*.py

# Remove archives and logs
echo "📦 Removing archives and logs..."
rm -rf archive/
rm -rf __pycache__/
rm -f *.log
rm -f *.pid
rm -f experiment_summary.json

# Remove deprecated experiment directories
echo "🔬 Removing deprecated experiments..."
rm -rf basic/
rm -rf D-NeRF/
rm -rf neural_radiance_fields/
rm -rf openCV/
rm -rf openobj_nerf/
rm -rf ov_nerf/
rm -rf semantic_nerf/
rm -rf integrated_3d_systems/
rm -rf magvit_options/
rm -rf neural_video_experiments/
rm -rf contracts/
rm -rf example_code/

# Keep only latest sequential_results
echo "🎯 Keeping only latest trained models..."
cd experiments/trajectory_video_understanding/
ls -t sequential_results_* | tail -n +2 | xargs rm -rf 2>/dev/null || true
cd ../..

# Clean up data directory (keep structure, remove data)
echo "💾 Cleaning data directory..."
find data/ -type f ! -name "README.md" -delete 2>/dev/null || true

# Clean Python caches
echo "🐍 Cleaning Python caches..."
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find . -type f -name "*.pyc" -delete 2>/dev/null || true

echo "✅ Cleanup complete!"
echo "📦 Backup saved to: $BACKUP_DIR"
echo ""
echo "Next steps:"
echo "1. Review changes: git status"
echo "2. Test: bash scripts/prove.sh"
echo "3. Commit: git add . && git commit -m 'Clean up for template'"
```

---

## Setup Instructions

### 1. Clone/Copy Template

```bash
# Option A: Clone from GitHub (once template is published)
git clone <template-repo-url> my-mono-to-3d-project
cd my-mono-to-3d-project

# Option B: Copy from existing project (after cleanup)
cp -r /path/to/cleaned/mono_to_3d /path/to/new/project
cd /path/to/new/project
rm -rf .git
git init
```

### 2. Verify Structure

```bash
# Check essential files exist
ls cursorrules requirements.md README.md requirements.txt
ls scripts/prove.sh scripts/tdd_capture.sh
ls -R experiments/trajectory_video_understanding/
```

### 3. Set Up Environment

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Set up git hooks
chmod +x scripts/*.sh
chmod +x .git/hooks/pre-push
```

### 4. Verify Tests

```bash
# Run test suite
bash scripts/prove.sh

# Should see:
# ✅ All tests passed!
# ✅ Proof bundle created: artifacts/proof/<git_sha>/
```

### 5. Configure EC2 (if needed)

```bash
# Edit EC2 connection details in scripts/
vim scripts/sync_results.sh

# Update SSH key path and host
SSH_KEY="/Users/yourusername/keys/YourKeyPair.pem"
EC2_HOST="ubuntu@your.ec2.ip.address"
```

---

## Key Differences from Generic Template

### 1. Domain-Specific Code
- Camera system (src/camera/)
- 3D tracking (src/tracking/)
- Trajectory analysis (experiments/)

### 2. Experiment Structure
- `experiments/` directory for ML experiments
- Each experiment has src/, tests/, results/
- Trained models stored in experiments/*/results/

### 3. EC2 Workflow
- All computation on EC2
- MacBook for editing only
- Sync scripts for results

### 4. Data Management
- Real trajectory data in experiments/*/persistence_augmented_dataset/
- Trained models in experiments/*/sequential_results_*/
- Results use timestamp prefix (YYYYMMDD_HHMM_*)

---

## Current State (January 26, 2026)

### ✅ Working Components

**Vision Model (MagVIT):**
- Location: `experiments/trajectory_video_understanding/sequential_results_*/magvit/`
- Model: `final_model.pt` (100% validation accuracy)
- Task: Trajectory persistence classification (Persistent vs Transient)

**VLM Integration:**
- Location: `experiments/trajectory_video_understanding/vision_language_integration/`
- Components: LLM interfaces, visual grounding (planned), Q&A system
- Models: TinyLlama (local), GPT-4 (API)

**Dataset:**
- Location: `experiments/trajectory_video_understanding/persistence_augmented_dataset/`
- Format: PyTorch .pt files + JSON metadata
- Samples: Thousands of augmented trajectories

### 🚧 In Progress

**Visual Grounding:**
- Plan: Connect MagVIT embeddings (512-dim) → LLM
- Status: Architecture planning complete
- Next: Implement simple adapter (2-3 days)

**Liquid Neural Networks:**
- Plan: Explore LNN for 3D dynamics and trajectory prediction
- Status: Strategic assessment complete
- Next: Decide on priority (product vs research)

### 📋 Deferred

**3D Integration:**
- Status: Deferred until visual grounding complete
- Reason: Visual grounding higher impact for immediate VLM quality

---

## Quick Start Development

### Scenario 1: Continue VLM Work

```bash
cd experiments/trajectory_video_understanding/vision_language_integration/

# Write tests first
vim tests/test_visual_adapter.py

# Capture RED phase
bash ../../../scripts/tdd_capture.sh red

# Implement
vim visual_adapter.py

# Capture GREEN phase
bash ../../../scripts/tdd_capture.sh green

# Create proof bundle
cd ../../..
bash scripts/prove.sh

# Commit
git add .
git commit -m "Add visual adapter with TDD evidence"
```

### Scenario 2: New Experiment

```bash
mkdir -p experiments/my_new_experiment/{src,tests,results}

# Create README
echo "# My New Experiment" > experiments/my_new_experiment/README.md

# Follow TDD workflow (same as Scenario 1)
```

### Scenario 3: Add Core Functionality

```bash
# Write tests first (root level)
vim tests/test_new_feature.py

# Follow TDD workflow
bash scripts/tdd_capture.sh red
# implement in src/
bash scripts/tdd_capture.sh green
bash scripts/prove.sh
git commit
```

---

## Maintenance

### Regular Cleanup

**Every few weeks:**
1. Review `CHAT_HISTORY/` - keep latest 2-3 sessions
2. Review experiment results/ - archive old results
3. Review notebooks/ - remove exploratory notebooks
4. Run proof bundle: `bash scripts/prove.sh`

### Documentation Updates

**After major milestones:**
1. Update `docs/` with latest architecture/decisions
2. Update `CHAT_HISTORY/` with session summary
3. Update this template if structure changes

### Dependency Updates

**Monthly:**
```bash
pip list --outdated
# Carefully update requirements.txt
pip install -r requirements.txt
bash scripts/prove.sh  # Ensure tests still pass
```

---

## Migration Guide

### From Full Project to Template

If you have the full mono_to_3d project and want to create a clean template:

```bash
# 1. Create backup
cp -r mono_to_3d mono_to_3d_backup

# 2. Run cleanup script (see "Cleanup Script" section above)
cd mono_to_3d
bash cleanup_for_template.sh

# 3. Verify
bash scripts/prove.sh
git status

# 4. Commit clean version
git add .
git commit -m "Clean template from full project"
```

### From Template to New Project

If you have the template and want to start a new mono_to_3d variant:

```bash
# 1. Copy template
cp -r mono_to_3d_template my_new_project

# 2. Customize
cd my_new_project
# Edit cursorrules, requirements.md, README.md

# 3. Initialize git
rm -rf .git
git init
git add .
git commit -m "Initial commit from mono_to_3d template"

# 4. Start developing
# Follow TDD workflow, create experiments/, etc.
```

---

## FAQ

**Q: Why keep experiments/ but remove basic/, example_code/?**
A: experiments/ contains active research with real models/data. basic/ and example_code/ are learning examples, not production code.

**Q: Why remove historical session documents?**
A: Keep latest 2-3 for context. Archive older ones. They're valuable for review but clutter the workspace.

**Q: Why keep docs/ARCHITECTURE_PLANNING_LNN.md but remove PARALLEL_EXECUTION_STATUS.md?**
A: Architecture planning is forward-looking (for future work). Status documents are backward-looking (historical).

**Q: Can I keep some of the "removed" files?**
A: Yes! This template is a recommendation. Adjust based on your needs. Just maintain clarity about what's active vs historical.

**Q: What if I need a file that was removed?**
A: All files are in git history. Use `git log --all --full-history -- path/to/file` to find when it was removed, then `git checkout <commit> -- path/to/file` to restore.

---

## Resources

- **Generic Template:** `TEMPLATE_GENERIC_CURSOR_AI_PROJECT.md`
- **Full Requirements:** `requirements.md`
- **Latest Architecture:** `docs/ARCHITECTURE_PLANNING_LNN.md`
- **Latest VLM Work:** `experiments/trajectory_video_understanding/vision_language_integration/`

---

## License

[Same as parent project]

---

## Credits

**Based on:** mono_to_3d project (January 2026)  
**Cleaned:** Removed historical artifacts while preserving essential structure  
**Purpose:** Clean starting point for continued development

---

**Template Version:** 1.0  
**Last Updated:** January 26, 2026  
**Maintained by:** [Your name/organization]

