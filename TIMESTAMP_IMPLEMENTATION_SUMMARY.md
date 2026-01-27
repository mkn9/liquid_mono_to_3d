# Timestamp Implementation Summary

## Overview

Implemented mandatory timestamped output filenames across the project per **requirements.md Section 5.4**.

**Date**: 2026-01-20  
**TDD Compliance**: ✅ Full RED-GREEN-REFACTOR cycle completed

---

## Changes Made

### 1. Documentation Updates

#### `requirements.md` Section 5.4
- ✅ Enhanced with comprehensive timestamp naming convention
- ✅ Format: `YYYYMMDD_HHMM_descriptive_name.ext`
- ✅ Documented results/ directory structure
- ✅ Provided Python helper function
- ✅ Listed exceptions (artifacts/, config files)

#### `cursorrules`
- ✅ Added "OUTPUT FILE NAMING REQUIREMENT" section
- ✅ Quick reference format and examples
- ✅ Python helper snippet included

### 2. Shared Utility Module (TDD-Validated)

#### `output_utils_shared.py`
**Location**: Project root  
**Purpose**: Centralized timestamped filename generation for all experiments

**Functions:**
```python
get_timestamped_filename(base_name: str, extension: str) -> str
    # Returns: "20260120_1430_descriptive_name.ext"

get_results_path(base_dir: Path, filename: str) -> Path
    # Creates results/ subdirectory and returns full path with timestamp

save_figure(fig, results_dir: Path, filename: str, dpi: int = 150, **kwargs) -> Path
    # Save matplotlib figure with timestamp

save_data(data, results_dir: Path, filename: str, **kwargs) -> Path
    # Save data (CSV, JSON, NPZ, NPY) with timestamp
```

**TDD Evidence:**
- ✅ RED Phase: 7/7 tests FAILED (module not found) → `artifacts/tdd_output_utils_red.txt`
- ✅ GREEN Phase: 7/7 tests PASSED → `artifacts/tdd_output_utils_green.txt`
- ✅ REFACTOR Phase: 7/7 tests PASSED → `artifacts/tdd_output_utils_refactor.txt`

**Test Coverage:**
- ✅ Invariant tests: Format validation, directory creation
- ✅ Golden tests: Exact format with mocked datetime
- ✅ Integration tests: Multiple file types (PNG, CSV, JSON, NPZ)

### 3. Updated Scripts

#### `experiments/magvit-3d-trajectories/generate_dataset.py`
**Changes:**
- ✅ Imported `output_utils_shared.save_figure`
- ✅ Replaced all `plt.savefig()` calls with `save_figure()`
- ✅ Files now saved to `results/` with timestamps

**Example output:**
```
results/20260120_1430_magvit_3d_trajectory_types.png
results/20260120_1430_magvit_3d_trajectories_by_shape.png
results/20260120_1430_magvit_3d_errors_2d.png
results/20260120_1430_magvit_3d_cameras.png
```

#### `generate_sphere_trajectories.py`
**Changes:**
- ✅ Imported `output_utils_shared.save_figure` and `save_data`
- ✅ Updated `save_trajectory_data()` to use `save_data()`
- ✅ Updated `plot_3d_trajectory()` to use `save_figure()`
- ✅ All CSV and PNG files now timestamped

**Example output:**
```
output/results/20260120_1430_horizontal_forward.csv
output/results/20260120_1430_horizontal_forward_3d_plot.png
output/results/20260120_1430_trajectory_summary.csv
```

---

## Directory Structure

All major directories now have `results/` subdirectories for timestamped outputs:

```
mono_to_3d/
├── output_utils_shared.py          # Shared utility (TDD-validated)
├── test_output_utils_shared.py     # Comprehensive test suite
├── artifacts/                       # TDD evidence
│   ├── tdd_output_utils_red.txt
│   ├── tdd_output_utils_green.txt
│   └── tdd_output_utils_refactor.txt
├── experiments/
│   ├── magvit-3d-trajectories/
│   │   ├── output_utils_shared.py   # Copy for local import
│   │   ├── generate_dataset.py      # UPDATED
│   │   └── results/                 # Timestamped outputs
│   ├── track_persistence/
│   │   └── results/
│   └── videogpt-2d-trajectories/
│       └── results/
├── neural_radiance_fields/
│   └── results/
└── basic/
    ├── output_utils.py              # Original utility (still valid)
    └── output/
        └── results/                  # Timestamped outputs
```

---

## Benefits

✅ **Chronological Ordering**: Files sort naturally by date/time when listed  
✅ **No Conflicts**: Timestamp ensures uniqueness across runs  
✅ **Easy Tracking**: Quickly identify when results were generated  
✅ **Version Control**: Compare results from different time points  
✅ **Reproducibility**: Clear temporal relationship between related outputs

---

## Usage Examples

### Example 1: Save Figure in Experiment

```python
from output_utils_shared import save_figure
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
ax.plot([1, 2, 3], [1, 2, 3])

# Saves to: experiments/my_exp/results/20260120_1430_plot.png
saved_path = save_figure(fig, "experiments/my_exp", "plot.png")
```

### Example 2: Save Data (CSV, JSON, NPZ)

```python
from output_utils_shared import save_data
import pandas as pd

df = pd.DataFrame({'x': [1, 2, 3], 'y': [4, 5, 6]})

# Saves to: experiments/my_exp/results/20260120_1430_data.csv
saved_path = save_data(df, "experiments/my_exp", "data.csv", index=False)
```

### Example 3: Get Timestamped Filename

```python
from output_utils_shared import get_timestamped_filename

filename = get_timestamped_filename("trajectory_comparison", "png")
# Returns: "20260120_1430_trajectory_comparison.png"
```

---

## Exceptions

Files that should **NOT** have timestamps:
- ✅ TDD artifacts in `artifacts/` directory (`tdd_red.txt`, etc.)
- ✅ Configuration files (`config.yaml`)
- ✅ Source code (`test_*.py`, `*.py`)
- ✅ README files (`README.md`)
- ✅ Fixed reference files (`golden_output.npz`)

---

## Next Steps for Future Scripts

When creating new visualization or data generation scripts:

1. **Import the shared utility:**
   ```python
   from output_utils_shared import save_figure, save_data
   ```

2. **Use `save_figure()` instead of `plt.savefig()`:**
   ```python
   saved_path = save_figure(fig, results_dir, "my_plot.png")
   ```

3. **Use `save_data()` for CSV/JSON/NPZ:**
   ```python
   saved_path = save_data(df, results_dir, "data.csv")
   ```

4. **Verify timestamp format:**
   - Check that output files match: `YYYYMMDD_HHMM_*.ext`
   - Files should be in `results/` subdirectory

---

## TDD Evidence Summary

### Phase 1: RED
```
7 failed in 0.12s
All tests failed with ModuleNotFoundError (expected)
```

### Phase 2: GREEN
```
7 passed in 0.65s
All tests passed after implementation
```

### Phase 3: REFACTOR
```
7 passed in 0.64s
Tests still pass after code improvements
```

**Evidence Location**: `artifacts/tdd_output_utils_*.txt`

---

## Compliance Checklist

- [x] Requirements.md Section 5.4 updated with comprehensive guidelines
- [x] cursorrules updated with timestamp requirement
- [x] Shared utility module created (`output_utils_shared.py`)
- [x] Full TDD cycle completed (RED → GREEN → REFACTOR)
- [x] All 7 tests passing
- [x] TDD artifacts captured and saved
- [x] `generate_dataset.py` updated to use timestamps
- [x] `generate_sphere_trajectories.py` updated to use timestamps
- [x] Local and EC2 files synchronized
- [x] Documentation created (this file)

---

## Conclusion

The project now has a **robust, TDD-validated** timestamp naming system that ensures:
1. All results files are chronologically ordered
2. No filename conflicts
3. Easy identification of when results were generated
4. Consistent naming across all experiments

**All changes comply with requirements.md Section 3.3 (TDD) and Section 5.4 (Output File Naming).**

