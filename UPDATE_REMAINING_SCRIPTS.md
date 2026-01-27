# Remaining Scripts to Update

## Scripts That Need Timestamp Updates

The following scripts currently save files without timestamps and should be updated when next modified:

### High Priority (Active Experiments)

1. **`experiments/magvit-3d-trajectories/visualize_verified_dataset.py`**
   - Lines ~125-130: Multiple `plt.savefig()` calls
   - Action: Replace with `save_figure(fig, output_dir, filename)`

2. **`experiments/magvit-3d-trajectories/visualize_camera_views.py`**
   - Contains `plt.savefig()` calls
   - Action: Import and use `output_utils_shared.save_figure()`

3. **`experiments/magvit-3d-trajectories/visualize_shapes.py`**
   - Contains `plt.savefig()` calls  
   - Action: Import and use `output_utils_shared.save_figure()`

4. **`experiments/magvit-3d-trajectories/visualize_trajectory_types.py`**
   - Contains `plt.savefig()` calls
   - Action: Import and use `output_utils_shared.save_figure()`

### Medium Priority (Neural Video Experiments)

5. **`neural_video_experiments/videogpt/code/videogpt_trajectory_generator.py`**
   - Contains file saving logic
   - Action: Update to use `output_utils_shared`

6. **`neural_video_experiments/magvit/code/magvit_trajectory_generator.py`**
   - Contains file saving logic
   - Action: Update to use `output_utils_shared`

7. **`neural_video_experiments/collective/code/run_all_experiments.py`**
   - Contains file saving logic
   - Action: Update to use `output_utils_shared`

### Lower Priority (Legacy/Archive)

8. **D-NeRF Scripts**
   - `D-NeRF/dnerf_prediction_demo.py`
   - `D-NeRF/dnerf_data_augmentation.py`
   - `D-NeRF/dnerf_integration.py`
   - Action: Update when next used

9. **Root-level Utility Scripts**
   - `dnerf_real_3d_visualization.py`
   - `3d_error_visualization.py`
   - `simple_3d_tracker.py`
   - Action: Update opportunistically

### Already Compliant

✅ `basic/output_utils.py` - Already has timestamp helpers  
✅ `output_utils_shared.py` - New TDD-validated shared utility  
✅ `experiments/magvit-3d-trajectories/generate_dataset.py` - UPDATED  
✅ `generate_sphere_trajectories.py` - UPDATED

---

## Update Template

For each script, follow this pattern:

```python
# 1. Add import at top
from output_utils_shared import save_figure, save_data

# 2. Replace plt.savefig() with save_figure()
# OLD:
plt.savefig(output_dir / 'plot.png', dpi=150, bbox_inches='tight')

# NEW:
save_figure(fig, output_dir.parent, 'plot.png', dpi=150)

# 3. Replace df.to_csv() with save_data()
# OLD:
df.to_csv(output_dir / 'data.csv', index=False)

# NEW:
save_data(df, output_dir.parent, 'data.csv', index=False)
```

---

## TDD Requirement

**IMPORTANT**: When updating these scripts, follow TDD per requirements.md Section 3.3:

1. Write tests FIRST in `test_*.py` file (RED phase)
2. Run `bash scripts/tdd_capture.sh` to capture RED evidence
3. Update the script code (GREEN phase)
4. Verify tests pass and capture GREEN evidence
5. Refactor if needed and capture REFACTOR evidence

**Do NOT skip TDD workflow even for "simple" changes!**

---

## Verification

After updating each script, verify:

- [ ] Import statement added
- [ ] All `plt.savefig()` replaced with `save_figure()`
- [ ] All `.to_csv()` replaced with `save_data()`
- [ ] Output files have `YYYYMMDD_HHMM` prefix
- [ ] Files saved to `results/` subdirectory
- [ ] TDD evidence captured in `artifacts/`
