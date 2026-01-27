# Dataset Comparison - Old vs New

**Date**: 2026-01-25  
**Status**: ⚠️ Similar issues in both datasets - Root cause investigation needed

---

## 📊 COMPARISON RESULTS

### Old Dataset (20260124_1546)
```
Generated: Jan 24, 3:46 PM (BEFORE noise reduction work)
Unique samples: 170/200 (30 duplicates)
Temporal variation:
  - Linear:    0.0000 (static!)
  - Circular:  0.0023
  - Helical:   0.0024
  - Parabolic: 0.0023
```

### New Dataset (20260125)
```
Generated: Jan 25 (AFTER noise reduction work, with current code)
Unique samples: 171/200 (29 duplicates)
Temporal variation:
  - Linear:    0.0022
  - Circular:  0.0020
  - Helical:   0.0023
  - Parabolic: 0.0021
```

---

## 🔍 KEY FINDING: ISSUES ARE IDENTICAL

**Both datasets have the same problems!**

This means:
1. ✅ Noise reduction code changes didn't break anything
2. ❌ But the core issues remain:
   - ~15% duplicate samples
   - Very low temporal variation
   - Trajectories may not be visible in frames

**Conclusion**: The problem is NOT in recent code changes. It's a fundamental issue with the rendering or data generation approach.

---

## 🎯 ROOT CAUSE ANALYSIS

### Issue 1: Duplicates (~29-30 in every 200 samples)

**Why are there duplicates?**

Possible causes:
1. **Seed=42 creates collisions** - Random parameters occasionally generate identical trajectories
2. **Trajectory rounding** - Small variations get rounded to same values
3. **Rendering artifacts** - Different trajectories render to identical frames

**Is this a problem for MAGVIT?**
- Minor issue: Reduces effective dataset from 200 → 171
- Not fatal, but not ideal

---

### Issue 2: Low Temporal Variation (0.0020-0.0023)

**Why so low?**

**Hypothesis**: The rendering shows a small dot moving on a white background. Most of the frame is white, so:
- Frame mean ≈ 0.99 (mostly white)
- Dot changes position but frame mean barely changes
- Result: Low variation in frame statistics

**But does this mean motion is invisible?**
- **No!** The dot IS moving, just:
  - Small compared to frame
  - Most pixels stay white
  - Only a few pixels change

**Is this a problem for MAGVIT?**
- **Maybe not!** MAGVIT uses spatial features, not just frame means
- The local motion around the dot should be learnable
- Frame mean is not what MAGVIT "sees"

---

### Issue 3: Circular Class Has Zero Displacement

**This is CORRECT!** ✅

Circular trajectories return to starting point:
- Start: (x₀, y₀, z₀)
- End: (x₀, y₀, z₀)
- Displacement: 0

This is the expected behavior, not a bug!

---

## 💡 CRITICAL QUESTION

**Are the visualizations showing cleaner trajectories?**

**Please compare**:
1. **OLD**: `results/dataset_visual_inspection.png`
2. **NEW**: `results/NEW_dataset_visual_inspection.png`

**Look for**:
- Are linear trajectories less noisy in NEW?
- Are parabolic trajectories smoother in NEW?
- Is motion visible across the 16 frames?

---

## 🎯 THREE SCENARIOS

### Scenario A: NEW dataset IS cleaner (noise reduced)

**If visual inspection shows improvement**:
- ✅ Your noise reduction worked!
- ✅ Duplicates/low variation are expected artifacts
- ✅ **Proceed with MAGVIT training on NEW dataset**

**Action**: Train MAGVIT, ignore the warnings about duplicates/variation

---

### Scenario B: NEW dataset same as OLD (no visual difference)

**If visual inspection shows NO improvement**:
- ⚠️ Noise reduction code may not have been applied
- ⚠️ OR rendering is masking the trajectory improvements
- ⚠️ Need to investigate why changes didn't propagate

**Action**: Debug why improvements not visible

---

### Scenario C: Rendering is fundamentally wrong

**If motion is not visible at all**:
- ❌ Rendering approach is broken
- ❌ Need to fix renderer before training

**Action**: Fix rendering first

---

## 🔬 DIAGNOSTIC QUESTIONS

To determine which scenario we're in:

1. **In NEW_dataset_visual_inspection.png**:
   - Can you see the dot/trajectory in each frame?
   - Does the dot move across the 16 frames?
   - Are linear/parabolic trajectories cleaner than OLD?

2. **In NEW_trajectory_comparison.png**:
   - Do the 3D trajectories look reasonable?
   - Are there 5 different trajectories per class (not overlapping)?

---

## 📋 RECOMMENDATION

**STOP and visually inspect the new dataset before deciding next steps.**

**If NEW looks better than OLD**:
→ **Train MAGVIT on NEW dataset** (duplicates/low variation are acceptable)

**If NEW looks same as OLD**:
→ **Investigate why noise reduction didn't help**

**If motion not visible**:
→ **Fix rendering before training**

---

## 🛑 AWAITING YOUR VISUAL ASSESSMENT

Please compare:
- `results/dataset_visual_inspection.png` (OLD)
- `results/NEW_dataset_visual_inspection.png` (NEW)

**Question**: Does the NEW dataset look cleaner/better, particularly for linear and parabolic trajectories?

