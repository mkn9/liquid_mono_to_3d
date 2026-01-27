# Documentation Integrity Protocol

**Created:** January 18, 2026  
**Reason:** Response to integrity failure where aspirational documentation was presented as completed work  
**Status:** MANDATORY for all project documentation

---

## Purpose

This protocol exists because of a serious integrity failure where:
- Documentation claimed "50 samples generated" when only 3 existed
- Visualization files were referenced that did not exist
- "100% success rate" was claimed without verification
- Aspirational plans were presented as completed work

**This must never happen again.**

---

## Core Principle

> **VERIFY EVERY CLAIM BEFORE DOCUMENTING IT**

If you cannot verify a claim with concrete evidence, do not present it as fact.

---

## Mandatory Verification Steps

### 1. File Existence Claims

**BEFORE writing:** "File X exists" or "Visualization Y was generated"

**REQUIRED VERIFICATION:**
```bash
# Check file exists
ls path/to/file.png
# OR
find . -name "file.png"
# OR use read_file tool

# Show in documentation:
"Verification: ls output shows file.png exists [timestamp]"
```

**NEVER claim files exist without this verification.**

### 2. Data Count Claims

**BEFORE writing:** "N samples generated" or "Dataset contains X items"

**REQUIRED VERIFICATION:**
```python
import numpy as np
data = np.load('dataset.npz')
print(f"Actual count: {len(data['key'])}")
print(f"Actual shape: {data['key'].shape}")
```

**NEVER claim sample counts without loading and checking the data.**

### 3. Success/Completion Claims  

**BEFORE writing:** "100% success" or "All tests passed" or "Successfully completed"

**REQUIRED VERIFICATION:**
```bash
# Run and show actual output
python -m pytest -v
# OR
python script.py > output.log 2>&1
cat output.log
```

**NEVER claim success without showing actual results.**

### 4. Visualization/Plot Claims

**BEFORE writing:** "Figure X shows..." or "Plots demonstrate..."

**REQUIRED VERIFICATION:**
```bash
# List all visualization files
ls -lh output/*.png
# OR
find . -name "*.png" -type f
```

**NEVER reference visualizations without confirming they exist.**

### 5. Performance/Metric Claims

**BEFORE writing:** "Achieved X accuracy" or "Runtime of Y seconds"

**REQUIRED VERIFICATION:**
- Show actual measurement output
- Include timestamps
- Provide calculation details

**NEVER claim metrics without showing how they were measured.**

---

## Language Standards

### ✅ ACCEPTABLE Language (Verified Work)

When work HAS been completed and verified:
- "Verification confirms N samples exist" + [show evidence]
- "File listing shows output.png was created" + [show ls output]
- "Test execution demonstrates 36/36 tests passed" + [show test output]
- "Actual data shape is (3, 16, 3)" + [show data loading output]

### ✅ ACCEPTABLE Language (Unverified/Planned Work)

When work has NOT been completed or cannot be verified:
- "Code has been written to generate N samples"
- "Designed to produce visualizations"
- "Intended to create the following outputs"
- "**NOT YET EXECUTED**"
- "**PLAN:** Will generate..."
- "**DESIGN:** Intended to produce..."

### ❌ PROHIBITED Language (Integrity Violations)

**NEVER use these without verification:**
- "Successfully generated N samples" ← requires verification
- "Results show..." ← must show actual results
- "File X contains..." ← must verify file exists
- "100% success rate" ← must measure and prove
- "All tests passed" ← must show test output
- "Visualizations demonstrate..." ← must verify visualizations exist
- Past tense for unexecuted work ← must clarify not yet done

---

## The Three-State Model

ALL work falls into one of three states. Be explicit about which state:

### State 1: CODE WRITTEN
- Code files exist in repository
- Functions/classes are implemented
- Tests are written
- **Documentation:** "Code has been written to..."

### State 2: CODE EXECUTED
- Code has been run
- Output files were generated
- Logs/results are available
- **Documentation:** "Execution produced..." + [show evidence]

### State 3: RESULTS VERIFIED
- Output files confirmed to exist
- Data counts validated
- Metrics measured and documented
- **Documentation:** "Verification confirms..." + [show proof]

**NEVER conflate State 1 with State 2 or 3.**

---

## Verification Checklist

Before committing documentation, answer ALL these questions:

### File Claims
- [ ] Did I verify every claimed file exists? (Show ls/find output)
- [ ] Did I check file sizes/timestamps? (Show ls -lh output)
- [ ] Did I actually read file contents if referencing them?

### Data Claims  
- [ ] Did I load data files to verify counts? (Show shape output)
- [ ] Did I check actual array/dataframe dimensions?
- [ ] Did I examine sample data contents?

### Success Claims
- [ ] Did I run tests and capture output? (Show pytest output)
- [ ] Did I measure actual metrics? (Show calculations)
- [ ] Did I verify success criteria were met?

### Visualization Claims
- [ ] Did I verify image files exist? (Show ls *.png)
- [ ] Did I check image file sizes are non-zero?
- [ ] Did I view or inspect the images?

### Language Check
- [ ] Did I distinguish "code written" from "code executed"?
- [ ] Did I avoid past tense for unexecuted work?
- [ ] Did I label plans/designs clearly as such?
- [ ] Did I provide evidence for every factual claim?

---

## Red Flags - Stop and Verify

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

## Examples

### ❌ WRONG (Integrity Violation)

```markdown
## Results

Successfully generated 50 3D trajectory samples for cubes, cylinders, and cones.

Key visualizations:
- magvit_3d_trajectories.png - Shows 3D trajectory plots
- magvit_3d_cameras.png - Camera positions

All experiments completed with 100% success rate.
```

**Problem:** No verification. Claims files exist, claims sample count, claims success - all without evidence.

### ✅ CORRECT (Evidence-Based)

```markdown
## Results

**STATUS: Minimal execution performed, not full dataset**

Code has been written to generate 3D trajectories for cubes, cylinders, and cones.

**Actual execution verification:**
```bash
$ ls -lh neural_video_experiments/magvit/results/*.npz
magvit_3d_dataset.npz  5.5K
```

**Actual data count verification:**
```python
$ python3
>>> import numpy as np
>>> data = np.load('magvit_3d_dataset.npz')
>>> print(f"Actual samples: {len(data['labels'])}")
Actual samples: 3
>>> print(f"Shape: {data['trajectories_3d'].shape}")
Shape: (3, 16, 3)
```

**Finding: Only 3 samples were generated, not the planned 50.**

**Visualization check:**
```bash
$ find . -name "*magvit_3d*.png"
[no results - files do not exist]
```

**Finding: No visualization files were created.**

**Status: Code exists and was minimally tested (3 samples), but full execution (50 samples with visualizations) has NOT been completed.**
```

---

## Enforcement

### Self-Checking
- Review your documentation before committing
- Run verification commands for every claim
- Add verification output to documentation

### Peer Review
- Documentation should be reviewable against actual files/data
- Claims should be falsifiable
- Evidence should be provided

### Automated Checks
- Consider automated verification scripts
- Check file existence matches documentation
- Validate data counts against claims

### Consequences of Violations
- Document must be amended with integrity warnings
- Original false version preserved in git history
- Public acknowledgment of failure required
- Additional safeguards implemented

---

## Git History as Truth

**Remember:** Git preserves everything.

- False claims will be discoverable
- Corrections will be visible
- Integrity failures will be on record
- Verification creates audit trail

**Make verification easy:** Provide evidence in documentation so future readers can verify independently.

---

## Summary

**The Rule:** VERIFY BEFORE DOCUMENTING

**The Test:** Can you prove every claim with concrete evidence?

**The Standard:** If uncertain, CHECK. Never assume. Never guess. Never present plans as completions.

**The Commitment:** Every claim must be verifiable, every file must exist, every count must be accurate.

**This is non-negotiable.**

---

*This protocol was created in response to a specific integrity failure and must be followed for all future documentation.*

