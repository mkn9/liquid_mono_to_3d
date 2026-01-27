# Parallel Development Workflow with Git Branches

## Purpose

This document defines the **mandatory workflow** for parallel development using git branches. Following this prevents:
- ❌ Fake component implementations
- ❌ Identical code across supposedly different branches
- ❌ Claims that can't be verified
- ❌ Shortcuts that violate requirements

## Prerequisites

1. **Component Specification File**: `.component_spec.yaml`
   - Defines what each claimed component must contain
   - See `.component_spec.yaml.example` for template

2. **Branch Specification File**: `.branch_specifications.json`
   - Maps branch names to required components
   - Created at project start

3. **Enforcement Scripts**: `scripts/`
   - `enforce_parallel_branches.sh` - Verifies branch independence
   - `verify_component_claims.py` - Verifies components exist
   - Installed as git hooks

## Workflow Steps

### Phase 1: Planning & Specification

#### 1.1 Create Component Specifications

```bash
# Copy example and customize
cp .component_spec.yaml.example .component_spec.yaml

# Edit to define YOUR project's component requirements
vim .component_spec.yaml
```

Example spec for MAGVIT:
```yaml
MAGVIT:
  type: model
  required_imports:
    - magvit
    - magvit2
  required_classes:
    - MAGVIT_VQ_VAE
    - VectorQuantizer
  forbidden_patterns:
    - "# fake magvit"
    - "class.*SimplifiedMAGVIT"
  min_lines_of_code: 100
```

#### 1.2 Create Branch Specifications

```bash
# Create branch specification
cat > .branch_specifications.json << EOF
{
  "branch1/i3d-magvit-gpt4": {
    "components": ["I3D", "MAGVIT", "GPT-4"],
    "description": "I3D video encoder with MAGVIT compression and GPT-4 LLM"
  },
  "branch2/slowfast-magvit-gpt4": {
    "components": ["SlowFast", "MAGVIT", "GPT-4"],
    "description": "SlowFast dual-pathway with MAGVIT and GPT-4"
  }
}
EOF
```

#### 1.3 Document Required Branches

```bash
# List all required branches
cat > .required_branches << EOF
branch1/i3d-magvit-gpt4
branch2/slowfast-magvit-gpt4
branch3/i3d-clip-mistral
branch4/slowfast-phi2
EOF
```

#### 1.4 Install Git Hooks

```bash
# Install pre-commit hook
cp scripts/pre-commit.hook .git/hooks/pre-commit
chmod +x .git/hooks/pre-commit

# Install pre-push hook (if not already present)
chmod +x scripts/pre-push.hook
ln -sf ../../scripts/pre-push.hook .git/hooks/pre-push
```

### Phase 2: Branch Creation & Isolation

#### 2.1 Create Base Branch (if needed)

```bash
# Create and checkout base branch
git checkout -b parallel-dev/base main

# Set up shared utilities/infrastructure
mkdir -p shared/
# ... add shared code ...

git add -A
git commit -m "Base: Shared utilities for parallel development"
git push origin parallel-dev/base
```

#### 2.2 Create Each Branch from Base

```bash
# For EACH branch, create separately:

# Branch 1
git checkout parallel-dev/base
git checkout -b branch1/i3d-magvit-gpt4
mkdir -p branch1/{models,tests,configs,results}
touch branch1/models/.gitkeep
git add branch1/
git commit -m "Branch 1: Initialize structure for I3D+MAGVIT+GPT4"
git push origin branch1/i3d-magvit-gpt4

# Branch 2
git checkout parallel-dev/base
git checkout -b branch2/slowfast-magvit-gpt4
mkdir -p branch2/{models,tests,configs,results}
touch branch2/models/.gitkeep
git add branch2/
git commit -m "Branch 2: Initialize structure for SlowFast+MAGVIT+GPT4"
git push origin branch2/slowfast-magvit-gpt4

# Repeat for branches 3 & 4...
```

**CRITICAL:** Always return to base before creating next branch!

### Phase 3: Parallel Implementation

#### 3.1 Work on Each Branch Independently

```bash
# Work on Branch 1
git checkout branch1/i3d-magvit-gpt4

# Implement ONLY Branch 1 components
vim branch1/models/i3d_model.py
vim branch1/models/magvit_encoder.py
vim branch1/models/gpt4_interface.py

# Test implementation
python -m pytest branch1/tests/

# Commit
git add branch1/
git commit -m "Branch 1: Implement I3D backbone with pretrained weights"
git push origin branch1/i3d-magvit-gpt4
```

#### 3.2 Switch to Next Branch

```bash
# Switch to Branch 2 (completely separate work)
git checkout branch2/slowfast-magvit-gpt4

# Implement ONLY Branch 2 components (different from Branch 1!)
vim branch2/models/slowfast_model.py
vim branch2/models/magvit_encoder.py  # Can be similar but in branch2/
vim branch2/models/gpt4_interface.py

# Test implementation
python -m pytest branch2/tests/

# Commit
git add branch2/
git commit -m "Branch 2: Implement SlowFast dual-pathway architecture"
git push origin branch2/slowfast-magvit-gpt4
```

#### 3.3 Rotate Through All Branches

```bash
# Systematic rotation:
# Day 1: Branch 1 (morning), Branch 2 (afternoon)
# Day 2: Branch 3 (morning), Branch 4 (afternoon)
# Day 3: Branch 1 (morning), Branch 2 (afternoon)
# ... continue until all complete
```

### Phase 4: Parallel Execution (on EC2)

#### 4.1 Set Up Tmux Sessions

```bash
# SSH to EC2
ssh -i ~/keys/key.pem ubuntu@ec2-instance

# Create tmux session for each branch
tmux new-session -d -s branch1
tmux new-session -d -s branch2
tmux new-session -d -s branch3
tmux new-session -d -s branch4

# Attach to first branch
tmux attach -t branch1
```

#### 4.2 Run Each Branch in Its Session

```bash
# Inside tmux session for branch1:
cd ~/project
git checkout branch1/i3d-magvit-gpt4
source venv/bin/activate
cd branch1
python train.py --config configs/branch1.yaml

# Detach: Ctrl+B, then D

# Attach to next session:
tmux attach -t branch2

# Inside tmux session for branch2:
git checkout branch2/slowfast-magvit-gpt4
cd branch2
python train.py --config configs/branch2.yaml

# ... repeat for all branches
```

#### 4.3 Monitor All Sessions

```bash
# List all sessions
tmux ls

# Monitor all in parallel (use tmux-sessions plugin or script)
watch -n 10 'tmux ls; echo "==="; tail -5 branch*/logs/training.log'

# Or create monitoring script:
cat > monitor_all.sh << 'EOF'
#!/bin/bash
while true; do
  clear
  echo "=== Branch Status ==="
  for branch in branch1 branch2 branch3 branch4; do
    echo "$branch:"
    tmux capture-pane -t $branch -p | tail -3
    echo
  done
  sleep 10
done
EOF
chmod +x monitor_all.sh
./monitor_all.sh
```

### Phase 5: Verification & Validation

#### 5.1 Run Enforcement Checks

```bash
# Check branch independence
bash scripts/enforce_parallel_branches.sh

# Check component claims
python scripts/verify_component_claims.py \
  --spec .component_spec.yaml \
  --branch-spec .branch_specifications.json \
  --search-path branch1 \
  --search-path branch2 \
  --search-path branch3 \
  --search-path branch4
```

#### 5.2 Review Generated Reports

```bash
# Branch enforcement report
cat BRANCH_ENFORCEMENT_REPORT.md

# Check that branches have:
# ✓ Diverged (not identical)
# ✓ Unique implementations
# ✓ Required components present
# ✓ Results generated
```

#### 5.3 Manual Code Review

For each branch:
```bash
git checkout branch1/i3d-magvit-gpt4
git log --oneline | head -10  # Review commit history
git diff parallel-dev/base..HEAD --stat  # See what changed from base
```

Verify:
- ✓ Commits are unique to this branch
- ✓ Code is branch-specific (not copy-paste)
- ✓ Components match branch name
- ✓ Tests exist and pass

### Phase 6: Results & Comparison

#### 6.1 Collect Results from Each Branch

```bash
# Ensure results are committed on each branch
for branch in branch1/i3d-magvit-gpt4 branch2/slowfast-magvit-gpt4 \
              branch3/i3d-clip-mistral branch4/slowfast-phi2; do
  git checkout $branch
  git add branch*/results/
  git commit -m "Results: Training complete" || true
  git push origin $branch
done
```

#### 6.2 Generate Cross-Branch Comparison

```bash
# Create comparison script
python scripts/compare_all_branches.py \
  --branches .required_branches \
  --output CROSS_BRANCH_COMPARISON.md
```

## Enforcement Mechanisms

### Automatic Checks (Git Hooks)

**Pre-commit Hook:**
- ✓ Checks current branch is in `.required_branches`
- ✓ Verifies no forbidden patterns in new code
- ✓ Ensures branch-specific directory structure

**Pre-push Hook:**
- ✓ Runs component verification
- ✓ Checks TDD evidence exists
- ✓ Verifies tests pass

### Manual Checks (Run Regularly)

```bash
# Daily verification
make verify-parallel-dev

# This runs:
# 1. scripts/enforce_parallel_branches.sh
# 2. scripts/verify_component_claims.py
# 3. pytest across all branches
```

## Red Flags - Signs of Fake/Shortcut Implementation

### 🚩 Red Flag Checklist

During code review, watch for:

1. **Naming Mismatch**
   - ❌ Branch named "magvit" but no `import magvit` anywhere
   - ❌ Class named `SimplifiedMAGVIT` or `FakeMAGVIT`

2. **Minimal Implementation**
   - ❌ "Component" file is < 50 lines
   - ❌ Only wrapper functions, no actual model code

3. **Template/Placeholder Code**
   - ❌ Comments like `# TODO: implement MAGVIT`
   - ❌ Comments like `# Simplified version`
   - ❌ Docstring: "Placeholder for..."

4. **Copy-Paste Across Branches**
   - ❌ Same file content in branch1/ and branch2/
   - ❌ Only difference is directory name

5. **No External Dependencies**
   - ❌ Claims GPT-4 but no `openai` import
   - ❌ Claims CLIP but no `clip` or `transformers`

6. **Missing Core Functionality**
   - ❌ MAGVIT without `encode()` and `decode()`
   - ❌ LLM without API call or model loading

## Checklist Before Claiming "Complete"

### ✅ Completion Criteria

Branch can only be marked complete when:

- [ ] Branch exists and has diverged from base
- [ ] All required components from spec are present
- [ ] Component verification script passes
- [ ] TDD tests exist and pass (RED-GREEN-REFACTOR evidence)
- [ ] Training completed and results committed
- [ ] Results include metrics, visualizations, model checkpoints
- [ ] No forbidden patterns found in code
- [ ] Manual code review completed
- [ ] Cross-branch comparison shows architectural differences

## Example: Correct Workflow

```bash
# 1. Setup
cp .component_spec.yaml.example .component_spec.yaml
vim .component_spec.yaml  # Customize for project
cat > .required_branches << EOF
branch1/i3d-magvit-gpt4
branch2/slowfast-magvit-gpt4
EOF

# 2. Create branches
git checkout -b branch1/i3d-magvit-gpt4
mkdir -p branch1/{models,tests}
git commit -m "Branch 1: Initialize"
git push origin branch1/i3d-magvit-gpt4

git checkout main
git checkout -b branch2/slowfast-magvit-gpt4
mkdir -p branch2/{models,tests}
git commit -m "Branch 2: Initialize"
git push origin branch2/slowfast-magvit-gpt4

# 3. Implement Branch 1
git checkout branch1/i3d-magvit-gpt4
# ... write tests first (TDD) ...
# ... implement I3D, MAGVIT, GPT-4 ...
git add branch1/
git commit -m "Branch 1: Full implementation with tests"
git push

# 4. Implement Branch 2 (separately!)
git checkout branch2/slowfast-magvit-gpt4
# ... write tests first (TDD) ...
# ... implement SlowFast, MAGVIT, GPT-4 ...
git add branch2/
git commit -m "Branch 2: Full implementation with tests"
git push

# 5. Verify
bash scripts/enforce_parallel_branches.sh
python scripts/verify_component_claims.py

# 6. Train (parallel on EC2)
# ... use tmux as shown above ...

# 7. Results
# ... commit results on each branch ...
# ... generate comparison report ...
```

## Summary

**Golden Rules:**
1. ✅ **Always specify components** before starting
2. ✅ **Create branches separately** from base
3. ✅ **Work on one branch at a time** (rotate systematically)
4. ✅ **Use real components** matching names/claims
5. ✅ **Verify claims** with automated scripts
6. ✅ **Execute truly in parallel** (tmux on EC2)
7. ✅ **Commit results per branch**
8. ✅ **Review before claiming complete**

**Never:**
- ❌ Create "simplified" versions when real component requested
- ❌ Copy-paste across branches
- ❌ Name something after a component it doesn't use
- ❌ Skip TDD for any component
- ❌ Claim completion without verification

---

**This workflow is mandatory for maintaining trust and quality.**

