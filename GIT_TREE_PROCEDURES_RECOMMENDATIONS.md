# Git Tree Procedures Recommendations
**Date:** December 15, 2025

---

## Overview

This document provides recommendations for organizing git tree procedures in `main_macbook.py` and `config.yaml` to manage parallel development workflows.

---

## Recommended Structure

### **1. `config.yaml` - Configuration File**

**Purpose:** Centralized configuration for all git operations, tasks, and execution parameters.

**Key Sections:**
- **Project metadata** - EC2 connection, paths
- **Git branch management** - Branch naming, operations
- **Task definitions** - All tasks with their branches, scripts, dependencies
- **Execution configuration** - Where to run (EC2/local), parallel limits
- **Output management** - File naming, transfer settings
- **Workflow procedures** - Branch creation, switching, merging, commit policies

**Benefits:**
- ✅ Single source of truth for all configuration
- ✅ Easy to modify without code changes
- ✅ Version controlled
- ✅ Human-readable

---

### **2. `main_macbook.py` - Orchestration Script**

**Purpose:** Main script that reads `config.yaml` and executes git operations and tasks.

**Key Classes:**
- **`GitTreeManager`** - Branch creation, switching, status
- **`TaskExecutor`** - Run tasks on EC2 or locally
- **`ResultTransfer`** - Transfer results from EC2 to MacBook
- **`TestRunner`** - Run tests before execution
- **`MainOrchestrator`** - Coordinates all operations

**Command-Line Interface:**
```bash
# Create all branches
python main_macbook.py --create-branches

# List branches
python main_macbook.py --list-branches

# Run all tasks
python main_macbook.py --run-tasks

# Run specific tasks
python main_macbook.py --run-tasks option1_direct_extraction option7_feature_fusion

# Transfer results
python main_macbook.py --transfer-results
```

---

## Design Principles

### **1. Separation of Concerns**

- **Config (`config.yaml`)**: What to do, where to do it
- **Code (`main_macbook.py`)**: How to do it
- **Benefits**: Easy to modify workflow without code changes

### **2. Declarative Configuration**

- Define tasks, branches, and procedures in YAML
- Script reads config and executes
- **Benefits**: Clear, maintainable, version-controlled

### **3. Modular Classes**

- Each class handles one responsibility
- Easy to test and extend
- **Benefits**: Maintainable, testable code

### **4. Error Handling**

- Graceful failures
- Continue on non-critical errors
- Clear error messages
- **Benefits**: Robust execution

---

## Configuration Structure Details

### **Project Metadata**
```yaml
project:
  name: "mono_to_3d"
  root_dir: "."
  ec2_host: "ubuntu@34.196.155.11"
  ec2_key: "/Users/mike/keys/AutoGenKeyPair.pem"
  ec2_path: "~/mono_to_3d"
```

**Why:** Centralized connection info, easy to update

### **Git Branch Management**
```yaml
git:
  default_branch: "main"
  branch_prefix: "magvit"
  naming:
    pattern: "{prefix}/{task_name}"
```

**Why:** Consistent branch naming, easy to create/manage

### **Task Definitions**
```yaml
tasks:
  option1_direct_extraction:
    branch: "magvit-option1-direct-extraction"
    script: "magvit_options/option1_direct_extraction.py"
    dependencies: []
    parallel: true
    priority: 1
```

**Why:** All task info in one place, dependencies clear, execution order defined

### **Execution Configuration**
```yaml
execution:
  location: "ec2"  # or "local"
  ec2:
    max_parallel: 5
    gpu_required: true
```

**Why:** Easy to switch between EC2 and local, control parallelism

---

## Workflow Procedures

### **Branch Creation**
- Defined in `config.yaml` under `workflow.create_branches`
- Executed by `GitTreeManager.create_branch()`
- **Benefits**: Consistent branch creation, no manual git commands

### **Task Execution**
- Defined in `config.yaml` under `tasks`
- Executed by `TaskExecutor.run_task()`
- **Benefits**: Automatic branch switching, script execution, result handling

### **Result Transfer**
- Defined in `config.yaml` under `output`
- Executed by `ResultTransfer.transfer_results()`
- **Benefits**: Automatic result syncing, no manual scp commands

---

## Usage Examples

### **Example 1: Setup New Project**
```bash
# 1. Create config.yaml with task definitions
# 2. Create all branches
python main_macbook.py --create-branches

# 3. Verify branches created
python main_macbook.py --list-branches
```

### **Example 2: Run Parallel Tasks**
```bash
# Run all parallel tasks
python main_macbook.py --run-tasks

# Script will:
# - Run tests first
# - Checkout appropriate branch for each task
# - Execute task on EC2
# - Transfer results to MacBook
```

### **Example 3: Run Specific Tasks**
```bash
# Run only Option 1 and Option 7
python main_macbook.py --run-tasks option1_direct_extraction option7_feature_fusion
```

### **Example 4: Manual Result Transfer**
```bash
# Transfer all results from EC2
python main_macbook.py --transfer-results
```

---

## Benefits of This Approach

### **1. Centralized Management**
- All git operations in one place
- No scattered git commands
- Easy to see what's happening

### **2. Reproducible Workflows**
- Config file defines exact procedures
- Version controlled
- Easy to share and document

### **3. Error Prevention**
- Validates branch existence before checkout
- Checks dependencies before execution
- Prevents common git mistakes

### **4. Automation**
- No manual branch switching
- Automatic result transfer
- Test execution before tasks

### **5. Flexibility**
- Easy to add new tasks (just add to config)
- Easy to change execution location
- Easy to modify workflows

---

## Integration with Existing Workflows

### **Compatible With:**
- ✅ Existing `setup_magvit_options.sh` (can be replaced or enhanced)
- ✅ Existing `setup_all_experiments.py` (can use same config)
- ✅ Existing test suites (configured in `config.yaml`)
- ✅ Existing output file naming (configured in `config.yaml`)

### **Enhancements:**
- 🔄 Unified interface for all git operations
- 🔄 Centralized configuration
- 🔄 Better error handling
- 🔄 Automatic result transfer

---

## Migration Path

### **Phase 1: Create Files**
1. Create `config.yaml` with current task definitions
2. Create `main_macbook.py` with basic functionality
3. Test with one task

### **Phase 2: Replace Scripts**
1. Replace `setup_magvit_options.sh` with `main_macbook.py --create-branches`
2. Replace manual git commands with `main_macbook.py --run-tasks`
3. Replace manual scp with `main_macbook.py --transfer-results`

### **Phase 3: Enhance**
1. Add more workflow procedures
2. Add monitoring and progress tracking
3. Add logging and reporting

---

## Best Practices

### **1. Keep Config Simple**
- Don't over-complicate YAML structure
- Use clear, descriptive names
- Group related settings

### **2. Validate Config**
- Check required fields exist
- Validate paths and connections
- Provide clear error messages

### **3. Handle Errors Gracefully**
- Don't fail on non-critical errors
- Continue execution when possible
- Log all errors for debugging

### **4. Document Changes**
- Update config.yaml comments
- Document new tasks in config
- Keep README updated

### **5. Version Control**
- Commit config.yaml changes
- Tag releases with config versions
- Document breaking changes

---

## Recommended Next Steps

1. **Create `config.yaml`** with current task definitions
2. **Create `main_macbook.py`** with basic functionality
3. **Test** with one task to verify workflow
4. **Expand** to all tasks
5. **Document** usage in README

---

## Conclusion

**Recommended Structure:**
- ✅ `config.yaml` - All configuration (tasks, branches, execution)
- ✅ `main_macbook.py` - All git operations and task execution
- ✅ Modular classes for each responsibility
- ✅ Command-line interface for easy use

**Benefits:**
- Centralized management
- Reproducible workflows
- Error prevention
- Automation
- Flexibility

**This approach provides a clean, maintainable way to manage git tree procedures and parallel task execution.**

---

**Last Updated:** December 15, 2025





