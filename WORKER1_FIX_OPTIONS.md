# Worker 1 (Clutter Integration) - Detailed Fix Options

**Branch:** `clutter-transient-objects`  
**Issue:** Missing `magvit_options.task1_trajectory_generator.TrajectoryGenerator`  
**File:** `experiments/clutter_transient_objects/task_clutter_integration.py`  

---

## Problem Analysis

### What the Code Needs
The task requires a `TrajectoryGenerator` class with:
1. **Constructor:** `TrajectoryGenerator(num_points=100)`
2. **Method:** `generator.generate(class_name)` 
3. **Supported classes:** `['straight', 'left_turn', 'right_turn', 'u_turn', 'loop']`
4. **Output:** NumPy array of 3D trajectory points `(N, 3)`

### Current Usage Pattern
```python
# Line 105-106
generator = TrajectoryGenerator(num_points=100)
trajectory = generator.generate('straight')  # Returns (N, 3) array

# Line 199-206
generator = TrajectoryGenerator(num_points=100)
classes = ['straight', 'left_turn', 'right_turn', 'u_turn', 'loop']
for class_name in classes:
    trajectory = generator.generate(class_name)
```

---

## Option 1: Create Simple Trajectory Generator ⭐ RECOMMENDED

**Complexity:** Low  
**Time:** 15-20 minutes  
**Risk:** Minimal  
**Best For:** Quick unblocking and testing clutter functionality

### Implementation
Create a new file: `magvit_options/task1_trajectory_generator.py`

```python
#!/usr/bin/env python3
"""
Simple Trajectory Generator for Clutter Integration Testing
"""
import numpy as np
from typing import Literal

TrajectoryType = Literal['straight', 'left_turn', 'right_turn', 'u_turn', 'loop']

class TrajectoryGenerator:
    """Simple 3D trajectory generator for testing clutter integration."""
    
    def __init__(self, num_points: int = 100):
        """
        Initialize trajectory generator.
        
        Args:
            num_points: Number of points in generated trajectory
        """
        self.num_points = num_points
        
    def generate(self, trajectory_type: TrajectoryType) -> np.ndarray:
        """
        Generate 3D trajectory of specified type.
        
        Args:
            trajectory_type: Type of trajectory to generate
            
        Returns:
            Array of shape (num_points, 3) with (x, y, z) coordinates
        """
        t = np.linspace(0, 1, self.num_points)
        
        if trajectory_type == 'straight':
            # Simple straight line
            x = t
            y = np.ones_like(t) * 0.5
            z = np.ones_like(t) * 0.5
            
        elif trajectory_type == 'left_turn':
            # 90-degree left turn
            angle = t * np.pi / 2  # 0 to 90 degrees
            radius = 0.3
            x = 0.5 + radius * np.cos(angle + np.pi)
            y = 0.5 + radius * np.sin(angle + np.pi)
            z = np.ones_like(t) * 0.5
            
        elif trajectory_type == 'right_turn':
            # 90-degree right turn
            angle = t * np.pi / 2  # 0 to 90 degrees
            radius = 0.3
            x = 0.5 + radius * np.cos(angle)
            y = 0.5 - radius * np.sin(angle)
            z = np.ones_like(t) * 0.5
            
        elif trajectory_type == 'u_turn':
            # 180-degree U-turn
            angle = t * np.pi  # 0 to 180 degrees
            radius = 0.25
            x = 0.5 + radius * np.cos(angle + np.pi)
            y = 0.5 + radius * np.sin(angle + np.pi)
            z = np.ones_like(t) * 0.5
            
        elif trajectory_type == 'loop':
            # Full circular loop
            angle = t * 2 * np.pi  # 0 to 360 degrees
            radius = 0.3
            x = 0.5 + radius * np.cos(angle)
            y = 0.5 + radius * np.sin(angle)
            z = np.ones_like(t) * 0.5
            
        else:
            raise ValueError(f"Unknown trajectory type: {trajectory_type}")
        
        # Stack into (N, 3) array
        trajectory = np.column_stack([x, y, z])
        
        return trajectory


def generate_synthetic_trajectory_videos(output_dir, num_per_class=10):
    """
    Generate synthetic trajectory videos (placeholder for compatibility).
    
    Args:
        output_dir: Directory to save videos
        num_per_class: Number of videos per trajectory class
    """
    from pathlib import Path
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    generator = TrajectoryGenerator(num_points=100)
    classes = ['straight', 'left_turn', 'right_turn', 'u_turn', 'loop']
    
    results = {
        'classes': classes,
        'num_per_class': num_per_class,
        'total_generated': 0
    }
    
    for class_name in classes:
        class_dir = output_dir / class_name
        class_dir.mkdir(exist_ok=True)
        
        for i in range(num_per_class):
            trajectory = generator.generate(class_name)
            # Save trajectory (simplified - just save as .npy)
            np.save(class_dir / f"{class_name}_{i:03d}.npy", trajectory)
            results['total_generated'] += 1
    
    return results
```

### Steps to Implement
1. Create `magvit_options/` directory if it doesn't exist
2. Create `magvit_options/__init__.py` (empty file)
3. Create `magvit_options/task1_trajectory_generator.py` with above code
4. Test import: `python -c "from magvit_options.task1_trajectory_generator import TrajectoryGenerator; print('✅ Success')"`
5. Run the clutter integration task

### Pros
✅ Simple, self-contained solution  
✅ No external dependencies  
✅ Matches expected API exactly  
✅ Quick to implement and test  
✅ Easy to understand and maintain  

### Cons
❌ Simplified trajectories (not as sophisticated as original might have been)  
❌ Doesn't include advanced features (noise, variations, etc.)  

---

## Option 2: Use Existing VideoGPT3D Generator

**Complexity:** Medium  
**Time:** 30-40 minutes  
**Risk:** Medium (requires adapter code)  
**Best For:** Reusing existing, tested code

### Implementation
Modify `task_clutter_integration.py` to use `VideoGPT3DTrajectoryDataGenerator`:

```python
# Replace lines 82-87 with:
try:
    from experiments.videogpt_3d_implementation.code.videogpt_3d_trajectory_generator import (
        VideoGPT3DTrajectoryDataGenerator
    )
    
    # Create adapter class
    class TrajectoryGenerator:
        """Adapter for VideoGPT3D generator."""
        def __init__(self, num_points=100):
            self.num_points = num_points
            self.generator = VideoGPT3DTrajectoryDataGenerator(
                seq_length=num_points,
                img_size=128
            )
            # Map class names to patterns
            self.pattern_map = {
                'straight': 'linear_3d',
                'left_turn': 'circular_3d',
                'right_turn': 'circular_3d',
                'u_turn': 'helical',
                'loop': 'circular_3d'
            }
        
        def generate(self, class_name):
            """Generate trajectory using VideoGPT3D generator."""
            pattern = self.pattern_map.get(class_name, 'linear_3d')
            start_pos = (0.3, 0.5, 0.5)
            end_pos = (0.7, 0.5, 0.5)
            return self.generator.generate_trajectory(pattern, start_pos, end_pos)
    
    logger.info("✅ TrajectoryGenerator (VideoGPT3D adapter) imported successfully")
except ImportError as e:
    logger.error(f"❌ Failed to import VideoGPT3D generator: {e}")
    logger.error(traceback.format_exc())
    sys.exit(1)
```

### Steps to Implement
1. Verify VideoGPT3D generator exists on EC2
2. Modify import section in `task_clutter_integration.py`
3. Add adapter class
4. Test trajectory generation
5. Run full clutter integration task

### Pros
✅ Reuses existing, tested code  
✅ More sophisticated trajectory generation  
✅ Already has 3D patterns (helical, parabolic, etc.)  
✅ VideoGPT-compatible output  

### Cons
❌ Requires adapter layer  
❌ Pattern mapping may not be perfect  
❌ More complex dependency chain  
❌ Harder to debug if issues arise  

---

## Option 3: Search Git History for Original

**Complexity:** Medium-High  
**Time:** 20-60 minutes (depending on if found)  
**Risk:** High (may not exist)  
**Best For:** If you need the exact original implementation

### Implementation Steps
1. **Search git history:**
   ```bash
   git log --all --full-history --source --name-status -- '*task1_trajectory_generator*'
   git log --all --grep="trajectory.*generator" --oneline
   ```

2. **Check all branches:**
   ```bash
   for branch in $(git branch -a | grep -v HEAD); do
       echo "=== $branch ==="
       git ls-tree -r --name-only $branch | grep trajectory_generator
   done
   ```

3. **Search for deleted files:**
   ```bash
   git log --diff-filter=D --summary | grep task1_trajectory_generator
   ```

4. **If found, restore:**
   ```bash
   git checkout <commit_hash> -- magvit_options/task1_trajectory_generator.py
   ```

### Pros
✅ Gets exact original implementation  
✅ Preserves original intent and features  
✅ May include documentation and tests  

### Cons
❌ File may not exist in git history  
❌ Time-consuming search  
❌ Original may have other dependencies  
❌ May be outdated or incompatible  

---

## Option 4: Refactor Task to Remove Dependency

**Complexity:** Medium  
**Time:** 30-45 minutes  
**Risk:** Low  
**Best For:** If TrajectoryGenerator isn't critical

### Implementation
Modify `task_clutter_integration.py` to generate trajectories inline:

```python
# Remove lines 82-87 (TrajectoryGenerator import)

# Replace trajectory generation with inline functions:
def generate_simple_trajectory(trajectory_type, num_points=100):
    """Generate simple 3D trajectory inline."""
    t = np.linspace(0, 1, num_points)
    
    if trajectory_type == 'straight':
        x = t
        y = np.ones_like(t) * 0.5
        z = np.ones_like(t) * 0.5
    elif trajectory_type == 'left_turn':
        angle = t * np.pi / 2
        radius = 0.3
        x = 0.5 + radius * np.cos(angle + np.pi)
        y = 0.5 + radius * np.sin(angle + np.pi)
        z = np.ones_like(t) * 0.5
    # ... etc for other types
    
    return np.column_stack([x, y, z])

# Then replace all instances of:
#   generator = TrajectoryGenerator(num_points=100)
#   trajectory = generator.generate('straight')
# With:
#   trajectory = generate_simple_trajectory('straight', num_points=100)
```

### Steps to Implement
1. Add inline trajectory generation function
2. Find/replace all `TrajectoryGenerator` usage
3. Remove import statement
4. Test all trajectory types
5. Run full task

### Pros
✅ Removes external dependency  
✅ Self-contained solution  
✅ Easy to modify and customize  
✅ No import issues  

### Cons
❌ Code duplication if used elsewhere  
❌ Less modular design  
❌ Harder to reuse in other tasks  

---

## Option 5: Mock Generator for Testing Only

**Complexity:** Very Low  
**Time:** 5-10 minutes  
**Risk:** Minimal  
**Best For:** Quick testing of clutter functionality

### Implementation
Add mock directly in `task_clutter_integration.py`:

```python
# Replace lines 81-87 with:
try:
    from magvit_options.task1_trajectory_generator import TrajectoryGenerator
    logger.info("✅ TrajectoryGenerator imported successfully")
except ImportError as e:
    logger.warning(f"⚠️  TrajectoryGenerator not found, using mock: {e}")
    
    # Mock TrajectoryGenerator for testing
    class TrajectoryGenerator:
        """Mock trajectory generator for testing clutter integration."""
        def __init__(self, num_points=100):
            self.num_points = num_points
        
        def generate(self, class_name):
            """Generate simple mock trajectory."""
            t = np.linspace(0, 1, self.num_points)
            # Simple straight line for all types (testing only)
            x = t
            y = np.ones_like(t) * 0.5
            z = np.ones_like(t) * 0.5
            return np.column_stack([x, y, z])
    
    logger.info("✅ Using mock TrajectoryGenerator for testing")
```

### Steps to Implement
1. Modify import section to catch exception
2. Add mock class definition
3. Run task to test clutter functionality
4. Replace with proper implementation later

### Pros
✅ Extremely quick to implement  
✅ Allows testing clutter functionality immediately  
✅ No new files needed  
✅ Graceful fallback  

### Cons
❌ All trajectories are the same (straight line)  
❌ Not suitable for production  
❌ Temporary solution only  
❌ Doesn't test trajectory variety  

---

## Comparison Matrix

| Option | Complexity | Time | Risk | Production Ready | Reusable | Recommended |
|--------|-----------|------|------|------------------|----------|-------------|
| **1. Create Simple Generator** | Low | 15-20m | Low | ✅ Yes | ✅ Yes | ⭐⭐⭐⭐⭐ |
| **2. Use VideoGPT3D** | Medium | 30-40m | Medium | ✅ Yes | ✅ Yes | ⭐⭐⭐ |
| **3. Search Git History** | High | 20-60m | High | ❓ Maybe | ❓ Maybe | ⭐⭐ |
| **4. Refactor Task** | Medium | 30-45m | Low | ✅ Yes | ❌ No | ⭐⭐⭐ |
| **5. Mock for Testing** | Very Low | 5-10m | Low | ❌ No | ❌ No | ⭐⭐ |

---

## Recommended Approach: Option 1 + Option 5

### Phase 1: Immediate Unblocking (5 minutes)
Use **Option 5** (Mock) to immediately test clutter functionality:
- Verify clutter objects work correctly
- Test persistent/transient object rendering
- Validate video generation pipeline

### Phase 2: Proper Implementation (15 minutes)
Implement **Option 1** (Simple Generator):
- Create proper `magvit_options/task1_trajectory_generator.py`
- Implement all 5 trajectory types correctly
- Test with real trajectory variations
- Commit and push to branch

### Why This Approach?
✅ **Immediate progress** - Can test clutter now  
✅ **Proper solution** - Clean, reusable implementation  
✅ **Low risk** - Simple, well-understood code  
✅ **Future-proof** - Easy to enhance later  

---

## Implementation Commands

### Quick Start (Option 5 - Mock)
```bash
# On EC2
cd /home/ubuntu/mono_to_3d
git checkout clutter-transient-objects

# Edit task_clutter_integration.py (add mock as shown above)
nano experiments/clutter_transient_objects/task_clutter_integration.py

# Run task
python experiments/clutter_transient_objects/task_clutter_integration.py
```

### Proper Fix (Option 1 - Simple Generator)
```bash
# On EC2
cd /home/ubuntu/mono_to_3d
git checkout clutter-transient-objects

# Create directory structure
mkdir -p magvit_options
touch magvit_options/__init__.py

# Create generator file
nano magvit_options/task1_trajectory_generator.py
# (paste implementation from Option 1)

# Test import
python -c "from magvit_options.task1_trajectory_generator import TrajectoryGenerator; print('✅')"

# Run full task
python experiments/clutter_transient_objects/task_clutter_integration.py

# Commit
git add magvit_options/
git commit -m "[clutter] Add task1_trajectory_generator with 5 trajectory types"
git push origin clutter-transient-objects
```

---

## Decision Factors

Choose based on your priorities:

### Priority: Speed → **Option 5** (Mock)
Get unblocked in 5 minutes, test clutter functionality now

### Priority: Quality → **Option 1** (Simple Generator)
Clean, proper implementation that's production-ready

### Priority: Reuse → **Option 2** (VideoGPT3D)
Leverage existing sophisticated trajectory generation

### Priority: Authenticity → **Option 3** (Git History)
Find and restore original implementation if it exists

### Priority: Independence → **Option 4** (Refactor)
Remove external dependencies entirely

---

## My Recommendation

**Start with Option 5, then implement Option 1**

This gives you:
1. ✅ Immediate progress (5 minutes)
2. ✅ Clutter functionality testing
3. ✅ Proper long-term solution (15 more minutes)
4. ✅ Clean, maintainable code
5. ✅ Production-ready implementation

**Total time: 20 minutes to complete solution**

---

## Next Steps

1. **Choose your option** (I recommend Option 1 + 5)
2. **I can implement it** - Just say "implement option X"
3. **Test on EC2** - Run the clutter integration task
4. **Review results** - Check generated videos with clutter
5. **Commit and push** - Save the working solution

**Ready to proceed? Which option would you like me to implement?**

