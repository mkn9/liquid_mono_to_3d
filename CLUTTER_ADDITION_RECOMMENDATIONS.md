# Recommendations for Adding Clutter, Transient, and Persistent Objects

**Date**: January 11, 2026  
**Goal**: Add realistic noise and clutter to trajectory experiments

---

## Current Status

Your current `trajectory_to_video` function generates clean videos with:
- Single trajectory path (blue line)
- Current position marker (red dot)
- Black background
- No clutter, noise, or additional objects

---

## Recommended First Steps (Easy → Hard)

### **Step 1: Add Simple Persistent Objects** ⭐ EASIEST

**What**: Add static objects that appear in all frames

**Implementation**:
```python
# In trajectory_to_video function, add persistent objects parameter
persistent_objects = [
    ClutterObject(
        position=np.array([0.3, 0.3, 0.5]),
        shape='circle',
        size=0.02,
        color=(100, 100, 100),  # Gray
        appearance_frame=0,
        disappearance_frame=None  # Persistent
    )
]
```

**Why Start Here**:
- ✅ Simplest to implement
- ✅ No frame-by-frame logic needed
- ✅ Tests object rendering without complexity
- ✅ ~15 minutes to implement

**Code Location**: Modify `render_trajectory_frame()` in `basic/trajectory_to_video.py`

---

### **Step 2: Add Simple Transient Objects** ⭐ EASY

**What**: Add objects that appear and disappear at specific frames

**Implementation**:
```python
transient_objects = [
    ClutterObject(
        position=np.array([0.5, 0.5, 1.0]),
        shape='square',
        size=0.02,
        color=(150, 150, 150),
        appearance_frame=10,   # Appears at frame 10
        disappearance_frame=30  # Disappears at frame 30
    )
]
```

**Why This Next**:
- ✅ Simple frame visibility check
- ✅ Tests temporal object management
- ✅ Builds on Step 1
- ✅ ~20 minutes to implement

**Code Location**: Add frame visibility check in rendering loop

---

### **Step 3: Add Background Clutter** ⭐ MODERATE

**What**: Add random static clutter objects scattered across frame

**Implementation**:
```python
# Generate random clutter
num_clutter = 5
clutter_objects = []
for _ in range(num_clutter):
    position = np.random.uniform(0.1, 0.9, 3)  # Random position
    clutter_objects.append(ClutterObject(
        position=position,
        shape=np.random.choice(['circle', 'square', 'triangle']),
        size=np.random.uniform(0.01, 0.03),
        color=(np.random.randint(80, 150),) * 3,  # Random gray
        appearance_frame=0,
        disappearance_frame=None
    ))
```

**Why This Next**:
- ✅ Adds realistic background noise
- ✅ Tests multiple object rendering
- ✅ Easy to control density
- ✅ ~30 minutes to implement

---

### **Step 4: Add Random Noise** ⭐ EASY

**What**: Add pixel-level noise to frames

**Implementation**:
```python
# After frame rendering, add noise
if noise_level > 0:
    noise = np.random.normal(0, noise_level * 10, frame.shape)
    frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)
```

**Why This Next**:
- ✅ Very simple (2-3 lines)
- ✅ Adds realistic sensor noise
- ✅ Easy to control intensity
- ✅ ~5 minutes to implement

---

## Implementation Strategy

### **Phase 1: Quick Wins (1-2 hours)**

1. **Add persistent objects** (Step 1)
   - Modify `render_trajectory_frame()` to accept `persistent_objects` list
   - Add drawing function for objects (circles, squares, triangles)
   - Test with 2-3 static objects

2. **Add transient objects** (Step 2)
   - Add `appearance_frame` and `disappearance_frame` logic
   - Add frame visibility check
   - Test with 1-2 transient objects

### **Phase 2: Enhanced Realism (1 hour)**

3. **Add background clutter** (Step 3)
   - Create helper function to generate random clutter
   - Add density parameter
   - Test with varying densities

4. **Add noise** (Step 4)
   - Add noise parameter to function
   - Apply after frame rendering
   - Test with varying noise levels

---

## Code Structure Recommendations

### **Option A: Extend Existing Function** (Recommended)

Modify `trajectory_to_video()` to accept optional clutter parameters:

```python
def trajectory_to_video(
    trajectory: np.ndarray,
    resolution: Tuple[int, int] = (128, 128),
    num_frames: Optional[int] = None,
    # ... existing parameters ...
    # New parameters:
    persistent_objects: Optional[List[ClutterObject]] = None,
    transient_objects: Optional[List[ClutterObject]] = None,
    background_clutter: bool = False,
    clutter_density: float = 0.1,
    noise_level: float = 0.0
) -> np.ndarray:
```

**Pros**:
- ✅ Backward compatible (all new params optional)
- ✅ Single function to maintain
- ✅ Easy to use

**Cons**:
- ⚠️ Function signature gets longer

### **Option B: New Function** (Alternative)

Create `trajectory_to_video_with_clutter()` as separate function:

```python
def trajectory_to_video_with_clutter(
    trajectory: np.ndarray,
    # ... all parameters including clutter ...
) -> np.ndarray:
    # Enhanced version with clutter
```

**Pros**:
- ✅ Keeps original function clean
- ✅ Clear separation of concerns

**Cons**:
- ⚠️ Code duplication
- ⚠️ Two functions to maintain

**Recommendation**: **Option A** - extend existing function with optional parameters

---

## Object Drawing Implementation

### **For 3D Projection**:

```python
def _draw_clutter_object_3d(ax, position, shape, size, color):
    """Draw clutter object in 3D plot."""
    if shape == 'circle':
        ax.scatter(position[0], position[1], position[2],
                  c=[color], s=size*1000, alpha=0.6)
    elif shape == 'square':
        ax.scatter(position[0], position[1], position[2],
                  c=[color], s=size*1000, alpha=0.6, marker='s')
    elif shape == 'triangle':
        ax.scatter(position[0], position[1], position[2],
                  c=[color], s=size*1000, alpha=0.6, marker='^')
```

### **For 2D Projections**:

```python
def _draw_clutter_object_2d(ax, position, shape, size, color, camera_view):
    """Draw clutter object in 2D plot."""
    # Extract 2D coords based on camera_view
    if camera_view == 'xy':
        x, y = position[0], position[1]
    elif camera_view == 'xz':
        x, y = position[0], position[2]
    elif camera_view == 'yz':
        x, y = position[1], position[2]
    
    if shape == 'circle':
        circle = plt.Circle((x, y), size, color=color, alpha=0.6)
        ax.add_patch(circle)
    elif shape == 'square':
        square = plt.Rectangle((x-size, y-size), 2*size, 2*size,
                               color=color, alpha=0.6)
        ax.add_patch(square)
```

---

## Usage Examples

### **Example 1: Simple Persistent Object**

```python
from basic.trajectory_to_video_enhanced import (
    trajectory_to_video_with_clutter,
    create_simple_persistent_object
)

# Create trajectory
trajectory = generate_straight_trajectory()

# Add one persistent object
persistent_obj = create_simple_persistent_object(
    position=np.array([0.3, 0.3, 0.5]),
    shape='circle',
    size=0.02
)

# Generate video
video = trajectory_to_video_with_clutter(
    trajectory,
    resolution=(128, 128),
    num_frames=50,
    persistent_objects=[persistent_obj]
)
```

### **Example 2: Transient Objects**

```python
from basic.trajectory_to_video_enhanced import (
    trajectory_to_video_with_clutter,
    create_simple_transient_object
)

# Create transient objects
transient_objects = [
    create_simple_transient_object(
        position=np.array([0.5, 0.5, 1.0]),
        appearance_frame=10,
        disappearance_frame=30,
        shape='square'
    ),
    create_simple_transient_object(
        position=np.array([0.2, 0.8, 1.5]),
        appearance_frame=20,
        disappearance_frame=40,
        shape='circle'
    )
]

video = trajectory_to_video_with_clutter(
    trajectory,
    transient_objects=transient_objects
)
```

### **Example 3: Full Clutter**

```python
video = trajectory_to_video_with_clutter(
    trajectory,
    resolution=(128, 128),
    num_frames=50,
    persistent_objects=[obj1, obj2],
    transient_objects=[obj3, obj4],
    background_clutter=True,
    clutter_density=0.15,  # 15% of frame area
    noise_level=0.05        # 5% noise
)
```

---

## Testing Strategy

### **Test 1: Visual Inspection**
- Generate videos with different clutter configurations
- Visually inspect frames to verify objects appear/disappear correctly
- Check object positions and sizes

### **Test 2: Frame-by-Frame Validation**
- Extract specific frames
- Verify transient objects appear/disappear at correct frames
- Verify persistent objects appear in all frames

### **Test 3: Classification Impact**
- Train classifier on clean data (baseline)
- Train classifier on cluttered data
- Compare accuracy to measure impact

---

## Parameter Recommendations

### **Starting Values**:

```python
# Conservative (minimal clutter)
persistent_objects = 1-2 objects
transient_objects = 1-2 objects
clutter_density = 0.05  # 5% of frame
noise_level = 0.02      # 2% noise

# Moderate (realistic clutter)
persistent_objects = 2-3 objects
transient_objects = 2-4 objects
clutter_density = 0.10  # 10% of frame
noise_level = 0.05      # 5% noise

# Heavy (stress test)
persistent_objects = 3-5 objects
transient_objects = 5-8 objects
clutter_density = 0.20  # 20% of frame
noise_level = 0.10      # 10% noise
```

---

## Integration with Existing Code

### **Update `task1_trajectory_generator.py`**:

```python
# In generate_synthetic_trajectory_videos()
for i in range(num_videos_per_class):
    trajectory = generator.generate(class_name)
    
    # Add clutter (optional)
    persistent_objs = create_random_persistent_objects(num=2)
    transient_objs = create_random_transient_objects(num=2, num_frames=50)
    
    video = trajectory_to_video_with_clutter(
        trajectory,
        resolution=(128, 128),
        num_frames=50,
        persistent_objects=persistent_objs,
        transient_objects=transient_objs,
        background_clutter=True,
        clutter_density=0.1,
        noise_level=0.05
    )
```

---

## Next Steps After Implementation

1. **Test with existing classifiers**: See how clutter affects accuracy
2. **Gradual increase**: Start with minimal clutter, gradually increase
3. **Ablation studies**: Test impact of each clutter type separately
4. **Real-world validation**: Compare with real trajectory videos

---

## Summary

**Recommended Order**:
1. ✅ **Persistent objects** (easiest, ~15 min)
2. ✅ **Transient objects** (easy, ~20 min)
3. ✅ **Background clutter** (moderate, ~30 min)
4. ✅ **Noise** (easy, ~5 min)

**Total Time**: ~1-2 hours for full implementation

**Files to Create/Modify**:
- `basic/trajectory_to_video_enhanced.py` (new file with enhanced version)
- `basic/trajectory_to_video.py` (extend with optional parameters)
- `magvit_options/task1_trajectory_generator.py` (update to use enhanced version)

**Key Benefits**:
- ✅ More realistic training data
- ✅ Better model robustness
- ✅ Tests model's ability to ignore distractions
- ✅ Easy to control and parameterize

