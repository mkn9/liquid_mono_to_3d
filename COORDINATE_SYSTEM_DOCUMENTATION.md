# 3D Tracking System - Coordinate System Documentation

## ⚠️ CRITICAL: Read This First

**Our coordinate system is NOT standard computer vision convention!**

## 🎯 Definitive Coordinate System Specification

### Camera Configuration
- **Camera 1 Position**: [0.0, 0.0, 2.5]
- **Camera 2 Position**: [1.0, 0.0, 2.5]  
- **Rotation Matrices**: Identity matrices (R₁ = R₂ = I₃ₓ₃)
- **Baseline**: 1.0 meter (X-axis separation)

### Camera Orientation
- **Primary Direction**: +Y (forward)
- **Level**: No tilt in Z direction
- **No Squint**: No tilt in X direction
- **Mathematical**: Identity rotation means cameras look in +Y direction

### Object Visibility Rules
- **In Front**: Objects with Y > camera_Y (Y > 0) are visible
- **Behind**: Objects with Y < camera_Y (Y < 0) are behind cameras
- **At Camera Level**: Objects with Y = camera_Y project to infinity

### Trajectory Constraints
- **Y Coordinate**: Must be > 0 for visibility (typically Y = 1.0)
- **Z Coordinate**: Should avoid Z = 2.5 (camera height) to prevent singularities
- **X Coordinate**: Determines left/right position in image

## 🚫 Common Mistakes to Avoid

### ❌ Wrong Assumptions
1. **"Cameras look in -Z direction"** ← WRONG for our system
2. **"Standard CV convention applies"** ← WRONG for our system  
3. **"Identity rotation = -Z looking"** ← WRONG for our system
4. **"Objects need negative coordinates"** ← WRONG for our system

### ✅ Correct Understanding
1. **Identity rotation = +Y looking** in our coordinate system
2. **Positive Y coordinates** = objects in front of cameras
3. **Y determines depth**, not Z in our camera arrangement
4. **Z determines vertical position** in world coordinates

## 🔧 Implementation Guidelines

### Camera Setup Template
```python
# CORRECT camera setup
R1 = np.eye(3, dtype=np.float64)  # Identity = +Y looking
R2 = np.eye(3, dtype=np.float64)  # Identity = +Y looking

# Comments should say:
# "Cameras look in +Y direction (forward)"
# NOT "Cameras look in -Z direction"
```

### Test Point Selection
```python
# CORRECT test points (Y > 0 for visibility)
test_points = [
    [0.0, 1.0, 2.5],  # In front of cameras
    [0.5, 2.0, 2.5],  # Further in front
    [0.3, 0.5, 2.5],  # Closer to cameras
]

# AVOID these points
avoid_points = [
    [0.0, 0.0, 2.5],  # At camera Y level (projects to infinity)
    [0.0, -1.0, 2.5], # Behind cameras (Y < 0)
    [0.0, 1.0, 2.5],  # At camera Z level (potential singularity)
]
```

### Validation Checklist
- [ ] Comments mention "+Y direction", not "-Z direction"
- [ ] Test points have Y > 0 for visibility
- [ ] Avoid Y = 0 (camera level) and Z = 2.5 (camera height)
- [ ] Rightward movement (increasing X) → increasing pixel X coordinates
- [ ] Forward movement (increasing Y) → objects get larger in image

## 📝 File Update Checklist

When updating any file, verify:
- [ ] No mentions of "-Z direction" for camera orientation
- [ ] No assumptions about "standard CV convention"
- [ ] Test trajectories use Y > 0 coordinates
- [ ] Comments accurately reflect +Y looking cameras
- [ ] FOV calculations account for +Y optical axis

## 🧪 Validation Commands

```bash
# Run coordinate system validation
python test_3d_tracker_correct_coordinate_system.py

# Check for incorrect comments
grep -r "\-Z direction" *.py *.ipynb
grep -r "standard.*CV.*convention" *.py *.ipynb

# Verify trajectory coordinates
grep -r "np\.array.*\[.*,.*0\." *.py *.ipynb  # Look for Y=0 points
```

## 📚 Mathematical Proof

Our coordinate system proof:
1. Camera at [0, 0, 2.5] with identity rotation
2. Test point at [0, 1, 2.5] (1 unit in +Y)  
3. Relative position: [0, 1, 0] (positive Y)
4. After identity rotation: [0, 1, 0] (unchanged)
5. Projection: positive Y means "in front" → camera looks in +Y

**This is definitive mathematical proof that our cameras look in +Y direction.**

---

**Remember**: When in doubt, refer to this document and run the validation tests! 