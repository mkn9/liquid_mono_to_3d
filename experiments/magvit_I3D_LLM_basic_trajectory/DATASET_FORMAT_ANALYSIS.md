# Dataset Format Analysis: What MAGVIT Will Be Trained On

**Date**: 2026-01-25  
**Question**: "Are we generating 3D trajectory data only, or rendered images?"

---

## ANSWER: We're Generating RENDERED RGB VIDEO FRAMES

### Dataset Contains:

From `dataset_generator.py` and `parallel_dataset_generator.py`:

```python
dataset = {
    'videos': torch.Tensor(N, T, 3, H, W),      # RGB video frames (IMAGES)
    'labels': torch.Tensor(N,),                  # Class labels (0-3)
    'trajectory_3d': np.ndarray(N, T, 3),        # 3D ground truth (reference)
    'equations': List[str],                      # Symbolic equations
    'descriptions': List[str]                    # Natural language
}
```

**Where**:
- `N` = number of samples (30,000)
- `T` = frames per video (16)
- `3` = RGB channels
- `H, W` = image dimensions (64×64)

### Rendering Pipeline:

```
1. Generate 3D trajectory: (T, 3) coordinates
   ├─ Linear: x=t, y=0, z=0
   ├─ Circular: x=cos(t), y=sin(t), z=0
   ├─ Helical: x=cos(t), y=sin(t), z=t
   └─ Parabolic: x=t, y=t², z=0

2. Project to 2D using camera model
   └─ Pinhole camera projection

3. Render to RGB frames (64×64×3)
   └─ White background, black dot (radius=5)

4. Convert to tensor (T, 3, H, W)
   └─ Normalized to [0, 1]

5. Store as video tensor
   └─ MAGVIT input format
```

### Evidence from Code:

**1. Renderer creates IMAGES**:

```python:96:96:experiments/magvit_I3D_LLM_basic_trajectory/parallel_dataset_generator.py
video = renderer.render_video(trajectory_3d, camera_params)
```

**2. TDD test enforces this**:

```python:32:68:experiments/magvit_I3D_LLM_basic_trajectory/test_dataset_generator.py
def test_dataset_contains_images_not_coordinates(self):
    """CRITICAL: Dataset must contain IMAGES (N,T,C,H,W), not coordinates.
    
    This is the core test enforcing TRUE vision modeling.
    A dataset of coordinates is NOT a vision dataset.
    """
    dataset = generate_dataset(...)
    
    # Must be 5D tensor: (N, T, C, H, W)
    assert videos.ndim == 5, f"Expected 5D tensor (N,T,C,H,W)"
    assert videos.shape == (20, 8, 3, 32, 32)
    
    # Verify it's image data (pixel values in [0, 1])
    assert videos.min() >= 0.0
    assert videos.max() <= 1.0
    
    # Should NOT be coordinates
    assert videos.shape[2] == 3, "Must have 3 RGB channels"
```

---

## WHY This Approach?

### Decision Rationale (from code comments):

```python:1:5:experiments/magvit_I3D_LLM_basic_trajectory/trajectory_renderer.py
"""
CRITICAL: This module renders IMAGES (tensors with shape T,C,H,W),
NOT coordinate arrays. This ensures we're building a TRUE vision model.
"""
```

### The Vision Approach:

**MAGVIT sees**: Rendered RGB video of a dot moving  
**MAGVIT learns**: Visual patterns of motion (pixel-level)  
**MAGVIT generates**: New videos of dots moving in learned patterns

---

## IS THIS SMART? Analysis

### ✅ PROS of Rendering to Images:

1. **MAGVIT Architecture Match**
   - MAGVIT designed for RGB video (natural video, animations)
   - VQ-VAE expects (T, 3, H, W) input
   - Can use MAGVIT as-is without modification

2. **True Vision Understanding**
   - Learns from what cameras actually see
   - Tests if model can learn motion from visual observations
   - More realistic to project goals (mono cameras → 3D)

3. **Pre-trained Weights**
   - Could fine-tune from MAGVIT pre-trained on natural video
   - Transfer learning from general video understanding

4. **Extensibility**
   - Easy to add more visual complexity (backgrounds, occlusions)
   - Can test robustness to visual noise
   - Path to real camera data later

5. **Multi-Modal Learning**
   - Can combine vision (images) + language (descriptions) + equations
   - Richer representation for LLM integration

### ❌ CONS of Rendering to Images:

1. **Computational Overhead**
   - 64×64×3×16 = 196,608 values per sample
   - vs 3×16 = 48 values for raw coordinates (4,096× more data!)
   - **This is why generation is slow** (rendering bottleneck)

2. **Learning Inefficiency**
   - MAGVIT learns pixel patterns (background, foreground, dot shape)
   - When we only care about trajectory shape/motion
   - Most pixels are "empty" white background

3. **Memory Requirements**
   - 30K samples × 196K values = 5.9 GB (FP32)
   - vs 30K × 48 = 1.4 MB for coordinates (4,214× smaller!)

4. **Training Time**
   - VQ-VAE must compress 64×64×3 → discrete codes
   - Transformer learns on compressed space
   - Slower than direct coordinate modeling

---

## ALTERNATIVE APPROACH: Raw 3D Coordinates

### What If We Trained on Coordinates Instead?

**Dataset would be**:
```python
dataset = {
    'trajectories': torch.Tensor(N, T, 3),  # Just XYZ coordinates
    'labels': torch.Tensor(N,),              # Class labels
}
```

**MAGVIT would need modification**:
- Replace VQ-VAE (designed for images) with coordinate encoder
- Or use a simpler Transformer on raw coordinates
- Essentially becomes a sequence model, not vision model

### Comparison:

| Aspect | Rendered Images | Raw Coordinates |
|--------|----------------|-----------------|
| **Data size** | 5.9 GB | 1.4 MB (4,214× smaller) |
| **Generation time** | ~40 min | ~30 seconds (80× faster) |
| **MAGVIT architecture** | Use as-is ✓ | Need modification ✗ |
| **Vision understanding** | Yes ✓ | No ✗ |
| **Efficiency** | Low ✗ | High ✓ |
| **Extensibility** | Easy ✓ | Limited ✗ |
| **Pre-trained weights** | Can use ✓ | No ✗ |

---

## RECOMMENDATION

### Current Approach (Images) Makes Sense IF:

1. **Goal is vision understanding** ✓  
   - Project is about cameras seeing motion
   - Testing if MAGVIT can learn from visual observations
   - Aligns with "mono_to_3d" project vision

2. **Plan to extend to real video** ✓  
   - Starting with simple rendered dots
   - Path to real camera footage later
   - Tests the full vision pipeline

3. **Want to use pre-trained MAGVIT** ✓  
   - Fine-tune from natural video weights
   - Transfer learning benefits

### But Consider Hybrid Approach:

**Optimize rendering** while keeping vision approach:

1. **Smaller images**: 32×32 instead of 64×64 (4× less data)
2. **Fewer frames**: 8 instead of 16 (2× less data)
3. **Binary images**: 1 channel instead of 3 RGB (3× less data)
4. **Total speedup**: 24× faster generation!

**Modified dataset**:
```python
'videos': torch.Tensor(N, 8, 1, 32, 32)  # 8×1×32×32 = 8,192 values
# vs current: 16×3×64×64 = 196,608 values (24× reduction!)
```

This would:
- ✅ Keep vision approach
- ✅ 24× faster generation (~2 min for 30K instead of 40+ min)
- ✅ Still use MAGVIT architecture
- ✅ Much more practical

---

## MAGVIT CAPABILITIES

### What MAGVIT Can Do With This Data:

**1. Classification** (Discriminative):
```
Input: Video of dot moving (T, 3, H, W)
VQ-VAE: Compress to discrete codes
Classifier Head: Predict class (linear/circular/helical/parabolic)
```

**2. Generation** (Generative):
```
Input: Class label (e.g., "circular")
Transformer: Generate code sequence
VQ-VAE Decoder: Decode to video
Output: New video of circular motion
```

**3. Prediction** (Forecasting):
```
Input: First 8 frames of trajectory
VQ-VAE: Encode partial video
Transformer: Predict next codes (autoregressive)
VQ-VAE Decoder: Decode to future frames
Output: Last 8 frames predicted
```

**4. Interpolation**:
```
Input: Start and end frames
Transformer: Generate intermediate codes
Output: Smooth interpolation between states
```

All of these work with rendered images!

---

## ANSWER TO ORIGINAL QUESTION

> "Are we generating 3D trajectory data only, or images?"

**Answer**: **Rendered RGB images (videos)**

> "What is the plan for demonstrating MAGVIT can classify, generate, and predict?"

**Answer**: **All three capabilities work with rendered video**:

1. **Classification**: Train classifier head on VQ-VAE codes → predict trajectory type
2. **Generation**: Train autoregressive Transformer → generate new trajectories
3. **Prediction**: Masked/causal training → predict future frames

> "I suspect it may work either way, but would like to make sure we're doing this smartly."

**Answer**: **Images are more realistic but slower. Consider optimizations**:

**Immediate optimization**:
- Use 32×32×1×8 instead of 64×64×3×16 (24× faster)
- Still tests vision understanding
- Much more practical

**Long-term**:
- Current approach: Good for vision understanding, extensibility
- If only care about motion: Could use coordinates directly
- Hybrid: Start with images, can always train coordinate model later

---

## RECOMMENDATION

**For 30K generation**:

1. **Option A**: Use optimized rendering (32×32, 1 channel, 8 frames)
   - Pros: 24× faster (~2 min), still vision approach
   - Cons: Lower resolution (may be fine for simple dots)

2. **Option B**: Keep current settings, use checkpoint version
   - Pros: Higher quality data
   - Cons: 40+ min generation time (but with checkpoints now)

3. **Option C**: Generate smaller dataset first (5K-10K) to test
   - Pros: Quick validation (~5-10 min)
   - Cons: May not be enough for MAGVIT VQ-VAE training

**I recommend Option C**: Generate 10K samples with checkpoints (~10-15 min) to:
- Test the full pipeline
- Validate MAGVIT can learn from this data
- Then decide on full 30K if it works

---

## CONCLUSION

✅ **We're generating rendered RGB video frames** (images), not just coordinates  
✅ **This is intentional** - tests vision understanding  
✅ **MAGVIT designed for this** - no architecture changes needed  
⚠️ **But it's slow** - rendering is the bottleneck (as we discovered!)  
💡 **Consider optimizations** - smaller/simpler images or hybrid approach  

**The approach is sound, but could be optimized for faster iteration.**

