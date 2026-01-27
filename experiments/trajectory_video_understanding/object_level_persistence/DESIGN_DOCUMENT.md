# Object-Level Persistence Detection - Design Document

**Created**: 2026-01-26  
**Goal**: Identify and filter transient objects within frames while continuing to track persistent objects through all frames

---

## 1. Problem Statement

### Current System (WRONG):
- Classifies entire **videos** as persistent/transient
- Stops processing when transient frames detected
- **Result**: Lose tracking data for real objects after noise appears ❌

### Target System (CORRECT):
- Classifies individual **objects** within each frame as persistent/transient
- Continues processing all frames
- Reduces compute on transient objects, focuses on persistent ones
- **Result**: Complete tracking of real objects, efficient filtering of noise ✓

---

## 2. Architecture Overview

```
Input: Video (T=16 frames, each 64x64x3)
       Each frame may contain multiple spheres (1 real + 0-2 transient)

Step 1: OBJECT DETECTION
   For each frame:
   - Detect all sphere objects (bounding boxes + features)
   - Output: List of detected objects per frame

Step 2: OBJECT TRACKING  
   Across frames:
   - Link detections into tracks (track IDs)
   - Output: Track for each unique object

Step 3: PERSISTENCE CLASSIFICATION (PER TRACK)
   For each track:
   - Feature: Track length (1-3 frames = transient, longer = persistent)
   - Feature: Visual consistency across frames
   - Feature: Motion pattern
   - Output: Persistence score per track

Step 4: ATTENTION GATING (PER TRACK)
   For each track:
   - IF persistent score > threshold: 
       → High compute (full feature extraction, attention)
   - IF transient score > threshold:
       → Low compute (minimal processing, low attention)
   
Step 5: 3D RECONSTRUCTION
   For persistent tracks only:
   - Use full tracking data from all frames
   - Reconstruct 3D trajectory
```

---

## 3. Key Components

### 3.1 Object Detector
**Purpose**: Detect all sphere objects in each frame

**Architecture**:
- Input: Single frame (3, 64, 64)
- Backbone: ResNet or simple CNN
- Head: Object detection (bounding boxes, confidence)
- Output: List of [bbox, features, confidence] per object

**Training**:
- Use existing dataset with known sphere positions
- Supervision from metadata (sphere centers)

---

### 3.2 Object Tracker
**Purpose**: Link detections across frames into tracks

**Algorithm Options**:
1. **Simple**: IoU-based matching (bounding box overlap)
2. **Intermediate**: Feature similarity + motion model
3. **Advanced**: SORT/DeepSORT tracking

**Output**: 
- Track ID for each detection
- Track history: {track_id: [(frame, bbox, features), ...]}

---

### 3.3 Persistence Classifier (Per-Track)
**Purpose**: Classify each track as persistent (real) or transient (noise)

**Features**:
1. **Track length**: 
   - Transient: 1-3 frames
   - Persistent: 4+ frames (ideally 16)

2. **Visual consistency**:
   - Color consistency (white for real, red for transient)
   - Size consistency
   - Shape consistency

3. **Motion pattern**:
   - Smooth trajectory (persistent)
   - Erratic or sudden appearance (transient)

**Architecture**:
- Input: Track features (sequence of frame features)
- Encoder: LSTM or Transformer to aggregate track history
- Classifier: Binary (persistent vs transient)
- Output: Persistence probability per track

---

### 3.4 Attention Mechanism (Per-Object)
**Purpose**: Allocate compute based on persistence

**Mechanism**:
```python
for track in tracks:
    if track.persistence_score > 0.8:  # Persistent
        compute_budget = 1.0  # Full processing
        attention_weight = HIGH
    else:  # Transient
        compute_budget = 0.2  # Minimal processing
        attention_weight = LOW
    
    # Apply compute budget
    features = extract_features(track, compute_budget)
    
    # Use attention for feature weighting
    weighted_features = attention_weight * features
```

---

## 4. Data Representation

### Frame-Level:
```python
{
    'frame_idx': 0,
    'detections': [
        {
            'bbox': [x, y, w, h],
            'center': [cx, cy],
            'features': tensor(256,),
            'confidence': 0.95
        },
        ...
    ]
}
```

### Track-Level:
```python
{
    'track_id': 0,
    'start_frame': 0,
    'end_frame': 15,
    'length': 16,
    'detections': [
        {'frame': 0, 'bbox': [x, y, w, h], 'features': ...},
        {'frame': 1, 'bbox': [x, y, w, h], 'features': ...},
        ...
    ],
    'persistence_score': 0.95,
    'is_persistent': True,
    'color': 'white',  # or 'red' for transient
    'trajectory_3d': [(x, y, z), ...]  # Reconstructed 3D path
}
```

---

## 5. Training Strategy

### Phase 1: Object Detection
- Dataset: All frames with ground truth sphere positions
- Loss: Detection loss (bbox + confidence)
- Metric: IoU, Detection accuracy

### Phase 2: Persistence Classification
- Dataset: Extracted tracks with labels
- Loss: Binary cross-entropy
- Metric: Accuracy, F1 score (per track, not per frame)

### Phase 3: Attention Learning
- Dataset: Full videos with mixed persistent/transient tracks
- Loss: Multi-task:
  - Persistence classification loss
  - Attention efficiency loss (penalize high attention on transient tracks)
  - 3D reconstruction loss (for persistent tracks)
- Metric: 
  - Attention ratio (persistent/transient)
  - Compute savings
  - 3D reconstruction accuracy

---

## 6. Evaluation Metrics

### Object Detection:
- Detection rate: % of objects detected
- False positives: Spurious detections

### Tracking:
- Track purity: % of detections correctly assigned to tracks
- Track completeness: % of ground truth track covered

### Persistence Classification:
- Per-track accuracy
- Precision/Recall for persistent tracks
- Precision/Recall for transient tracks

### Attention Efficiency:
- Avg attention on persistent tracks (should be HIGH)
- Avg attention on transient tracks (should be LOW)
- Compute savings: % reduction from equal attention baseline

### 3D Reconstruction (Final Goal):
- Trajectory accuracy for persistent tracks
- No false trajectories from transient tracks

---

## 7. Key Differences from Current System

| Aspect | Current (Video-Level) | New (Object-Level) |
|--------|----------------------|-------------------|
| **Classification Unit** | Entire video | Individual track |
| **Early Stopping** | Stop video at frame 4 | Never stop, process all frames |
| **Attention** | Per-frame | Per-object/track |
| **Output** | "Video is clean/noisy" | "Track 0 is persistent, Track 1 is transient" |
| **Use Case** | Pre-filtering | Real-time tracking with noise filtering |
| **3D Reconstruction** | Incomplete (missing frames) | Complete (all frames) |

---

## 8. Implementation Plan

### Step 1: Object Detection Module ✓ (Week 1)
- Implement sphere detector
- Train on existing dataset
- TDD: Test detection accuracy

### Step 2: Object Tracking Module ✓ (Week 1)
- Implement IoU-based tracker
- Link detections into tracks
- TDD: Test track assignment

### Step 3: Persistence Classifier ✓ (Week 2)
- Extract track features
- Train track-level classifier
- TDD: Test persistence classification

### Step 4: Attention Mechanism ✓ (Week 2)
- Implement per-track attention
- Integrate with feature extraction
- TDD: Test attention allocation

### Step 5: Integration & Evaluation ✓ (Week 3)
- End-to-end pipeline
- Efficiency metrics
- 3D reconstruction quality

---

## 9. Success Criteria

1. ✅ **Detection**: >95% of objects detected in each frame
2. ✅ **Tracking**: >90% track purity and completeness
3. ✅ **Classification**: >90% accuracy in identifying transient vs persistent tracks
4. ✅ **Attention Efficiency**: 
   - Persistent tracks get 3x+ more attention than transient
   - 50%+ compute savings from filtering transient tracks
5. ✅ **3D Reconstruction**: Complete trajectories for all persistent tracks

---

## 10. Example Workflow

```
Video Input: 16 frames

Frame 0: [White sphere detected at (10, 20)]
         → Track 0 created

Frame 1: [White sphere at (11, 21)]
         → Track 0 continued

Frame 2: [White sphere at (12, 22), Red sphere at (50, 50)]
         → Track 0 continued
         → Track 1 created (NEW)

Frame 3: [White sphere at (13, 23), Red sphere at (51, 49)]
         → Track 0 continued
         → Track 1 continued

Frame 4: [White sphere at (14, 24)]
         → Track 0 continued
         → Track 1 ended (length=2, TRANSIENT)

...

Frame 15: [White sphere at (25, 35)]
         → Track 0 ended (length=16, PERSISTENT)

Analysis:
- Track 0: Length=16, Color=white, Motion=smooth
  → PERSISTENT (score=0.98)
  → HIGH ATTENTION
  → USE FOR 3D RECONSTRUCTION ✓

- Track 1: Length=2, Color=red, Motion=erratic
  → TRANSIENT (score=0.05)
  → LOW ATTENTION
  → IGNORE FOR 3D RECONSTRUCTION ✗

Output:
- Persistent tracks: [Track 0]
- 3D trajectory: [(x0,y0,z0), (x1,y1,z1), ..., (x15,y15,z15)]
- Compute savings: 40% (reduced processing on Track 1)
```

---

## 11. Next Steps

1. Create git branch: `object-level-persistence`
2. Implement TDD tests for object detector
3. Implement object detector
4. Implement tracker
5. Implement persistence classifier
6. Integrate and evaluate

**This is the correct architecture for tracking + noise filtering!** 🎯

