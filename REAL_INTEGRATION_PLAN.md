# Real Track Persistence Integration Plan

## Current State Analysis

### What We Have
✅ Stereo 3D tracking system (`simple_3d_tracker.py`)  
✅ Synthetic 2D track generation  
✅ Triangulation pipeline  
✅ MagVIT pretrained model  
✅ Toy proof-of-concept for persistence (NOT integrated)

### What's Missing
❌ Realistic 2D tracks with clutter/noise  
❌ MagVIT visual features in actual pipeline  
❌ Transformer attention for persistent vs transient  
❌ Integration into 3D reconstruction  
❌ Multi-sensor track fusion  
❌ LLM reasoning on attention patterns

---

## Implementation Plan: 6 Workers in Parallel

### Worker 1: Realistic 2D Track Simulation
**Branch:** `track-persistence/realistic-2d-tracks`

**Goal:** Generate realistic 2D track sequences that simulate real object detector output (YOLO-like)

**Tasks:**
1. Create `realistic_track_generator.py`:
   - Persistent objects (20-50 frames)
   - Transient detections (2-5 frames): glare, reflections, shadows
   - False positives (1 frame): noise
   - Bounding boxes with pixel data
   - Multiple objects per scene
   
2. Track properties:
   - `track_id`: Unique identifier
   - `frames`: List of frame numbers
   - `bboxes`: List of [x, y, w, h] bounding boxes
   - `pixels`: Crop of image region (for MagVIT)
   - `is_persistent`: Ground truth label
   - `duration`: Number of frames
   
3. Generate dataset:
   - 1000 track sequences
   - Mix of persistent (60%), brief (30%), noise (10%)
   - Save as `.npz` with metadata

**Output:** Dataset of realistic 2D tracks

---

### Worker 2: MagVIT Visual Feature Extraction
**Branch:** `track-persistence/magvit-features`

**Goal:** Extract visual features from 2D track sequences using pretrained MagVIT

**Tasks:**
1. Create `extract_track_features.py`:
   - Load 2D track sequence (bbox crops over time)
   - Format as video tensor: `(1, T, H, W, 3)`
   - Pass through MagVIT encoder
   - Extract latent features: `(T, D)` where D=256
   
2. Feature types:
   - **Appearance**: What does the object look like?
   - **Motion**: How does it move?
   - **Temporal consistency**: Does appearance persist?
   
3. Cache features:
   - Save as `track_{id}_features.npy`
   - Speeds up training

**Output:** MagVIT features for each 2D track

---

### Worker 3: Transformer Attention Model
**Branch:** `track-persistence/transformer-attention`

**Goal:** Build Transformer model that classifies tracks using attention

**Tasks:**
1. Create `attention_persistence_model.py`:
   ```
   Input: MagVIT features (T, 256)
   ↓
   Positional Encoding
   ↓
   Transformer (4 layers, 8 heads, 256 dim)
   ↓
   Attention weights (T,) - which frames matter?
   ↓
   Classification head → persistent (1) or not (0)
   ```
   
2. Key features:
   - Extract attention weights from last layer
   - Visualize which frames get high attention
   - Interpret: "Model focuses on frames 5-8 where object is stable"
   
3. Training:
   - Loss: Binary cross-entropy
   - Metrics: Precision, recall, F1
   - Target: >95% accuracy on held-out tracks

**Output:** Trained attention-based persistence classifier

---

### Worker 4: Pipeline Integration
**Branch:** `track-persistence/pipeline-integration`

**Goal:** Integrate persistence filter into `simple_3d_tracker.py`

**Tasks:**
1. Modify `simple_3d_tracker.py`:
   ```python
   # BEFORE (current):
   sensor1_track, sensor2_track = generate_synthetic_tracks()
   reconstructed_3d = triangulate_tracks(sensor1_track, sensor2_track, P1, P2)
   
   # AFTER (with persistence filter):
   sensor1_tracks, sensor2_tracks = generate_realistic_tracks()  # Multiple tracks
   
   # Filter each track
   filtered_sensor1 = []
   filtered_sensor2 = []
   for track1, track2 in zip(sensor1_tracks, sensor2_tracks):
       is_persistent = persistence_filter.predict(track1, track2)
       if is_persistent:
           filtered_sensor1.append(track1)
           filtered_sensor2.append(track2)
   
   # Only triangulate persistent tracks
   for track1, track2 in zip(filtered_sensor1, filtered_sensor2):
       reconstructed_3d = triangulate_tracks(track1, track2, P1, P2)
   ```
   
2. Create `PersistenceFilter` class:
   - `load_model(checkpoint)`
   - `extract_features(track)` using MagVIT
   - `predict(track1, track2)` → bool + attention weights
   - `get_attention_explanation()` → str

**Output:** Integrated system with filtering

---

### Worker 5: Multi-Sensor 3D Test Scenarios
**Branch:** `track-persistence/test-scenarios`

**Goal:** Test on realistic 3D tracking scenarios

**Tasks:**
1. Create test scenarios:
   - **Scenario 1: Clean scene** (2 persistent objects)
   - **Scenario 2: Cluttered scene** (5 objects, 10 transients)
   - **Scenario 3: Noisy detections** (3 objects, 20 false positives)
   
2. Metrics:
   - **Precision**: % of kept tracks that are truly persistent
   - **Recall**: % of persistent tracks that are kept
   - **3D Quality**: Reduction in spurious 3D points
   - **Runtime**: Filtering overhead
   
3. Visualizations:
   - Before/after 3D point clouds
   - Attention weights over time
   - Filter decisions with explanations

**Output:** Test results and visualizations

---

### Worker 6: LLM Attention Analysis
**Branch:** `track-persistence/llm-analysis`

**Goal:** Use LLM to explain attention patterns and suggest improvements

**Tasks:**
1. Create `llm_attention_analyzer.py`:
   - Collect attention weights from all tracks
   - Identify patterns:
     - "Persistent tracks have high attention on middle frames"
     - "Brief tracks show declining attention"
     - "Model ignores first 2 frames (initialization noise)"
   
2. Failure analysis:
   - Find false positives/negatives
   - Ask LLM: "Why did the model get this wrong?"
   - Generate hypotheses for improvement
   
3. Research questions:
   - "What visual features indicate persistence?"
   - "How does motion pattern affect classification?"
   - "When does the model fail?"

**Output:** LLM-generated insights and improvement suggestions

---

## Timeline

| Worker | Task | Duration |
|--------|------|----------|
| 1 | Realistic 2D tracks | 4 hours |
| 2 | MagVIT features | 3 hours |
| 3 | Transformer attention | 5 hours |
| 4 | Pipeline integration | 3 hours |
| 5 | Test scenarios | 4 hours |
| 6 | LLM analysis | 2 hours |

**Total (parallel):** ~6 hours wall-clock time

---

## Success Criteria

1. ✅ Takes realistic 2D tracks (not toy shapes)
2. ✅ Uses MagVIT visual features
3. ✅ Transformer attention classifies persistence
4. ✅ Integrated into `simple_3d_tracker.py`
5. ✅ Reduces spurious 3D points by >80%
6. ✅ LLM explains attention patterns
7. ✅ >95% precision/recall on test scenarios

---

## Ready to start?

All 6 workers will execute in parallel on separate branches.

