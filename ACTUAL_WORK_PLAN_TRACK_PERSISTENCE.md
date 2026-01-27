# Actual Work Plan: Track Persistence Integration

## What Was Actually Assigned

Integrate track persistence filtering into the **existing 3D reconstruction system** to filter out noise and keep only valid persistent tracks.

## What We Have vs What We Need

### ✅ What We Have
- Existing 3D tracking system (`simple_3d_tracker.py`)
- Stereo camera setup with 2D track generation
- MagVIT pretrained model (from previous work)
- Transformer architecture understanding
- Basic concept of persistence classification

### ❌ What We Built Instead (Wrong Direction)
- Toy synthetic 2D shapes (128x128 matplotlib dots)
- Model trained on artificial data
- No connection to real camera system
- No MagVIT visual features
- No integration with 3D pipeline

### 🎯 What Actually Needs to Be Built

## Phase 1: Real 2D Track Extraction
**Goal:** Get actual 2D tracks from the existing stereo camera system

**Tasks:**
1. Extract 2D track sequences from camera views
2. Get bounding boxes and pixel data over time
3. Format as video sequences for the model
4. Link to ground truth 3D positions

**Output:** Real camera track data, not synthetic shapes

---

## Phase 2: MagVIT Visual Feature Integration
**Goal:** Use MagVIT to extract visual features from track videos

**Tasks:**
1. Take 2D track bounding boxes over time (N frames)
2. Extract visual features using pretrained MagVIT encoder
3. Replace statistical features with visual embeddings
4. Feed to Transformer for persistence classification

**Output:** Model using actual visual features, not just statistics

---

## Phase 3: Transformer Attention for Persistence
**Goal:** Apply attention mechanism to identify persistent tracks

**Tasks:**
1. Transformer processes temporal sequence of MagVIT features
2. Attention weights show which frames matter for persistence
3. Classification: persistent (keep) vs non-persistent (filter)
4. Threshold tuning for precision/recall trade-off

**Output:** Attention-based persistence classifier

---

## Phase 4: Integration with 3D Reconstruction
**Goal:** Filter 2D tracks BEFORE 3D triangulation

**Tasks:**
1. Hook into existing `simple_3d_tracker.py`
2. For each 2D track pair (cam1, cam2):
   - Extract track video/features
   - Run through persistence model
   - Get persistence score + attention weights
3. Filter: Only triangulate persistent tracks
4. Result: Cleaner 3D reconstruction with less noise

**Output:** Integrated system with improved 3D tracking

---

## Phase 5: LLM Reasoning & Analysis
**Goal:** Use LLM to analyze attention patterns and performance

**Tasks:**
1. Collect attention weights from Transformer
2. Identify which temporal patterns indicate persistence
3. Analyze failure cases (false positives/negatives)
4. Generate insights about what visual/temporal cues matter
5. Suggest improvements based on learned patterns

**Output:** Interpretable system with LLM-driven insights

---

## Correct Architecture

```
┌─────────────────────────────────────────────────┐
│         Stereo Camera System                    │
│  (existing simple_3d_tracker.py)                │
└────────────┬────────────────────────────────────┘
             │
             ├──> Camera 1: 2D tracks (bounding boxes over time)
             └──> Camera 2: 2D tracks (bounding boxes over time)
                           │
                           ▼
             ┌─────────────────────────────┐
             │  Track Persistence Filter   │
             │  (NEW - what we need to     │
             │   actually build)           │
             └─────────────┬───────────────┘
                           │
                ┌──────────┴───────────┐
                │                      │
                ▼                      ▼
        ┌──────────────┐      ┌──────────────┐
        │   MagVIT     │      │  Statistical │
        │   Encoder    │      │  Features    │
        │  (visual)    │      │  (optional)  │
        └──────┬───────┘      └──────┬───────┘
               │                     │
               └──────────┬──────────┘
                          │
                          ▼
                ┌──────────────────┐
                │   Transformer    │
                │   (Attention)    │
                │  4 layers, 8     │
                │  heads           │
                └─────────┬────────┘
                          │
                          ▼
                ┌──────────────────┐
                │  Persistence     │
                │  Classification  │
                │  Head            │
                └─────────┬────────┘
                          │
                ┌─────────┴──────────┐
                │                    │
                ▼                    ▼
        KEEP (persistent)    FILTER (non-persistent)
                │
                ▼
    ┌───────────────────────┐
    │  3D Triangulation     │
    │  (only persistent     │
    │   tracks)             │
    └───────────┬───────────┘
                │
                ▼
    ┌───────────────────────┐
    │  Clean 3D Tracks      │
    │  (noise filtered)     │
    └───────────────────────┘
```

---

## Why Current Work Was Wrong

1. **No Real Data:** Trained on matplotlib dots, not camera footage
2. **No MagVIT:** Using only statistical features, not visual
3. **No Integration:** Standalone toy system, not connected to 3D pipeline
4. **No Real Testing:** 98.67% on synthetic shapes means nothing
5. **Wrong Problem:** Classifying dot patterns, not real track persistence

---

## Correct Success Criteria

1. ✅ Takes REAL 2D tracks from stereo cameras as input
2. ✅ Uses MagVIT visual features from track appearance
3. ✅ Transformer attention identifies persistent vs transient
4. ✅ Filters tracks BEFORE 3D triangulation
5. ✅ Measurably improves 3D reconstruction quality:
   - Fewer spurious 3D points
   - Smoother trajectories
   - Less noise
6. ✅ LLM analysis explains attention patterns
7. ✅ Integrated into existing `simple_3d_tracker.py`

---

## Next Steps (In Order)

### Step 1: Understand Existing System
- Read `simple_3d_tracker.py` to understand current pipeline
- Identify where 2D tracks are generated
- Find injection point for persistence filter

### Step 2: Extract Real 2D Tracks
- Modify system to save 2D track sequences
- Get bounding boxes + pixel data over time
- Create dataset from real camera runs

### Step 3: Integrate MagVIT
- Load pretrained MagVIT from previous work
- Extract features from 2D track videos
- Replace toy statistical features

### Step 4: Train on Real Data
- Generate persistence labels from track duration
- Train Transformer on MagVIT features
- Validate on held-out camera data

### Step 5: Deploy to Pipeline
- Integrate filter into `simple_3d_tracker.py`
- Test on real scenes
- Measure 3D reconstruction improvement

### Step 6: LLM Analysis
- Analyze attention patterns
- Explain what makes tracks persistent
- Generate improvement suggestions

---

## Timeline Estimate

- Step 1: 2 hours (understand existing code)
- Step 2: 4 hours (extract real tracks, create dataset)
- Step 3: 3 hours (MagVIT integration)
- Step 4: 6 hours (training on real data)
- Step 5: 4 hours (pipeline integration)
- Step 6: 2 hours (LLM analysis)

**Total: ~21 hours of actual work**

---

## Current Status

❌ Built toy proof-of-concept with synthetic shapes  
✅ Learned Transformer + persistence concept works  
🎯 **Need to rebuild with real data and MagVIT**

Ready to start with Step 1?

