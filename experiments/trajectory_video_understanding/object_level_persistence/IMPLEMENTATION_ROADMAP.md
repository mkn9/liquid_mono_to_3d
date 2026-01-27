# Object-Level Persistence Detection - Implementation Roadmap

**Date**: 2026-01-26  
**Status**: Planning Phase  
**Approach**: Test-Driven Development + Parallel Git Branches

---

## 🎯 Overview

Redesign from video-level classification to **object-level tracking and persistence detection**.

**Core Principle**: Track ALL objects through ALL frames, but allocate compute intelligently based on persistence.

---

## 📋 Implementation Phases

### **Phase 1: Object Detection** (Priority: HIGH)
**Goal**: Detect all sphere objects in each frame

**Tasks**:
1. ✅ Create TDD tests for object detector
2. ✅ Implement sphere detector (CNN-based)
3. ✅ Train on existing dataset (ground truth from metadata)
4. ✅ Validate: >95% detection rate

**Deliverables**:
- `object_detector.py` - Detection module
- `test_object_detector.py` - TDD tests
- Trained model: `sphere_detector.pt`
- Evidence: `artifacts/tdd_object_detector_*.txt`

**Estimated Time**: 2-3 days

---

### **Phase 2: Object Tracking** (Priority: HIGH)
**Goal**: Link detections across frames into tracks

**Tasks**:
1. ✅ Create TDD tests for tracker
2. ✅ Implement IoU-based tracker (simple, fast)
3. ✅ Test on videos with multiple objects
4. ✅ Validate: >90% track purity

**Deliverables**:
- `object_tracker.py` - Tracking module
- `test_object_tracker.py` - TDD tests
- Evidence: `artifacts/tdd_object_tracker_*.txt`

**Estimated Time**: 2-3 days

---

### **Phase 3: Track-Level Persistence Classifier** (Priority: HIGH)
**Goal**: Classify each track as persistent or transient

**Tasks**:
1. ✅ Create TDD tests for classifier
2. ✅ Extract track features (length, color, motion)
3. ✅ Implement LSTM/Transformer classifier
4. ✅ Train on labeled tracks
5. ✅ Validate: >90% classification accuracy

**Deliverables**:
- `track_persistence_classifier.py` - Classification module
- `test_track_classifier.py` - TDD tests
- Trained model: `track_classifier.pt`
- Evidence: `artifacts/tdd_track_classifier_*.txt`

**Estimated Time**: 3-4 days

---

### **Phase 4: Per-Object Attention Mechanism** (Priority: MEDIUM)
**Goal**: Allocate compute based on persistence scores

**Tasks**:
1. ✅ Create TDD tests for attention gating
2. ✅ Implement dynamic compute allocation
3. ✅ Integrate with feature extraction
4. ✅ Measure attention efficiency

**Deliverables**:
- `per_object_attention.py` - Attention module
- `test_per_object_attention.py` - TDD tests
- Evidence: `artifacts/tdd_attention_*.txt`

**Estimated Time**: 2 days

---

### **Phase 5: Integration & Pipeline** (Priority: HIGH)
**Goal**: End-to-end object-level tracking pipeline

**Tasks**:
1. ✅ Integrate all modules
2. ✅ Create end-to-end pipeline
3. ✅ Test on full dataset
4. ✅ Generate efficiency metrics

**Deliverables**:
- `object_level_pipeline.py` - Main pipeline
- `test_pipeline.py` - Integration tests
- Evidence: `artifacts/tdd_pipeline_*.txt`

**Estimated Time**: 2 days

---

### **Phase 6: Evaluation & Visualization** (Priority: MEDIUM)
**Goal**: Comprehensive evaluation and visualizations

**Tasks**:
1. ✅ Per-track accuracy metrics
2. ✅ Attention efficiency analysis
3. ✅ Visualization: Object tracks over time
4. ✅ Visualization: Attention heatmaps per object
5. ✅ Comprehensive report

**Deliverables**:
- Evaluation scripts
- Visualizations showing:
  - Object tracks colored by persistence score
  - Attention allocation per object
  - Compute savings breakdown
- Final report

**Estimated Time**: 2-3 days

---

## 🔧 Technical Decisions

### Object Detection Approach
**Decision**: Simple CNN with regression head for sphere centers

**Rationale**:
- Spheres are simple, consistent shapes
- Known size (don't need full object detection)
- Just need center positions + confidence
- Fast inference for real-time processing

**Alternative Considered**: Full YOLO/Faster-RCNN (overkill for spheres)

---

### Tracking Algorithm
**Decision**: IoU-based tracking with Kalman filter (simple SORT)

**Rationale**:
- Spheres have predictable motion
- IoU sufficient for matching across adjacent frames
- Kalman filter handles occlusion/missing detections
- Fast, proven approach

**Alternative Considered**: DeepSORT with re-identification (unnecessary complexity)

---

### Persistence Features
**Decision**: Combination of track length, visual features, motion

**Features**:
1. **Track length**: Primary indicator (transient = 1-3 frames)
2. **Color consistency**: White = persistent, Red = transient
3. **Motion smoothness**: Smooth = persistent, erratic = transient
4. **Appearance stability**: Consistent size/shape = persistent

**Why not just use track length?**: 
- Need to identify transients EARLY (frame 1-2)
- Track length alone requires waiting until track ends
- Visual/motion features enable early prediction

---

### Attention Mechanism
**Decision**: Simple gating based on persistence score

```python
if persistence_score > 0.8:
    compute_budget = 1.0  # Full
else:
    compute_budget = 0.2  # Minimal
```

**Rationale**:
- Simple, interpretable
- Direct control over compute allocation
- Can adjust thresholds based on efficiency targets

**Alternative Considered**: Learned attention (adds complexity, may not converge well)

---

## 📊 Key Metrics

### Detection Metrics
- **Detection Rate**: % objects detected per frame (target: >95%)
- **False Positive Rate**: Spurious detections (target: <5%)

### Tracking Metrics
- **Track Purity**: % detections correctly assigned (target: >90%)
- **Track Completeness**: % of object lifetime tracked (target: >95%)

### Classification Metrics
- **Accuracy**: Per-track classification (target: >90%)
- **Precision (Persistent)**: Avoid misclassifying transients as persistent (target: >95%)
- **Recall (Persistent)**: Catch all real objects (target: >98%)

### Efficiency Metrics
- **Attention Ratio**: Persistent/Transient attention (target: >3x)
- **Compute Savings**: % reduction from equal baseline (target: >40%)

### End-to-End Metrics
- **3D Reconstruction Accuracy**: For persistent tracks (target: matches current baseline)
- **False Track Rate**: Transient tracks in final output (target: <2%)

---

## 🔀 Git Workflow

### Branch Structure
```
main
└── early-persistence/magvit (current work)
    └── object-level-persistence (new redesign)
        ├── object-detection
        ├── object-tracking
        ├── track-classification
        ├── attention-mechanism
        └── integration
```

### Workflow
1. Create feature branch for each phase
2. TDD: Write tests first (RED)
3. Implement (GREEN)
4. Refactor (REFACTOR)
5. Capture evidence in `artifacts/`
6. Merge to `object-level-persistence`
7. Periodic sync to MacBook

---

## ⚙️ Standard Procedures (ALL APPLY)

✅ **TDD Process**: RED-GREEN-REFACTOR with evidence capture  
✅ **Periodic Saving**: Sync results to MacBook every 15 minutes  
✅ **Heartbeat Monitoring**: Progress updates during long runs  
✅ **Git Workflow**: Feature branches with proper commits  
✅ **EC2 Computation**: All training/evaluation on EC2  
✅ **Documentation**: Update docs with each phase  

---

## 🎯 Success Criteria (Phase-by-Phase)

### Phase 1 Success:
- [ ] Detector finds >95% of spheres
- [ ] <5% false positives
- [ ] TDD evidence captured

### Phase 2 Success:
- [ ] Tracks maintain >90% purity
- [ ] Objects tracked through >95% of lifetime
- [ ] TDD evidence captured

### Phase 3 Success:
- [ ] >90% classification accuracy per track
- [ ] Can identify transients by frame 3
- [ ] TDD evidence captured

### Phase 4 Success:
- [ ] Persistent tracks get 3x+ attention
- [ ] >40% compute savings measured
- [ ] TDD evidence captured

### Phase 5 Success:
- [ ] End-to-end pipeline runs on full dataset
- [ ] All metrics meet targets
- [ ] TDD evidence captured

### Phase 6 Success:
- [ ] Comprehensive visualizations generated
- [ ] Final report shows all metrics
- [ ] System ready for 3D reconstruction integration

---

## 📝 Questions to Answer Before Starting

1. **Dataset**: Use existing augmented dataset with transient spheres?
   - **Answer**: YES - already has ground truth

2. **Training Split**: How to split data?
   - **Proposal**: 70% train, 15% val, 15% test (by video, not frame)

3. **Compute Budget**: What thresholds for attention gating?
   - **Proposal**: Start with 1.0 (persistent) vs 0.2 (transient), tune later

4. **Integration**: How to integrate with existing 3D reconstruction?
   - **Proposal**: Output persistent tracks in same format as current system

5. **Timeline**: Acceptable timeline?
   - **Estimate**: 2-3 weeks for complete implementation

---

## 🚀 Next Immediate Steps

1. ✅ Review design document with user
2. ✅ Get approval to proceed
3. Create `object-level-persistence` git branch
4. Create TDD tests for Phase 1 (object detection)
5. Implement Phase 1
6. Iterate through phases

---

## ⚠️ Risks & Mitigation

### Risk 1: Detection accuracy insufficient
**Impact**: Poor tracking downstream  
**Mitigation**: Start with simple sphere detector, validate early

### Risk 2: Tracking fails with multiple objects
**Impact**: Confused tracks, wrong associations  
**Mitigation**: Test incrementally (1 object, 2 objects, 3 objects)

### Risk 3: Classification can't distinguish early
**Impact**: Need full track to classify  
**Mitigation**: Use visual features (color) as early indicator

### Risk 4: Integration breaks existing pipeline
**Impact**: Lost time on integration  
**Mitigation**: Maintain separate branch, test carefully before merge

---

## 📚 References

- SORT tracking: https://arxiv.org/abs/1602.00763
- Object detection for tracking: https://arxiv.org/abs/1703.07402
- Attention mechanisms: https://arxiv.org/abs/1706.03762

---

**Ready to proceed?** Let's start with Phase 1: Object Detection! 🎯

