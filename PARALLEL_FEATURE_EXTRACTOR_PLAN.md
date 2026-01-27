# Parallel Feature Extractor Development Plan

**Date**: 2026-01-25  
**Status**: READY TO IMPLEMENT  
**Goal**: Compare multiple video feature extractors for trajectory classification/prediction

---

## 🎯 Overview

### **Core Strategy**
Implement **4 parallel branches** with different feature extractors, keeping rest of pipeline identical:

```
Common Pipeline:
  Input: Video (T, H, W, 3)
         ↓
  [SWAPPABLE FEATURE EXTRACTOR] ← Different per branch
         ↓
  Features: (T, D)
         ↓
  Transformer Backbone (shared architecture)
         ↓
  Task Heads: Classification, Prediction, Generation
```

### **Why This Approach?**
✅ **Fair comparison** - Same data, same tasks, only feature extractor changes  
✅ **Parallel development** - No blocking, multiple workers  
✅ **Best-of-breed** - Can choose winner or ensemble  
✅ **Risk mitigation** - If one approach fails, others continue  

---

## 📊 Feature Extractor Options

### **Option 1: MagVIT** (Already Working)
**Branch**: `feature-extractors/magvit`

**Architecture**:
- Pretrained MagVIT-2 VQ-VAE encoder
- Input: (T, 3, H, W) → Output: (T, 256)
- Captures appearance + motion via learned codebook

**Pros**:
- ✅ Already implemented and tested
- ✅ State-of-art for video generation
- ✅ Compact representations

**Cons**:
- ⚠️ Requires pretrained model
- ⚠️ Not specifically trained for trajectories

**Estimated Effort**: 1 day (refinement only)

---

### **Option 2: I3D (Inflated 3D ConvNets)**
**Branch**: `feature-extractors/i3d`

**Architecture**:
- I3D pretrained on Kinetics-400/600
- Input: (T, H, W, 3) → Output: (T, 1024)
- 3D convolutions capture spatiotemporal patterns

**Pros**:
- ✅ Strong baseline for video understanding
- ✅ Pretrained on action recognition
- ✅ Well-established architecture

**Cons**:
- ⚠️ Heavy computation (3D convs)
- ⚠️ Large model size (~100MB)

**Implementation**:
```python
import torch
import torch.nn as nn
from torchvision.models.video import i3d_r50

class I3DFeatureExtractor:
    def __init__(self, pretrained=True):
        self.model = i3d_r50(pretrained=pretrained)
        # Remove classification head, keep features
        self.features = nn.Sequential(*list(self.model.children())[:-1])
        
    def extract(self, video):
        # video: (B, T, C, H, W)
        with torch.no_grad():
            features = self.features(video)  # (B, T, 1024)
        return features
```

**Estimated Effort**: 2-3 days

---

### **Option 3: Slow/Fast Networks**
**Branch**: `feature-extractors/slowfast`

**Architecture**:
- Two-pathway: Slow (semantic) + Fast (motion)
- Slow: Low frame rate, high spatial resolution
- Fast: High frame rate, low spatial resolution
- Lateral connections fuse pathways

**Pros**:
- ✅ Excellent motion understanding
- ✅ Efficient (parallel pathways)
- ✅ State-of-art on action recognition

**Cons**:
- ⚠️ More complex implementation
- ⚠️ Requires careful frame sampling

**Implementation**:
```python
from pytorchvideo.models import slowfast_r50

class SlowFastFeatureExtractor:
    def __init__(self, pretrained=True):
        self.model = slowfast_r50(pretrained=pretrained)
        # Remove head, keep backbone
        
    def extract(self, video):
        # Prepare slow/fast inputs
        slow_pathway = video[:, ::4]  # Every 4th frame
        fast_pathway = video[:, ::1]  # Every frame
        
        features = self.model([slow_pathway, fast_pathway])
        return features  # (B, T, 2304) [slow+fast concat]
```

**Estimated Effort**: 3-4 days

---

### **Option 4: Basic Transformer (From Scratch)**
**Branch**: `feature-extractors/transformer`

**Architecture**:
- Lightweight 3D patch embedding
- Simple transformer encoder (4 layers)
- Trained from scratch on our data

**Pros**:
- ✅ Fully trainable on our task
- ✅ No pretrained model dependency
- ✅ Lightweight and fast
- ✅ Good learning exercise

**Cons**:
- ⚠️ May need more training data
- ⚠️ Might underperform pretrained models initially

**Implementation**:
```python
class BasicTransformerFeatureExtractor(nn.Module):
    def __init__(self, patch_size=8, d_model=256, n_layers=4):
        super().__init__()
        # 3D patch embedding
        self.patch_embed = nn.Conv3d(3, d_model, 
                                      kernel_size=(2, patch_size, patch_size),
                                      stride=(2, patch_size, patch_size))
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead=8)
        self.transformer = nn.TransformerEncoder(encoder_layer, n_layers)
        
    def forward(self, video):
        # video: (B, T, C, H, W) → (B, C, T, H, W)
        video = video.permute(0, 2, 1, 3, 4)
        
        # Patch embedding: (B, d_model, T', H', W')
        features = self.patch_embed(video)
        
        # Flatten spatial dims: (B, d_model, T' * H' * W')
        B, D, T, H, W = features.shape
        features = features.view(B, D, T * H * W).permute(2, 0, 1)
        
        # Transformer: (T*H*W, B, D)
        features = self.transformer(features)
        
        # Pool to (T, B, D)
        features = features.view(T, H * W, B, D).mean(dim=1)
        
        return features.permute(1, 0, 2)  # (B, T, D)
```

**Estimated Effort**: 4-5 days

---

## 🌳 Git Branch Strategy

### **Branch Structure**

```
master (current working state)
│
├── feature-extractors/magvit          [Worker 1]
│   └── Refine MagVIT integration
│
├── feature-extractors/i3d             [Worker 2]
│   └── Implement I3D extractor
│
├── feature-extractors/slowfast        [Worker 3]
│   └── Implement Slow/Fast extractor
│
├── feature-extractors/transformer     [Worker 4]
│   └── Implement basic transformer
│
└── data/generate-10k                  [Worker 5]
    └── Generate additional 10K samples
```

### **Branch Naming Convention**

```
feature-extractors/<extractor-name>
data/<task-name>
```

### **Each Branch Contains**

```
experiments/trajectory_classification/
├── feature_extractors/
│   ├── base.py              # Abstract base class
│   ├── magvit_extractor.py  # Option 1
│   ├── i3d_extractor.py     # Option 2
│   ├── slowfast_extractor.py # Option 3
│   └── transformer_extractor.py # Option 4
├── models/
│   └── unified_model.py     # Common transformer + task heads
├── train.py                 # Training script (same for all)
├── evaluate.py              # Evaluation script (same for all)
└── tests/
    └── test_feature_extractor.py
```

---

## 📋 Implementation Plan

### **Phase 1: Setup (Week 1)**

#### **Day 1: Architecture Foundation**
- [ ] Create abstract `FeatureExtractor` base class
- [ ] Define common interface: `.extract(video) → features`
- [ ] Create unified training script
- [ ] Write comprehensive tests for base class

#### **Day 2: Branch Creation**
- [ ] Create all 5 git branches from master
- [ ] Set up directory structure in each
- [ ] Create README in each branch explaining approach

### **Phase 2: Parallel Development (Week 1-2)**

#### **Worker 1: MagVIT** (1 day)
```bash
git checkout -b feature-extractors/magvit
# Refactor existing MagVIT code into new structure
# Test compatibility with unified pipeline
# Baseline results
```

#### **Worker 2: I3D** (2-3 days)
```bash
git checkout -b feature-extractors/i3d
# Install pytorchvideo or torchvision.models.video
# Implement I3DFeatureExtractor
# Test on sample data
# Full training run
```

#### **Worker 3: Slow/Fast** (3-4 days)
```bash
git checkout -b feature-extractors/slowfast
# Install pytorchvideo
# Implement SlowFastFeatureExtractor
# Handle dual-pathway input preparation
# Full training run
```

#### **Worker 4: Basic Transformer** (4-5 days)
```bash
git checkout -b feature-extractors/transformer
# Implement from scratch
# Test patch embedding
# Train from scratch (needs more epochs)
# Compare to pretrained options
```

#### **Worker 5: Data Generation** (2-3 days)
```bash
git checkout -b data/generate-10k
# Use existing parallel generator with checkpoints
# Generate 10,000 samples (4 trajectory classes)
# Validate data quality
# Merge into shared dataset directory
```

### **Phase 3: Evaluation & Comparison (Week 2-3)**

#### **Unified Evaluation**
```python
# experiments/trajectory_classification/compare_extractors.py

extractors = {
    'magvit': MagVITExtractor(),
    'i3d': I3DExtractor(),
    'slowfast': SlowFastExtractor(),
    'transformer': TransformerExtractor()
}

results = {}
for name, extractor in extractors.items():
    model = UnifiedModel(extractor)
    results[name] = evaluate_all_tasks(model, test_data)
    
# Compare:
# - Classification accuracy
# - Prediction MSE
# - Generation quality (FVD)
# - Inference speed
# - Model size
```

---

## 📊 Comparison Metrics

### **Task Performance**

| Metric | MagVIT | I3D | Slow/Fast | Transformer |
|--------|--------|-----|-----------|-------------|
| **Classification Acc** | TBD | TBD | TBD | TBD |
| **Prediction MSE** | TBD | TBD | TBD | TBD |
| **Generation FVD** | TBD | TBD | TBD | TBD |

### **Efficiency**

| Metric | MagVIT | I3D | Slow/Fast | Transformer |
|--------|--------|-----|-----------|-------------|
| **Inference (ms)** | TBD | TBD | TBD | TBD |
| **Model Size (MB)** | TBD | TBD | TBD | TBD |
| **Training Time** | TBD | TBD | TBD | TBD |

### **Feature Quality**

| Metric | MagVIT | I3D | Slow/Fast | Transformer |
|--------|--------|-----|-----------|-------------|
| **Feature Dim** | 256 | 1024 | 2304 | 256 |
| **Pretrained?** | Yes | Yes | Yes | No |
| **Domain** | Video gen | Actions | Actions | Ours |

---

## 🎓 Recommendations for Proceeding

### **✅ 1. Start with Modular Architecture (Critical)**

**Create Abstract Base First:**
```python
# experiments/trajectory_classification/feature_extractors/base.py

from abc import ABC, abstractmethod
import torch.nn as nn

class FeatureExtractor(ABC):
    """Abstract base for all feature extractors."""
    
    @abstractmethod
    def extract(self, video: torch.Tensor) -> torch.Tensor:
        """
        Extract features from video.
        
        Args:
            video: (B, T, C, H, W) or (B, T, H, W, C)
            
        Returns:
            features: (B, T, D) where D is feature dimension
        """
        pass
    
    @property
    @abstractmethod
    def feature_dim(self) -> int:
        """Return feature dimension D."""
        pass
    
    @abstractmethod
    def preprocess(self, video: torch.Tensor) -> torch.Tensor:
        """Preprocess video for this extractor."""
        pass
```

**Why This Matters:**
- ✅ All extractors have same interface
- ✅ Easy to swap in unified pipeline
- ✅ Fair comparison (same inputs/outputs)
- ✅ TDD: Test interface compliance

---

### **✅ 2. Recommended Development Order**

#### **Priority 1: MagVIT (Week 1, Day 1-2)**
- Already working, quick win
- Establishes baseline
- Tests unified pipeline works

#### **Priority 2: Basic Transformer (Week 1, Day 3-5)**
- No external dependencies
- Good learning experience
- Fully trainable on our data
- Lightweight

#### **Priority 3: I3D (Week 2, Day 1-3)**
- Well-established baseline
- Easy to implement (torchvision)
- Strong performance expected

#### **Priority 4: Slow/Fast (Week 2, Day 4-7)**
- Most complex
- Highest potential performance
- Good if time permits

#### **Parallel: Data Generation (Anytime)**
- Independent of feature extractor work
- Can run on EC2 while development happens
- 10K samples: ~1-2 hours with parallel generator

---

### **✅ 3. Testing Strategy**

#### **Unit Tests (Per Extractor)**
```python
def test_feature_extractor_interface():
    """Test extractor follows interface."""
    extractor = I3DFeatureExtractor()
    
    # Test input/output shapes
    video = torch.randn(2, 16, 3, 64, 64)  # B=2, T=16
    features = extractor.extract(video)
    
    assert features.shape == (2, 16, extractor.feature_dim)
    assert torch.all(torch.isfinite(features))

def test_feature_extractor_determinism():
    """Test same input → same output."""
    extractor = I3DFeatureExtractor()
    video = torch.randn(1, 16, 3, 64, 64)
    
    features1 = extractor.extract(video)
    features2 = extractor.extract(video)
    
    torch.testing.assert_close(features1, features2)
```

#### **Integration Tests (Cross-Branch)**
```python
def test_all_extractors_compatible():
    """Test all extractors work with unified pipeline."""
    extractors = [
        MagVITExtractor(),
        I3DExtractor(),
        SlowFastExtractor(),
        TransformerExtractor()
    ]
    
    video = load_test_video()
    
    for extractor in extractors:
        features = extractor.extract(video)
        model = UnifiedModel(extractor)
        output = model(video)
        assert output['classification'].shape == (1, 4)  # 4 classes
```

---

### **✅ 4. Data Strategy**

#### **Current Data**
- ~3-5K samples from previous work
- Need more for robust training

#### **10K Generation Plan**
```bash
# On EC2 (parallel, with checkpoints)
cd experiments/trajectory_classification
./scripts/run_with_keepalive.sh "python generate_10k_dataset.py \
    --output data/10k_trajectories \
    --checkpoint-interval 2000 \
    --workers 4"
```

**Timeline:**
- 10K samples: ~2 hours (with 4 workers)
- Validation: 30 min
- **Total: 2.5 hours**

#### **When to Generate 30K?**
Wait until after Phase 1 comparison:
- If models overfit on 10K → need 30K
- If 10K is sufficient → save time/compute
- If one approach clearly wins → generate 30K for that approach only

---

### **✅ 5. Deferring LLM Integration (Smart Decision)**

**Why Defer?**
✅ **Focus**: Get core pipeline working first  
✅ **Independence**: Feature extraction doesn't need LLM  
✅ **Clear milestones**: Complete Phase 1, then add LLM  
✅ **Risk management**: Reduce complexity in initial implementation  

**LLM Phase (Later):**
```
Phase 1: Feature Extraction + Tasks ← DO THIS FIRST
    ↓
    (Validate, compare, choose winner)
    ↓
Phase 2: LLM Integration ← DO AFTER PHASE 1
    - Take attention weights from Phase 1
    - Feed to GPT-4/local LLM
    - Generate natural language explanations
    - Already designed (llm_attention_analyzer.py exists)
```

**Timeline:**
- Phase 1: 2-3 weeks
- Gap: 1 week (analysis, documentation)
- Phase 2: 1 week (LLM integration)

---

## 📅 Detailed Timeline

### **Week 1: Foundation + Parallel Start**

| Day | Worker 1 | Worker 2 | Worker 3 | Worker 4 | Worker 5 |
|-----|----------|----------|----------|----------|----------|
| Mon | Abstract base | Branch setup | Branch setup | Branch setup | Generate 10K |
| Tue | MagVIT refactor | I3D research | SlowFast research | Transformer design | (generating) |
| Wed | MagVIT tests | I3D implement | SlowFast implement | Transformer implement | Validate data |
| Thu | MagVIT train | I3D test | SlowFast test | Transformer test | Merge to shared |
| Fri | MagVIT eval | I3D train | SlowFast train | Transformer train | Document |

### **Week 2: Training + Comparison**

| Day | All Workers |
|-----|-------------|
| Mon | Continue training |
| Tue | Monitor convergence |
| Wed | Evaluate all models |
| Thu | Compare results |
| Fri | Document findings |

### **Week 3: Analysis + Next Steps**

| Day | Task |
|-----|------|
| Mon | Choose winner or ensemble |
| Tue | Ablation studies |
| Wed | Document architecture decisions |
| Thu | Plan Phase 2 (LLM integration) |
| Fri | Begin Phase 2 or 30K generation |

---

## 🎯 Success Criteria

### **Phase 1 Complete When:**
- [ ] All 4 extractors implemented
- [ ] All pass interface tests
- [ ] All trained on 10K dataset
- [ ] Evaluation metrics collected
- [ ] Comparison report generated
- [ ] Winner(s) identified
- [ ] Documentation complete

### **Metrics to Compare:**
1. **Task Performance**
   - Classification accuracy (primary)
   - Prediction MSE
   - Generation quality

2. **Efficiency**
   - Inference speed
   - Training time
   - Model size

3. **Practical**
   - Implementation complexity
   - Dependency requirements
   - Debugability

---

## 🚀 Getting Started (Next Steps)

### **Immediate Actions:**

1. **Create Base Architecture** (1-2 hours)
   ```bash
   mkdir -p experiments/trajectory_classification/feature_extractors
   # Create base.py with abstract interface
   # Write tests for interface compliance
   ```

2. **Create Git Branches** (30 min)
   ```bash
   git checkout -b feature-extractors/magvit
   git checkout master
   git checkout -b feature-extractors/i3d
   # ... etc
   ```

3. **Start Data Generation** (Background)
   ```bash
   # On EC2
   ./scripts/run_on_ec2_with_keepalive.sh \
     "cd mono_to_3d/experiments/trajectory_classification && \
      python generate_10k_dataset.py"
   ```

4. **Assign Workers**
   - Worker 1: MagVIT (you or me?)
   - Worker 2: I3D
   - Worker 3: Slow/Fast
   - Worker 4: Basic Transformer
   - Worker 5: Data generation (EC2 background)

---

## 💡 Key Insights

### **Why Parallel Development Works Here:**
✅ **Independent**: Each extractor can be developed separately  
✅ **Same interface**: Easy to compare and swap  
✅ **Fast feedback**: See which approach works best  
✅ **Risk hedging**: Multiple bets, choose winner  

### **Why Defer LLM:**
✅ **Separation of concerns**: Feature extraction ≠ explanation  
✅ **Test separately**: Validate numerical tasks first  
✅ **Layer B design**: LLM only interprets, doesn't compute  
✅ **Already designed**: Worker 6 from Jan 18 ready to use  

### **Why 10K Before 30K:**
✅ **Faster iteration**: 2 hours vs ~6 hours  
✅ **Sufficient for comparison**: Can see trends  
✅ **Resource efficient**: Don't waste compute if not needed  
✅ **Progressive scaling**: Add more data only if needed  

---

## ✅ Final Recommendation

**START WITH:**
1. ✅ Create abstract base class (today)
2. ✅ Implement MagVIT extractor first (tomorrow)
3. ✅ Start 10K generation on EC2 (parallel, tonight)
4. ✅ Implement basic transformer (this week)
5. ✅ Add I3D and Slow/Fast (next week)

**DEFER:**
- ⏸️ LLM integration (until Phase 1 complete)
- ⏸️ 30K generation (until we know we need it)

**EXPECTED OUTCOME:**
- Week 2-3: Have 4 working extractors
- Week 3: Clear comparison data
- Week 3-4: Choose winner, proceed to LLM phase

---

**Ready to start? Want me to:**
1. Create the abstract base class code?
2. Generate 10K samples on EC2 now?
3. Implement MagVIT extractor in new structure?

Let me know where you'd like to begin! 🚀

