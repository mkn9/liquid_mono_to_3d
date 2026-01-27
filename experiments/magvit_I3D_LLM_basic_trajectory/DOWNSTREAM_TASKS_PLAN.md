# Downstream Tasks Implementation Plan

**Date**: 2026-01-25  
**Status**: MAGVIT encoder/decoder trained (PSNR 25 dB)  
**Next**: Implement classification, generation, and prediction

---

## 🎯 THREE TASKS TO IMPLEMENT

### Task 1: Classification (Highest Priority) ⭐
**Goal**: Classify trajectories using MAGVIT codes  
**Why First**: Simplest, validates if codes preserve class information  
**Time**: ~2 hours with TDD  
**Difficulty**: ⭐☆☆ Easy

### Task 2: Generation (Medium Priority)
**Goal**: Generate new trajectory videos  
**Why Second**: Medium complexity, useful for data augmentation  
**Time**: ~3 hours with TDD  
**Difficulty**: ⭐⭐☆ Medium

### Task 3: Prediction (Lower Priority)
**Goal**: Predict future frames given past frames  
**Why Last**: Most complex, requires temporal modeling  
**Time**: ~4 hours with TDD  
**Difficulty**: ⭐⭐⭐ Hard

---

## 📋 TASK 1: CLASSIFICATION (RECOMMENDED START)

### Overview:
Train a classifier on MAGVIT codes to predict trajectory class (Linear/Circular/Helical/Parabolic)

### Why Start Here:
1. ✅ **Simplest** - Just need a classifier head
2. ✅ **Fast** - Can implement in 2 hours
3. ✅ **Validates codes** - Tells us if codes preserve class info
4. ✅ **Tests reconstruction** - If classifier works, 25 dB is sufficient

### Architecture:
```
Video → MAGVIT.encode() → Codes → Classifier → Class Label
         [frozen]                  [trainable]
```

### Implementation Steps (TDD):

#### Step 1: Write Tests FIRST (30 min)
```python
# test_classification.py

def test_extract_codes_from_trained_magvit():
    """Test extracting codes from trained MAGVIT encoder"""
    # Load trained MAGVIT
    # Extract codes for all videos
    # Verify codes shape and no NaNs
    
def test_classifier_trains_on_codes():
    """Test classifier can train on MAGVIT codes"""
    # Create simple classifier
    # Train on codes + labels
    # Verify loss decreases
    
def test_classification_accuracy_above_threshold():
    """Test classifier achieves >80% accuracy on test set"""
    # Train classifier
    # Evaluate on test set
    # Assert accuracy > 0.80
    
def test_per_class_accuracy_balanced():
    """Test all classes have reasonable accuracy"""
    # Compute per-class accuracy
    # Assert no class < 50% accuracy
```

#### Step 2: Run TDD RED Phase (5 min)
```bash
bash scripts/tdd_capture.sh
# Should fail - code doesn't exist yet
```

#### Step 3: Implement Classification (60 min)
```python
# classify_from_codes.py

class CodeClassifier(nn.Module):
    """Classifier on MAGVIT codes"""
    def __init__(self, code_dim, num_classes=4):
        super().__init__()
        # Pool codes (B, C, T, H, W) -> (B, C)
        self.pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        # Classify pooled features
        self.classifier = nn.Sequential(
            nn.Linear(code_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, codes):
        # codes: (B, C, T, H, W)
        pooled = self.pool(codes).squeeze()  # (B, C)
        return self.classifier(pooled)  # (B, num_classes)

def extract_all_codes(magvit_model, dataset):
    """Extract codes for all videos"""
    magvit_model.eval()
    all_codes = []
    with torch.no_grad():
        for videos, labels in dataloader:
            codes = magvit_model.encode(videos)
            all_codes.append(codes.cpu())
    return torch.cat(all_codes, dim=0)

def train_classifier(codes, labels, epochs=50):
    """Train classifier on codes"""
    # Split train/val
    # Create classifier
    # Training loop
    # Return trained classifier + metrics

def evaluate_classifier(classifier, codes, labels):
    """Evaluate classifier"""
    # Compute accuracy
    # Per-class accuracy
    # Confusion matrix
    # Return metrics dict
```

#### Step 4: Run TDD GREEN Phase (5 min)
```bash
bash scripts/tdd_capture.sh
# Should pass - implementation complete
```

#### Step 5: Run Full Classification (15 min)
```python
# Full pipeline:
# 1. Load trained MAGVIT
# 2. Extract codes for all 200 videos
# 3. Train classifier (160 train / 40 test)
# 4. Evaluate accuracy
# 5. Visualize confusion matrix
```

### Expected Results:
- **Accuracy**: 85-95% (classes are distinct)
- **Training time**: ~1 minute
- **If accuracy < 70%**: Codes don't preserve class info well

### Success Criteria:
- ✅ Overall accuracy > 80%
- ✅ All classes > 60% accuracy
- ✅ Confusion matrix shows clear diagonal

### If This Fails:
- MAGVIT reconstruction quality insufficient
- Need better training (longer, perceptual loss)
- Or codes don't separate classes

---

## 📋 TASK 2: GENERATION

### Overview:
Generate new trajectory videos by sampling codes and decoding

### Architecture:
```
Random Noise → Prior Model → Sampled Codes → MAGVIT.decode() → Generated Video
                 [trainable]                   [frozen]
```

### Two Approaches:

#### Approach A: Simple (Gaussian Prior)
**Time**: 1 hour  
**Complexity**: Low

```python
# Learn mean/std of code distribution
code_mean = codes.mean(dim=0)
code_std = codes.std(dim=0)

# Sample new codes
sampled_codes = torch.randn_like(code_mean) * code_std + code_mean

# Decode
generated_video = magvit.decode(sampled_codes)
```

**Pros**: Simple, fast  
**Cons**: May not respect discrete codebook structure

#### Approach B: Conditional (Class-Conditioned)
**Time**: 3 hours  
**Complexity**: Medium

```python
# Learn per-class code distributions
for class_id in range(4):
    class_codes = codes[labels == class_id]
    class_prior[class_id] = fit_distribution(class_codes)

# Generate for specific class
def generate(target_class):
    sampled_codes = class_prior[target_class].sample()
    return magvit.decode(sampled_codes)
```

**Pros**: Can control trajectory type  
**Cons**: More complex, needs per-class modeling

### Implementation Steps (TDD):

#### Step 1: Write Tests (30 min)
```python
def test_can_sample_from_code_distribution():
    """Test sampling from learned code prior"""
    
def test_sampled_codes_have_valid_shape():
    """Test sampled codes match expected dimensions"""
    
def test_generated_videos_are_valid():
    """Test generated videos have correct shape and range"""
    
def test_conditional_generation_produces_correct_class():
    """Test class-conditional generation produces requested class"""
    # Generate 10 videos for each class
    # Use trained classifier to verify class
    # Assert >70% match requested class
```

#### Step 2: Implement (1-2 hours)
- Fit code distribution
- Sample new codes
- Decode to videos
- Visualize generated samples

#### Step 3: Evaluate Quality
- Visual inspection
- Diversity (are samples different?)
- Realism (do they look like trajectories?)
- Class accuracy (if conditional)

### Expected Results:
- Generated videos should show trajectory-like motion
- If conditional: should match requested class
- Diversity: samples should be different from training data

### Success Criteria:
- ✅ Generated videos are valid (no NaN/Inf)
- ✅ Visual inspection: recognizable as trajectories
- ✅ If conditional: >70% match requested class

---

## 📋 TASK 3: PREDICTION

### Overview:
Predict future frames given past frames

### Architecture:
```
Past Frames → MAGVIT.encode() → Past Codes → Temporal Model → Future Codes → MAGVIT.decode() → Predicted Frames
              [frozen]                         [trainable]                     [frozen]
```

### Implementation Approaches:

#### Approach A: Simple Linear Predictor
**Time**: 2 hours  
**Complexity**: Medium

```python
# Predict future codes from past codes
class TemporalPredictor(nn.Module):
    def __init__(self, code_dim, num_future=4):
        self.predictor = nn.LSTM(
            input_size=code_dim,
            hidden_size=256,
            num_layers=2
        )
        self.fc = nn.Linear(256, code_dim * num_future)
    
    def forward(self, past_codes):
        # past_codes: (B, T_past, C)
        # output: (B, T_future, C)
```

#### Approach B: Transformer-Based
**Time**: 4 hours  
**Complexity**: High

```python
# Use transformer for temporal modeling
class TransformerPredictor(nn.Module):
    def __init__(self):
        self.transformer = nn.TransformerEncoder(...)
```

### Implementation Steps (TDD):

#### Step 1: Write Tests (45 min)
```python
def test_predictor_outputs_correct_shape():
    """Test predictor outputs future codes with correct shape"""
    
def test_predictor_loss_decreases():
    """Test predictor training reduces prediction error"""
    
def test_prediction_mse_below_threshold():
    """Test prediction MSE < 0.01 on test set"""
    
def test_predicted_frames_are_temporally_coherent():
    """Test predicted frames show smooth motion"""
```

#### Step 2: Prepare Data (30 min)
```python
# Create train/test splits for prediction
def create_prediction_dataset(videos, n_past=12, n_future=4):
    """
    Split videos into past/future
    videos: (N, T, C, H, W) where T=16
    Returns: (past_videos, future_videos)
    """
    past = videos[:, :n_past]  # First 12 frames
    future = videos[:, n_past:n_past+n_future]  # Next 4 frames
    return past, future
```

#### Step 3: Implement (2-3 hours)
- Extract codes for past frames
- Train temporal predictor
- Predict future codes
- Decode to frames
- Compute prediction error

#### Step 4: Evaluate
- MSE between predicted and actual future frames
- Visual comparison (side-by-side)
- Temporal coherence (no sudden jumps)

### Expected Results:
- **Prediction MSE**: 0.005-0.01 (worse than reconstruction)
- **Visual quality**: Reasonable but less sharp
- **Temporal coherence**: Should show smooth continuation

### Success Criteria:
- ✅ Prediction MSE < 2× reconstruction MSE
- ✅ Visual inspection: predictions look plausible
- ✅ No catastrophic failures (NaN/divergence)

---

## 🎯 RECOMMENDED IMPLEMENTATION ORDER

### Phase 1: Classification (Day 1, ~2 hours) ⭐⭐⭐
**Why First**: 
- Simplest to implement
- Validates if MAGVIT codes are useful
- Tests if 25 dB reconstruction is sufficient
- Fast feedback on whether approach works

**Decision Point**: 
- If accuracy > 80% → Proceed to Phase 2
- If accuracy < 70% → Improve MAGVIT first

### Phase 2: Generation (Day 1-2, ~3 hours) ⭐⭐
**Why Second**:
- Medium complexity
- Useful for data augmentation
- Validates code space is learnable
- Fun to visualize!

**Decision Point**:
- If generates recognizable trajectories → Proceed to Phase 3
- If generates noise → Revisit MAGVIT training

### Phase 3: Prediction (Day 2, ~4 hours) ⭐
**Why Last**:
- Most complex
- Requires temporal modeling
- Builds on classification/generation learnings
- Optional if first two work well

---

## 📊 ESTIMATED TIMELINE

### Option A: Sequential (Recommended for Learning)
```
Day 1 Morning:   Classification (2 hours)
Day 1 Afternoon: Generation (3 hours)
Day 2 Morning:   Prediction (4 hours)
Total: 9 hours
```

### Option B: Parallel (Faster but riskier)
```
Branch 1: Classification (2 hours)
Branch 2: Generation (3 hours)
Branch 3: Prediction (4 hours)
Total: 4 hours (parallel)
```

### Option C: Minimal Viable (Fastest)
```
Only Classification: 2 hours
- Proves MAGVIT codes work
- Validates approach
- Defer generation/prediction
```

---

## 💡 MY STRONG RECOMMENDATION

### **Start with Classification ONLY** (Option C)

**Why**:
1. **Fast**: 2 hours total
2. **Critical validation**: Tests if codes preserve class info
3. **Informs next steps**: If this fails, need better MAGVIT
4. **Low risk**: Simple implementation
5. **High value**: Answers "does this approach work?"

**Then decide**:
- Classification works (>80% acc) → Proceed to generation/prediction
- Classification fails (<70% acc) → Improve MAGVIT first

### Implementation Plan for Classification:

#### Immediate Next Steps:
```bash
1. Write test_classification.py (30 min)
   - Following TDD per cursorrules
   
2. Run TDD RED phase (5 min)
   - bash scripts/tdd_capture.sh
   - Capture failures
   
3. Implement classify_from_codes.py (60 min)
   - Extract codes
   - Train classifier
   - Evaluate accuracy
   
4. Run TDD GREEN phase (5 min)
   - Verify tests pass
   
5. Run full evaluation (15 min)
   - Train on 160 samples
   - Test on 40 samples
   - Report accuracy + confusion matrix

Total: ~2 hours
```

---

## 🚨 CRITICAL: FOLLOW TDD THIS TIME

**Per your requirements and cursorrules**:
1. ✅ Write tests FIRST
2. ✅ Run RED phase (capture failures)
3. ✅ Implement code
4. ✅ Run GREEN phase (capture passes)
5. ✅ Refactor if needed

**Don't repeat mistake**: We trained MAGVIT without proper TDD. Let's do classification correctly!

---

## 📋 SUCCESS METRICS

### Classification:
- [ ] Overall accuracy > 80%
- [ ] All classes > 60% accuracy
- [ ] Clear confusion matrix diagonal
- [ ] TDD evidence captured

### Generation:
- [ ] Generated videos are valid
- [ ] Visual quality recognizable
- [ ] If conditional: >70% class match
- [ ] Samples are diverse

### Prediction:
- [ ] Prediction MSE < 0.01
- [ ] Temporal coherence maintained
- [ ] Visual quality acceptable
- [ ] No catastrophic failures

---

## 🎯 FINAL RECOMMENDATION

**START NOW: Classification with proper TDD**

1. Write tests for classification (30 min)
2. Run TDD RED phase (5 min)
3. Implement classifier (60 min)
4. Run TDD GREEN phase (5 min)
5. Evaluate results (15 min)

**Total: 2 hours to answer**: "Do MAGVIT codes enable trajectory classification?"

**Then we'll know**:
- If 25 dB reconstruction is sufficient
- If codes preserve class information
- Whether to proceed with generation/prediction
- Or if we need better MAGVIT first

**Shall I proceed with implementing classification with full TDD?**

