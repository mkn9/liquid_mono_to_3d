# Next Steps Options for Development

**Date:** 2026-01-24  
**Current State Review:** After implementing proof bundle system

---

## Executive Summary

**What's Proven (via TDD):**
- ✅ `TrajectoryRenderer` - Generates 2D video frames from 3D trajectories
- ✅ `DatasetGenerator` - Creates synthetic trajectory datasets
- ✅ `EvaluationMetrics` - Classification accuracy, RMSE, symbolic equation similarity

**What's Claimed But Not Real:**
- ❌ MAGVIT integration (basic Conv3d, not actual VQ-VAE)
- ❌ CLIP integration (completely absent)
- ❌ GPT-4/Mistral/Phi-2 integration (templates, no API calls)
- ❌ True I3D/SlowFast (oversimplified models)
- ❌ Parallel git branches (never used)

**Current Proof Bundle Status:**
- `bash scripts/prove.sh` → Exit 0 ✅
- All foundation tests pass
- No component contracts enabled (would expose lies)

---

## Option 1: Continue with Simplified Models (Honest About Scope)

**Description:** Accept that this is a proof-of-concept using simplified components, but be honest about it.

### Actions:
1. **Rename components to reflect reality:**
   - `simple_model.py` → `simple_3dcnn_classifier.py`
   - Remove "MAGVIT", "I3D", "SlowFast" from documentation
   - Update `BRANCH_SPECIFICATIONS.md` to say "Simple 3D CNN"

2. **Fix LLM integration with templates explicitly labeled:**
   ```python
   def generate_description(trajectory_class, params):
       """Generate description using TEMPLATE (not actual LLM).
       
       NOTE: This is a placeholder for future LLM integration.
       Real LLM integration requires: pip install openai + API key
       """
       return template_based_description(trajectory_class, params)
   ```

3. **Add honest contract:**
   ```yaml
   # contracts/simple_baseline.yaml
   name: "simple_baseline_model"
   description: "Simplified 3D CNN baseline (not claiming MAGVIT/CLIP)"
   tests:
     - cmd: "pytest tests/test_simple_baseline.py -v"
   ```

4. **Run prove.sh to validate honesty**

### Pros:
- ✅ Fast path forward
- ✅ Honest about capabilities
- ✅ Foundation already TDD-validated
- ✅ Can iterate and improve later
- ✅ Establishes baseline performance

### Cons:
- ⚠️ Not using state-of-art components (yet)
- ⚠️ Limited to simple classification/forecasting
- ⚠️ No symbolic equation generation (templates only)

### Timeline:
- **Week 1:** Honest renaming + documentation
- **Week 2:** Baseline performance validation
- **Week 3:** Analysis and reporting

### Proof Bundle Gate:
```bash
# Enable honest baseline contract
mv contracts/simple_baseline.yaml.DISABLED contracts/simple_baseline.yaml
bash scripts/prove.sh  # Must pass
```

---

## Option 2: Implement Real MAGVIT (Research-Grade)

**Description:** Build actual MAGVIT VQ-VAE integration for video compression/representation.

### Actions:
1. **Research & select MAGVIT implementation:**
   - Option A: Use existing PyTorch MAGVIT implementation (e.g., from GitHub)
   - Option B: Implement from paper (CVPR 2023)
   - Requires: VQ-VAE architecture, codebook, encoder/decoder

2. **TDD workflow for MAGVIT:**
   ```python
   # test_magvit_integration.py
   def test_magvit_encodes_video_to_codes():
       """MAGVIT must compress video to discrete codes."""
       video = torch.randn(1, 16, 3, 64, 64)  # B, T, C, H, W
       magvit = MAGVIT_VQ_VAE(codebook_size=1024)
       codes = magvit.encode(video)
       assert codes.shape == (1, 16, 8, 8)  # Compressed
       assert codes.dtype == torch.long  # Discrete codes
   
   def test_magvit_decodes_codes_to_video():
       """MAGVIT must reconstruct video from codes."""
       codes = torch.randint(0, 1024, (1, 16, 8, 8))
       magvit = MAGVIT_VQ_VAE(codebook_size=1024)
       reconstructed = magvit.decode(codes)
       assert reconstructed.shape == (1, 16, 3, 64, 64)
   
   def test_magvit_reconstruction_quality():
       """MAGVIT reconstruction should preserve trajectory motion."""
       # Generate trajectory video
       trajectory_3d = generate_circular_trajectory(n_frames=16)
       original_video = render_trajectory(trajectory_3d)
       
       # Encode/decode
       magvit = MAGVIT_VQ_VAE(codebook_size=1024)
       codes = magvit.encode(original_video)
       reconstructed = magvit.decode(codes)
       
       # Check reconstruction quality (PSNR > 20 dB)
       psnr = calculate_psnr(original_video, reconstructed)
       assert psnr > 20.0
   ```

3. **Enable MAGVIT contract:**
   ```bash
   mv contracts/magvit_integration.yaml.DISABLED contracts/magvit_integration.yaml
   ```

4. **Train MAGVIT on trajectory videos:**
   - Use existing trajectory dataset (1200 samples)
   - Train VQ-VAE for video compression
   - Validate reconstruction quality

5. **Integrate with classification pipeline:**
   - Replace simple 3D CNN with MAGVIT encoder
   - Use discrete codes as input to classifier
   - Measure performance improvement

### Pros:
- ✅ Real research-grade component
- ✅ Learned video representations (not hand-crafted)
- ✅ Potential for generation (not just classification)
- ✅ Publishable if done correctly

### Cons:
- ⚠️ Complex implementation (VQ-VAE, codebook, training)
- ⚠️ Requires GPU resources for training
- ⚠️ May not improve classification (could be overkill)
- ⚠️ Longer development timeline (2-4 weeks)

### Timeline:
- **Week 1:** Research, select implementation, RED phase tests
- **Week 2:** GREEN phase implementation, basic encode/decode working
- **Week 3:** REFACTOR, train on trajectory data
- **Week 4:** Integration with classifier, validation

### Proof Bundle Gate:
```bash
bash scripts/prove.sh
# Will fail until:
# - from magvit2 import MAGVIT_VQ_VAE works
# - tests/test_magvit_integration.py passes
# - Reconstruction quality meets threshold
```

### Resource Requirements:
- GPU: Tesla T4 or better (for VQ-VAE training)
- Time: 2-4 weeks development + 1 week training
- References: MAGVIT paper, existing implementations

---

## Option 3: Implement Real LLM Integration (GPT-4 or Local)

**Description:** Add actual LLM calls for symbolic equation generation and natural language descriptions.

### Actions:

#### Option 3A: GPT-4 API Integration

1. **TDD workflow for GPT-4:**
   ```python
   # test_gpt4_integration.py
   @pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="No API key")
   def test_gpt4_generates_symbolic_equation():
       """GPT-4 must generate symbolic equation from trajectory features."""
       from openai import OpenAI
       
       trajectory_features = {
           "class": "circular",
           "center": [0.0, 0.0, 1.0],
           "radius": 0.5,
           "period": 2.0
       }
       
       client = OpenAI()
       response = client.chat.completions.create(
           model="gpt-4",
           messages=[{
               "role": "user",
               "content": f"Generate symbolic equation for circular trajectory: {trajectory_features}"
           }]
       )
       
       equation = response.choices[0].message.content
       assert "cos" in equation or "sin" in equation  # Circular motion
       assert len(equation) > 20  # Non-trivial response
   
   def test_gpt4_generates_natural_language_description():
       """GPT-4 must generate human-readable trajectory description."""
       # Similar test for NL description
       ...
   ```

2. **Enable GPT-4 contract:**
   ```bash
   mv contracts/gpt4_integration.yaml.DISABLED contracts/gpt4_integration.yaml
   ```

3. **Cost estimation:**
   - GPT-4: ~$0.03/1K tokens input, $0.06/1K tokens output
   - ~500 tokens per trajectory analysis
   - 1200 trajectories = ~$50-100 total

4. **Implementation:**
   - Replace template functions with actual API calls
   - Add retry logic, error handling
   - Cache responses to avoid redundant calls
   - Add cost tracking

#### Option 3B: Local LLM (Mistral/Phi-2)

1. **Setup local LLM:**
   ```bash
   pip install transformers accelerate
   # Download Mistral-7B-Instruct or Phi-2
   ```

2. **TDD workflow:**
   ```python
   def test_local_llm_loads_successfully():
       """Local LLM must load without errors."""
       from transformers import AutoModelForCausalLM, AutoTokenizer
       
       model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
       tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
       
       assert model is not None
       assert tokenizer is not None
   
   def test_local_llm_generates_equation():
       """Local LLM must generate equation given prompt."""
       # Similar structure to GPT-4 test
       ...
   ```

3. **Pros/Cons:**
   - ✅ No API costs
   - ✅ Full control, privacy
   - ⚠️ Requires 16GB+ VRAM (Mistral-7B)
   - ⚠️ Slower inference than GPT-4
   - ⚠️ May need prompt engineering for good results

### Pros (Either Option):
- ✅ Real LLM integration (not templates)
- ✅ Can generate novel equations/descriptions
- ✅ Proves concept works end-to-end
- ✅ Publishable/presentable results

### Cons:
- ⚠️ API costs (Option 3A) or GPU requirements (Option 3B)
- ⚠️ Rate limits, error handling complexity
- ⚠️ Output validation needed (LLMs can hallucinate)
- ⚠️ May not significantly improve classification accuracy

### Timeline:
- **Week 1:** RED phase tests, API/model setup
- **Week 2:** GREEN phase implementation, basic calls working
- **Week 3:** REFACTOR, batch processing, caching
- **Week 4:** Full dataset processing, validation

### Proof Bundle Gate:
```bash
bash scripts/prove.sh
# Will fail until:
# - Actual API calls work (or local model loads)
# - Generated equations are non-template
# - All tests pass with real LLM responses
```

---

## Option 4: Parallel Git Branch Development (Done Right)

**Description:** Actually use git branches for parallel exploration (last time this was claimed but not done).

### Actions:

1. **Create actual git branches:**
   ```bash
   # Start from proven foundation
   git checkout -b exp/simple-baseline
   git checkout -b exp/magvit-integration
   git checkout -b exp/gpt4-integration
   git checkout -b exp/clip-integration
   ```

2. **Branch-specific contracts:**
   ```yaml
   # contracts/branch_simple_baseline.yaml
   name: "branch_simple_baseline"
   forbidden_patterns:
     - "MAGVIT"
     - "GPT-4"
     - "CLIP"
   
   # contracts/branch_magvit.yaml
   name: "branch_magvit"
   imports:
     - "from magvit2 import MAGVIT_VQ_VAE"
   tests:
     - "pytest tests/test_magvit.py"
   ```

3. **Enforcement:**
   - Each branch must pass `bash scripts/prove.sh`
   - Contract validates branch claims
   - Git hooks prevent cross-contamination

4. **Merge strategy:**
   - Prove all branches independently
   - Compare results objectively
   - Merge best-performing approach
   - Document decisions with evidence

### Pros:
- ✅ True parallel exploration
- ✅ Isolates risk per approach
- ✅ Can compare architectures fairly
- ✅ Each branch provable independently

### Cons:
- ⚠️ More git overhead
- ⚠️ Need to coordinate dataset across branches
- ⚠️ Merge conflicts possible
- ⚠️ Requires discipline to maintain isolation

### Timeline:
- **Week 1:** Branch setup, contracts, enforcement scripts
- **Weeks 2-4:** Parallel development (one approach per branch)
- **Week 5:** Comparison, merge decisions

### Proof Bundle Gate:
```bash
# Each branch must prove independently
git checkout exp/simple-baseline
bash scripts/prove.sh  # Must pass

git checkout exp/magvit-integration
bash scripts/prove.sh  # Must pass

# Etc.
```

---

## Option 5: Focus on Application (Real 2D→3D Reconstruction)

**Description:** Pivot back to original project goal: reconstruct 3D trajectories from real 2D tracking data.

### Rationale:
The current trajectory classification work is interesting but may have drifted from the core mission: **"3D Track Reconstruction from Multiple 2D Mono Tracks"**

### Actions:

1. **Assess existing 2D tracking data:**
   ```bash
   # What real data do we have?
   ls data/*/2d_tracks/
   ```

2. **Review core reconstruction modules:**
   - Camera calibration (Section 2.1 in requirements.md)
   - Triangulation (Section 2.3)
   - Track persistence filtering (Section 2.4)

3. **Run prove.sh on core modules:**
   ```bash
   # Check if main application code has tests
   pytest tests/test_camera_calibration.py
   pytest tests/test_triangulation.py
   ```

4. **Generate end-to-end demo:**
   - Load real 2D tracks from 2+ cameras
   - Triangulate to 3D
   - Visualize 3D trajectory
   - Validate against ground truth (if available)

5. **Document with proof bundle:**
   - Real input data (not synthetic)
   - Actual 3D reconstruction
   - Validation metrics
   - Visualization outputs

### Pros:
- ✅ Aligns with project mission
- ✅ Produces tangible, useful output
- ✅ Can validate against real scenarios
- ✅ Clear success criteria (3D tracks match reality)

### Cons:
- ⚠️ Shifts focus away from vision-language modeling
- ⚠️ May need real calibration data
- ⚠️ Trajectory classification work becomes secondary

### Timeline:
- **Week 1:** Audit existing code, identify gaps
- **Week 2:** TDD for missing pieces (if any)
- **Week 3:** End-to-end integration
- **Week 4:** Validation, visualization, documentation

### Proof Bundle Gate:
```bash
# contracts/3d_reconstruction_demo.yaml
name: "3d_reconstruction_demo"
commands:
  - cmd: "python main_demo.py --input data/sample_2d_tracks/"
    outputs:
      - "results/3d_trajectory.json"
      - "results/3d_visualization.png"
tests:
  - "pytest tests/test_reconstruction_pipeline.py"

bash scripts/prove.sh  # Must pass
```

---

## Option 6: Hybrid - Incremental Honest Implementation

**Description:** Accept current simplified state, then incrementally add real components **one at a time** with full TDD.

### Sequence:

#### Phase 1: Establish Honest Baseline (Week 1)
- Rename components to reflect reality (Option 1)
- Run prove.sh, commit proof bundle
- Document baseline performance

#### Phase 2: Add Real LLM (Weeks 2-3)
- Implement Option 3A or 3B
- TDD → prove.sh → commit
- Compare before/after performance

#### Phase 3: Add MAGVIT (Weeks 4-6) [Optional]
- Implement Option 2
- TDD → prove.sh → commit
- Measure impact on classification

#### Phase 4: Return to Core Application (Weeks 7-8)
- Option 5: Real 2D→3D reconstruction
- End-to-end demo with proof bundle

### Pros:
- ✅ Incremental, low-risk
- ✅ Each phase independently proven
- ✅ Can stop at any phase if not valuable
- ✅ Learns from each iteration

### Cons:
- ⚠️ Longer total timeline
- ⚠️ May pivot multiple times
- ⚠️ Risk of scope creep

### Timeline:
- **Total:** 6-8 weeks for full hybrid approach
- **Minimum:** 1 week for honest baseline

---

## Recommendation Matrix

| Option | Complexity | Timeline | Honesty | Research Value | Practical Value |
|--------|------------|----------|---------|----------------|-----------------|
| 1. Simplified Models | Low | 1-3 weeks | ✅ High | Low | Medium |
| 2. Real MAGVIT | High | 4-6 weeks | ✅ High | High | Low-Medium |
| 3. Real LLM | Medium | 2-4 weeks | ✅ High | Medium | Medium-High |
| 4. Parallel Branches | Medium | 4-5 weeks | ✅ High | Medium | Medium |
| 5. Core Application | Low-Medium | 3-4 weeks | ✅ High | Low-Medium | ✅ High |
| 6. Hybrid Incremental | Medium | 6-8 weeks | ✅ High | Medium-High | High |

---

## Decision Criteria

### Choose Option 1 if:
- Need quick results
- Want to establish baseline
- Prefer honest simplicity over complexity

### Choose Option 2 if:
- Research-oriented (publications)
- Have GPU resources
- Interested in video generation/representation

### Choose Option 3 if:
- Want to prove LLM value for trajectory analysis
- Have API budget or GPU for local LLM
- Symbolic equation generation is core deliverable

### Choose Option 4 if:
- Want to compare multiple approaches fairly
- Have time for proper parallel development
- Previous parallel claims need redemption

### Choose Option 5 if:
- Core 2D→3D reconstruction is primary goal
- Have real tracking data available
- Want practical, deployable solution

### Choose Option 6 if:
- Want flexibility to adapt
- Prefer incremental validation
- Long-term project with multiple goals

---

## Next Step: Validate with prove.sh

**Regardless of option chosen, first step is always:**

```bash
bash scripts/prove.sh
```

Current state must be proven before moving forward. All options start from a known, validated baseline.

---

## Questions to Guide Decision

1. **What is the primary goal?**
   - Research publication → Option 2 or 6
   - Practical application → Option 5
   - Quick validation → Option 1
   - LLM exploration → Option 3

2. **What resources are available?**
   - GPU + time → Option 2 (MAGVIT)
   - API budget → Option 3A (GPT-4)
   - Limited resources → Option 1 (Simplified)
   - GPU for inference → Option 3B (Local LLM)

3. **What is the timeline?**
   - 1-2 weeks → Option 1
   - 3-4 weeks → Option 3 or 5
   - 4-6 weeks → Option 2 or 4
   - 6-8 weeks → Option 6

4. **What needs to be proven?**
   - Vision-language models work for trajectories → Option 3
   - Video representations help classification → Option 2
   - System works on real data → Option 5
   - Multiple architectures can be compared → Option 4

---

## Current Proof Bundle Status

```bash
$ bash scripts/prove.sh
✓ PROOF BUNDLE CREATED
Location: artifacts/proof/3e84e84e9f987f5b6a3204741b79f1008a736871
```

**Foundation is proven. Ready for next step.**

