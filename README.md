# Liquid Mono to 3D - Vision-Language Trajectory Understanding

A production-ready implementation of vision-language models for trajectory video understanding, built with Cursor AI and following TDD best practices.

## 🌟 What This Is

This is a **cleaned template** derived from the `mono_to_3d` research project, containing:
- ✅ **Production-ready code** (MagVIT + TinyLlama VLM integration)
- ✅ **Latest experiments** (trajectory persistence detection, visual grounding)
- ✅ **TDD infrastructure** (comprehensive test suite, evidence capture)
- ✅ **Documentation** (architecture planning, strategic assessments)
- ✅ **Last 12 chat sessions** (development history and rationale)

## ⚠️ CRITICAL COMPUTATION RULE ⚠️

**ALL COMPUTATION MUST BE PERFORMED ON THE EC2 INSTANCE**

- **MacBook**: File editing, documentation, and SSH connections ONLY
- **EC2 Instance**: All code execution, testing, and computational tasks
- **Never install**: pytorch, numpy, pytest, or any ML/scientific packages on MacBook
- **Always verify**: You are connected to EC2 before running any Python code

```bash
# CORRECT: Connect to EC2 first
ssh -i /path/to/key.pem ubuntu@your-ec2-ip

# INCORRECT: Running computation on MacBook
python test_semantic_nerf.py  # ❌ DON'T DO THIS
```

## 🚀 Quick Start

### 1. Launch EC2 Instance

**Recommended**: `g5.2xlarge` (A10G GPU, 24GB VRAM)
- Cost: $0.40/hr (Spot) or $1.21/hr (On-Demand)
- See `AWS_SPOT_FAILOVER_SETUP.md` for production setup

### 2. Clone and Setup

```bash
# SSH to EC2
ssh -i your-key.pem ubuntu@your-ec2-ip

# Clone repo
git clone https://github.com/mkn9/liquid_mono_to_3d.git
cd liquid_mono_to_3d

# Run setup script
bash scripts/setup_ec2_instance.sh
```

### 3. Set Up Data and Models

**See `DATA_SETUP.md` for detailed instructions.**

Quick option (if you have S3 backup):
```bash
aws s3 sync s3://your-bucket/data/ ./experiments/trajectory_video_understanding/persistence_augmented_dataset/
aws s3 sync s3://your-bucket/models/ ./experiments/trajectory_video_understanding/sequential_results_20260125_2148_FULL/
```

### 4. Verify Setup

```bash
# Run tests
cd experiments/trajectory_video_understanding
pytest

# Run VLM demo
cd vision_language_integration
python demo_real_magvit.py --num-examples 3
```

## 📁 Project Structure

```
liquid_mono_to_3d/
├── experiments/
│   ├── trajectory_video_understanding/     # Main VLM experiment
│   │   ├── vision_language_integration/   # VLM code (MagVIT + TinyLlama)
│   │   ├── early_persistence_detection/   # Persistence classification
│   │   ├── parallel_workers/              # Multi-worker training
│   │   └── persistence_augmented_dataset/ # Training data (gitignored)
│   └── magvit_I3D_LLM_basic_trajectory/   # Basic trajectory experiments
├── scripts/
│   ├── setup_ec2_instance.sh              # EC2 setup automation
│   └── tdd_capture.sh                     # TDD evidence capture
├── docs/
│   ├── requirements.md                    # Comprehensive requirements (TDD, protocols)
│   └── cursorrules                        # Cursor AI development rules
├── CHAT_HISTORY_*.md                      # Last 12 development sessions
├── ARCHITECTURE_PLANNING_LNN.md           # Visual grounding + LNN plans
├── VLM_STRATEGIC_ASSESSMENT.md            # VLM performance analysis
├── AWS_SPOT_FAILOVER_SETUP.md             # Production AWS setup
└── DATA_SETUP.md                          # Data and model setup guide
```

## 🧪 Key Features

### 1. Vision-Language Model (VLM)
- **MagVIT** (ResNet-18 + Transformer): 100% validation accuracy on trajectory persistence
- **TinyLlama-1.1B**: Local LLM for natural language descriptions and Q&A
- **Real integration**: No mock data or fake responses

### 2. Test-Driven Development (TDD)
- **Red-Green-Refactor**: All code follows strict TDD workflow
- **Evidence capture**: `artifacts/tdd_*.txt` for all test phases
- **100% test coverage** on core modules

### 3. Trajectory Understanding
- **Persistence detection**: Classify persistent vs. transient objects in videos
- **Augmented dataset**: 2000 synthetic trajectory samples
- **Early detection**: Identify transient objects within first few frames

### 4. Production-Ready Infrastructure
- **AWS Spot + On-Demand failover**: 1-2 min downtime per interruption
- **Checkpoint system**: Resume training after interruptions
- **Scalable**: Multi-worker parallel training support

## 📖 Documentation

### Core Documentation
- **`requirements.md`**: Comprehensive project requirements, TDD methodology, scientific integrity protocols
- **`cursorrules`**: Cursor AI development directives (what to do)
- **`DATA_SETUP.md`**: Data and model setup instructions

### Architecture and Planning
- **`ARCHITECTURE_PLANNING_LNN.md`**: Visual grounding roadmap, LNN integration plans
- **`VLM_STRATEGIC_ASSESSMENT.md`**: VLM performance expectations vs. general VLMs
- **`AWS_SPOT_FAILOVER_SETUP.md`**: Production AWS Spot instance setup
- **`GPU_CLOUD_PROVIDER_COMPARISON.md`**: Cost comparison (AWS vs. Vast.ai, etc.)

### Development History
- **`CHAT_HISTORY_*.md`**: Last 12 development sessions with full context

## 🎯 Current State (January 2026)

### ✅ Completed
1. **MagVIT Model**: 100% validation accuracy on persistence classification
2. **VLM Integration**: Real MagVIT + TinyLlama with actual data
3. **TDD Infrastructure**: Comprehensive test suite with evidence capture
4. **Documentation**: Architecture planning, strategic assessments

### 🚧 Next Steps (See `ARCHITECTURE_PLANNING_LNN.md`)
1. **Visual Grounding**: Connect MagVIT embeddings to LLM (Week 1-2)
2. **3D Integration**: Connect 3D trajectory models (Week 9-10)
3. **Liquid Neural Networks**: Explore LNNs for continuous dynamics (Research)

## 💰 Cost Optimization

**Current Setup**: AWS `g5.2xlarge`
- Spot: $0.40/hr (with 8.7% interruption rate)
- Spot + On-Demand Backup: $0.43/hr avg (1-2 min downtime per interruption)
- Annual cost (5 hrs/week): $112 (vs $315 pure on-demand)

**See**: `AWS_SPOT_FAILOVER_SETUP.md` for production setup

## 🤝 Contributing

This is a template derived from active research. If you use this:
1. Follow TDD workflow (see `requirements.md` Section 3.4)
2. Capture evidence for all test phases (`scripts/tdd_capture.sh`)
3. Run tests on EC2, not MacBook
4. Update chat history for significant changes

## 📝 Key Principles (From `requirements.md`)

1. **Honesty**: No mock data, no fake results, no hallucinations
2. **TDD**: Tests BEFORE code, always
3. **Evidence**: Capture all test phases (red, green, refactor, structural)
4. **EC2 Computation**: All code execution on EC2, never MacBook
5. **Documentation**: Update chat history for all significant work

## 📄 License

MIT License (or specify your license)

## 🙏 Acknowledgments

Built with Cursor AI, following best practices from:
- Test-Driven Development (Kent Beck)
- Clean Code (Robert C. Martin)
- Domain-Driven Design (Eric Evans)

---

**For detailed setup and usage, see `DATA_SETUP.md` and `requirements.md`.**
