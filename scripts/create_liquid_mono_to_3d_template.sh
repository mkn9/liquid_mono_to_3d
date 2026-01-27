#!/bin/bash
# Script to create liquid_mono_to_3d template from mono_to_3d
# This creates a COPY - does NOT modify the original mono_to_3d repo

set -e  # Exit on error

ORIGINAL_DIR="/Users/mike/Dropbox/Documents/Machine_Learning/LLM_and_Multimodal_Models/repos/mono_to_3d"
TEMPLATE_DIR="/Users/mike/Dropbox/Documents/Machine_Learning/LLM_and_Multimodal_Models/repos/liquid_mono_to_3d"

echo "🚀 Creating liquid_mono_to_3d template..."
echo "   Original: $ORIGINAL_DIR"
echo "   Template: $TEMPLATE_DIR"
echo ""

# Remove template dir if it exists
if [ -d "$TEMPLATE_DIR" ]; then
    echo "⚠️  Template directory already exists. Removing..."
    rm -rf "$TEMPLATE_DIR"
fi

# Create fresh template directory
echo "📁 Creating template directory structure..."
mkdir -p "$TEMPLATE_DIR"

# Copy entire repo first (excluding .git)
echo "📋 Copying repo structure..."
rsync -av --exclude='.git' --exclude='*.pyc' --exclude='__pycache__' --exclude='.DS_Store' \
    "$ORIGINAL_DIR/" "$TEMPLATE_DIR/"

cd "$TEMPLATE_DIR"

# =============================================================================
# CHAT HISTORY CLEANUP - Keep last 12 sessions
# =============================================================================
echo ""
echo "📝 Processing chat history (keeping last 12 sessions)..."

# Find all chat history files sorted by modification time (newest first)
CHAT_FILES=$(find . -type f \( -name "CHAT_*.md" -o -name "*chat*.md" -o -name "*CHAT*.md" \) \
    ! -path "./.git/*" -printf '%T@ %p\n' | sort -rn | awk '{print $2}')

# Keep only the 12 most recent
CHAT_COUNT=0
while IFS= read -r chat_file; do
    CHAT_COUNT=$((CHAT_COUNT + 1))
    if [ $CHAT_COUNT -le 12 ]; then
        echo "   ✅ Keeping: $chat_file"
    else
        echo "   ❌ Removing: $chat_file"
        rm -f "$chat_file"
    fi
done <<< "$CHAT_FILES"

# =============================================================================
# NOTEBOOK CLEANUP - Keep latest with mono_to_3d manipulations
# =============================================================================
echo ""
echo "📓 Processing notebooks (keeping latest with mono_to_3d work)..."

# Keep these specific latest notebooks
KEEP_NOTEBOOKS=(
    "3d_tracker_interactive_FIXED_FINAL_COMPLETE_WORKING.ipynb"
    "3d_tracker_9.ipynb"
    "3d_visualization.ipynb"
    "sensor_impact_analysis.ipynb"
)

# Remove all other notebooks at root level
find . -maxdepth 1 -name "*.ipynb" -type f | while read -r notebook; do
    BASENAME=$(basename "$notebook")
    KEEP=false
    for keep_nb in "${KEEP_NOTEBOOKS[@]}"; do
        if [ "$BASENAME" == "$keep_nb" ]; then
            KEEP=true
            break
        fi
    done
    
    if [ "$KEEP" = true ]; then
        echo "   ✅ Keeping: $notebook"
    else
        echo "   ❌ Removing: $notebook"
        rm -f "$notebook"
    fi
done

# =============================================================================
# REMOVE HEAVY DATA/RESULTS - Keep structure, remove contents
# =============================================================================
echo ""
echo "🗑️  Cleaning up data and results (keeping structure)..."

# Remove large data files but keep directory structure
find . -type f \( -name "*.pt" -o -name "*.pth" -o -name "*.ckpt" -o -name "*.h5" \) \
    ! -path "./.git/*" | while read -r file; do
    echo "   ❌ Removing data: $file"
    rm -f "$file"
done

# Remove result images but keep a few examples
find ./experiments -type f -name "*.png" | tail -n +20 | while read -r file; do
    rm -f "$file"
done

# Remove old/deprecated experiment directories
echo ""
echo "🧹 Removing deprecated experiments..."
DEPRECATED_DIRS=(
    "./old_experiments"
    "./deprecated"
    "./neural_video_experiments"
    "./magvit_options"
)

for dir in "${DEPRECATED_DIRS[@]}"; do
    if [ -d "$dir" ]; then
        echo "   ❌ Removing: $dir"
        rm -rf "$dir"
    fi
done

# =============================================================================
# KEEP ESSENTIAL CODE AND STRUCTURE
# =============================================================================
echo ""
echo "✅ Keeping essential code and structure..."

KEEP_DIRS=(
    "./experiments/trajectory_video_understanding"
    "./experiments/magvit_I3D_LLM_basic_trajectory"
    "./scripts"
    "./docs"
)

for dir in "${KEEP_DIRS[@]}"; do
    if [ -d "$dir" ]; then
        echo "   ✅ Keeping: $dir"
    fi
done

# =============================================================================
# CREATE .gitignore for data/models
# =============================================================================
echo ""
echo "📄 Creating/updating .gitignore..."

cat > .gitignore << 'EOF'
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
venv/
env/
ENV/
.venv

# Jupyter
.ipynb_checkpoints/
*.ipynb_checkpoints

# Data and Models (CRITICAL - DO NOT COMMIT)
*.pt
*.pth
*.ckpt
*.h5
*.hdf5
*.pkl
*.pickle
*.npy
*.npz

# Large result files
*.mp4
*.avi
*.mov
results/*/predictions/
results/*/checkpoints/

# OS
.DS_Store
Thumbs.db

# IDE
.vscode/
.idea/
*.swp
*.swo

# Logs
*.log
wandb/

# Temporary
tmp/
temp/
*.tmp
EOF

echo "   ✅ .gitignore created"

# =============================================================================
# CREATE DATA SETUP INSTRUCTIONS
# =============================================================================
echo ""
echo "📋 Creating data setup instructions..."

cat > DATA_SETUP.md << 'EOF'
# Data and Model Setup for liquid_mono_to_3d

## ⚠️ Data and Models Are NOT in Git

All data files and trained models are excluded from version control (see `.gitignore`).

You must set up data and models separately after cloning this repo.

---

## Option 1: Copy from S3 (Recommended for EC2)

```bash
# Sync augmented dataset
aws s3 sync s3://YOUR-BUCKET/mono_to_3d/data/ ./experiments/trajectory_video_understanding/persistence_augmented_dataset/

# Sync trained models
aws s3 sync s3://YOUR-BUCKET/mono_to_3d/models/ ./experiments/trajectory_video_understanding/sequential_results_20260125_2148_FULL/
```

---

## Option 2: Copy from Existing Instance

```bash
# From existing EC2 instance to new instance
rsync -avz -e "ssh -i your-key.pem" \
  ubuntu@old-instance-ip:~/mono_to_3d/experiments/trajectory_video_understanding/persistence_augmented_dataset/ \
  ./experiments/trajectory_video_understanding/persistence_augmented_dataset/

rsync -avz -e "ssh -i your-key.pem" \
  ubuntu@old-instance-ip:~/mono_to_3d/experiments/trajectory_video_understanding/sequential_results_20260125_2148_FULL/ \
  ./experiments/trajectory_video_understanding/sequential_results_20260125_2148_FULL/
```

---

## Option 3: Regenerate Data

If you don't have access to existing data, regenerate it:

```bash
cd experiments/trajectory_video_understanding/early_persistence_detection
python generate_augmented_dataset.py --num-samples 2000 --output-dir ../persistence_augmented_dataset
```

**Note:** You'll need to retrain models if regenerating data.

---

## Required Data Structure

After setup, you should have:

```
experiments/trajectory_video_understanding/
├── persistence_augmented_dataset/
│   └── output_samples/
│       ├── augmented_traj_00000.pt
│       ├── augmented_traj_00000.json
│       └── ... (2000 samples)
└── sequential_results_20260125_2148_FULL/
    └── magvit/
        ├── final_model.pt
        └── training_metrics.json
```

---

## Verification

```bash
# Check data
ls -lh experiments/trajectory_video_understanding/persistence_augmented_dataset/output_samples/ | head

# Check models
ls -lh experiments/trajectory_video_understanding/sequential_results_20260125_2148_FULL/magvit/

# Test model loading
cd experiments/trajectory_video_understanding/vision_language_integration
python demo_real_magvit.py --num-examples 1
```

If all commands succeed, your setup is complete! ✅
EOF

echo "   ✅ DATA_SETUP.md created"

# =============================================================================
# CREATE DEPLOYMENT SETUP SCRIPT
# =============================================================================
echo ""
echo "🚀 Creating deployment setup script..."

cat > scripts/setup_ec2_instance.sh << 'EOF'
#!/bin/bash
# Quick setup script for new EC2 instance

set -e

echo "🚀 Setting up liquid_mono_to_3d on EC2..."

# Update system
echo "📦 Updating system packages..."
sudo apt-get update -qq

# Install Python dependencies
echo "🐍 Installing Python environment..."
python3 -m pip install --upgrade pip

# Install PyTorch with CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install other dependencies
pip install transformers accelerate sentencepiece
pip install opencv-python matplotlib seaborn pandas numpy scipy
pip install pytest pytest-cov

# Verify CUDA
echo ""
echo "🔍 Verifying CUDA setup..."
python3 -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}')"

echo ""
echo "✅ Setup complete!"
echo ""
echo "Next steps:"
echo "1. Copy data and models (see DATA_SETUP.md)"
echo "2. Run tests: cd experiments/trajectory_video_understanding && pytest"
echo "3. Start training or inference"
EOF

chmod +x scripts/setup_ec2_instance.sh
echo "   ✅ scripts/setup_ec2_instance.sh created"

# =============================================================================
# UPDATE README for liquid_mono_to_3d
# =============================================================================
echo ""
echo "📝 Updating README for liquid_mono_to_3d..."

cat > README.md << 'EOF'
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
EOF

echo "   ✅ README.md updated"

# =============================================================================
# SUMMARY
# =============================================================================
echo ""
echo "============================================"
echo "✅ liquid_mono_to_3d template created!"
echo "============================================"
echo ""
echo "Location: $TEMPLATE_DIR"
echo ""
echo "What was kept:"
echo "  ✅ Last 12 chat history sessions"
echo "  ✅ Latest notebooks (mono_to_3d work)"
echo "  ✅ Core code (MagVIT, VLM, experiments)"
echo "  ✅ Documentation (requirements, cursorrules, planning docs)"
echo "  ✅ Test infrastructure"
echo "  ✅ Scripts (setup, TDD capture)"
echo ""
echo "What was removed:"
echo "  ❌ Old chat history (keeping only last 12)"
echo "  ❌ Data files (*.pt, *.pth, *.ckpt)"
echo "  ❌ Old notebooks (keeping only latest)"
echo "  ❌ Deprecated experiments"
echo "  ❌ Heavy result files"
echo ""
echo "New files created:"
echo "  📄 .gitignore (excludes data/models)"
echo "  📄 DATA_SETUP.md (setup instructions)"
echo "  📄 README.md (updated for liquid_mono_to_3d)"
echo "  📄 scripts/setup_ec2_instance.sh (EC2 automation)"
echo ""
echo "Next steps:"
echo "  1. Review the template: cd $TEMPLATE_DIR"
echo "  2. Initialize git: cd $TEMPLATE_DIR && git init"
echo "  3. Create GitHub repo: mkn9/liquid_mono_to_3d"
echo "  4. Push: git remote add origin git@github.com:mkn9/liquid_mono_to_3d.git"
echo "           git add ."
echo "           git commit -m '🎉 Initial commit: liquid_mono_to_3d template'"
echo "           git push -u origin main"
echo ""
echo "✨ Original mono_to_3d repo is UNTOUCHED ✨"
echo ""

