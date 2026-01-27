# Neural Video Generation Multi-Experiment Setup

This repository contains a comprehensive multi-experiment setup for neural video generation using state-of-the-art models to learn and predict trajectories of geometric shapes.

## 🎯 Overview

We've implemented three parallel experiments using git branches to organize and run simultaneous development:

### 1. MAGVIT 2D Trajectories
- **Branch**: `experiment/magvit-2d-trajectories`
- **Shapes**: Squares, Circles, Triangles
- **Framework**: [MAGVIT](https://github.com/google-research/magvit) (JAX/Flax)
- **Paper**: [MAGVIT: Masked Generative Video Transformer](https://arxiv.org/abs/2204.02896)
- **Goal**: Learn to predict 2D trajectories from initial frames

### 2. VideoGPT 2D Trajectories 
- **Branch**: `experiment/videogpt-2d-trajectories`
- **Shapes**: Squares, Circles, Triangles
- **Framework**: [VideoGPT](https://github.com/wilson1yan/VideoGPT) (PyTorch)
- **Goal**: Learn to predict 2D trajectories from initial frames

### 3. MAGVIT 3D Trajectories
- **Branch**: `experiment/magvit-3d-trajectories`
- **Shapes**: Cubes, Cylinders, Cones
- **Framework**: MAGVIT adapted for 3D using mono_to_3d codebase
- **Goal**: Learn to predict 3D trajectories from multi-view observations

## 🏗️ Architecture

```
mono_to_3d/
├── experiments/
│   ├── magvit-2d-trajectories/
│   │   ├── setup_magvit_2d.py
│   │   ├── train_magvit_2d.py
│   │   ├── data/
│   │   ├── models/
│   │   └── results/
│   ├── videogpt-2d-trajectories/
│   │   ├── setup_videogpt_2d.py
│   │   ├── train_videogpt_2d.py
│   │   ├── data/
│   │   ├── models/
│   │   └── results/
│   └── magvit-3d-trajectories/
│       ├── setup_magvit_3d.py
│       ├── train_magvit_3d.py
│       ├── data/
│       ├── models/
│       └── results/
├── setup_all_experiments.py
├── launch_all_experiments.sh
├── monitor_experiments.py
└── experiment_summary.json
```

## 🚀 Quick Start

### Option 1: Automated Setup and Launch
```bash
# Run complete setup and launch all experiments
python setup_all_experiments.py

# When prompted, choose 'y' to start all experiments immediately
```

### Option 2: Manual Setup and Launch
```bash
# 1. Setup all experiments
python setup_all_experiments.py

# 2. Launch all experiments in parallel
./launch_all_experiments.sh

# 3. Monitor experiment progress
python monitor_experiments.py
```

## 📋 Detailed Setup Instructions

### Prerequisites
```bash
# Ensure you're in the project root with virtual environment activated
source activate_mono_to_3d_env.sh

# Install additional dependencies
pip install psutil jax jaxlib flax optax h5py einops open3d trimesh
```

### Individual Experiment Setup

#### MAGVIT 2D Setup
```bash
git checkout experiment/magvit-2d-trajectories
cd experiments/magvit-2d-trajectories
python setup_magvit_2d.py
python train_magvit_2d.py
```

#### VideoGPT 2D Setup
```bash
git checkout experiment/videogpt-2d-trajectories
cd experiments/videogpt-2d-trajectories
python setup_videogpt_2d.py
python train_videogpt_2d.py
```

#### MAGVIT 3D Setup
```bash
git checkout experiment/magvit-3d-trajectories
cd experiments/magvit-3d-trajectories
python setup_magvit_3d.py
python train_magvit_3d.py
```

## 🎮 Trajectory Patterns

All experiments generate multiple trajectory patterns for robust learning:

### 2D Patterns (MAGVIT 2D, VideoGPT 2D)
- **Linear**: Straight line motion across the frame
- **Circular**: Circular motion around a center point
- **Sine Wave**: Sinusoidal motion with varying frequency
- **Parabolic**: Parabolic arc motion

### 3D Patterns (MAGVIT 3D)
- **Linear 3D**: Straight line motion in 3D space
- **Circular 3D**: Horizontal circular motion
- **Helical**: Spiral motion with height variation
- **Parabolic 3D**: 3D parabolic trajectories

## 🎨 Generated Data Specifications

### 2D Experiments
- **Resolution**: 128x128 pixels
- **Sequence Length**: 16 frames
- **Color Channels**: RGB (3 channels)
- **Data Format**: NumPy arrays and HDF5
- **Shapes**: Squares (red), Circles (green), Triangles (blue)

### 3D Experiments
- **Resolution**: 128x128 pixels per camera view
- **Sequence Length**: 16 frames
- **Camera Views**: 3 cameras at different positions
- **Data Format**: Multi-view video dictionaries
- **Shapes**: Cubes, Cylinders, Cones with 3D coordinates

## 🔧 Configuration

Each experiment has its own `config.json` with optimized hyperparameters:

### MAGVIT 2D Configuration
```json
{
  "model": {
    "vocab_size": 1024,
    "hidden_dim": 512,
    "num_layers": 8,
    "sequence_length": 16
  },
  "training": {
    "batch_size": 32,
    "learning_rate": 1e-4,
    "num_epochs": 100
  }
}
```

### VideoGPT 2D Configuration
```json
{
  "model": {
    "n_codes": 2048,
    "n_hiddens": 240,
    "resolution": 128,
    "sequence_length": 16
  },
  "training": {
    "batch_size": 16,
    "learning_rate": 3e-4,
    "num_epochs": 100
  }
}
```

### MAGVIT 3D Configuration
```json
{
  "model": {
    "vocab_size": 2048,
    "hidden_dim": 768,
    "num_layers": 12,
    "num_cameras": 3
  },
  "training": {
    "batch_size": 16,
    "learning_rate": 5e-5,
    "num_epochs": 150
  }
}
```

## 📊 Monitoring and Logging

### Real-time Monitoring
```bash
# Monitor all running experiments
python monitor_experiments.py

# Check system processes
ps aux | grep python

# View GPU usage (if available)
nvidia-smi
```

### Log Files
All experiments log to `logs/` directory with timestamps:
- `logs/magvit-2d_YYYYMMDD_HHMMSS.log`
- `logs/videogpt-2d_YYYYMMDD_HHMMSS.log`
- `logs/magvit-3d_YYYYMMDD_HHMMSS.log`

## 🌿 Git Branch Management

The setup automatically manages git branches:

```bash
# View all experiment branches
git branch

# Switch to specific experiment
git checkout experiment/magvit-2d-trajectories
git checkout experiment/videogpt-2d-trajectories
git checkout experiment/magvit-3d-trajectories

# Return to master
git checkout master
```

## 📈 Expected Results

### Training Progression
1. **Data Generation**: ~2-5 minutes per experiment
2. **Model Setup**: ~5-10 minutes (including dependency installation)
3. **Training**: Variable (depends on hardware and configuration)

### Performance Targets
- **2D Experiments**: Learn to predict simple geometric trajectories
- **3D Experiments**: Learn to predict 3D motion from multi-view observations
- **Evaluation**: Visual comparison of predicted vs. actual trajectories

## 🐛 Troubleshooting

### Common Issues

#### Git Branch Conflicts
```bash
# If branch switching fails
git stash
git checkout master
git checkout <target-branch>
git stash pop
```

#### Dependency Issues
```bash
# Reinstall dependencies
pip install --upgrade -r requirements.txt
pip install jax jaxlib flax optax orbax-checkpoint
```

#### GPU Memory Issues
```bash
# Reduce batch sizes in config.json
# Monitor GPU usage: nvidia-smi
# Consider running experiments sequentially instead of parallel
```

#### Process Management
```bash
# Kill all experiment processes
pkill -f "train_"

# Check running processes
ps aux | grep python
```

## 🔬 Extending the Experiments

### Adding New Trajectory Patterns
1. Edit the `generate_trajectory_patterns()` method in data generators
2. Add new mathematical functions for motion
3. Test with visualization before training

### Adding New Shapes
1. Create new shape classes (e.g., `Pentagon2D`, `Sphere3D`)
2. Implement rendering methods
3. Update shape lists in data generators

### Modifying Model Architecture
1. Edit configuration files in each experiment
2. Adjust hyperparameters based on complexity
3. Monitor training curves for optimization

## 📚 References

- [MAGVIT Paper](https://arxiv.org/abs/2204.02896)
- [MAGVIT Code](https://github.com/google-research/magvit)
- [VideoGPT Code](https://github.com/wilson1yan/VideoGPT)
- [JAX Documentation](https://jax.readthedocs.io/)
- [PyTorch Documentation](https://pytorch.org/docs/)

## 📧 Support

For issues related to:
- **Setup Problems**: Check this README and troubleshooting section
- **Training Issues**: Review log files in `logs/` directory
- **Model Performance**: Adjust hyperparameters in `config.json` files

## 🏆 Success Metrics

The experiments are successful when:
1. ✅ All three experiments setup without errors
2. ✅ Data generation completes for all trajectory types
3. ✅ Models begin training and show loss reduction
4. ✅ Predicted trajectories visually match expected patterns
5. ✅ Multi-experiment monitoring shows stable parallel execution

---

**Happy Experimenting!** 🚀🎬🧠 