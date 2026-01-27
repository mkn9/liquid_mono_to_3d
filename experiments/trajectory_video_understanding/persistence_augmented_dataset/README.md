# Persistence-Augmented Trajectory Dataset

## Overview

This dataset extends the original 10,000 trajectory samples by adding **non-persistent transient spheres** to each video. The transient spheres:

- Persist for only **1-3 frames** (compared to 16 frames for real trajectories)
- Move with similar motion patterns (linear, circular, helical, parabolic)
- Appear at random times within the video
- Are visually distinct (reddish color) from the primary trajectory

## Purpose

This augmented dataset enables training of **track persistence classifiers** that can distinguish between:
- **Persistent tracks**: Real objects that maintain consistent trajectories
- **Transient tracks**: Noise, reflections, or temporary detections that disappear quickly

## Dataset Structure

### Files

Each sample consists of two files:

```
augmented_traj_XXXXX.pt     # Video tensor (T, C, H, W) with both real and transient spheres
augmented_traj_XXXXX.json   # Metadata including transient locations and timing
```

### Metadata Format

```json
{
  "trajectory_type": "linear",
  "trajectory_class": 0,
  "augmented": true,
  "source_sample_idx": 42,
  "num_transients": 3,
  "transient_frames": [5, 6, 10, 11, 12],
  "transient_details": [
    {
      "start_frame": 5,
      "duration": 2,
      "trajectory_type": "circular",
      "start_position": [0.3, -0.2, 0.1]
    },
    ...
  ]
}
```

### Key Metadata Fields

- `num_transients`: Number of transient spheres in this video
- `transient_frames`: List of frame indices containing transient spheres
- `transient_details`: Full parameters for each transient (start, duration, trajectory type)
- `source_sample_idx`: Index of the original trajectory sample

## Generation Process

### TDD Implementation

Following Test-Driven Development:

1. **RED Phase**: Tests written first, all failing
   - Evidence: `artifacts/tdd_persistence_red.txt`
   
2. **GREEN Phase**: Implementation to pass all tests
   - Evidence: `artifacts/tdd_persistence_green.txt`
   - All 9 tests passing

3. **REFACTOR Phase**: Code optimization and documentation

### Batch Processing

```bash
# Run with automatic monitoring and syncing
./run_generation_with_monitoring.sh
```

Features:
- **Checkpointing**: Resume from interruption every 100 samples
- **Progress tracking**: Real-time progress file updates
- **MacBook visibility**: Results synced every 30 seconds
- **Error handling**: Continues processing on individual sample errors

### Manual Execution

```bash
python batch_augment_dataset.py \
    --source_dir /path/to/trajectory_dataset_10k \
    --output_dir /path/to/output \
    --num_samples 10000 \
    --transients_per_video 3 \
    --checkpoint_interval 100 \
    --sync_interval 500
```

## Statistics

- **Total Samples**: 10,000
- **Transients per Video**: 1-6 (average ~3)
- **Transient Duration**: 1-3 frames
- **Transient Trajectory Types**: Linear, Circular, Helical, Parabolic
- **Total Transients**: ~30,000 across all videos

## Usage Example

```python
import torch
import json

# Load augmented sample
video = torch.load('augmented_traj_00042.pt')
with open('augmented_traj_00042.json', 'r') as f:
    metadata = json.load(f)

print(f"Video shape: {video.shape}")  # (16, 3, 64, 64)
print(f"Number of transients: {metadata['num_transients']}")
print(f"Frames with transients: {metadata['transient_frames']}")

# Check which frames have transients
for frame_idx in range(len(video)):
    has_transient = frame_idx in metadata['transient_frames']
    print(f"Frame {frame_idx}: {'TRANSIENT' if has_transient else 'clean'}")
```

## Transient Sphere Characteristics

### Visual Properties
- **Color**: Reddish (0.9, 0.3, 0.3) for easy identification
- **Size**: Adaptive based on simulated depth
- **Rendering**: Filled circles projected from 3D space

### Motion Properties
- **Linear**: Straight-line motion with random velocity
- **Circular**: Circular motion in XY plane
- **Helical**: Spiral motion with Z-axis progression
- **Parabolic**: Arc motion simulating gravity

### Temporal Properties
- **Duration**: Uniformly distributed between 1-3 frames
- **Start Time**: Random within video duration
- **Frequency**: 1-6 transients per video (average 3)

## Training Applications

### Track Persistence Classification

Binary classification task:
- **Input**: Video segment or track features
- **Output**: Persistent (1) or Transient (0)
- **Labels**: 
  - Frames with `transient_frames` indices → Transient (0)
  - Other frames → Persistent (1)

### Attention Mechanism Training

The augmented dataset is ideal for training attention mechanisms to:
- Focus on persistent trajectories
- Ignore transient noise
- Learn temporal consistency patterns

### Multi-Task Learning

Combined tasks:
1. **Trajectory Classification**: Identify trajectory type (linear, circular, etc.)
2. **Persistence Detection**: Identify which objects are persistent vs. transient
3. **Position Prediction**: Predict future positions of persistent tracks only

## Quality Assurance

### TDD Coverage

✅ All core functionality tested:
- Transient parameter generation
- Sphere rendering on frames
- Trajectory generation (4 types)
- Video augmentation
- Data loading and saving
- Checkpointing and resume

### Validation

- Visual inspection of sample outputs
- Metadata consistency checks
- Duration distribution verification
- Spatial overlap prevention (TODO: future enhancement)

## File Organization

```
persistence_augmented_dataset/
├── README.md                          # This file
├── generate_transient_dataset.py      # Core generation logic
├── batch_augment_dataset.py           # Batch processing script
├── run_generation_with_monitoring.sh  # Execution wrapper with sync
├── tests/
│   └── test_transient_generator.py    # TDD tests (9 tests, all passing)
├── output/                            # Generated augmented samples (10K)
│   ├── augmented_traj_XXXXX.pt
│   ├── augmented_traj_XXXXX.json
│   ├── checkpoint.json                # Resume checkpoint
│   ├── PROGRESS.txt                   # Real-time progress log
│   └── GENERATION_SUMMARY.json        # Final statistics
└── results/                           # Synced to MacBook
    ├── PROGRESS.txt
    ├── checkpoint.json
    └── GENERATION_SUMMARY.json
```

## Performance

- **Generation Rate**: ~10-50 samples/second (depends on transient count)
- **Estimated Time**: 3-17 minutes for 10,000 samples
- **Disk Space**: ~same as original dataset (~20-30 GB)

## Future Enhancements

1. **Spatial Overlap Detection**: Prevent transients from overlapping with main trajectory
2. **Occlusion Modeling**: Simulate transients passing behind main object
3. **Variable Appearance**: Different colors/sizes for transients
4. **Realistic Noise**: Add camera noise, motion blur to transients
5. **Temporal Clustering**: Group transients to simulate burst noise

## References

- Original trajectory dataset: `../data/trajectory_dataset_10k/`
- Training scripts: `../branch_X_*/train.py`
- Feature extractors: `../shared/base_extractor.py`

---

**Author**: AI Assistant  
**Date**: 2026-01-26  
**TDD Evidence**: `artifacts/tdd_persistence_*.txt`

