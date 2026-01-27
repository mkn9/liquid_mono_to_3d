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
