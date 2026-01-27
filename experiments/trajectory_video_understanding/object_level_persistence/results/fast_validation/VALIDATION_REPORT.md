# Fast Attention Validation Report

**Date**: 2026-01-26 05:18:10  
**Goal**: Validate transformer attention on persistent vs transient objects

---

## Training Results

- **Epochs**: 10
- **Final Training Accuracy**: 99.57%
- **Final Validation Accuracy**: 54.95%
- **Persistent Object Accuracy**: 56.18%
- **Transient Object Accuracy**: 48.92%

---

## Attention Analysis

Analyzed 20 validation samples:

- **Average Classification Accuracy**: 57.08%
- **Average Persistent Attention**: 1.0015
- **Average Transient Attention**: 0.6931
- **Attention Ratio (Persistent/Transient)**: 1.01x

---

## Conclusion

⚠️ **INCONCLUSIVE**: No clear attention preference detected.

Attention ratio: 1.01x (close to 1.0)

The model may need more training or architectural adjustments.


---

## Files Generated

- **Training logs**: `/home/ubuntu/mono_to_3d/experiments/trajectory_video_understanding/object_level_persistence/results/fast_validation/training/`
- **Attention visualizations**: `/home/ubuntu/mono_to_3d/experiments/trajectory_video_understanding/object_level_persistence/results/fast_validation/attention_visualizations/`
- **Model checkpoint**: `/home/ubuntu/mono_to_3d/experiments/trajectory_video_understanding/object_level_persistence/results/fast_validation/training/best_model.pt`
- **Training history**: `/home/ubuntu/mono_to_3d/experiments/trajectory_video_understanding/object_level_persistence/results/fast_validation/training/training_history.json`
- **Attention metrics**: `/home/ubuntu/mono_to_3d/experiments/trajectory_video_understanding/object_level_persistence/results/fast_validation/attention_visualizations/attention_metrics.json`

---

## Next Steps


⚠️ **Further investigation needed.**

**Recommended actions:**
1. Analyze individual sample visualizations
2. Check if data labeling is correct (white=persistent, red=transient)
3. Consider adjusting model architecture or training hyperparameters
4. Increase training epochs or dataset size
