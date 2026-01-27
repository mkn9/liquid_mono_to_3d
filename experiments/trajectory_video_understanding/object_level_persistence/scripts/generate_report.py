"""Generate summary report from training and visualization results."""

import json
from pathlib import Path
import numpy as np
import argparse
from datetime import datetime


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_root', type=str, required=True)
    args = parser.parse_args()
    
    output_root = Path(args.output_root)
    
    # Load training history
    with open(output_root / 'training/training_history.json') as f:
        history = json.load(f)
    
    # Load attention metrics
    with open(output_root / 'attention_visualizations/attention_metrics.json') as f:
        metrics = json.load(f)
    
    # Generate report
    report = f"""# Fast Attention Validation Report

**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  
**Goal**: Validate transformer attention on persistent vs transient objects

---

## Training Results

- **Epochs**: {len(history['train_acc'])}
- **Final Training Accuracy**: {history['train_acc'][-1]:.2%}
- **Final Validation Accuracy**: {history['val_acc'][-1]:.2%}
- **Persistent Object Accuracy**: {history['val_persistent_acc'][-1]:.2%}
- **Transient Object Accuracy**: {history['val_transient_acc'][-1]:.2%}

---

## Attention Analysis

Analyzed {len(metrics)} validation samples:

- **Average Classification Accuracy**: {np.mean([m['accuracy'] for m in metrics]):.2%}
- **Average Persistent Attention**: {np.mean([m['avg_persistent_attention'] for m in metrics]):.4f}
- **Average Transient Attention**: {np.mean([m['avg_transient_attention'] for m in metrics]):.4f}
- **Attention Ratio (Persistent/Transient)**: {np.mean([m['attention_ratio'] for m in metrics if m['attention_ratio'] > 0]):.2f}x

---

## Conclusion

"""
    
    # Check result
    ratio = np.mean([m['attention_ratio'] for m in metrics if m['attention_ratio'] > 0])
    if ratio > 1.5:
        report += f'✅ **SUCCESS**: Transformer successfully attends MORE to persistent objects!\n\n'
        report += f'The model allocates **{ratio:.2f}x** more attention to persistent objects than transient ones.\n\n'
        report += 'This demonstrates that the transformer has learned to:\n'
        report += '1. Focus computational resources on meaningful (persistent) tracks\n'
        report += '2. Reduce attention on noise (transient) objects\n'
        report += '3. Allocate compute efficiently based on object importance\n'
    elif ratio < 0.67:
        report += f'⚠️ **UNEXPECTED**: Transformer attends more to transient objects.\n\n'
        report += f'Attention ratio: {ratio:.2f}x (should be > 1.0)\n\n'
        report += 'This suggests the model may be learning to detect anomalies rather than persistence.\n'
    else:
        report += f'⚠️ **INCONCLUSIVE**: No clear attention preference detected.\n\n'
        report += f'Attention ratio: {ratio:.2f}x (close to 1.0)\n\n'
        report += 'The model may need more training or architectural adjustments.\n'
    
    report += f"""

---

## Files Generated

- **Training logs**: `{output_root}/training/`
- **Attention visualizations**: `{output_root}/attention_visualizations/`
- **Model checkpoint**: `{output_root}/training/best_model.pt`
- **Training history**: `{output_root}/training/training_history.json`
- **Attention metrics**: `{output_root}/attention_visualizations/attention_metrics.json`

---

## Next Steps

"""
    
    if ratio > 1.5:
        report += """
✅ **Validation successful!** The attention mechanism is working as intended.

**Recommended actions:**
1. Scale up to full dataset (10,000 samples)
2. Implement proper object tracker (IoU-based)
3. Add early stopping mechanism based on attention confidence
4. Measure compute savings from attention gating
"""
    else:
        report += """
⚠️ **Further investigation needed.**

**Recommended actions:**
1. Analyze individual sample visualizations
2. Check if data labeling is correct (white=persistent, red=transient)
3. Consider adjusting model architecture or training hyperparameters
4. Increase training epochs or dataset size
"""
    
    # Save report
    with open(output_root / 'VALIDATION_REPORT.md', 'w') as f:
        f.write(report)
    
    print(report)
    print(f"\n✅ Report saved to: {output_root}/VALIDATION_REPORT.md")


if __name__ == '__main__':
    main()

