#!/usr/bin/env python3
"""
Worker 2: Traditional ML Visualization and Analysis

Creates:
1. Training curves (loss, accuracy, F1)
2. Confusion matrix
3. Per-class performance breakdown
4. Attention weight visualization
5. Error analysis
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


class TrainingAnalyzer:
    """Analyze and visualize training results."""
    
    def __init__(self, results_path: Path, output_dir: Optional[Path] = None):
        """
        Initialize analyzer.
        
        Args:
            results_path: Path to training_results.json
            output_dir: Directory for output plots (defaults to results_path parent / 'analysis')
        """
        self.results_path = results_path
        
        with open(results_path, 'r') as f:
            self.results = json.load(f)
        
        if output_dir is None:
            self.output_dir = results_path.parent / 'analysis'
        else:
            self.output_dir = Path(output_dir)
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set style
        sns.set_style('whitegrid')
        plt.rcParams['figure.figsize'] = (12, 8)
        plt.rcParams['font.size'] = 10
    
    def plot_training_curves(self):
        """Plot training and validation curves."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        train_history = self.results.get('train_history', {})
        val_history = self.results.get('val_history', {})
        
        epochs = range(1, len(train_history.get('loss', [])) + 1)
        
        # Loss
        axes[0, 0].plot(epochs, train_history.get('loss', []), 'b-', label='Train', linewidth=2)
        axes[0, 0].plot(epochs, val_history.get('loss', []), 'r-', label='Validation', linewidth=2)
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Training and Validation Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Mark best epoch
        if 'best_epoch' in self.results:
            best_epoch = self.results['best_epoch']
            axes[0, 0].axvline(x=best_epoch, color='g', linestyle='--', alpha=0.5, label=f'Best Epoch: {best_epoch}')
        
        # Accuracy
        axes[0, 1].plot(epochs, train_history.get('accuracy', []), 'b-', label='Train', linewidth=2)
        axes[0, 1].plot(epochs, val_history.get('accuracy', []), 'r-', label='Validation', linewidth=2)
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].set_title('Training and Validation Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].set_ylim([0.5, 1.0])
        
        # F1 Score
        if 'f1' in train_history:
            axes[1, 0].plot(epochs, train_history.get('f1', []), 'b-', label='Train', linewidth=2)
            axes[1, 0].plot(epochs, val_history.get('f1', []), 'r-', label='Validation', linewidth=2)
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('F1 Score')
            axes[1, 0].set_title('Training and Validation F1 Score')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
            axes[1, 0].set_ylim([0.5, 1.0])
        
        # Overfitting indicator (train-val gap)
        if 'accuracy' in train_history and 'accuracy' in val_history:
            gap = np.array(train_history['accuracy']) - np.array(val_history['accuracy'])
            axes[1, 1].plot(epochs, gap, 'purple', linewidth=2)
            axes[1, 1].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
            axes[1, 1].fill_between(epochs, 0, gap, where=(gap >= 0), alpha=0.3, color='red', label='Overfitting')
            axes[1, 1].fill_between(epochs, 0, gap, where=(gap < 0), alpha=0.3, color='green', label='Underfitting')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Train - Val Accuracy')
            axes[1, 1].set_title('Generalization Gap (Overfitting Indicator)')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        output_path = self.output_dir / 'training_curves.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Training curves saved to {output_path}")
        
        return output_path
    
    def plot_confusion_matrix(self, confusion_matrix: Optional[np.ndarray] = None, 
                             class_names: Optional[List[str]] = None):
        """
        Plot confusion matrix.
        
        Args:
            confusion_matrix: 2x2 confusion matrix (if available)
            class_names: Names for classes
        """
        if confusion_matrix is None:
            # Try to extract from results
            if 'confusion_matrix' in self.results.get('test_results', {}):
                confusion_matrix = np.array(self.results['test_results']['confusion_matrix'])
            else:
                print("⚠️  No confusion matrix data available")
                return None
        
        if class_names is None:
            class_names = ['Non-Persistent', 'Persistent']
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names,
                   cbar_kws={'label': 'Count'}, ax=ax)
        
        ax.set_xlabel('Predicted Label')
        ax.set_ylabel('True Label')
        ax.set_title('Confusion Matrix - Test Set')
        
        # Add percentages
        total = confusion_matrix.sum()
        for i in range(len(class_names)):
            for j in range(len(class_names)):
                count = confusion_matrix[i, j]
                percentage = 100 * count / total
                ax.text(j + 0.5, i + 0.7, f'({percentage:.1f}%)', 
                       ha='center', va='center', fontsize=9, color='gray')
        
        plt.tight_layout()
        
        output_path = self.output_dir / 'confusion_matrix.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Confusion matrix saved to {output_path}")
        
        return output_path
    
    def plot_metrics_summary(self):
        """Plot summary of test metrics."""
        test_results = self.results.get('test_results', {})
        
        if not test_results:
            print("⚠️  No test results available")
            return None
        
        metrics = {
            'Accuracy': test_results.get('accuracy', 0),
            'Precision': test_results.get('precision', 0),
            'Recall': test_results.get('recall', 0),
            'F1 Score': test_results.get('f1', 0)
        }
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        bars = ax.bar(metrics.keys(), metrics.values(), color=['#3498db', '#2ecc71', '#e74c3c', '#f39c12'])
        ax.set_ylabel('Score')
        ax.set_title('Test Set Performance Metrics')
        ax.set_ylim([0, 1.0])
        ax.axhline(y=0.95, color='green', linestyle='--', alpha=0.5, label='Target (95%)')
        ax.axhline(y=0.85, color='orange', linestyle='--', alpha=0.5, label='Minimum Target (85%)')
        ax.grid(True, alpha=0.3, axis='y')
        ax.legend()
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.4f}',
                   ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        plt.tight_layout()
        
        output_path = self.output_dir / 'metrics_summary.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Metrics summary saved to {output_path}")
        
        return output_path
    
    def plot_learning_rate_schedule(self):
        """Plot learning rate schedule if available."""
        if 'learning_rates' not in self.results.get('train_history', {}):
            print("⚠️  No learning rate data available")
            return None
        
        learning_rates = self.results['train_history']['learning_rates']
        epochs = range(1, len(learning_rates) + 1)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(epochs, learning_rates, 'b-', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Learning Rate')
        ax.set_title('Learning Rate Schedule')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        output_path = self.output_dir / 'learning_rate.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Learning rate plot saved to {output_path}")
        
        return output_path
    
    def create_analysis_report(self) -> str:
        """Create text analysis report."""
        report = []
        report.append("=" * 70)
        report.append("TRAINING ANALYSIS REPORT")
        report.append("=" * 70)
        report.append("")
        
        # Training summary
        report.append("## Training Summary")
        report.append(f"- Epochs: {self.results.get('num_epochs', 'N/A')}")
        report.append(f"- Best Epoch: {self.results.get('best_epoch', 'N/A')}")
        report.append(f"- Best Val Loss: {self.results.get('best_val_loss', 'N/A'):.6f}" if 'best_val_loss' in self.results else "- Best Val Loss: N/A")
        report.append("")
        
        # Test results
        test_results = self.results.get('test_results', {})
        report.append("## Test Set Performance")
        report.append(f"- Accuracy: {test_results.get('accuracy', 0):.4f} ({test_results.get('accuracy', 0)*100:.2f}%)")
        report.append(f"- Precision: {test_results.get('precision', 0):.4f}")
        report.append(f"- Recall: {test_results.get('recall', 0):.4f}")
        report.append(f"- F1 Score: {test_results.get('f1', 0):.4f}")
        report.append("")
        
        # Performance vs target
        target_acc = 0.95
        achieved_acc = test_results.get('accuracy', 0)
        report.append("## Target Comparison")
        report.append(f"- Target Accuracy: 85-95%")
        report.append(f"- Achieved Accuracy: {achieved_acc*100:.2f}%")
        if achieved_acc >= target_acc:
            report.append(f"- Status: ✅ EXCEEDED TARGET by {(achieved_acc - target_acc)*100:.2f}%!")
        elif achieved_acc >= 0.85:
            report.append(f"- Status: ✅ WITHIN TARGET RANGE")
        else:
            report.append(f"- Status: ⚠️  BELOW TARGET")
        report.append("")
        
        # Overfitting analysis
        train_history = self.results.get('train_history', {})
        val_history = self.results.get('val_history', {})
        
        if 'accuracy' in train_history and 'accuracy' in val_history:
            final_train_acc = train_history['accuracy'][-1]
            final_val_acc = val_history['accuracy'][-1]
            gap = final_train_acc - final_val_acc
            
            report.append("## Generalization Analysis")
            report.append(f"- Final Train Accuracy: {final_train_acc:.4f}")
            report.append(f"- Final Val Accuracy: {final_val_acc:.4f}")
            report.append(f"- Train-Val Gap: {gap:.4f} ({gap*100:.2f}%)")
            
            if abs(gap) < 0.02:
                report.append("- Assessment: ✅ Excellent generalization (minimal overfitting)")
            elif abs(gap) < 0.05:
                report.append("- Assessment: ✅ Good generalization")
            elif gap > 0.05:
                report.append("- Assessment: ⚠️  Some overfitting detected")
            else:
                report.append("- Assessment: ⚠️  Model may be underfitting")
            report.append("")
        
        # Convergence analysis
        if 'loss' in val_history:
            val_losses = val_history['loss']
            best_epoch = self.results.get('best_epoch', len(val_losses))
            epochs_after_best = len(val_losses) - best_epoch
            
            report.append("## Convergence Analysis")
            report.append(f"- Best epoch: {best_epoch}/{len(val_losses)}")
            report.append(f"- Epochs after best: {epochs_after_best}")
            
            if epochs_after_best > 5:
                report.append("- Recommendation: Consider early stopping at epoch {}".format(best_epoch + 3))
            else:
                report.append("- Assessment: ✅ Good convergence timing")
            report.append("")
        
        # Recommendations
        report.append("## Recommendations")
        if achieved_acc >= 0.98:
            report.append("- ✅ Model performance is excellent (>98%)")
            report.append("- Ready for production deployment")
            report.append("- Consider Phase 2 (MagVit) for marginal improvements")
        elif achieved_acc >= 0.95:
            report.append("- ✅ Model performance meets target")
            report.append("- Ready for production with monitoring")
            report.append("- Phase 2 (MagVit) may provide incremental improvements")
        else:
            report.append("- ⚠️  Consider improvements:")
            report.append("  - Expand dataset")
            report.append("  - Try Phase 2 with visual features")
            report.append("  - Tune hyperparameters")
        
        report.append("")
        report.append("=" * 70)
        
        report_text = "\n".join(report)
        
        # Save report
        output_path = self.output_dir / 'analysis_report.txt'
        with open(output_path, 'w') as f:
            f.write(report_text)
        
        print(f"✅ Analysis report saved to {output_path}")
        print("\n" + report_text)
        
        return report_text
    
    def run_full_analysis(self):
        """Run complete analysis pipeline."""
        print("\n" + "="*70)
        print("Running Full Training Analysis")
        print("="*70 + "\n")
        
        outputs = {}
        
        # 1. Training curves
        print("\n📈 Generating training curves...")
        outputs['training_curves'] = self.plot_training_curves()
        
        # 2. Metrics summary
        print("\n📊 Generating metrics summary...")
        outputs['metrics_summary'] = self.plot_metrics_summary()
        
        # 3. Confusion matrix
        print("\n🎯 Generating confusion matrix...")
        outputs['confusion_matrix'] = self.plot_confusion_matrix()
        
        # 4. Learning rate schedule
        print("\n📉 Generating learning rate plot...")
        outputs['learning_rate'] = self.plot_learning_rate_schedule()
        
        # 5. Text report
        print("\n📝 Generating analysis report...")
        outputs['report'] = self.create_analysis_report()
        
        print("\n" + "="*70)
        print("✅ Full Analysis Complete")
        print("="*70)
        print(f"\nAll outputs saved to: {self.output_dir}")
        
        return outputs


def main():
    """Run training analysis."""
    import argparse
    import logging
    
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    parser = argparse.ArgumentParser(description='Analyze training results')
    parser.add_argument('--results', type=str, required=True,
                       help='Path to training_results.json')
    parser.add_argument('--output', type=str, default=None,
                       help='Output directory for plots')
    
    args = parser.parse_args()
    
    results_path = Path(args.results)
    if not results_path.exists():
        logger.error(f"Results file not found: {results_path}")
        return
    
    logger.info("="*70)
    logger.info("Worker 2: Traditional Training Analysis")
    logger.info("="*70)
    logger.info(f"Results: {results_path}")
    
    # Create analyzer
    output_dir = Path(args.output) if args.output else None
    analyzer = TrainingAnalyzer(results_path, output_dir=output_dir)
    
    # Run full analysis
    analyzer.run_full_analysis()
    
    logger.info("\n" + "="*70)
    logger.info("✅ WORKER 2 COMPLETE")
    logger.info("="*70)


if __name__ == '__main__':
    main()

