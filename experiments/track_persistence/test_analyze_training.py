#!/usr/bin/env python3
"""Test traditional training analysis."""

import json
import sys
import tempfile
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Create mock training results
mock_results = {
    "num_epochs": 30,
    "best_epoch": 28,
    "best_val_loss": 0.038,
    "timestamp": "2026-01-16T16:38:00",
    "train_history": {
        "loss": list(np.linspace(0.5, 0.05, 30)),
        "accuracy": list(np.linspace(0.75, 0.985, 30)),
        "f1": list(np.linspace(0.73, 0.982, 30))
    },
    "val_history": {
        "loss": list(np.linspace(0.52, 0.055, 30)),
        "accuracy": list(np.linspace(0.73, 0.993, 30)),
        "f1": list(np.linspace(0.71, 0.990, 30))
    },
    "test_results": {
        "accuracy": 0.9867,
        "f1": 0.9846,
        "precision": 0.9823,
        "recall": 0.9871,
        "confusion_matrix": [[185, 5], [5, 180]]
    }
}

def test_analyzer_creation():
    """Test analyzer can be created."""
    try:
        from analyze_training import TrainingAnalyzer
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(mock_results, f)
            temp_path = Path(f.name)
        
        try:
            analyzer = TrainingAnalyzer(temp_path)
            assert analyzer.results == mock_results
            print("✅ TrainingAnalyzer initialized successfully")
            return True
        finally:
            temp_path.unlink()
        
    except Exception as e:
        print(f"❌ Failed to create analyzer: {e}")
        return False

def test_plot_training_curves():
    """Test training curves plotting."""
    try:
        from analyze_training import TrainingAnalyzer
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(mock_results, f)
            temp_path = Path(f.name)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            try:
                analyzer = TrainingAnalyzer(temp_path, output_dir=Path(tmpdir))
                output = analyzer.plot_training_curves()
                
                assert output.exists()
                assert output.name == 'training_curves.png'
                print("✅ Training curves plotted successfully")
                return True
            finally:
                temp_path.unlink()
        
    except Exception as e:
        print(f"❌ Training curves plotting failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_plot_confusion_matrix():
    """Test confusion matrix plotting."""
    try:
        from analyze_training import TrainingAnalyzer
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(mock_results, f)
            temp_path = Path(f.name)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            try:
                analyzer = TrainingAnalyzer(temp_path, output_dir=Path(tmpdir))
                output = analyzer.plot_confusion_matrix()
                
                assert output.exists()
                assert output.name == 'confusion_matrix.png'
                print("✅ Confusion matrix plotted successfully")
                return True
            finally:
                temp_path.unlink()
        
    except Exception as e:
        print(f"❌ Confusion matrix plotting failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_plot_metrics_summary():
    """Test metrics summary plotting."""
    try:
        from analyze_training import TrainingAnalyzer
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(mock_results, f)
            temp_path = Path(f.name)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            try:
                analyzer = TrainingAnalyzer(temp_path, output_dir=Path(tmpdir))
                output = analyzer.plot_metrics_summary()
                
                assert output.exists()
                assert output.name == 'metrics_summary.png'
                print("✅ Metrics summary plotted successfully")
                return True
            finally:
                temp_path.unlink()
        
    except Exception as e:
        print(f"❌ Metrics summary plotting failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_create_analysis_report():
    """Test text report generation."""
    try:
        from analyze_training import TrainingAnalyzer
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(mock_results, f)
            temp_path = Path(f.name)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            try:
                analyzer = TrainingAnalyzer(temp_path, output_dir=Path(tmpdir))
                report = analyzer.create_analysis_report()
                
                assert 'TRAINING ANALYSIS REPORT' in report
                assert '0.9867' in report or '98.67' in report
                
                report_file = Path(tmpdir) / 'analysis_report.txt'
                assert report_file.exists()
                
                print("✅ Analysis report generated successfully")
                return True
            finally:
                temp_path.unlink()
        
    except Exception as e:
        print(f"❌ Analysis report generation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    print("="*70)
    print("Testing Traditional Training Analysis")
    print("="*70)
    
    tests = [
        test_analyzer_creation,
        test_plot_training_curves,
        test_plot_confusion_matrix,
        test_plot_metrics_summary,
        test_create_analysis_report
    ]
    
    results = []
    for test in tests:
        print(f"\nRunning {test.__name__}...")
        results.append(test())
    
    print("\n" + "="*70)
    print(f"Results: {sum(results)}/{len(results)} tests passed")
    print("="*70)
    
    if all(results):
        print("✅ ALL TESTS PASSED")
        sys.exit(0)
    else:
        print("❌ SOME TESTS FAILED")
        sys.exit(1)

