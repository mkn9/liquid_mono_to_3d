"""
TDD Tests for Liquid NN Trajectory Visualizations
RED Phase: Tests written FIRST, will fail until implementation
"""

import pytest
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

class TestLiquidTrajectoryVisualization:
    """Test suite for Liquid NN trajectory visualization functions."""
    
    def test_create_trajectory_comparison_function_exists(self):
        """Test that create_trajectory_comparison function exists."""
        from create_liquid_trajectory_viz import create_trajectory_comparison
        assert callable(create_trajectory_comparison)
    
    def test_create_trajectory_comparison_returns_path(self):
        """Test that visualization function returns output path."""
        from create_liquid_trajectory_viz import create_trajectory_comparison
        output_path = create_trajectory_comparison()
        assert isinstance(output_path, Path)
        assert output_path.exists()
        assert output_path.suffix == '.png'
    
    def test_create_trajectory_comparison_has_correct_naming(self):
        """Test that output file follows naming convention YYYYMMDD_HHMM."""
        from create_liquid_trajectory_viz import create_trajectory_comparison
        output_path = create_trajectory_comparison()
        filename = output_path.name
        # Should start with YYYYMMDD_HHMM
        assert len(filename.split('_')[0]) == 8  # YYYYMMDD
        assert len(filename.split('_')[1]) == 4  # HHMM
        assert 'liquid_trajectory' in filename
    
    def test_create_trajectory_grid_function_exists(self):
        """Test that create_trajectory_grid function exists."""
        from create_liquid_trajectory_viz import create_trajectory_grid
        assert callable(create_trajectory_grid)
    
    def test_create_trajectory_grid_returns_path(self):
        """Test that grid visualization function returns output path."""
        from create_liquid_trajectory_viz import create_trajectory_grid
        output_path = create_trajectory_grid(num_samples=4)  # Small number for testing
        assert isinstance(output_path, Path)
        assert output_path.exists()
        assert output_path.suffix == '.png'
    
    def test_create_jitter_analysis_function_exists(self):
        """Test that create_jitter_analysis function exists."""
        from create_liquid_trajectory_viz import create_jitter_analysis
        assert callable(create_jitter_analysis)
    
    def test_create_jitter_analysis_returns_path(self):
        """Test that jitter analysis function returns output path."""
        from create_liquid_trajectory_viz import create_jitter_analysis
        output_path = create_jitter_analysis()
        assert isinstance(output_path, Path)
        assert output_path.exists()
        assert output_path.suffix == '.png'
    
    def test_visualizations_save_to_correct_directory(self):
        """Test that all visualizations save to results directory."""
        from create_liquid_trajectory_viz import create_trajectory_comparison
        output_path = create_trajectory_comparison()
        assert 'results' in str(output_path)
        assert 'experiments/liquid_vlm_integration' in str(output_path)
    
    def test_calculate_jerk_function(self):
        """Test jerk calculation helper function."""
        from create_liquid_trajectory_viz import calculate_jerk
        
        # Simple linear trajectory (should have low jerk)
        linear = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2], [3, 3, 3], [4, 4, 4]])
        jerk_linear = calculate_jerk(linear)
        
        # Noisy trajectory (should have high jerk)
        noisy = linear + np.random.randn(*linear.shape) * 0.5
        jerk_noisy = calculate_jerk(noisy)
        
        assert isinstance(jerk_linear, float)
        assert isinstance(jerk_noisy, float)
        assert jerk_noisy > jerk_linear  # Noisy should have higher jerk


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

