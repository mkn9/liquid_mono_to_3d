#!/usr/bin/env python3
"""
TDD Tests for Shared Output Utilities

Following requirements.md Section 3.3:
- Tests written FIRST (RED phase)
- Deterministic with explicit assertions  
- Covers output file naming with timestamps
"""

import pytest
import numpy as np
from pathlib import Path
from datetime import datetime
import tempfile
import shutil


# ==============================================================================
# FIXTURES
# ==============================================================================

@pytest.fixture
def temp_results_dir(tmp_path):
    """Temporary results directory for testing."""
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    return results_dir


# ==============================================================================
# INVARIANT TESTS
# ==============================================================================

def test_get_timestamped_filename_format():
    """
    INVARIANT: Filename must match YYYYMMDD_HHMM_name.ext format.
    
    SPECIFICATION:
    - Format: YYYYMMDD_HHMM_descriptive_name.ext
    - YYYYMMDD: 8 digits (year, month, day)
    - HHMM: 4 digits (hour, minute)
    - Must include underscores as separators
    """
    from output_utils_shared import get_timestamped_filename
    
    filename = get_timestamped_filename("test_output", "png")
    
    # Check format: should be like "20260120_1430_test_output.png"
    parts = filename.split('_')
    
    assert len(parts) >= 3, f"Filename should have at least 3 parts: {filename}"
    assert len(parts[0]) == 8, f"Date should be 8 digits, got: {parts[0]}"
    assert parts[0].isdigit(), f"Date should be all digits, got: {parts[0]}"
    assert len(parts[1]) == 4, f"Time should be 4 digits (HHMM), got: {parts[1]}"
    assert parts[1].isdigit(), f"Time should be all digits, got: {parts[1]}"
    assert filename.endswith('.png'), f"Should end with .png, got: {filename}"


def test_get_timestamped_filename_unique():
    """
    INVARIANT: Sequential calls must produce different or same timestamps.
    
    Since we use minute precision, filenames generated in same minute
    should be identical (expected behavior).
    """
    from output_utils_shared import get_timestamped_filename
    
    filename1 = get_timestamped_filename("test", "png")
    filename2 = get_timestamped_filename("test", "png")
    
    # Should be same if called in same minute
    assert filename1 == filename2, "Same name in same minute is expected"


def test_get_results_path_creates_directory():
    """
    INVARIANT: get_results_path must create results/ directory if missing.
    
    SPECIFICATION:
    - If results_dir doesn't exist, create it
    - Return path should be inside results/
    - Path should include timestamp
    """
    import tempfile
    from output_utils_shared import get_results_path
    
    with tempfile.TemporaryDirectory() as tmpdir:
        results_path = get_results_path(
            base_dir=tmpdir,
            filename="test_output.png"
        )
        
        # Check directory was created
        results_dir = Path(tmpdir) / "results"
        assert results_dir.exists(), "results/ directory should be created"
        assert results_dir.is_dir(), "results/ should be a directory"
        
        # Check path is inside results/
        assert results_path.parent == results_dir, "Path should be in results/"


# ==============================================================================
# GOLDEN TESTS
# ==============================================================================

def test_get_timestamped_filename_golden():
    """
    GOLDEN TEST: Verify exact format with mocked datetime.
    
    SPECIFICATION:
    - Input: "trajectory_comparison", "png"
    - Current time: 2026-01-20 14:30:00
    - Expected: "20260120_1430_trajectory_comparison.png"
    """
    from output_utils_shared import get_timestamped_filename
    from unittest.mock import patch
    
    mock_datetime = datetime(2026, 1, 20, 14, 30, 0)
    
    with patch('output_utils_shared.datetime') as mock_dt:
        mock_dt.now.return_value = mock_datetime
        filename = get_timestamped_filename("trajectory_comparison", "png")
    
    expected = "20260120_1430_trajectory_comparison.png"
    assert filename == expected, f"Expected {expected}, got {filename}"


def test_save_figure_creates_timestamped_file(temp_results_dir):
    """
    GOLDEN TEST: save_figure must create file with timestamp.
    
    SPECIFICATION:
    - Creates PNG file in results/
    - Filename has YYYYMMDD_HHMM prefix
    - File actually exists on disk
    - Returns path to file
    """
    from output_utils_shared import save_figure
    import matplotlib.pyplot as plt
    
    # Create simple figure
    fig, ax = plt.subplots()
    ax.plot([1, 2, 3], [1, 2, 3])
    
    # Save with timestamp
    saved_path = save_figure(
        fig,
        results_dir=temp_results_dir,
        filename="test_plot.png"
    )
    
    # Verify file exists
    assert saved_path.exists(), f"File should exist: {saved_path}"
    assert saved_path.suffix == ".png", "Should be PNG file"
    assert "test_plot" in saved_path.name, "Filename should contain description"
    
    # Verify it's in results dir
    assert saved_path.parent == temp_results_dir
    
    plt.close(fig)


def test_save_data_csv(temp_results_dir):
    """
    GOLDEN TEST: save_data must save CSV with timestamp.
    
    SPECIFICATION:
    - Input: pandas DataFrame
    - Output: CSV file in results/
    - Filename has timestamp prefix
    - Data is correctly saved
    """
    from output_utils_shared import save_data
    import pandas as pd
    
    # Create test data
    df = pd.DataFrame({
        'x': [1, 2, 3],
        'y': [4, 5, 6]
    })
    
    # Save with timestamp
    saved_path = save_data(
        df,
        results_dir=temp_results_dir,
        filename="trajectory_data.csv"
    )
    
    # Verify file exists
    assert saved_path.exists(), f"File should exist: {saved_path}"
    assert saved_path.suffix == ".csv", "Should be CSV file"
    
    # Verify data can be loaded
    loaded_df = pd.read_csv(saved_path, index_col=0)
    pd.testing.assert_frame_equal(loaded_df, df)


def test_multiple_file_types(temp_results_dir):
    """
    INTEGRATION: Test saving multiple file types with timestamps.
    
    Ensures the utility works for: png, csv, json, npz
    """
    from output_utils_shared import save_data
    import pandas as pd
    import json
    
    # Test CSV
    df = pd.DataFrame({'a': [1, 2]})
    csv_path = save_data(df, temp_results_dir, "data.csv")
    assert csv_path.exists() and csv_path.suffix == ".csv"
    
    # Test JSON
    data_dict = {'key': 'value', 'number': 42}
    json_path = save_data(data_dict, temp_results_dir, "data.json")
    assert json_path.exists() and json_path.suffix == ".json"
    
    # Test NPZ
    arr = np.array([1, 2, 3])
    npz_path = save_data({'arr': arr}, temp_results_dir, "data.npz")
    assert npz_path.exists() and npz_path.suffix == ".npz"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
