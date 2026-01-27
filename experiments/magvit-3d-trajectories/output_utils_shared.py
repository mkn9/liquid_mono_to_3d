#!/usr/bin/env python3
"""
Shared Output Utilities for All Experiments

Provides timestamped filename generation for all results/ directories.
Complies with requirements.md Section 5.4: YYYYMMDD_HHMM_descriptive_name.ext

This module can be imported by any experiment to ensure consistent
timestamped output naming across the entire project.
"""

from pathlib import Path
from datetime import datetime
from typing import Union, Any
import json


def get_timestamped_filename(base_name: str, extension: str) -> str:
    """
    Generate timestamped filename following YYYYMMDD_HHMM format.
    
    Args:
        base_name: Descriptive name (e.g., "trajectory_comparison")
        extension: File extension without dot (e.g., "png", "csv")
        
    Returns:
        Timestamped filename string (e.g., "20260120_1430_trajectory_comparison.png")
        
    Example:
        >>> get_timestamped_filename("test_output", "png")
        "20260120_1430_test_output.png"
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    ext = extension.lstrip('.')
    return f"{timestamp}_{base_name}.{ext}"


def _needs_timestamp(filename: str) -> bool:
    """Check if filename already has timestamp prefix."""
    first_part = filename.split('_')[0]
    return not (first_part.isdigit() and len(first_part) == 8)


def get_results_path(base_dir: Union[str, Path], filename: str) -> Path:
    """
    Get full path in results/ subdirectory, creating directory if needed.
    
    Args:
        base_dir: Base directory (if already 'results/', used directly; 
                  otherwise results/ subdirectory is created)
        filename: Filename to save (timestamp added if not present)
        
    Returns:
        Full Path object for file in results/ subdirectory
        
    Examples:
        >>> get_results_path("experiments/my_exp", "output.png")
        Path("experiments/my_exp/results/20260120_1430_output.png")
        
        >>> get_results_path("experiments/my_exp/results", "output.png")
        Path("experiments/my_exp/results/20260120_1430_output.png")
    """
    base_path = Path(base_dir)
    
    # Handle case where base_dir already points to results/
    if base_path.name == "results":
        results_dir = base_path
    else:
        results_dir = base_path / "results"
    
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Add timestamp if not already present
    if _needs_timestamp(filename):
        name_parts = filename.rsplit('.', 1)
        if len(name_parts) == 2:
            base_name, ext = name_parts
            filename = get_timestamped_filename(base_name, ext)
    
    return results_dir / filename


def save_figure(fig, results_dir: Union[str, Path], filename: str, 
                dpi: int = 150, **kwargs) -> Path:
    """
    Save matplotlib figure with timestamped filename to results/ directory.
    
    Args:
        fig: Matplotlib figure object
        results_dir: Directory containing (or to contain) results/ subdirectory
        filename: Base filename (e.g., "trajectory_plot.png")
        dpi: Resolution in dots per inch (default: 150)
        **kwargs: Additional arguments passed to fig.savefig()
        
    Returns:
        Path object pointing to saved file
        
    Example:
        >>> fig, ax = plt.subplots()
        >>> ax.plot([1,2,3])
        >>> save_figure(fig, "experiments/magvit", "trajectory.png")
        Path("experiments/magvit/results/20260120_1430_trajectory.png")
    """
    output_path = get_results_path(results_dir, filename)
    fig.savefig(output_path, dpi=dpi, bbox_inches='tight', **kwargs)
    print(f"✅ Saved figure: {output_path}")
    return output_path


def save_data(data: Any, results_dir: Union[str, Path], filename: str, 
              **kwargs) -> Path:
    """
    Save data with timestamped filename to results/ directory.
    
    Supports multiple formats:
    - CSV: pandas DataFrame or array-like data
    - JSON: dict or JSON-serializable object
    - NPZ: numpy arrays (dict of arrays or single array)
    - NPY: single numpy array
    
    Args:
        data: Data to save (DataFrame, dict, numpy array, etc.)
        results_dir: Directory containing (or to contain) results/ subdirectory
        filename: Base filename determining format (e.g., "data.csv")
        **kwargs: Additional arguments passed to format-specific save function
        
    Returns:
        Path object pointing to saved file
        
    Example:
        >>> df = pd.DataFrame({'x': [1,2,3], 'y': [4,5,6]})
        >>> save_data(df, "experiments/magvit", "trajectory.csv")
        Path("experiments/magvit/results/20260120_1430_trajectory.csv")
    """
    output_path = get_results_path(results_dir, filename)
    extension = output_path.suffix.lstrip('.')
    
    if extension == 'csv':
        import pandas as pd
        if isinstance(data, pd.DataFrame):
            data.to_csv(output_path, **kwargs)
        else:
            pd.DataFrame(data).to_csv(output_path, **kwargs)
    elif extension == 'json':
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2, **kwargs)
    elif extension == 'npz':
        import numpy as np
        if isinstance(data, dict):
            np.savez(output_path, **data)
        else:
            np.savez(output_path, data=data)
    elif extension == 'npy':
        import numpy as np
        np.save(output_path, data, **kwargs)
    else:
        raise ValueError(
            f"Unsupported extension: {extension}. "
            f"Supported: csv, json, npz, npy"
        )
    
    print(f"✅ Saved data: {output_path}")
    return output_path


if __name__ == "__main__":
    # Demonstration
    print("=== Output Utilities Demo ===\n")
    
    # Example 1: Generate timestamped filename
    filename = get_timestamped_filename("test_output", "png")
    print(f"1. Timestamped filename: {filename}")
    
    # Example 2: Get results path
    path = get_results_path("experiments/test", "output.png")
    print(f"2. Results path: {path}")
