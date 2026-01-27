#!/usr/bin/env python3
"""
Output utility functions for consistent file naming in basic/ folder.
All outputs should use timestamped naming: YYYYMMDD_HHMMSS_descriptive_name.ext
"""

import os
from datetime import datetime
from pathlib import Path

# Base output directory
OUTPUT_DIR = Path(__file__).parent / 'output'

def get_output_path(descriptive_name, extension='png', create_dir=True):
    """
    Generate a timestamped output file path.
    
    Args:
        descriptive_name: Brief description of the output (e.g., '3d_track_comparison')
        extension: File extension (default: 'png')
        create_dir: Create output directory if it doesn't exist (default: True)
    
    Returns:
        Path object for the output file
    
    Example:
        path = get_output_path('3d_track_comparison', 'png')
        # Returns: basic/output/20251210_120530_3d_track_comparison.png
    """
    if create_dir:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Generate timestamp: YYYYMMDD_HHMMSS
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Create filename: YYYYMMDD_HHMMSS_descriptive_name.ext
    filename = f'{timestamp}_{descriptive_name}.{extension}'
    
    return OUTPUT_DIR / filename

def save_figure(fig, descriptive_name, extension='png', dpi=150, **kwargs):
    """
    Save a matplotlib figure with timestamped filename.
    
    Args:
        fig: Matplotlib figure object
        descriptive_name: Brief description of the output
        extension: File extension (default: 'png')
        dpi: Resolution (default: 150)
        **kwargs: Additional arguments passed to fig.savefig()
    
    Returns:
        Path to saved file
    """
    output_path = get_output_path(descriptive_name, extension)
    fig.savefig(output_path, dpi=dpi, bbox_inches='tight', **kwargs)
    print(f'Saved: {output_path}')
    return output_path

def save_data(data, descriptive_name, extension='csv', **kwargs):
    """
    Save data (DataFrame, array, etc.) with timestamped filename.
    
    Args:
        data: Data to save (pandas DataFrame, numpy array, etc.)
        descriptive_name: Brief description of the output
        extension: File extension (default: 'csv')
        **kwargs: Additional arguments passed to save function
    
    Returns:
        Path to saved file
    """
    output_path = get_output_path(descriptive_name, extension)
    
    if extension == 'csv':
        import pandas as pd
        if isinstance(data, pd.DataFrame):
            data.to_csv(output_path, **kwargs)
        else:
            pd.DataFrame(data).to_csv(output_path, **kwargs)
    elif extension == 'npy':
        import numpy as np
        np.save(output_path, data, **kwargs)
    elif extension == 'json':
        import json
        with open(output_path, 'w') as f:
            json.dump(data, f, **kwargs)
    else:
        raise ValueError(f'Unsupported extension: {extension}')
    
    print(f'Saved: {output_path}')
    return output_path

# Example usage:
if __name__ == '__main__':
    # Example 1: Get a path for saving a figure
    path = get_output_path('test_visualization', 'png')
    print(f'Example output path: {path}')
    
    # Example 2: Get a path for saving data
    path = get_output_path('track_data', 'csv')
    print(f'Example data path: {path}')
