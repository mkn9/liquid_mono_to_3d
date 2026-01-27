#!/usr/bin/env python3
"""
Shared Utilities for Future Prediction Experiments
==================================================
Common functions for all 3 parallel branches.
"""

import torch
import torch.nn as nn
import numpy as np
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List
import json
import sys

# Setup logging
def setup_logging(task_name: str, output_dir: Path) -> logging.Logger:
    """Setup logging for a task."""
    log_dir = output_dir / 'logs'
    log_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = log_dir / f"{timestamp}_{task_name}.log"
    
    logger = logging.getLogger(task_name)
    logger.setLevel(logging.DEBUG)
    logger.handlers = []  # Clear existing handlers
    
    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter('%(levelname)s - %(message)s')
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    logger.info(f"Logging set up for: {task_name}")
    logger.info(f"Log file: {log_file}")
    return logger


def save_results(results: Dict[str, Any], output_dir: Path, task_name: str):
    """Save results to JSON file."""
    results_dir = output_dir / 'results'
    results_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filepath = results_dir / f"{timestamp}_{task_name}_results.json"
    
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    logging.getLogger(task_name).info(f"✅ Results saved to: {filepath}")
    return filepath


def load_trajectory_dataset(data_path: str, logger: logging.Logger) -> Dict[str, np.ndarray]:
    """Load trajectory video dataset."""
    logger.info(f"Loading dataset from: {data_path}")
    
    try:
        # Load from npz file
        data = np.load(data_path)
        
        dataset = {
            'train_videos': data['train_features'] if 'train_features' in data else None,
            'train_labels': data['train_labels'] if 'train_labels' in data else None,
            'test_videos': data['test_features'] if 'test_features' in data else None,
            'test_labels': data['test_labels'] if 'test_labels' in data else None,
            'class_names': data['class_names'] if 'class_names' in data else None
        }
        
        logger.info(f"✅ Dataset loaded:")
        for key, val in dataset.items():
            if val is not None:
                logger.info(f"   {key}: {val.shape if hasattr(val, 'shape') else len(val)}")
        
        return dataset
        
    except Exception as e:
        logger.error(f"❌ Failed to load dataset: {e}")
        return {}


def create_video_dataset_for_prediction(num_samples: int = 100, device: str = 'cuda') -> torch.Tensor:
    """
    Create synthetic trajectory videos for testing.
    
    Returns:
        videos: (N, 3, T, H, W) tensor of videos
    """
    T = 50  # 50 frames total
    H, W = 128, 128
    
    videos = torch.zeros(num_samples, 3, T, H, W, device=device)
    
    for i in range(num_samples):
        # Random trajectory parameters
        start_x = np.random.randint(20, 40)
        start_y = np.random.randint(20, 40)
        velocity_x = np.random.uniform(-2, 2)
        velocity_y = np.random.uniform(-2, 2)
        
        for t in range(T):
            # Calculate position
            x = int(start_x + velocity_x * t)
            y = int(start_y + velocity_y * t)
            
            # Keep in bounds
            x = max(5, min(H-6, x))
            y = max(5, min(W-6, y))
            
            # Draw circle (red channel for simplicity)
            for dx in range(-5, 6):
                for dy in range(-5, 6):
                    if dx*dx + dy*dy <= 25:  # radius 5
                        videos[i, 0, t, x+dx, y+dy] = 1.0
    
    return videos


class SimpleTransformer(nn.Module):
    """Simple Transformer for future frame prediction."""
    
    def __init__(
        self,
        embed_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 12,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.embed_dim = embed_dim
        
        # Positional encoding
        self.pos_encoding = nn.Parameter(torch.randn(1, 1000, embed_dim))
        
        # Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output projection
        self.output_proj = nn.Linear(embed_dim, embed_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, C) sequence of encoded features
        Returns:
            out: (B, T, C) predicted features
        """
        B, T, C = x.shape
        
        # Add positional encoding
        x = x + self.pos_encoding[:, :T, :]
        
        # Transform
        out = self.transformer(x)
        out = self.output_proj(out)
        
        return out


def compute_metrics(pred: torch.Tensor, target: torch.Tensor, logger: logging.Logger) -> Dict[str, float]:
    """Compute evaluation metrics."""
    
    # MSE
    mse = nn.functional.mse_loss(pred, target).item()
    
    # PSNR
    psnr = 10 * torch.log10(torch.tensor(1.0) / (mse + 1e-10)).item()
    
    # MAE
    mae = nn.functional.l1_loss(pred, target).item()
    
    metrics = {
        'mse': mse,
        'psnr': psnr,
        'mae': mae
    }
    
    logger.info(f"Metrics: MSE={mse:.6f}, PSNR={psnr:.2f}, MAE={mae:.6f}")
    
    return metrics


def run_test_suite(test_functions: List[callable], logger: logging.Logger) -> Dict[str, Any]:
    """Run a suite of test functions."""
    results = {
        'total_tests': len(test_functions),
        'passed': 0,
        'failed': 0,
        'details': {}
    }
    
    logger.info("=" * 60)
    logger.info("Running Test Suite")
    logger.info("=" * 60)
    
    for test_func in test_functions:
        test_name = test_func.__name__
        logger.info(f"\n--- Test: {test_name} ---")
        
        try:
            result = test_func(logger)
            results['details'][test_name] = {'status': 'passed', 'output': result}
            results['passed'] += 1
            logger.info(f"✅ PASSED: {test_name}")
        except Exception as e:
            import traceback
            error_msg = str(e)
            traceback_str = traceback.format_exc()
            results['details'][test_name] = {
                'status': 'failed',
                'error': error_msg,
                'traceback': traceback_str
            }
            results['failed'] += 1
            logger.error(f"❌ FAILED: {test_name}")
            logger.error(f"Error: {error_msg}")
            logger.error(traceback_str)
    
    logger.info("\n" + "=" * 60)
    logger.info("Test Suite Summary")
    logger.info("=" * 60)
    logger.info(f"Total: {results['total_tests']}, Passed: {results['passed']}, Failed: {results['failed']}")
    logger.info(f"Success Rate: {results['passed']/results['total_tests']*100:.1f}%")
    
    return results

