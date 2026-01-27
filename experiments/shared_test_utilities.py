#!/usr/bin/env python3
"""
Shared Test Utilities for Parallel Tasks
=========================================
Common testing and debugging utilities for all task scripts.
"""

import sys
import os
import traceback
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional
import json
import numpy as np

# Setup logging
def setup_task_logging(task_name: str, output_dir: Path) -> logging.Logger:
    """Setup logging for a task."""
    log_dir = output_dir / 'logs'
    log_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = log_dir / f"{timestamp}_{task_name}.log"
    
    logger = logging.getLogger(task_name)
    logger.setLevel(logging.DEBUG)
    
    # File handler
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)
    
    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    
    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    return logger


def validate_environment(logger: logging.Logger) -> Dict[str, bool]:
    """Validate computation environment."""
    results = {
        'ec2_environment': False,
        'python_version': False,
        'required_modules': {},
        'gpu_available': False
    }
    
    # Check if running on EC2 (basic check)
    import socket
    hostname = socket.gethostname()
    results['ec2_environment'] = 'ip-' in hostname or 'ubuntu' in hostname.lower()
    logger.info(f"Hostname: {hostname}, EC2: {results['ec2_environment']}")
    
    # Check Python version
    import sys
    python_version = sys.version_info
    results['python_version'] = python_version >= (3, 7)
    logger.info(f"Python version: {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    # Check required modules
    required_modules = ['numpy', 'matplotlib', 'pathlib', 'json']
    for module in required_modules:
        try:
            __import__(module)
            results['required_modules'][module] = True
            logger.debug(f"✅ {module} available")
        except ImportError:
            results['required_modules'][module] = False
            logger.warning(f"❌ {module} not available")
    
    # Check GPU (if PyTorch available)
    try:
        import torch
        results['gpu_available'] = torch.cuda.is_available()
        if results['gpu_available']:
            logger.info(f"✅ GPU available: {torch.cuda.get_device_name(0)}")
        else:
            logger.warning("⚠️  GPU not available")
    except ImportError:
        logger.info("PyTorch not available (GPU check skipped)")
    
    return results


def run_test_suite(test_func, logger: logging.Logger, test_name: str = "Test Suite") -> Dict[str, Any]:
    """Run a test suite and capture results."""
    logger.info(f"=" * 60)
    logger.info(f"Running {test_name}")
    logger.info(f"=" * 60)
    
    results = {
        'test_name': test_name,
        'timestamp': datetime.now().isoformat(),
        'tests_run': 0,
        'tests_passed': 0,
        'tests_failed': 0,
        'errors': []
    }
    
    try:
        test_results = test_func(logger)
        results['tests_run'] = test_results.get('tests_run', 0)
        results['tests_passed'] = test_results.get('tests_passed', 0)
        results['tests_failed'] = test_results.get('tests_failed', 0)
        results['test_details'] = test_results.get('details', {})
        
        if results['tests_failed'] > 0:
            results['errors'] = test_results.get('errors', [])
            logger.warning(f"⚠️  {results['tests_failed']} tests failed")
        else:
            logger.info(f"✅ All {results['tests_passed']} tests passed")
            
    except Exception as e:
        results['tests_failed'] += 1
        results['errors'].append({
            'test': test_name,
            'error': str(e),
            'traceback': traceback.format_exc()
        })
        logger.error(f"❌ Test suite failed: {e}")
        logger.debug(traceback.format_exc())
    
    return results


def save_test_results(results: Dict[str, Any], output_dir: Path, task_name: str):
    """Save test results to JSON file."""
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_path = output_dir / f"{timestamp}_{task_name}_test_results.json"
    
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    return results_path


def debug_print(message: str, logger: Optional[logging.Logger] = None, level: str = "INFO"):
    """Debug print with optional logger."""
    if logger:
        if level == "DEBUG":
            logger.debug(message)
        elif level == "WARNING":
            logger.warning(message)
        elif level == "ERROR":
            logger.error(message)
        else:
            logger.info(message)
    else:
        print(f"[{level}] {message}")


def validate_output(output: Any, expected_type: type, logger: logging.Logger, name: str = "output") -> bool:
    """Validate output type and basic properties."""
    if not isinstance(output, expected_type):
        logger.error(f"❌ {name} type mismatch: expected {expected_type}, got {type(output)}")
        return False
    
    logger.debug(f"✅ {name} type correct: {type(output)}")
    return True


def validate_array(array: np.ndarray, expected_shape: Optional[tuple] = None, 
                   logger: logging.Logger = None, name: str = "array") -> bool:
    """Validate numpy array properties."""
    if not isinstance(array, np.ndarray):
        if logger:
            logger.error(f"❌ {name} is not a numpy array: {type(array)}")
        return False
    
    if expected_shape and array.shape != expected_shape:
        if logger:
            logger.error(f"❌ {name} shape mismatch: expected {expected_shape}, got {array.shape}")
        return False
    
    if logger:
        logger.debug(f"✅ {name} shape correct: {array.shape}")
    
    return True

