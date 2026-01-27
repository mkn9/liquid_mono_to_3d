#!/bin/bash
# Run All Tests on EC2
# Usage: bash RUN_TESTS_EC2.sh

set -e  # Exit on error

echo "=================================================="
echo "Track Persistence - Test Execution on EC2"
echo "=================================================="
echo ""

# Check if virtual environment is activated
if [ -z "$VIRTUAL_ENV" ]; then
    echo "⚠️  Warning: No virtual environment detected"
    echo "Please activate your virtual environment first:"
    echo "  source venv/bin/activate"
    echo ""
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Check for required packages
echo "Checking dependencies..."
python -c "import pytest" 2>/dev/null || { echo "❌ pytest not found. Run: pip install pytest"; exit 1; }
python -c "import torch" 2>/dev/null || { echo "❌ torch not found. Run: pip install torch"; exit 1; }
python -c "import cv2" 2>/dev/null || { echo "❌ opencv not found. Run: pip install opencv-python"; exit 1; }
echo "✅ All dependencies found"
echo ""

# Navigate to test directory
cd "$(dirname "$0")"

echo "=================================================="
echo "Test Suite 1: Structural Tests (No GPU required)"
echo "=================================================="
python -m pytest test_structural.py -v
echo ""

echo "=================================================="
echo "Test Suite 2: Track Generator Tests"
echo "=================================================="
python -m pytest test_realistic_track_generator.py -v
echo ""

echo "=================================================="
echo "Test Suite 3: Attention Model Tests (Requires PyTorch)"
echo "=================================================="
python -m pytest test_attention_model.py -v
echo ""

echo "=================================================="
echo "Test Suite 4: Integrated Tracker Tests"
echo "=================================================="
python -m pytest test_integrated_tracker.py -v
echo ""

echo "=================================================="
echo "ALL TESTS COMPLETE"
echo "=================================================="
echo ""
echo "To run with coverage:"
echo "  python -m pytest . -v --cov=. --cov-report=html"
echo ""
echo "Coverage report will be in: htmlcov/index.html"

