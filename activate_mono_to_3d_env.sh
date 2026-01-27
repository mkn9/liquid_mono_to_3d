#!/bin/bash
# Activation script for mono_to_3d isolated environment

echo "🔒 Activating isolated mono_to_3d virtual environment..."
echo "   This protects your base system from any accidental modifications"
echo ""

# Check if virtual environment exists
if [ ! -d "mono_to_3d_env" ]; then
    echo "❌ Virtual environment not found!"
    echo "   Run: python -m venv mono_to_3d_env"
    echo "   Then: pip install -r requirements.txt"
    exit 1
fi

# Activate the environment
source mono_to_3d_env/bin/activate

echo "✅ Successfully activated mono_to_3d_env"
echo "   Python: $(python --version)"
echo "   Pip: $(pip --version | cut -d' ' -f1-2)"
echo ""
echo "🚨 SAFETY REMINDER:"
echo "   - All EC2 computation should be done via SSH"
echo "   - This MacBook env is for commands/scripts only"
echo "   - Never run intensive ML training locally"
echo ""
echo "🎯 Ready for safe development!"
