"""Branch 4: SlowFast + Phi2 - same as Branch 2"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "branch2"))
from simple_model import SimplifiedSlowFast as SimplifiedSlowFastPhi2

