"""Branch 3: I3D + CLIP alignment - same as Branch 1 for now"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "branch1"))
from simple_model import SimplifiedI3D as SimplifiedI3DCLIP

