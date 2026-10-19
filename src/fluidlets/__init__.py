"""Fluidlets: regularized fundamental solutions for fluids and fields."""
from .stokeslet3d import stokeslet3d_velocity
from .kernels import Stokeslet3D, Stokeslet3DWall
from .core import assemble, evaluate, resistance

__version__ = "0.0.1"
__all__ = [
    "stokeslet3d_velocity",
    "Stokeslet3D",
    "Stokeslet3DWall",
    "assemble",
    "evaluate",
    "resistance",
]
