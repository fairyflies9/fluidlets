"""Fluidlets: regularized fundamental solutions for fluids and fields."""
from .stokeslet3d import stokeslet3d_velocity
from .kernels import Brinkmanlet3D, Stokeslet3D, WallStokeslet3D
from .core import assemble, assemble_nearest, evaluate, nearest_neighbors, resistance

__version__ = "0.1.0"
__all__ = [
    "stokeslet3d_velocity",
    "Stokeslet3D",
    "WallStokeslet3D",
    "Brinkmanlet3D",
    "assemble",
    "assemble_nearest",
    "nearest_neighbors",
    "evaluate",
    "resistance",
]
