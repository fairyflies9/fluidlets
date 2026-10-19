"""Kernel-agnostic assembly, matrix-free evaluation and solvers."""
from .assemble import assemble, assemble_nearest, nearest_neighbours
from .matvec import evaluate
from .solve import resistance

__all__ = ["assemble", "assemble_nearest", "nearest_neighbours", "evaluate", "resistance"]
