"""Kernel interface shared by all regularized fundamental solutions.

A kernel is defined by one Numba-compiled ``block`` function that fills the
(p, q) matrix G(x, x0) mapping a source strength (length q) at x0 to the
field (length p) at x, for unit viscosity. The loops in ``fluidlets.core``
call ``block`` for every target/source pair and divide by mu, so adding a
kernel never touches ``core``.
"""
from __future__ import annotations

import numpy as np


class Kernel:
    """Base class. Subclasses set ``dim``, ``out_shape`` and ``block``."""

    dim: int
    out_shape: tuple[int, int]
    #: @njit function block(x, x0, eps, out) filling out[:p, :q] for mu = 1
    block = None

    def velocity(self, x: np.ndarray, x0: np.ndarray, f: np.ndarray, eps: float, mu=1.0) -> np.ndarray:
        """Reference (non-Numba) field at one target from one source."""
        raise NotImplementedError


class VectorKernel(Kernel):
    """Fluid kernel: a force (dim) at a source gives a velocity (dim) at a target."""

    @property
    def out_shape(self) -> tuple[int, int]:
        return (self.dim, self.dim)
