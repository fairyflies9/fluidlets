"""Convenience function kept from the 0.0.1 release.

``stokeslet3d_velocity(targets, sources, forces, eps, mu)`` is the same as
``fluidlets.evaluate(fluidlets.Stokeslet3D(), targets, sources, forces, eps, mu)``;
the formula lives in ``fluidlets.kernels.stokeslet3d``.
"""
from __future__ import annotations

import numpy as np

from .core.matvec import evaluate
from .kernels.stokeslet3d import Stokeslet3D


def stokeslet3d_velocity(targets, sources, forces, eps, mu=1.0) -> np.ndarray:
    """Velocity at targets induced by 3D regularized Stokeslets.

    Parameters
    ----------
    targets : (M, 3) array
    sources : (N, 3) array
    forces : (N, 3) array
    eps : float or (N,) array
        Blob width(s).
    mu : float or complex
        Dynamic viscosity.

    Returns
    -------
    (M, 3) array of velocities.
    """
    return evaluate(Stokeslet3D(), targets, sources, forces, eps, mu)
