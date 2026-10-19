"""Dense kernel matrices.

Entry layout follows the [point, component] convention: row p*i + a is
component a at target i, column q*j + b is component b of source j, so
``K @ forces.ravel()`` gives ``velocities.ravel()``.
"""
from __future__ import annotations

import numpy as np
from numba import njit

from ._common import blob_widths, field_dtype, points


@njit
def _fill(targets, sources, eps, block, p, q, K):
    G = np.empty((p, q))
    for i in range(targets.shape[0]):
        for j in range(sources.shape[0]):
            block(targets[i], sources[j], eps[j], G)
            for a in range(p):
                for b in range(q):
                    K[p * i + a, q * j + b] = G[a, b]


def assemble(kernel, targets, sources, eps, mu=1.0) -> np.ndarray:
    """Dense (p M) x (q N) matrix of ``kernel`` from N sources to M targets.

    Parameters
    ----------
    kernel : fluidlets.kernels.Kernel
    targets : (M, dim) array
    sources : (N, dim) array
    eps : float or (N,) array
        Blob width of each source.
    mu : float or complex
        Viscosity; a complex mu gives a complex matrix.
    """
    targets = points(targets, kernel.dim)
    sources = points(sources, kernel.dim)
    eps = blob_widths(eps, sources.shape[0])
    p, q = kernel.out_shape
    K = np.empty((p * targets.shape[0], q * sources.shape[0]))
    _fill(targets, sources, eps, kernel.block, p, q, K)
    return K.astype(field_dtype(mu)) / mu
