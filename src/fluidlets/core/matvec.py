"""Matrix-free kernel sums u(x_i) = sum_j G(x_i, x_j) f_j / mu."""
from __future__ import annotations

import numpy as np
from numba import njit, prange

from ._common import blob_widths, field_dtype, points


@njit(parallel=True)
def _apply(targets, sources, eps, block, p, q, strengths, out):
    # each target row is independent, so rows run in parallel
    for i in prange(targets.shape[0]):
        G = np.empty((p, q))
        for j in range(sources.shape[0]):
            block(targets[i], sources[j], eps[j], G)
            for a in range(p):
                acc = out[i, a] * 0.0
                for b in range(q):
                    acc += G[a, b] * strengths[j, b]
                out[i, a] += acc


def evaluate(kernel, targets, sources, strengths, eps, mu=1.0) -> np.ndarray:
    """Field at targets from regularized sources, without storing the matrix.

    Parameters
    ----------
    kernel : fluidlets.kernels.Kernel
    targets : (M, dim) array
    sources : (N, dim) array
    strengths : (N, q) array
        Forces (vector kernels) or charges (scalar kernels).
    eps : float or (N,) array
    mu : float or complex

    Returns
    -------
    (M, p) array, complex if mu or strengths are complex.
    """
    targets = points(targets, kernel.dim)
    sources = points(sources, kernel.dim)
    eps = blob_widths(eps, sources.shape[0])
    p, q = kernel.out_shape
    dtype = field_dtype(mu, strengths)
    strengths = np.ascontiguousarray(np.asarray(strengths, dtype=dtype).reshape(sources.shape[0], q))
    out = np.zeros((targets.shape[0], p), dtype=dtype)
    _apply(targets, sources, eps, kernel.block, p, q, strengths, out)
    return out / mu
