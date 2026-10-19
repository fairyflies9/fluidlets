"""Dense kernel matrices.

Entry layout follows the [point, component] convention: row p*i + a is
component a at target i, column q*j + b is component b of source j, so
``K @ forces.ravel()`` gives ``velocities.ravel()``.
"""
from __future__ import annotations

import numpy as np
from numba import njit, prange

from ._common import blob_widths, field_dtype, points


@njit(parallel=True)
def _fill(targets, sources, eps, block, p, q, K):
    # each target row is independent, so rows run in parallel
    for i in prange(targets.shape[0]):
        G = np.empty((p, q))
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


@njit(parallel=True)
def _fill_nearest(targets, quad, nearest, eps, block, p, q, K):
    for i in prange(targets.shape[0]):
        G = np.empty((p, q))
        for k in range(quad.shape[0]):
            block(targets[i], quad[k], eps[k], G)
            j = nearest[k]
            for a in range(p):
                for b in range(q):
                    K[p * i + a, q * j + b] += G[a, b]


def nearest_neighbors(quad_points, force_points) -> np.ndarray:
    """Index of the nearest force point for every quadrature point, shape (Q,)."""
    from scipy.spatial import cKDTree

    return cKDTree(np.asarray(force_points, float)).query(np.asarray(quad_points, float))[1]


def assemble_nearest(kernel, targets, quad_points, nearest, n_force, eps, mu=1.0) -> np.ndarray:
    """Kernel matrix with forces on a coarse set and quadrature on a fine set.

    Nearest-neighbour discretization of Gallagher & Smith (Phys. Rev. Fluids 3,
    053101, 2018; Smith 2018): each quadrature point X_k carries the force of
    its nearest force point, so

        K[i, j] = sum over k with nearest[k] = j of G(x_i, X_k) / mu.

    With identical force and quadrature points (nearest = identity) this is
    the ordinary ``assemble``.

    Parameters
    ----------
    kernel : fluidlets.kernels.Kernel
    targets : (M, dim) array
        Collocation points (usually the force points).
    quad_points : (Q, dim) array
    nearest : (Q,) int array
        From ``nearest_neighbors(quad_points, force_points)``.
    n_force : int
        Number of force points N.
    eps : float or (Q,) array
        Blob width of each quadrature point.
    mu : float or complex
    """
    targets = points(targets, kernel.dim)
    quad = points(quad_points, kernel.dim)
    eps = blob_widths(eps, quad.shape[0])
    nearest = np.ascontiguousarray(nearest, dtype=np.int64)
    p, q = kernel.out_shape
    K = np.zeros((p * targets.shape[0], q * n_force))
    _fill_nearest(targets, quad, nearest, eps, kernel.block, p, q, K)
    return K.astype(field_dtype(mu)) / mu
