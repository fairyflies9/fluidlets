"""3D regularized Stokeslet (Cortez, Fauci & Medovikov, Phys. Fluids 17, 031504, 2005).

With the blob phi_eps(r) = 15 eps^4 / (8 pi (r^2 + eps^2)^(7/2)), the velocity
at x due to a force f at x0 is

    u = H1(r) f + H2(r) (f . d) d,     d = x - x0,  r = |d|,
    H1 = (r^2 + 2 eps^2) / (8 pi mu (r^2 + eps^2)^(3/2)),
    H2 = 1 / (8 pi mu (r^2 + eps^2)^(3/2)).

This is the kernel in legacy/sdc_cuda3/StokesletEval.m (there written as
(1/4pi) [1/(2 sqrt(r^2+eps^2)) + eps^2/(2 (r^2+eps^2)^(3/2))] with mu = 1);
docs/legacy_verification.md shows the two agree to roundoff.
"""
from __future__ import annotations

import numpy as np
from numba import njit

from .base import VectorKernel


@njit(cache=True)
def H1(r2, eps2):
    return (r2 + 2.0 * eps2) / (8.0 * np.pi * (r2 + eps2) ** 1.5)


@njit(cache=True)
def H2(r2, eps2):
    return 1.0 / (8.0 * np.pi * (r2 + eps2) ** 1.5)


@njit(cache=True)
def block(x, x0, eps, out):
    """G_ij = H1 delta_ij + H2 d_i d_j, for mu = 1."""
    d0 = x[0] - x0[0]
    d1 = x[1] - x0[1]
    d2 = x[2] - x0[2]
    r2 = d0 * d0 + d1 * d1 + d2 * d2
    eps2 = eps * eps
    h1 = H1(r2, eps2)
    h2 = H2(r2, eps2)
    out[0, 0] = h1 + h2 * d0 * d0
    out[0, 1] = h2 * d0 * d1
    out[0, 2] = h2 * d0 * d2
    out[1, 0] = out[0, 1]
    out[1, 1] = h1 + h2 * d1 * d1
    out[1, 2] = h2 * d1 * d2
    out[2, 0] = out[0, 2]
    out[2, 1] = out[1, 2]
    out[2, 2] = h1 + h2 * d2 * d2


class Stokeslet3D(VectorKernel):
    """Regularized Stokeslet in free space."""

    dim = 3
    block = staticmethod(block)

    def velocity(self, x, x0, f, eps, mu=1.0):
        d = np.asarray(x, float) - np.asarray(x0, float)
        f = np.asarray(f)
        r2 = d @ d
        q = (r2 + eps**2) ** 1.5
        return ((r2 + 2 * eps**2) * f + (f @ d) * d) / (8 * np.pi * mu * q)
