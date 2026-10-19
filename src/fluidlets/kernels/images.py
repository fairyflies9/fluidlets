"""Regularized Stokeslet above a no-slip plane wall at z = 0.

Image system of Ainley, Durkin, Embid, Boindala & Cortez (J. Comput. Phys.
227, 4600-4616, 2008), as implemented in legacy/sdc_cuda3/ImageEval.m. For a
source at height h0 = x0_3 the image point is x0* = (x0_1, x0_2, -h0) and,
with d = x - x0* and Q = r^2 + eps^2, the image adds (for mu = 1, times 1/4pi):

    Stokeslet  -[H1 I + H2 d d^T],          H1 = 1/(2 Q^(1/2)) + eps^2/(2 Q^(3/2)), H2 = 1/(2 Q^(3/2))
    dipole     h0^2 terms,                  Hdp1 = -h0^2 (Q^-3/2 - 3 eps^2 Q^-5/2), Hdp2 = 3 h0^2 Q^-5/2
    doublet    2 h0 terms,                  Hdb1 = 2h0(-3/2 eps^2 Q^-5/2 - 1/2 Q^-3/2), Hdb2 = h0 Q^-3/2, Hdb3 = -3 h0 Q^-5/2
    rotlet     Hr1 = 3 h0 eps^2 Q^-5/2.

The sign pattern of each term follows ImageEval.m exactly. The velocity of
Stokeslet + images vanishes on z = 0 to roundoff, and converges to Blake's
(1971) singular solution at O(eps^2) (docs/legacy_verification.md).
"""
from __future__ import annotations

import numpy as np
from numba import njit

from .base import VectorKernel
from .stokeslet3d import Stokeslet3D, block as stokeslet_block


@njit(cache=True)
def image_block(x, x0, eps, out):
    """Image part only (no free-space Stokeslet), for mu = 1."""
    h0 = x0[2]
    d0 = x[0] - x0[0]
    d1 = x[1] - x0[1]
    d2 = x[2] + h0                      # displacement from the image point
    eps2 = eps * eps
    Q = d0 * d0 + d1 * d1 + d2 * d2 + eps2
    q1 = Q**0.5
    q3 = Q**1.5
    q5 = Q**2.5

    H1 = 0.5 / q1 + 0.5 * eps2 / q3
    H2 = 0.5 / q3
    Hdp1 = -h0 * h0 * (1.0 / q3 - 3.0 * eps2 / q5)
    Hdp2 = -h0 * h0 * (-3.0 / q5)
    Hdb1 = 2.0 * h0 * (-1.5 * eps2 / q5 - 0.5 / q3)
    Hdb2 = 2.0 * h0 * (0.5 / q3)
    Hdb3 = 2.0 * h0 * (-1.5 / q5)
    Hr1 = h0 * 3.0 * eps2 / q5

    d = (d0, d1, d2)
    for i in range(3):
        for j in range(3):
            delta = 1.0 if i == j else 0.0
            # negative Stokeslet at the image point
            a = -(delta * H1 + d[i] * d[j] * H2)
            # dipole: + on the z column, - elsewhere
            sgn = 1.0 if j == 2 else -1.0
            a += sgn * (delta * Hdp1 + d[i] * d[j] * Hdp2)
            out[i, j] = a

    # doublet
    out[0, 0] -= d2 * (Hdb2 + d0 * d0 * Hdb3)
    out[0, 1] -= d0 * d1 * d2 * Hdb3
    out[0, 2] += d0 * (Hdb2 + d2 * d2 * Hdb3)
    out[1, 0] -= d1 * d0 * d2 * Hdb3
    out[1, 1] -= d2 * (Hdb2 + d1 * d1 * Hdb3)
    out[1, 2] += d1 * (Hdb2 + d2 * d2 * Hdb3)
    out[2, 0] -= d0 * (Hdb1 + d2 * d2 * Hdb3)
    out[2, 1] -= d1 * (Hdb1 + d2 * d2 * Hdb3)
    out[2, 2] += d2 * (Hdb1 + 2.0 * Hdb2 + d2 * d2 * Hdb3)

    # rotlet
    out[0, 0] += d2 * Hr1
    out[1, 1] += d2 * Hr1
    out[2, 0] -= d0 * Hr1
    out[2, 1] -= d1 * Hr1

    for i in range(3):
        for j in range(3):
            out[i, j] /= 4.0 * np.pi


@njit(cache=True)
def block(x, x0, eps, out):
    """Free-space Stokeslet plus wall images, for mu = 1."""
    image_block(x, x0, eps, out)
    tmp = np.empty((3, 3))
    stokeslet_block(x, x0, eps, tmp)
    for i in range(3):
        for j in range(3):
            out[i, j] += tmp[i, j]


class Stokeslet3DWall(VectorKernel):
    """Regularized Stokeslet above a no-slip wall at z = 0 (sources need z > 0)."""

    dim = 3
    block = staticmethod(block)

    def velocity(self, x, x0, f, eps, mu=1.0):
        free = Stokeslet3D().velocity(x, x0, f, eps, mu)
        return free + image_velocity_reference(x, x0, f, eps, mu)


def image_velocity_reference(x, x0, f, eps, mu=1.0):
    """Reference (NumPy) image velocity at one target from one source.

    Written as matrix algebra rather than the element-by-element form of
    image_block, so the two implementations check each other.
    """
    x = np.asarray(x, float); x0 = np.asarray(x0, float); f = np.asarray(f)
    h0 = x0[2]
    d = x - np.array([x0[0], x0[1], -h0])
    e2 = eps**2
    Q = d @ d + e2
    I = np.eye(3)
    D = np.outer(d, d)
    Z = np.diag([-1.0, -1.0, 1.0])            # flips sign of x, y columns
    ez = np.array([0.0, 0.0, 1.0])

    stokeslet = -((0.5 / Q**0.5 + 0.5 * e2 / Q**1.5) * I + 0.5 / Q**1.5 * D)
    dipole = (-h0**2 * (1 / Q**1.5 - 3 * e2 / Q**2.5) * I + 3 * h0**2 / Q**2.5 * D) @ Z
    Hdb1 = 2 * h0 * (-1.5 * e2 / Q**2.5 - 0.5 / Q**1.5)
    Hdb2 = h0 / Q**1.5
    Hdb3 = -3 * h0 / Q**2.5
    # doublet: -d3 (Hdb2 I + Hdb3 D) on the x, y columns of rows 1-2, etc.
    doublet = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            if j < 2:
                doublet[i, j] = -(d[2] * (Hdb2 * I[i, j] + d[i] * d[j] * Hdb3) if i < 2
                                  else d[j] * (Hdb1 + d[2] ** 2 * Hdb3))
            else:
                doublet[i, j] = d[i] * (Hdb2 + d[2] ** 2 * Hdb3) + (d[2] * (Hdb1 + Hdb2) if i == 2 else 0.0)
    Hr1 = 3 * h0 * e2 / Q**2.5
    rotlet = Hr1 * (d[2] * np.diag([1.0, 1.0, 0.0]) - np.outer(ez, [d[0], d[1], 0.0]))
    return (stokeslet + dipole + doublet + rotlet) @ f / (4 * np.pi * mu)
