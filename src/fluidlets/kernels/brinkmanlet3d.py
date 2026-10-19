"""3D regularized Brinkmanlet (Cortez, Cummins, Leiderman & Varela,
J. Comput. Phys. 229, 7609-7624, 2010, Sec. 2.1, "regularization 1").

Brinkman flow, mu (Delta - alpha^2) u - grad p = -f phi_eps, div u = 0, models
fluid moving through a sparse porous medium; 1/alpha is the screening length
(alpha^2 = mu / permeability, in units where mu is scaled out). For a force f
at x0 the velocity at x is

    u = H1(r) f + H2(r) (f . d) d,     d = x - x0,  r = |d|,

with H1 = -(r B'' + B')/r and H2 = (r B'' - B')/r^3, where (Delta - alpha^2) B = G
(paper Eqs. 1-4). The singular Brinkmanlet has B(r) = (1 - e^{-alpha r}) / (4 pi alpha^2 r).
Regularization 1 replaces r by R = sqrt(r^2 + eps^2) in B. The chain rule then gives

    H2(r) = h2(R),    H1(r) = h1(R) + eps^2 h2(R),

where h1, h2 are the singular functions, written with z = alpha R:

    h1(R) = [e^{-z} (1 + 1/z + 1/z^2) - 1/z^2] / (4 pi R),
    h2(R) = [3/z^2 - e^{-z} (1 + 3/z + 3/z^2)] / (4 pi R^3).

As alpha -> 0, h1 -> 1/(8 pi R) and h2 -> 1/(8 pi R^3), so the kernel becomes
exactly the Cortez-Fauci-Medovikov regularized Stokeslet (fluidlets.Stokeslet3D).
For small z the bracketed terms cancel badly, so they are evaluated from their
Taylor series there:

    e^{-z}(1 + 1/z + 1/z^2) - 1/z^2 = sum_n (-1)^n [1/(n+2)! - 1/(n+1)! + 1/n!] z^n,
    3/z^2 - e^{-z}(1 + 3/z + 3/z^2) = sum_n (-1)^(n+1) [3/(n+2)! - 3/(n+1)! + 1/n!] z^n.

alpha must be real and non-negative here; complex alpha (oscillatory unsteady
Stokes) is not supported yet.
"""
from __future__ import annotations

from functools import lru_cache
from math import factorial

import numpy as np
from numba import njit

from .base import VectorKernel

Z_SERIES = 0.5        # below this z = alpha R the Taylor series is used
N_TERMS = 24          # error < z^24/24! ~ 1e-31 at z = 0.5

C1 = np.array([(-1) ** n * (1 / factorial(n + 2) - 1 / factorial(n + 1) + 1 / factorial(n))
               for n in range(N_TERMS)])
C2 = np.array([(-1) ** (n + 1) * (3 / factorial(n + 2) - 3 / factorial(n + 1) + 1 / factorial(n))
               for n in range(N_TERMS)])


@njit(cache=True)
def f1(z):
    """e^{-z}(1 + 1/z + 1/z^2) - 1/z^2, which tends to 1/2 as z -> 0."""
    if z < Z_SERIES:
        s = 0.0
        for n in range(N_TERMS - 1, -1, -1):        # Horner
            s = s * z + C1[n]
        return s
    return np.exp(-z) * (1.0 + 1.0 / z + 1.0 / (z * z)) - 1.0 / (z * z)


@njit(cache=True)
def f2(z):
    """3/z^2 - e^{-z}(1 + 3/z + 3/z^2), which tends to 1/2 as z -> 0."""
    if z < Z_SERIES:
        s = 0.0
        for n in range(N_TERMS - 1, -1, -1):
            s = s * z + C2[n]
        return s
    return 3.0 / (z * z) - np.exp(-z) * (1.0 + 3.0 / z + 3.0 / (z * z))


@njit(cache=True)
def H1_H2(r2, eps2, alpha):
    """Regularized H1, H2 at squared distance r2, for mu = 1."""
    R = np.sqrt(r2 + eps2)
    z = alpha * R
    h1 = f1(z) / (4.0 * np.pi * R)
    h2 = f2(z) / (4.0 * np.pi * R * R * R)
    return h1 + eps2 * h2, h2


@lru_cache(maxsize=None)
def make_block(alpha: float):
    """Compiled block function with alpha fixed (one compilation per alpha value)."""

    @njit
    def block(x, x0, eps, out):
        d0 = x[0] - x0[0]
        d1 = x[1] - x0[1]
        d2 = x[2] - x0[2]
        H1, H2 = H1_H2(d0 * d0 + d1 * d1 + d2 * d2, eps * eps, alpha)
        out[0, 0] = H1 + H2 * d0 * d0
        out[0, 1] = H2 * d0 * d1
        out[0, 2] = H2 * d0 * d2
        out[1, 0] = out[0, 1]
        out[1, 1] = H1 + H2 * d1 * d1
        out[1, 2] = H2 * d1 * d2
        out[2, 0] = out[0, 2]
        out[2, 1] = out[1, 2]
        out[2, 2] = H1 + H2 * d2 * d2

    return block


class Brinkmanlet3D(VectorKernel):
    """Regularized Brinkmanlet in free space, with inverse screening length alpha >= 0.

    ``Brinkmanlet3D(0.0)`` is the regularized Stokeslet.
    """

    dim = 3

    def __init__(self, alpha: float):
        if np.iscomplexobj(alpha) or alpha < 0:
            raise ValueError("alpha must be real and non-negative")
        self.alpha = float(alpha)
        self.block = make_block(self.alpha)

    def __repr__(self) -> str:
        return f"Brinkmanlet3D(alpha={self.alpha})"

    def velocity(self, x, x0, f, eps, mu=1.0):
        """Reference (NumPy) velocity at one target from one source.

        Uses the closed forms only (no series), so it is accurate for
        alpha R not too small; for alpha = 0 it falls back to the Stokeslet.
        """
        d = np.asarray(x, float) - np.asarray(x0, float)
        f = np.asarray(f)
        R = np.sqrt(d @ d + eps**2)
        a = self.alpha
        if a == 0.0:
            h1, h2 = 1 / (8 * np.pi * R), 1 / (8 * np.pi * R**3)
        else:
            E = np.exp(-a * R)
            h1 = (E * (1 + 1 / (a * R) + 1 / (a * R) ** 2) - 1 / (a * R) ** 2) / (4 * np.pi * R)
            h2 = (3 / (a * R) ** 2 - E * (1 + 3 / (a * R) + 3 / (a * R) ** 2)) / (4 * np.pi * R**3)
        H1 = h1 + eps**2 * h2
        return (H1 * f + h2 * (f @ d) * d) / mu
