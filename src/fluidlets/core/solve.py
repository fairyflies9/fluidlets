"""Resistance problems: velocities prescribed at points, forces unknown.

Solves  sum_j G(x_i, x_j) f_j / mu = U_i  for the forces f_j, either with a
dense LU factorization or with matrix-free GMRES (as the legacy code did with
MATLAB's gmres and StokesSelf.m).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.linalg
import scipy.sparse.linalg as spla

from ._common import blob_widths, field_dtype, points as as_points
from .assemble import assemble
from .matvec import evaluate

#: above this many unknowns, method="auto" switches from dense LU to GMRES
DENSE_MAX = 6000


@dataclass
class SolveInfo:
    method: str
    iterations: int
    residual: float          # ||K f - U|| / ||U||


def resistance(kernel, points, velocities, eps, mu=1.0, method="auto",
               rtol=1e-10, restart=60, maxiter=200):
    """Forces that produce the given velocities at the given points.

    Parameters
    ----------
    kernel : fluidlets.kernels.Kernel
    points : (N, dim) array
        Points that are both sources and targets.
    velocities : (N, dim) array
    eps : float or (N,) array
    mu : float or complex
    method : {"auto", "lu", "gmres"}
    rtol, restart, maxiter : GMRES settings (restart = inner iterations).

    Returns
    -------
    forces : (N, dim) array
    info : SolveInfo

    Notes
    -----
    For a wall kernel, points exactly on the wall give zero rows and columns
    (Stokeslet and image cancel), so the dense matrix is singular there. GMRES
    started from zero leaves those forces at zero; use method="gmres".
    """
    X = as_points(points, kernel.dim)
    n = X.shape[0]
    eps = blob_widths(eps, n)
    U = np.asarray(velocities).reshape(-1)
    dtype = field_dtype(mu, U)
    if method == "auto":
        method = "lu" if U.size <= DENSE_MAX else "gmres"

    if method == "lu":
        K = assemble(kernel, X, X, eps, mu)
        f = scipy.linalg.solve(K, U.astype(dtype))
        iterations = 1
    elif method == "gmres":
        def matvec(v):
            return evaluate(kernel, X, X, v.reshape(n, -1), eps, mu).ravel()

        A = spla.LinearOperator((U.size, U.size), matvec=matvec, dtype=dtype)
        count = [0]
        f, flag = spla.gmres(A, U.astype(dtype), rtol=rtol, atol=0.0, restart=restart,
                             maxiter=maxiter, callback=lambda _: count.__setitem__(0, count[0] + 1),
                             callback_type="pr_norm")
        if flag != 0:
            raise RuntimeError(f"GMRES did not converge (flag {flag})")
        iterations = count[0]
    else:
        raise ValueError(f"unknown method {method!r}")

    resid = np.linalg.norm(evaluate(kernel, X, X, f.reshape(n, -1), eps, mu).ravel() - U) / np.linalg.norm(U)
    return f.reshape(n, -1), SolveInfo(method, iterations, float(resid))
