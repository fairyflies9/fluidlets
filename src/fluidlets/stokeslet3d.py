"""3D regularized Stokeslet (Cortez, Fauci & Medovikov, Phys. Fluids 2005).

u(x) = 1/(8*pi*mu) * sum_k [ f_k * (r^2 + 2 eps^2) + (f_k . d) d ] / (r^2 + eps^2)^(3/2),
with d = x - x_k, r = |d|. Equivalent to the H1/H2 form used in the 2011 UNC
glycocalyx code (StokesletEval.m) with mu = 1.
"""
import numpy as np


def stokeslet3d_velocity(targets, sources, forces, eps, mu=1.0):
    """Velocity at targets induced by regularized Stokeslets.

    Parameters
    ----------
    targets : (M, 3) array
    sources : (N, 3) array
    forces : (N, 3) array
    eps : float or (N,) array
        Blob width(s).
    mu : float
        Dynamic viscosity.

    Returns
    -------
    (M, 3) array of velocities.
    """
    targets = np.atleast_2d(np.asarray(targets, dtype=float))
    sources = np.atleast_2d(np.asarray(sources, dtype=float))
    forces = np.atleast_2d(np.asarray(forces, dtype=float))
    eps = np.broadcast_to(np.asarray(eps, dtype=float), (sources.shape[0],))

    d = targets[:, None, :] - sources[None, :, :]          # (M, N, 3)
    r2 = np.einsum("mni,mni->mn", d, d)
    e2 = eps[None, :] ** 2
    denom = (r2 + e2) ** 1.5
    fd = np.einsum("mni,ni->mn", d, forces)
    u = (forces[None, :, :] * ((r2 + 2 * e2) / denom)[..., None]
         + d * (fd / denom)[..., None])
    return u.sum(axis=1) / (8 * np.pi * mu)
