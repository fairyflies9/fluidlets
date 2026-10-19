"""Line-for-line NumPy transcriptions of the legacy MATLAB kernels.

Used only by the Phase 0 checks, so the legacy formulas can be evaluated at
arbitrary points (finite differences, eps -> 0 limits). Phase 1 ports these
properly into fluidlets.kernels and checks them against the same data.
"""
import numpy as np


def image_eval(targets, sources, strengths, deltas):
    """Transcription of legacy/sdc_cuda3/ImageEval.m (wall at z = 0, mu = 1)."""
    targets = np.atleast_2d(np.asarray(targets, dtype=float))
    sources = np.atleast_2d(np.asarray(sources, dtype=float))
    strengths = np.atleast_2d(np.asarray(strengths, dtype=float))
    d2s = np.broadcast_to(np.asarray(deltas, dtype=float), (sources.shape[0],)) ** 2

    V = np.zeros((targets.shape[0], 3))
    for (sx, sy, h0), f, d2 in zip(sources, strengths, d2s):
        # displacement from the image point (sx, sy, -h0)
        dx1 = targets[:, 0] - sx
        dx2 = targets[:, 1] - sy
        dx3 = targets[:, 2] + h0
        r2 = dx1**2 + dx2**2 + dx3**2
        q1, q3, q5 = (r2 + d2) ** 0.5, (r2 + d2) ** 1.5, (r2 + d2) ** 2.5

        H1 = 0.5 / q1 + 0.5 * d2 / q3                  # Stokeslet
        H2 = 0.5 / q3
        Hdp1 = -h0**2 * (1 / q3 - 3 * d2 / q5)         # dipole
        Hdp2 = -h0**2 * (-3 / q5)
        Hdb1 = 2 * h0 * (-1.5 * d2 / q5 - 0.5 / q3)    # doublet
        Hdb2 = 2 * h0 * (0.5 / q3)
        Hdb3 = 2 * h0 * (-1.5 / q5)
        Hr1 = h0 * 3 * d2 / q5                         # rotlet

        A = np.empty((targets.shape[0], 3, 3))
        dx = (dx1, dx2, dx3)
        for i in range(3):
            for j in range(3):
                A[:, i, j] = -((i == j) * H1 + dx[i] * dx[j] * H2)
        # dipole: sign flips on the z column
        for i in range(3):
            for j in range(3):
                s = 1.0 if j == 2 else -1.0
                A[:, i, j] += s * ((i == j) * Hdp1 + dx[i] * dx[j] * Hdp2)
        # doublet
        A[:, 0, 0] -= dx3 * (Hdb2 + dx1 * dx1 * Hdb3)
        A[:, 0, 1] -= dx1 * dx2 * dx3 * Hdb3
        A[:, 0, 2] += dx1 * (Hdb2 + dx3 * dx3 * Hdb3)
        A[:, 1, 0] -= dx2 * dx1 * dx3 * Hdb3
        A[:, 1, 1] -= dx3 * (Hdb2 + dx2 * dx2 * Hdb3)
        A[:, 1, 2] += dx2 * (Hdb2 + dx3 * dx3 * Hdb3)
        A[:, 2, 0] -= dx1 * (Hdb1 + dx3 * dx3 * Hdb3)
        A[:, 2, 1] -= dx2 * (Hdb1 + dx3 * dx3 * Hdb3)
        A[:, 2, 2] += dx3 * (Hdb1 + 2 * Hdb2 + dx3 * dx3 * Hdb3)
        # rotlet
        A[:, 0, 0] += dx3 * Hr1
        A[:, 1, 1] += dx3 * Hr1
        A[:, 2, 0] -= dx1 * Hr1
        A[:, 2, 1] -= dx2 * Hr1

        V += A @ f / (4 * np.pi)
    return V


def blake(x, x0, f, mu=1.0):
    """Singular Stokeslet above a no-slip wall at z = 0 (Blake 1971).

    G_ij = (d_ij/r + r_i r_j/r^3) - (d_ij/R + R_i R_j/R^3)
           + 2h (d_ja d_ak - d_j3 d_3k) d/dR_k [h R_i/R^3 - (d_i3/R + R_i R_3/R^3)],
    with r = x - x0, R = x - x0*, x0* the mirror image, a summed over 1, 2.
    The R_k-derivative is taken numerically (central differences) to keep the
    formula readable; it is smooth away from the image point.
    """
    x = np.asarray(x, float); x0 = np.asarray(x0, float); f = np.asarray(f, float)
    h = x0[2]
    r = x - x0
    R = x - np.array([x0[0], x0[1], -h])

    def stokeslet(d):
        n = np.linalg.norm(d)
        return np.eye(3) / n + np.outer(d, d) / n**3

    def phi(Rv):
        # phi_i = h R_i/R^3 - (d_i3/R + R_i R_3/R^3)
        n = np.linalg.norm(Rv)
        e3 = np.array([0.0, 0.0, 1.0])
        return h * Rv / n**3 - (e3 / n + Rv * Rv[2] / n**3)

    eps = 1e-6
    dphi = np.empty((3, 3))                       # dphi[i, k] = d phi_i / d R_k
    for k in range(3):
        e = np.zeros(3); e[k] = eps
        dphi[:, k] = (phi(R + e) - phi(R - e)) / (2 * eps)
    P = np.diag([1.0, 1.0, -1.0])                 # d_ja d_ak - d_j3 d_3k
    G = stokeslet(r) - stokeslet(R) + 2 * h * dphi @ P
    return G @ f / (8 * np.pi * mu)
