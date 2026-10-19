"""Discretized sphere surfaces."""
from __future__ import annotations

import numpy as np


def fibonacci_sphere(n: int, radius: float = 1.0, center=(0.0, 0.0, 0.0)) -> np.ndarray:
    """n nearly uniform points on a sphere (golden-angle spiral). Shape (n, 3)."""
    k = np.arange(n) + 0.5
    phi = np.arccos(1 - 2 * k / n)
    theta = np.pi * (1 + np.sqrt(5)) * k
    unit = np.c_[np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi)]
    return radius * unit + np.asarray(center, float)


def ring_sphere(n_phi: int, radius: float = 1.0, center=(0.0, 0.0, 0.0)) -> np.ndarray:
    """Rings of latitude, as in Ainley et al. (2008), Sec. 5.4.

    phi_k = pi k / n_phi for k = 0..n_phi; ring k holds N_k points, the
    smallest integer larger than 2 n_phi sin(phi_k), at theta_j = 2 pi j / N_k.
    The polar axis is z (Ainley et al. use x, normal to their wall x = 0).
    """
    pts = []
    for k in range(n_phi + 1):
        phi = np.pi * k / n_phi
        n_k = int(np.floor(2 * n_phi * np.sin(phi))) + 1
        theta = 2 * np.pi * np.arange(1, n_k + 1) / n_k
        pts.append(np.c_[np.sin(phi) * np.cos(theta), np.sin(phi) * np.sin(theta),
                         np.full(n_k, np.cos(phi))])
    return radius * np.vstack(pts) + np.asarray(center, float)
