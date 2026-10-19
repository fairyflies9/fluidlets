"""Input normalization shared by core routines."""
from __future__ import annotations

import numpy as np


def points(a, dim: int) -> np.ndarray:
    """(N, dim) float64 array of point coordinates."""
    a = np.atleast_2d(np.asarray(a, dtype=np.float64))
    if a.shape[1] != dim:
        raise ValueError(f"expected points of shape (N, {dim}), got {a.shape}")
    return np.ascontiguousarray(a)


def blob_widths(eps, n: int) -> np.ndarray:
    """One blob width per source (a scalar eps is broadcast)."""
    eps = np.asarray(eps, dtype=np.float64)
    return np.ascontiguousarray(np.broadcast_to(eps, (n,)))


def field_dtype(mu, *arrays) -> np.dtype:
    """float64, or complex128 if mu (e.g. a complex viscosity) or any input is complex."""
    return np.result_type(np.asarray(mu), np.float64, *[np.asarray(a) for a in arrays])
