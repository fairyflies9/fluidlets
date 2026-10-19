"""IB2d/IBAMR ``.vertex`` files: first line N, then one point per line."""
from __future__ import annotations

from pathlib import Path

import numpy as np


def read_vertex(path) -> np.ndarray:
    """Points as an (N, dim) array; dim is 2 (IB2d) or 3 (IBAMR)."""
    lines = Path(path).read_text().split("\n", 1)
    n = int(lines[0].split()[0])
    pts = np.loadtxt(lines[1].splitlines(), ndmin=2)
    if pts.shape[0] != n:
        raise ValueError(f"{path}: header says {n} points, found {pts.shape[0]}")
    return pts


def write_vertex(path, points) -> None:
    points = np.atleast_2d(points)
    with open(path, "w") as fh:
        fh.write(f"{points.shape[0]}\n")
        np.savetxt(fh, points, fmt="%.16e")
