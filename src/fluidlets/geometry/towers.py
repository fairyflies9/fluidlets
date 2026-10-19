"""Glycocalyx core-protein towers, ported from legacy/sdc_cuda3/build_*_tower.m.

Each tower is a stack of levels from z = origin_z to z = origin_z + height.
Defaults are the values hard-coded in the legacy builders (units: mm).
Points are returned level by level, as in the legacy code, with shape (N, 3).
"""
from __future__ import annotations

import numpy as np

HEX, TRI, LINE = 0, 1, 2       # tower type codes used in towers.txt


def hex_tower(origin, levels: int = 20, radius: float = 0.5, height: float = 10.0) -> np.ndarray:
    """Hexagonal tower: 6 points at angles pi/6 + k pi/3 plus the centre, on levels + 1 levels."""
    angles = np.pi / 6 + 2 * np.pi / 6 * np.arange(6)
    ring = np.c_[radius * np.cos(angles), radius * np.sin(angles)]
    level = np.vstack([ring, [0.0, 0.0]])                     # 7 points per level
    z = np.arange(levels + 1) * (height / levels)
    pts = np.array([[x, y, zk] for zk in z for x, y in level])
    return pts + np.asarray(origin, float)


def tri_tower(origin, levels: int = 10, radius: float = 0.037, height: float = 1.0) -> np.ndarray:
    """Triangular tower: 3 points per level on levels + 1 levels."""
    c, s = np.cos(np.pi / 3), np.sin(np.pi / 3)
    level = np.array([[-radius, 0.0], [radius * c, -radius * s], [radius * c, radius * s]])
    z = np.arange(levels + 1) * (height / levels)
    pts = np.array([[x, y, zk] for zk in z for x, y in level])
    return pts + np.asarray(origin, float)


def line_tower(origin, levels: int = 20, height: float = 10.0) -> np.ndarray:
    """A single vertical line of levels + 1 points."""
    z = np.arange(levels + 1) * (height / levels)
    return np.c_[np.zeros_like(z), np.zeros_like(z), z] + np.asarray(origin, float)


def towers_from_table(table) -> np.ndarray:
    """Build all towers from rows ``type x y z height levels`` (the towers.txt format).

    The legacy builders ignore the height and levels columns and use their
    hard-coded defaults; here the columns are used. For the legacy file
    (height 10, levels 20, hex towers) the two agree.
    """
    builders = {HEX: hex_tower, TRI: tri_tower, LINE: line_tower}
    towers = []
    for kind, x, y, z, height, levels in np.atleast_2d(table):
        towers.append(builders[int(kind)]((x, y, z), levels=int(levels), height=height))
    return np.vstack(towers)


def hex_array_layout(delta: float, height: float, levels: int = 20, columns: int = 13) -> np.ndarray:
    """Hexagonally packed array of hex towers, spacing delta (towers.txt rows).

    This is the full layout written below the early ``return`` in
    legacy generate_hex_layout.m ("according to the tower layout from
    Laura"), which the legacy code never reached. Column k sits at
    x = k delta sqrt(3)/2; the end columns hold 2 towers, odd columns 5
    (offset delta/2) and the other even columns 6.
    """
    dx, dy = delta * np.sqrt(3) / 2, delta
    rows = []
    for k in range(columns):
        if k in (0, columns - 1):
            y0, n = 2 * dy, 2
        elif k % 2:
            y0, n = 0.5 * dy, 5
        else:
            y0, n = 0.0, 6
        rows += [[HEX, k * dx, y0 + dy * i, 0.0, height, levels] for i in range(n)]
    return np.array(rows)
