"""Tower builders reproduce legacy build_*_tower.m and run_particles.m."""
from pathlib import Path

import numpy as np
import pytest
import scipy.io as sio

from fluidlets.geometry import hex_array_layout, hex_tower, line_tower, towers_from_table, tri_tower

DATA = Path(__file__).parent / "data"
LEGACY = Path(__file__).parents[2] / "legacy" / "sdc_cuda3"


def load(name):
    if not (DATA / name).exists():
        pytest.skip(f"{name} not generated")
    return sio.loadmat(DATA / name, squeeze_me=True)


def legacy_points(a):
    n = a.size // 3
    return a.reshape(3, n).T          # legacy layout: all x, then all y, then all z


def test_builders_match_matlab():
    d = load("geometry.mat")
    assert np.allclose(hex_tower([1, 2, 0.5]), legacy_points(d["hex"]), atol=1e-14)
    assert np.allclose(tri_tower([0.5, -1, 0]), legacy_points(d["tri"]), atol=1e-14)
    assert np.allclose(line_tower([3, 1, 0]), legacy_points(d["line"]), atol=1e-14)


def test_towers_txt_reproduces_run_particles_points():
    d = load("towers.mat")
    table = np.loadtxt(LEGACY / "towers.txt")
    assert np.allclose(towers_from_table(table), legacy_points(d["x0"]), atol=1e-14)


def test_full_hex_layout_shape():
    rows = hex_array_layout(4.0, 20.0)
    assert rows.shape == (2 + 6 * 5 + 5 * 6 + 2, 6)       # 2+5+6+5+...+5+2 = 64 towers
    xy = rows[:, 1:3]
    d = np.linalg.norm(xy[:, None] - xy[None], axis=-1)
    assert np.isclose(d[d > 0].min(), 4.0)                # nearest neighbours one delta apart
