"""Phase 1: the Python/Numba port reproduces the legacy MATLAB results.

Data come from the Phase 0 MATLAB runs (tests/legacy/data). Legacy arrays of
points are component-major (all x, then all y, then all z); they are
reshaped to the [point, component] layout used by Fluidlets.
"""
from pathlib import Path

import numpy as np
import pytest
import scipy.io as sio

import fluidlets as fl

DATA = Path(__file__).parent / "data"
LEGACY = Path(__file__).parents[2] / "legacy" / "sdc_cuda3"
WALL = fl.WallStokeslet3D()
DELTA = 0.5                     # blob width in run_particles.m


def load(name):
    path = DATA / name
    if not path.exists():
        pytest.skip(f"{name} not generated")
    return sio.loadmat(path, squeeze_me=True)


def legacy_points(a):
    n = a.size // 3
    return a.reshape(3, n).T


def test_stokeslet_port_matches_StokesletEval():
    d = load("kernels.mat")
    u = fl.evaluate(fl.Stokeslet3D(), d["targets"], d["sources"], d["strengths"], d["deltas"])
    assert np.allclose(u, d["V_stokeslet"], rtol=1e-13, atol=1e-15)


def test_wall_port_matches_StokesletEval_plus_ImageEval():
    d = load("kernels.mat")
    u = fl.evaluate(WALL, d["targets"], d["sources"], d["strengths"], d["deltas"])
    assert np.allclose(u, d["V_stokeslet"] + d["V_image"], rtol=1e-13, atol=1e-15)
    u = fl.evaluate(WALL, d["wall"], d["sources"], d["strengths"], d["deltas"])
    assert np.allclose(u, d["V_wall_total"], atol=1e-15)


@pytest.fixture(scope="module")
def towers():
    d = load("towers.mat")
    X = legacy_points(d["x0"])
    return d, X, legacy_points(d["Vel"]), legacy_points(d["Forces"])


def test_port_operator_equals_StokesSelf(towers):
    # Applying the port to the legacy GMRES forces must give exactly the
    # residual MATLAB reported with StokesSelf.m: same operator to roundoff.
    d, X, U, F_legacy = towers
    r = np.linalg.norm(fl.evaluate(WALL, X, X, F_legacy, DELTA) - U) / np.linalg.norm(U)
    assert abs(r - d["resid"]) < 1e-12


def test_port_gmres_reproduces_tower_forces(towers):
    d, X, U, F_legacy = towers
    F, info = fl.resistance(WALL, X, U, DELTA, method="gmres")
    assert info.residual < 1e-9
    on_wall = X[:, 2] == 0
    assert np.abs(F[on_wall]).max() < 1e-12
    # legacy stopped GMRES at 1e-6 on an ill-conditioned system, so forces
    # agree to ~1e-4 pointwise and the tower drags to ~1e-6
    above = ~on_wall
    assert np.linalg.norm(F[above] - F_legacy[above]) / np.linalg.norm(F_legacy[above]) < 1e-3
    for k in range(3):
        tower = slice(147 * k, 147 * (k + 1))
        assert np.isclose(F[tower, 0].sum(), F_legacy[tower, 0].sum(), rtol=1e-5)


def test_port_reproduces_2010_velocity_grid(towers):
    # The 32^3 grid written by particlesHex.m (compute_velocity_grid): x fastest,
    # then y, then z; u includes the background shear 0.1 z.
    d, X, U, F_legacy = towers
    xs, ys, zs = np.linspace(-1, 8, 32), np.linspace(-5, 25, 32), np.linspace(0, 20, 32)
    Z, Y, Xg = np.meshgrid(zs, ys, xs, indexing="ij")
    grid = np.c_[Xg.ravel(), Y.ravel(), Z.ravel()]
    V = fl.evaluate(WALL, grid, X, F_legacy, DELTA)
    V[:, 0] += 0.1 * grid[:, 2]
    stored = np.fromfile(LEGACY / "velocity_t_xyz_final.raw", "<f4").reshape(-1, 3)
    assert np.linalg.norm(V - stored) / np.linalg.norm(stored) < 1e-6    # float32 CUDA output


def test_wall_kernel_eps_convergence_to_blake():
    from legacy_kernels import blake
    x0, x, f = np.array([0.3, -0.2, 1.0]), np.array([1.2, 0.4, 0.6]), np.array([0.3, -1.0, 0.7])
    exact = blake(x, x0, f)
    eps = np.array([0.1, 0.05, 0.025, 0.0125])
    err = [np.abs(fl.evaluate(WALL, [x], [x0], [f], e)[0] - exact).max() for e in eps]
    assert np.all(np.log2(np.array(err[:-1]) / np.array(err[1:])) > 1.9)
