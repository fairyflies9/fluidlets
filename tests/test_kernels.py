"""Kernels and core: Numba path vs reference path, analytic checks, solves."""
import numpy as np
import pytest

import fluidlets as fl
from fluidlets.kernels.images import image_block, image_velocity_reference

rng = np.random.default_rng(7)
KERNELS = [fl.Stokeslet3D(), fl.Stokeslet3DWall()]


def above_wall(n):
    return np.c_[rng.uniform(-1, 1, (n, 2)), rng.uniform(0.2, 1.5, n)]


@pytest.mark.parametrize("kernel", KERNELS, ids=lambda k: type(k).__name__)
def test_numba_matches_reference(kernel):
    X, X0, F = above_wall(20), above_wall(15), rng.normal(size=(15, 3))
    eps = rng.uniform(0.05, 0.3, 15)
    ref = sum(np.array([kernel.velocity(x, X0[j], F[j], eps[j]) for x in X]) for j in range(15))
    assert np.allclose(fl.evaluate(kernel, X, X0, F, eps), ref, rtol=1e-12, atol=1e-14)


def test_image_block_matches_reference():
    G = np.empty((3, 3))
    for _ in range(10):
        x, x0, f = above_wall(1)[0], above_wall(1)[0], rng.normal(size=3)
        image_block(x, x0, 0.1, G)
        assert np.allclose(G @ f, image_velocity_reference(x, x0, f, 0.1), rtol=1e-12, atol=1e-15)


@pytest.mark.parametrize("kernel", KERNELS, ids=lambda k: type(k).__name__)
def test_assemble_matches_evaluate(kernel):
    X, F = above_wall(12), rng.normal(size=(12, 3))
    K = fl.assemble(kernel, X, X, 0.2)
    assert np.allclose(K @ F.ravel(), fl.evaluate(kernel, X, X, F, 0.2).ravel(), rtol=1e-13)


def test_wall_velocity_vanishes():
    wall = np.c_[rng.uniform(-2, 2, (50, 2)), np.zeros(50)]
    u = fl.evaluate(fl.Stokeslet3DWall(), wall, above_wall(10), rng.normal(size=(10, 3)), 0.1)
    assert np.abs(u).max() < 1e-14


def test_viscosity_scaling_and_complex_mu():
    X, F = above_wall(5), rng.normal(size=(5, 3))
    u1 = fl.evaluate(fl.Stokeslet3D(), X, X, F, 0.1)
    mu = 2.0 - 0.5j
    u = fl.evaluate(fl.Stokeslet3D(), X, X, F, 0.1, mu=mu)
    assert u.dtype == np.complex128
    assert np.allclose(u, u1 / mu)
    assert fl.assemble(fl.Stokeslet3D(), X, X, 0.1, mu=mu).dtype == np.complex128


def fib_sphere(n, a=1.0, c=(0, 0, 0)):
    k = np.arange(n) + 0.5
    phi, th = np.arccos(1 - 2 * k / n), np.pi * (1 + 5**0.5) * k
    return a * np.c_[np.cos(th) * np.sin(phi), np.sin(th) * np.sin(phi), np.cos(phi)] + c


def test_sphere_drag_lu_and_gmres():
    # Translating unit sphere, mu = 1: total force -> 6 pi as spacing -> 0
    # (legacy check_sphere_drag.m gives 1.0164 at N = 400, eps = spacing/2).
    n = 400
    X = fib_sphere(n)
    eps = 0.5 * np.sqrt(4 * np.pi / n)
    U = np.tile([1.0, 0, 0], (n, 1))
    F_lu, info_lu = fl.resistance(fl.Stokeslet3D(), X, U, eps, method="lu")
    F_gm, info_gm = fl.resistance(fl.Stokeslet3D(), X, U, eps, method="gmres")
    assert info_lu.residual < 1e-12 and info_gm.residual < 1e-9
    assert np.allclose(F_lu, F_gm, rtol=1e-6, atol=1e-8)
    drag = F_lu[:, 0].sum() / (6 * np.pi)
    assert 1.0 < drag < 1.02


def test_stokeslet_eps_convergence():
    # Regularized -> singular Stokeslet at fixed r with error O(eps^2).
    x, f = np.array([0.6, -0.3, 0.4]), np.array([1.0, 0.5, -0.2])
    r = np.linalg.norm(x)
    exact = (f / r + (f @ x) * x / r**3) / (8 * np.pi)
    eps = np.array([0.08, 0.04, 0.02, 0.01])
    err = [np.abs(fl.evaluate(fl.Stokeslet3D(), [x], [[0, 0, 0]], [f], e)[0] - exact).max() for e in eps]
    assert np.all(np.log2(np.array(err[:-1]) / np.array(err[1:])) > 1.9)
