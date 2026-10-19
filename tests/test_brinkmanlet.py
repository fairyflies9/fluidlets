"""3D regularized Brinkmanlet (Cortez, Cummins, Leiderman & Varela 2010, regularization 1)."""
from decimal import Decimal, getcontext

import numpy as np
import pytest

import fluidlets as fl
from fluidlets.geometry import fibonacci_sphere
from fluidlets.kernels.brinkmanlet3d import f1, f2

rng = np.random.default_rng(11)


def test_alpha_zero_is_the_regularized_stokeslet():
    X, X0, F = rng.normal(size=(25, 3)), rng.normal(size=(15, 3)), rng.normal(size=(15, 3))
    u_b = fl.evaluate(fl.Brinkmanlet3D(0.0), X, X0, F, 0.1)
    u_s = fl.evaluate(fl.Stokeslet3D(), X, X0, F, 0.1)
    assert np.allclose(u_b, u_s, rtol=1e-13, atol=1e-15)


@pytest.mark.parametrize("alpha", [0.5, 3.0, 20.0])
def test_numba_matches_reference(alpha):
    K = fl.Brinkmanlet3D(alpha)
    X, X0, F = rng.normal(size=(20, 3)), rng.normal(size=(12, 3)), rng.normal(size=(12, 3))
    eps = rng.uniform(0.05, 0.2, 12)
    ref = sum(np.array([K.velocity(x, X0[j], F[j], eps[j]) for x in X]) for j in range(12))
    assert np.allclose(fl.evaluate(K, X, X0, F, eps), ref, rtol=1e-12, atol=1e-15)


def test_series_and_closed_forms_match_high_precision():
    getcontext().prec = 50
    for z in [1e-8, 0.01, 0.3, 0.499, 0.501, 2.0, 30.0]:
        Z = Decimal(repr(z))
        E = (-Z).exp()
        exact1 = E * (1 + 1 / Z + 1 / Z**2) - 1 / Z**2
        exact2 = 3 / Z**2 - E * (1 + 3 / Z + 3 / Z**2)
        assert abs(float((Decimal(repr(f1(z))) - exact1) / exact1)) < 1e-14
        assert abs(float((Decimal(repr(f2(z))) - exact2) / exact2)) < 1e-14


def test_singular_functions_are_derivatives_of_B():
    # h1 = -B'' - B'/r and h2 = (B'' - B'/r)/r^2 with B = (1 - e^{-alpha r})/(4 pi alpha^2 r)
    a = 2.3
    B = lambda r: (1 - np.exp(-a * r)) / (4 * np.pi * a**2 * r)
    for r in [0.4, 1.0, 2.5]:
        h = 1e-4
        Bp = (B(r + h) - B(r - h)) / (2 * h)
        Bpp = (B(r + h) - 2 * B(r) + B(r - h)) / h**2
        assert np.isclose(-Bpp - Bp / r, f1(a * r) / (4 * np.pi * r), rtol=1e-6)
        assert np.isclose((Bpp - Bp / r) / r**2, f2(a * r) / (4 * np.pi * r**3), rtol=1e-6)


def test_divergence_free():
    K = fl.Brinkmanlet3D(4.0)
    src, F = rng.normal(size=(5, 3)), rng.normal(size=(5, 3))
    x0, h = np.array([0.3, -0.2, 0.1]), 1e-5
    div = sum((fl.evaluate(K, [x0 + h * e], src, F, 0.2)[0, i] - fl.evaluate(K, [x0 - h * e], src, F, 0.2)[0, i]) / (2 * h)
              for i, e in enumerate(np.eye(3)))
    assert abs(div) < 1e-6


def singular_brinkmanlet(x, f, a):
    r = np.linalg.norm(x)
    z = a * r
    h1 = (np.exp(-z) * (1 + 1 / z + 1 / z**2) - 1 / z**2) / (4 * np.pi * r)
    h2 = (3 / z**2 - np.exp(-z) * (1 + 3 / z + 3 / z**2)) / (4 * np.pi * r**3)
    return h1 * f + h2 * (f @ x) * x


def test_eps_convergence_to_singular_brinkmanlet():
    x, f, a = np.array([0.6, -0.3, 0.4]), np.array([1.0, 0.5, -0.2]), 3.0
    exact = singular_brinkmanlet(x, f, a)
    eps = np.array([0.08, 0.04, 0.02, 0.01])
    err = [np.abs(fl.evaluate(fl.Brinkmanlet3D(a), [x], [[0, 0, 0]], [f], e)[0] - exact).max() for e in eps]
    assert np.all(np.log2(np.array(err[:-1]) / np.array(err[1:])) > 1.9)


def test_screened_far_field():
    # Far away the Stokeslet's 1/r decay is screened: u -> (-f/r^3 + 3 (f.x) x / r^5) / (4 pi alpha^2)
    a, f = 2.0, np.array([0.3, -1.0, 0.5])
    x = np.array([12.0, 9.0, -8.0])
    r = np.linalg.norm(x)
    far = (-f / r**3 + 3 * (f @ x) * x / r**5) / (4 * np.pi * a**2)
    assert np.allclose(fl.evaluate(fl.Brinkmanlet3D(a), [x], [[0, 0, 0]], [f], 0.01)[0], far, rtol=1e-6)


def test_sphere_in_brinkman_flow():
    # Cortez et al. (2010), Sec. 3.1-3.2: summed forces on a unit sphere with
    # velocity U are 6 pi a U (1 + alpha a + alpha^2 a^2 / 3); a sphere spun at
    # Omega drives u = Omega x x e^{alpha (a - r)} a^3 (1 + alpha r) / (r^3 (1 + alpha a)).
    alpha, N = 1.0, 1500
    X = fibonacci_sphere(N)
    eps = 0.4 * np.sqrt(4 * np.pi / N)
    K = fl.Brinkmanlet3D(alpha)
    ez = np.array([0.0, 0.0, 1.0])
    F, _ = fl.resistance(K, X, np.tile(ez, (N, 1)), eps, method="lu")
    assert np.isclose(F[:, 2].sum(), 6 * np.pi * (1 + alpha + alpha**2 / 3), rtol=0.01)

    W, _ = fl.resistance(K, X, np.cross(ez, X), eps, method="lu")
    r = np.array([1.5, 2.0, 3.0])
    pts = np.c_[r, np.zeros(3), np.zeros(3)]
    u = fl.evaluate(K, pts, X, W, eps)
    exact = np.exp(alpha * (1 - r)) * (1 + alpha * r) / (r**3 * (1 + alpha))
    assert np.allclose(u[:, 1] / r, exact, rtol=0.03)
