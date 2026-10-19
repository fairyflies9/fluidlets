import numpy as np
from fluidlets import stokeslet3d_velocity


def test_far_field_matches_singular_stokeslet():
    x = np.array([[3.0, 1.0, -2.0]]); f = np.array([[1.0, -0.5, 2.0]])
    u = stokeslet3d_velocity(x, [[0, 0, 0]], f, eps=1e-6)
    r = np.linalg.norm(x)
    exact = (f / r + (f @ x.T) * x / r**3) / (8 * np.pi)
    assert np.allclose(u, exact, rtol=1e-8)


def test_divergence_free():
    rng = np.random.default_rng(0)
    src = rng.normal(size=(5, 3)); f = rng.normal(size=(5, 3))
    x0 = np.array([0.3, -0.2, 0.1]); h = 1e-5; div = 0.0
    for i in range(3):
        e = np.zeros(3); e[i] = h
        up = stokeslet3d_velocity([x0 + e], src, f, eps=0.2)
        um = stokeslet3d_velocity([x0 - e], src, f, eps=0.2)
        div += (up[0, i] - um[0, i]) / (2 * h)
    assert abs(div) < 1e-7
