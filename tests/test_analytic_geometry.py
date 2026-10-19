"""Exact solutions and sphere discretizations against published values."""
import numpy as np

from fluidlets.analytic import brenner_wall_drag, cooley_oneill_wall_drag, translating_sphere_velocity
from fluidlets.geometry import cube_sphere, ring_sphere

ALPHA = np.array([3, 2, 1, 0.5, 0.3])


def test_brenner_matches_ainley_table2():
    # Ainley et al. (2008), Table 2, last column (d/a = cosh alpha)
    ref = np.array([1.1253, 1.4129, 3.0361, 9.2518, 23.6605])
    assert np.allclose(brenner_wall_drag(np.cosh(ALPHA)), ref, atol=1e-4)
    assert np.isclose(cooley_oneill_wall_drag(np.cosh(0.1)), 201.8640, atol=1e-3)
    assert np.isclose(brenner_wall_drag(1e6)[0], 1.0, atol=1e-5)


def test_translating_sphere_solution():
    U = np.array([0.3, -0.2, 1.0])
    surface = cube_sphere(6)
    assert np.allclose(translating_sphere_velocity(surface, U), U)         # no slip on r = a
    far = translating_sphere_velocity([[0, 0, 1e4]], U)[0]
    assert np.abs(far).max() < 2e-4                                          # decays like a/r
    assert np.allclose(translating_sphere_velocity([[0, 0, 0]], U)[0], U)    # rigid inside


def test_sphere_point_counts_match_papers():
    assert len(cube_sphere(12)) == 6 * 12**2                                 # CFM (2005), "6 x 12 x 12"
    assert len(ring_sphere(19)) == 468 and len(ring_sphere(25)) == 812       # Ainley et al. (2008)
    for X in (cube_sphere(5, 2.0, (1, 1, 1)), ring_sphere(9, 2.0, (1, 1, 1))):
        assert np.allclose(np.linalg.norm(X - 1, axis=1), 2.0)
