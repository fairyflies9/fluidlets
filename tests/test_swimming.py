"""Nearest-neighbour assembly and the free-swimming solver."""
import numpy as np

import fluidlets as fl
from fluidlets.core import assemble_nearest, nearest_neighbours
from fluidlets.geometry import cube_sphere
from fluidlets.swimming import SwimmerShape, swim, swimming_problem

K = fl.Stokeslet3D()


def test_nearest_with_identical_points_is_ordinary_assembly():
    X = cube_sphere(4)
    nn = nearest_neighbours(X, X)
    assert np.array_equal(nn, np.arange(len(X)))
    assert np.allclose(assemble_nearest(K, X, X, nn, len(X), 0.1), fl.assemble(K, X, X, 0.1), rtol=1e-14)


def test_nearest_assembly_sums_grouped_quadrature():
    x = cube_sphere(3)                      # coarse force points
    Xq = cube_sphere(9)                     # fine quadrature points
    nn = nearest_neighbours(Xq, x)
    A = assemble_nearest(K, x, Xq, nn, len(x), 0.05)
    f = np.random.default_rng(0).normal(size=(len(x), 3))
    direct = fl.evaluate(K, x, Xq, f[nn], 0.05)      # every quadrature point carries its group's force
    assert np.allclose(A @ f.ravel(), direct.ravel(), rtol=1e-12)


def sphere_shape(slip):
    x = cube_sphere(4)
    return SwimmerShape(x=x, v=slip(x), X=cube_sphere(10))


def test_uniform_slip_swims_rigidly():
    # Surface points moving backwards at -V in the body frame: the exact
    # solution is a body moving forward at V through still fluid, zero force.
    V = np.array([0.2, -0.1, 0.7])
    st = swimming_problem(sphere_shape(lambda x: -np.tile(V, (len(x), 1))), np.zeros(3), np.eye(3), 0.05)
    assert np.allclose(st.U, V, atol=1e-12) and np.allclose(st.Omega, 0, atol=1e-12)
    assert np.abs(st.forces).max() < 1e-10


def test_rotational_slip_spins_rigidly():
    W = np.array([0.0, 0.3, 1.0])
    B = np.linalg.qr(np.random.default_rng(1).normal(size=(3, 3)))[0]
    B *= np.sign(np.linalg.det(B))                    # proper rotation
    x0 = np.array([1.0, 2.0, -1.0])
    W_body = B.T @ W
    st = swimming_problem(sphere_shape(lambda x: -np.cross(W_body, x)), x0, B, 0.05)
    assert np.allclose(st.Omega, W, atol=1e-12) and np.allclose(st.U, 0, atol=1e-12)


def test_rigid_body_without_slip_does_not_move():
    st = swimming_problem(sphere_shape(np.zeros_like), np.zeros(3), np.eye(3), 0.05)
    assert np.allclose(st.U, 0) and np.allclose(st.Omega, 0)


def test_trajectory_of_steady_slip_swimmer():
    V = np.array([0.0, 0.0, 0.5])
    W = np.array([0.0, 0.0, 0.25])          # body-frame spin about the swimming axis

    def shape_at(t):
        x = cube_sphere(3)
        return SwimmerShape(x=x, v=-(V + np.cross(W, x)), X=cube_sphere(6))

    tr = swim(shape_at, (0.0, 2.0), 0.05, t_eval=[2.0], rtol=1e-9, atol=1e-12)
    assert np.allclose(tr.x0[-1], [0, 0, 1.0], atol=1e-7)             # translates 0.5 * 2
    angle = 0.25 * 2.0
    assert np.allclose(tr.B[-1][:, 0], [np.cos(angle), np.sin(angle), 0], atol=1e-7)
