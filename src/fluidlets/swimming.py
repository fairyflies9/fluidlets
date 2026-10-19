"""Free swimming in Stokes flow: a body that moves by changing its shape.

Following Gallagher & Smith, "Meshfree and efficient modeling of swimming
cells", Phys. Rev. Fluids 3, 053101 (2018), Secs. II-III.

A swimmer is described in its body frame: points xi with body-frame
velocities xi_dot (the beat). The lab frame has origin x0 and orthonormal
axes B = [b1 b2 b3], so x = x0 + B xi. At every instant the unknowns are the
forces f_n on the fluid at N force points, the frame velocity U and the
angular velocity Omega. They satisfy (Eq. 8, with inertia neglected)

    (1/mu) sum_n K(x_m, n) f_n - U - Omega x (x_m - x0) = B xi_dot_m,   m = 1..N
    sum_q f(X_q) = 0,     sum_q X_q x f(X_q) = 0,

i.e. no slip on the swimmer, zero net force and zero net torque. K is the
nearest-neighbour kernel matrix (``fluidlets.core.assemble_nearest``): forces
live on the coarse points, the kernel is integrated over a finer set of
quadrature points X_q, each carrying the force of its nearest coarse point.
The trajectory then follows from dx0/dt = U, db_j/dt = Omega x b_j (Eq. 14).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.integrate import solve_ivp

from .core.assemble import assemble_nearest, nearest_neighbors
from .kernels import Stokeslet3D


def _cross_matrix(v: np.ndarray) -> np.ndarray:
    """(N, 3, 3) matrices C with C @ w = v x w for each row v."""
    v = np.atleast_2d(v)
    C = np.zeros((len(v), 3, 3))
    C[:, 0, 1], C[:, 0, 2] = -v[:, 2], v[:, 1]
    C[:, 1, 0], C[:, 1, 2] = v[:, 2], -v[:, 0]
    C[:, 2, 0], C[:, 2, 1] = -v[:, 1], v[:, 0]
    return C


@dataclass
class SwimmerShape:
    """Swimmer geometry at one instant, in the body frame.

    x : (N, 3) force (collocation) points
    v : (N, 3) their body-frame velocities (the beat)
    X : (Q, 3) quadrature points
    """
    x: np.ndarray
    v: np.ndarray
    X: np.ndarray


@dataclass
class SwimState:
    U: np.ndarray            # frame velocity (lab)
    Omega: np.ndarray        # angular velocity (lab)
    forces: np.ndarray       # (N, 3) force on the fluid at each force point (lab)
    x: np.ndarray            # (N, 3) force points (lab)
    X: np.ndarray            # (Q, 3) quadrature points (lab)
    nearest: np.ndarray      # (Q,) nearest force point of each quadrature point


def swimming_problem(shape: SwimmerShape, x0, B, eps, mu=1.0, kernel=None,
                     nearest=None) -> SwimState:
    """Solve the instantaneous free-swimming problem (Eq. 8 / Eq. 11).

    Parameters
    ----------
    shape : SwimmerShape in the body frame
    x0 : (3,) frame origin; B : (3, 3) frame axes as columns
    eps : float, blob width
    mu : float, viscosity
    kernel : fluidlets kernel (default Stokeslet3D)
    nearest : (Q,) int array, optional
        Nearest force point of each quadrature point. If omitted it is
        recomputed from the current shape.
    """
    kernel = Stokeslet3D() if kernel is None else kernel
    x0 = np.asarray(x0, float)
    x = x0 + shape.x @ B.T
    X = x0 + shape.X @ B.T
    v = shape.v @ B.T
    if nearest is None:
        nearest = nearest_neighbors(X, x)
    N = len(x)

    A = np.zeros((3 * N + 6, 3 * N + 6))
    A[:3 * N, :3 * N] = assemble_nearest(kernel, x, X, nearest, N, eps, mu)
    A[:3 * N, 3 * N:3 * N + 3] = -np.tile(np.eye(3), (N, 1))               # -U
    A[:3 * N, 3 * N + 3:] = _cross_matrix(x - x0).reshape(3 * N, 3)          # -Omega x r = r x Omega
    # Each force point stands for every quadrature point that maps to it.
    count = np.bincount(nearest, minlength=N).astype(float)
    Xsum = np.zeros((N, 3))
    np.add.at(Xsum, nearest, X)
    A[3 * N:3 * N + 3, :3 * N] = np.kron(count, np.eye(3))                  # net force
    A[3 * N + 3:, :3 * N] = np.hstack(list(_cross_matrix(Xsum)))              # net torque
    rhs = np.concatenate([v.ravel(), np.zeros(6)])
    sol = np.linalg.solve(A, rhs)
    return SwimState(sol[3 * N:3 * N + 3], sol[3 * N + 3:], sol[:3 * N].reshape(N, 3), x, X, nearest)


@dataclass
class Trajectory:
    t: np.ndarray            # (T,)
    x0: np.ndarray           # (T, 3) frame origin
    B: np.ndarray            # (T, 3, 3) frame axes as columns


def swim(shape_at: Callable[[float], SwimmerShape], t_span, eps, mu=1.0, x0=(0, 0, 0), B0=None,
         kernel=None, t_eval=None, rtol=1e-6, atol=1e-9, fixed_nearest=False) -> Trajectory:
    """Integrate a free swimmer's position and orientation over time (Eq. 14).

    Parameters
    ----------
    shape_at : function t -> SwimmerShape (body frame)
    t_span : (t0, t1)
    eps, mu, kernel : as in ``swimming_problem``
    x0, B0 : initial frame origin and axes (default identity)
    t_eval : times at which to report the trajectory
    rtol, atol : tolerances of the ODE solver (scipy RK45)
    fixed_nearest : bool
        If True, compute the nearest-neighbour map once, at t0, and keep it
        (suggested by Gallagher & Smith for bodies that come close, e.g. a
        flagellum sweeping past the cell body). Default False recomputes it at
        every instant, as their published code does.
    """
    B0 = np.eye(3) if B0 is None else np.asarray(B0, float)
    nearest = None
    if fixed_nearest:
        s0 = shape_at(t_span[0])
        nearest = nearest_neighbors(s0.X, s0.x)

    def rhs(t, z):
        b1, b2 = z[3:6], z[6:9]
        B = np.c_[b1, b2, np.cross(b1, b2)]
        st = swimming_problem(shape_at(t), z[:3], B, eps, mu, kernel, nearest)
        return np.concatenate([st.U, np.cross(st.Omega, b1), np.cross(st.Omega, b2)])

    z0 = np.concatenate([np.asarray(x0, float), B0[:, 0], B0[:, 1]])
    sol = solve_ivp(rhs, t_span, z0, t_eval=t_eval, rtol=rtol, atol=atol)
    if not sol.success:
        raise RuntimeError(sol.message)
    b1, b2 = sol.y[3:6].T, sol.y[6:9].T
    B = np.stack([b1, b2, np.cross(b1, b2)], axis=2)
    return Trajectory(sol.t, sol.y[:3].T, B)
