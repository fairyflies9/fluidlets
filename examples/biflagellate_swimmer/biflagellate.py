"""The model biflagellate of Gallagher & Smith (2018), Sec. V A.

Reads ``input3d`` from this folder and provides ``shape_at(t)``, the swimmer's
force points, their velocities and the quadrature points in the body frame.
Imported by run.py and the movie scripts.

Cell body: rigid ellipsoid with semi-axes head_axes. Flagella: unit length,
mirror images, beating in synchrony with tangent angle (their Eqs. 29-30)

    psi(s, t) = -2.5 s - (0.7 + 0.15 sin 2 pi s) cos(t - 2 pi s),   0 <= s <= 1.
"""
from pathlib import Path

import numpy as np
from scipy.integrate import cumulative_trapezoid

from fluidlets.geometry import cube_sphere
from fluidlets.io import read_input
from fluidlets.swimming import SwimmerShape

here = Path(__file__).parent
p = read_input(here / "input3d")
a1, a2, a3 = p["head_axes"]
ang = np.deg2rad(p["insertion_angle"])
eps, mu = p["eps"], p["mu"]

# ---------------------------------------------------------------- the beat
S_FINE = np.linspace(0.0, 1.0, 4001)
S_REF = 4 / 29        # the base chord 0 -> S_REF is held pointing forward, as in the authors' code


def flagellum_plane(s, t):
    """Flagellum shape (X, Y) in its beat plane, base at the origin."""
    if p["beat_variant"] == "authors_code":
        # The published ChlamyFromModel.m integrates over 30 panels and then drops
        # the first, so its flagellum is the arc 1/30 <= sigma <= 1 of the curve
        # (length 29/30) laid on 0 <= s <= 1. Reproduce that to compare numbers.
        s = 1 / 30 + np.asarray(s) * 29 / 30
        s_ref = 5 / 30
        base = 1 / 30
    else:
        s_ref, base = S_REF, 0.0
    psi = -2.5 * S_FINE - (0.7 + 0.15 * np.sin(2 * np.pi * S_FINE)) * np.cos(t - 2 * np.pi * S_FINE)
    cx = cumulative_trapezoid(np.cos(psi), S_FINE, initial=0.0)
    cy = cumulative_trapezoid(np.sin(psi), S_FINE, initial=0.0)
    cx -= np.interp(base, S_FINE, cx)
    cy -= np.interp(base, S_FINE, cy)
    # rotate so that the chord from the base to s_ref points along +Y
    rot = np.pi / 2 - np.arctan2(np.interp(s_ref, S_FINE, cy), np.interp(s_ref, S_FINE, cx))
    X = np.cos(rot) * cx - np.sin(rot) * cy
    Y = np.sin(rot) * cx + np.cos(rot) * cy
    return np.interp(s, S_FINE, X), np.interp(s, S_FINE, Y)


def rotate_z(P, theta):
    c, s = np.cos(theta), np.sin(theta)
    return P @ np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]]).T


def flagella(s, t):
    """Both flagella in the body frame: cis on the +x1 side, trans its mirror image."""
    X, Y = flagellum_plane(s, t)
    zero = np.zeros_like(X)
    cis = rotate_z(np.c_[X, Y, zero], -ang) + [a1 * np.sin(ang), a2 * np.cos(ang), 0]
    trans = rotate_z(np.c_[-X, Y, zero], ang) + [-a1 * np.sin(ang), a2 * np.cos(ang), 0]
    return np.vstack([cis, trans])


head = cube_sphere(int(p["n_head"])) * [a1, a2, a3]
head_quad = cube_sphere(int(p["n_head_quad"])) * [a1, a2, a3]
s_force = np.linspace(0, 1, int(p["n_flag"]))
s_quad = np.linspace(0, 1, int(p["n_flag_quad"]))


def shape_at(t, dt=1e-5):
    """Body-frame force points, their velocities (central differences in t) and quadrature points."""
    flag = flagella(s_force, t)
    flag_v = (flagella(s_force, t + dt) - flagella(s_force, t - dt)) / (2 * dt)
    return SwimmerShape(x=np.vstack([head, flag]),
                        v=np.vstack([np.zeros_like(head), flag_v]),
                        X=np.vstack([head_quad, flagella(s_quad, t)]))
