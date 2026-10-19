"""A model biflagellate alga swimming by beating two flagella.

Reproduces Sec. V A of Gallagher & Smith, "Meshfree and efficient modeling
of swimming cells", Phys. Rev. Fluids 3, 053101 (2018). The cell body is a
rigid ellipsoid; two flagella of unit length beat in synchrony, mirror images
of each other, with tangent angle (their Eqs. 29-30)

    psi(s, t) = -2.5 s - (0.7 + 0.15 sin 2 pi s) cos(t - 2 pi s),   0 <= s <= 1.

The cell has no motor force or torque overall, so at every instant we solve
for the forces on the fluid together with the cell's velocity and spin
(free-swimming problem), then integrate the cell's position and orientation.

Run from this folder:  python run.py
"""
from pathlib import Path

import numpy as np
from scipy.integrate import cumulative_trapezoid

import fluidlets as fl
from fluidlets.geometry import cube_sphere
from fluidlets.io import read_input, write_pvd, write_vtp, write_vtr
from fluidlets.swimming import SwimmerShape, swim, swimming_problem

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
    flag = flagella(s_force, t)
    flag_v = (flagella(s_force, t + dt) - flagella(s_force, t - dt)) / (2 * dt)
    return SwimmerShape(x=np.vstack([head, flag]),
                        v=np.vstack([np.zeros_like(head), flag_v]),
                        X=np.vstack([head_quad, flagella(s_quad, t)]))


# ---------------------------------------------------------------- swim
beats = p["beats"]
frames = int(p["frames_per_beat"]) * beats
t_out = np.linspace(0, 2 * np.pi * beats, frames + 1)
shape0 = shape_at(0.0)
print(f"{len(shape0.x)} force points, {len(shape0.X)} quadrature points, eps = {eps}")
traj = swim(shape_at, (t_out[0], t_out[-1]), eps, mu, t_eval=t_out, rtol=p["ode_rtol"], atol=1e-10)

dist = np.linalg.norm(traj.x0[-1] - traj.x0[0])
per_beat = traj.x0[int(p["frames_per_beat"])] - traj.x0[0]
print(f"distance after {beats} beat(s): {dist:.4e} flagellar lengths")
print(f"displacement over the first beat: {per_beat.round(5)} (forward is +x2)")
if np.isclose(p["insertion_angle"], 30) and beats == 1:
    ref = "5.52e-2 to 5.53e-2 (their Table II, 40 force points per flagellum)"
    note = "" if p["beat_variant"] == "authors_code" else "; set beat_variant = authors_code to compare (see README)"
    print(f"Gallagher & Smith (2018): {ref}{note}")

# ---------------------------------------------------------------- output
viz = here / "viz"
viz.mkdir(exist_ok=True)
axes = [np.linspace(*p[f"grid_{c}"][:2], int(p[f"grid_{c}"][2])) for c in "xyz"]
Z, Y, Xg = np.meshgrid(axes[2], axes[1], axes[0], indexing="ij")
grid = np.c_[Xg.ravel(), Y.ravel(), Z.ravel()]
flow_frames = set(np.atleast_1d(p["flow_frames"]).astype(int).tolist())
lag, eul = [], []
for k, t in enumerate(traj.t):
    st = swimming_problem(shape_at(t), traj.x0[k], traj.B[k], eps, mu)
    write_vtp(viz / f"lag.{k:04d}.vtp", st.x, {"force": st.forces})
    # the flow comes from every quadrature point, each carrying its nearest force point's force
    lag.append((t, f"lag.{k:04d}.vtp"))
    if k in flow_frames:
        u = fl.evaluate(fl.Stokeslet3D(), grid, st.X, st.forces[st.nearest], eps, mu)
        write_vtr(viz / f"eul.{k:04d}.vtr", *axes, {"u": u, "speed": np.linalg.norm(u, axis=1)})
        eul.append((t, f"eul.{k:04d}.vtr"))
write_pvd(viz / "lag.pvd", lag)
write_pvd(viz / "eul.pvd", eul)
np.savetxt(viz / "trajectory.csv", np.c_[traj.t, traj.x0], delimiter=",",
           header="t,x1,x2,x3", comments="")
print(f"wrote {len(lag)} swimmer frames and {len(eul)} flow fields to {viz}")
