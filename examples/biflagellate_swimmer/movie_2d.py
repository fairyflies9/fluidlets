"""Animated GIF of the swimmer and the flow in its beat plane (x3 = 0).

Needs matplotlib (with Pillow), which Fluidlets itself does not require:
    pip install matplotlib
Run from this folder:  python movie_2d.py
Writes viz/biflagellate.gif (two beats, about 2 minutes).
"""
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

import fluidlets as fl
from fluidlets.swimming import swim, swimming_problem

from biflagellate import a1, a2, eps, here, head_quad, mu, shape_at

BEATS, FRAMES_PER_BEAT = 2, 40
t_out = np.linspace(0, 2 * np.pi * BEATS, BEATS * FRAMES_PER_BEAT + 1)
traj = swim(shape_at, (0, t_out[-1]), eps, mu, t_eval=t_out, rtol=1e-6, atol=1e-10)

# flow on a grid in the beat plane, at every frame
xs, ys = np.linspace(-1.6, 1.6, 81), np.linspace(-1.4, 2.2, 91)
GX, GY = np.meshgrid(xs, ys)
grid = np.c_[GX.ravel(), GY.ravel(), np.zeros(GX.size)]
frames = []
for k, t in enumerate(traj.t):
    st = swimming_problem(shape_at(t), traj.x0[k], traj.B[k], eps, mu)
    u = fl.evaluate(fl.Stokeslet3D(), grid, st.X, st.forces[st.nearest], eps, mu)
    frames.append((st.X[len(head_quad):], u[:, 0].reshape(GX.shape), u[:, 1].reshape(GX.shape)))

fig, ax = plt.subplots(figsize=(5.2, 5.8), dpi=110)
vmax = np.median([np.hypot(U, V).max() for _, U, V in frames])
theta = np.linspace(0, 2 * np.pi, 100)


def draw(k):
    ax.clear()
    flag, U, V = frames[k]
    ax.pcolormesh(xs, ys, np.hypot(U, V), cmap="Blues", vmin=0, vmax=vmax, shading="auto")
    ax.streamplot(xs, ys, U, V, color="0.35", density=1.1, linewidth=0.5, arrowsize=0.6)
    c, b1, b2 = traj.x0[k], traj.B[k][:2, 0], traj.B[k][:2, 1]
    outline = c[:2, None] + np.outer(b1, a1 * np.cos(theta)) + np.outer(b2, a2 * np.sin(theta))
    ax.fill(outline[0], outline[1], color="#2e7d32", zorder=3)
    ax.plot(flag[:, 0], flag[:, 1], ".", color="#1b5e20", ms=1.5, zorder=4)
    ax.plot(traj.x0[: k + 1, 0], traj.x0[: k + 1, 1], color="#c62828", lw=1.5, zorder=5)
    ax.set_xlim(xs[0], xs[-1])
    ax.set_ylim(ys[0], ys[-1])
    ax.set_aspect("equal")
    ax.set_xlabel("x$_1$ (flagellar lengths)")
    ax.set_ylabel("x$_2$ (flagellar lengths)")
    ax.set_title(f"Model biflagellate (Gallagher & Smith 2018)\nbeat {traj.t[k] / (2 * np.pi):.2f}   "
                 f"swum {traj.x0[k, 1] - traj.x0[0, 1]:.3f}", fontsize=9)


out = here / "viz" / "biflagellate.gif"
out.parent.mkdir(exist_ok=True)
FuncAnimation(fig, draw, frames=len(frames)).save(out, writer=PillowWriter(fps=12))
print(f"wrote {out}")
