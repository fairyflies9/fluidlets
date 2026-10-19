"""Animated GIF of the swimmer and its 3D flow, rendered with ParaView's VTK.

Needs ParaView (for pvpython) and Pillow (pip install pillow). Steps:
  1. simulate two beats and save the 3D flow around the cell at each frame
     (viz/movie3d/frameNNN.npz);
     and carry passive tracer particles through the time-dependent flow;
  2. render each frame with render_3d.py under pvpython: cell body, flagella
     as tubes, tracers with short trails coloured by speed, a faint speed
     slice in the beat plane, and a camera orbiting the swimming axis;
  3. stitch the images into viz/biflagellate_3d.gif.
Run from this folder:  python movie_3d.py   (about 7 minutes)
Restyle without re-simulating:  python movie_3d.py --render-only   (edit render_3d.py first)
"""
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image

import fluidlets as fl
from fluidlets.swimming import swim, swimming_problem

from biflagellate import a1, a2, a3, eps, here, head_quad, mu, s_quad, shape_at

BEATS, FRAMES_PER_BEAT = 2, 30
SUBSTEPS = 6          # tracer time steps per frame
N_TRACERS = 500


def find_pvpython():
    found = shutil.which("pvpython")
    if found:
        return found
    roots = [Path(os.environ.get("ProgramFiles", "C:/Program Files")), Path("/Applications")]
    for root in roots:
        for cand in sorted(root.glob("ParaView*/bin/pvpython*")) + sorted(root.glob("ParaView*/Contents/bin/pvpython")):
            return str(cand)
    raise SystemExit("pvpython not found: install ParaView or put pvpython on your PATH")


pvpython = find_pvpython()
work = here / "viz" / "movie3d"
work.mkdir(parents=True, exist_ok=True)

if "--render-only" not in sys.argv:
    # 1. simulate the swimmer on a fine time grid (tracer steps and their midpoints)
    n_steps = BEATS * FRAMES_PER_BEAT * SUBSTEPS
    t_fine = np.linspace(0, 2 * np.pi * BEATS, 2 * n_steps + 1)
    traj = swim(shape_at, (0, t_fine[-1]), eps, mu, t_eval=t_fine, rtol=1e-6, atol=1e-10)
    states = [swimming_problem(shape_at(t), traj.x0[k], traj.B[k], eps, mu) for k, t in enumerate(t_fine)]
    print(f"solved the swimming problem at {len(states)} times")


    def velocity(k, x):
        """Fluid velocity at points x at fine time index k (exact Stokeslet sum)."""
        st = states[k]
        return fl.evaluate(fl.Stokeslet3D(), x, st.X, st.forces[st.nearest], eps, mu)


    # tracers: random points in a shell around the cell, outside the body
    rng = np.random.default_rng(3)
    pts = rng.uniform([-1.4, -1.0, -0.9], [1.4, 1.9, 0.9], (6 * N_TRACERS, 3))
    outside = ((pts / [a1 * 1.05, a2 * 1.05, a3 * 1.05]) ** 2).sum(axis=1) > 1
    tracers = pts[outside][:N_TRACERS] + traj.x0[0]

    gx, gy, gz = np.linspace(-1.6, 1.6, 41), np.linspace(-1.2, 2.0, 41), np.linspace(-1.2, 1.2, 31)
    Z, Y, X = np.meshgrid(gz, gy, gx, indexing="ij")
    local = np.c_[X.ravel(), Y.ravel(), Z.ravel()]
    nf = len(s_quad)
    dt = t_fine[2] - t_fine[0]
    for frame in range(BEATS * FRAMES_PER_BEAT + 1):
        k = 2 * SUBSTEPS * frame
        st, x0 = states[k], traj.x0[k]
        u_tr = velocity(k, tracers)
        u = velocity(k, local + x0)
        flag = st.X[len(head_quad):]
        np.savez(work / f"frame{frame:03d}.npz", t=t_fine[k], x0=x0, B=traj.B[k], start=traj.x0[0],
                 gx=gx + x0[0], gy=gy + x0[1], gz=gz + x0[2], u=u,
                 cis=flag[:nf], trans=flag[nf:], axes=np.array([a1, a2, a3]),
                 tracers=tracers, tracer_speed=np.linalg.norm(u_tr, axis=1))
        if frame == BEATS * FRAMES_PER_BEAT:
            break
        for j in range(SUBSTEPS):                      # midpoint (RK2) steps to the next frame
            i = k + 2 * j
            half = tracers + 0.5 * dt * velocity(i, tracers)
            tracers = tracers + dt * velocity(i + 1, half)
    print(f"saved {BEATS * FRAMES_PER_BEAT + 1} frames of 3D flow and tracers")

# 2. render with ParaView's VTK
subprocess.run([pvpython, str(here / "render_3d.py"), str(work)], check=True)

# 3. assemble the GIF
images = [Image.open(f).convert("RGB").resize((720, 608), Image.LANCZOS)
          for f in sorted(work.glob("img*.png"))]
frames = [im.quantize(colors=200, method=Image.MEDIANCUT) for im in images]
out = here / "viz" / "biflagellate_3d.gif"
frames[0].save(out, save_all=True, append_images=frames[1:], duration=90, loop=0, optimize=True)
print(f"wrote {out}")
