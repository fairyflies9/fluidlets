"""Glycocalyx towers in shear flow above a wall.

Three hexagonal towers stand on a wall in the shear flow u = 0.1 z. We find
the regularized forces that hold every tower point still (a resistance
problem), then evaluate the flow on a grid and write VTK files for ParaView.

Run from this folder:  python run.py
"""
from pathlib import Path

import numpy as np

import fluidlets as fl
from fluidlets.geometry import towers_from_table
from fluidlets.io import read_input, read_vertex, write_vertex, write_vtp, write_vtr

here = Path(__file__).parent
p = read_input(here / "input3d")
name = p["string_name"]

# Geometry: rebuild the .vertex file from the tower table if it is missing.
vertex_file = here / f"{name}.vertex"
if not vertex_file.exists():
    write_vertex(vertex_file, towers_from_table(np.loadtxt(here / "towers.txt")))
X = read_vertex(vertex_file)
kernel = fl.Stokeslet3DWall() if p["wall"] == "z0" else fl.Stokeslet3D()


def background(x):
    """Shear flow u = shear_rate * z in x."""
    u = np.zeros_like(x)
    u[:, 0] = p["shear_rate"] * x[:, 2]
    return u


# Resistance problem: the disturbance flow must cancel the shear at every tower point.
forces, info = fl.resistance(kernel, X, -background(X), p["eps"], p["mu"],
                             method="gmres", rtol=p["gmres_tol"])
print(f"{len(X)} points, GMRES {info.iterations} iterations, residual {info.residual:.1e}")

towers = np.loadtxt(here / "towers.txt", ndmin=2)
per_tower = len(X) // len(towers)
for k, row in enumerate(towers):
    drag = forces[k * per_tower:(k + 1) * per_tower].sum(axis=0)
    print(f"tower {k} at y = {row[2]:4.1f}: force on fluid = ({drag[0]:8.3f}, {drag[1]:7.3f}, {drag[2]:7.3f})")

# Flow on the evaluation grid (output only; it never affects the solve).
axes = [np.linspace(*p[f"grid_{c}"][:2], int(p[f"grid_{c}"][2])) for c in "xyz"]
Z, Y, Xg = np.meshgrid(axes[2], axes[1], axes[0], indexing="ij")
grid = np.c_[Xg.ravel(), Y.ravel(), Z.ravel()]
u = fl.evaluate(kernel, grid, X, forces, p["eps"], p["mu"]) + background(grid)

viz = here / "viz"
viz.mkdir(exist_ok=True)
write_vtp(viz / "lag.0000.vtp", X, {"force": forces})
write_vtr(viz / "eul.0000.vtr", *axes, {"u": u, "speed": np.linalg.norm(u, axis=1)})
print(f"wrote {viz / 'lag.0000.vtp'} and {viz / 'eul.0000.vtr'}")

# Optional check against the grid the 2010 CUDA code wrote for the same setup.
legacy = here.parents[1] / "legacy" / "sdc_cuda3" / "velocity_t_xyz_final.raw"
if legacy.exists() and u.shape == (32**3, 3):
    stored = np.fromfile(legacy, "<f4").reshape(-1, 3)
    print(f"relative difference from the 2010 legacy grid: {np.linalg.norm(u - stored) / np.linalg.norm(stored):.1e}")
