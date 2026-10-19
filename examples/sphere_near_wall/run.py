"""Drag on a sphere translating parallel or normal to a no-slip wall.

Reproduces Tables 1 and 2 of Ainley, Durkin, Embid, Boindala & Cortez,
"The method of images for regularized Stokeslets", J. Comput. Phys. 227
(2008) 4600-4616, Sec. 5.4: a sphere of radius 0.1 is discretized with rings
of points, every point is given the same velocity, and the regularized
Stokeslets plus their wall images are solved for the forces. The total force,
divided by Stokes' drag 6 pi mu a U, is compared with the exact results.

Run from this folder:  python run.py
"""
from pathlib import Path

import numpy as np

import fluidlets as fl
from fluidlets.analytic import brenner_wall_drag, cooley_oneill_wall_drag
from fluidlets.geometry import ring_sphere
from fluidlets.io import read_input, write_vtp, write_vtr

here = Path(__file__).parent
p = read_input(here / "input3d")
a, mu = p["radius"], p["mu"]
alphas = np.atleast_1d(p["alpha"])
d_over_a = np.cosh(alphas)
n_phis = np.atleast_1d(p["n_phi"]).astype(int)
wall = fl.Stokeslet3DWall()

# Exact values. Parallel: O'Neill (1964), as tabulated by Ainley et al. (Table 1).
ONEILL = dict(zip([10, 3, 2, 1, 0.5, 0.3, 0.1], [1.0000, 1.0591, 1.1738, 1.5675, 2.1515, 2.6475, 3.7863]))
exact_par = np.array([ONEILL.get(float(al), np.nan) for al in alphas])
# Normal: Brenner's series for d/a >= 1.02, Cooley & O'Neill's asymptotics below.
exact_norm = np.where(d_over_a >= 1.02, brenner_wall_drag(d_over_a), cooley_oneill_wall_drag(d_over_a))

# Ainley et al.'s own values, columns S = 468, 812, 1486, 2718.
PAPER_PAR = {468: [0.9933, 1.0515, 1.1644, 1.5459, 2.0614, 2.4226, 3.6581],
             812: [0.9960, 1.0545, 1.1681, 1.5537, 2.0851, 2.4621, 3.4548]}
PAPER_NORM = {468: [1.0240, 1.1556, 1.4605, 3.2441, 10.2872, 19.7071, 2010.44],
              812: [1.0187, 1.1488, 1.4497, 3.1952, 10.0943, 19.4680, 736.627]}


def drag(X, U, eps):
    forces, info = fl.resistance(wall, X, np.tile(U, (len(X), 1)), eps, mu,
                                 method="gmres", rtol=p["gmres_tol"], maxiter=1000)
    return forces, forces.sum(axis=0) @ U / (6 * np.pi * mu * a)


results = {}
vtk_cases = {}
for n_phi in n_phis:
    unit = ring_sphere(n_phi, a)
    S = len(unit)
    h = np.sqrt(4 * np.pi * a**2 / S)
    for direction, U, coef in [("parallel", np.array([1.0, 0, 0]), p["eps_coef_parallel"]),
                               ("normal", np.array([0, 0, 1.0]), p["eps_coef_normal"])]:
        eps = coef * h**0.9
        for k, d in enumerate(d_over_a):
            X = unit + [0, 0, d * a]
            forces, F = drag(X, U, eps)
            results[(direction, S, k)] = F
            if k == p["vtk_case"] and n_phi == n_phis[-1]:
                vtk_cases[direction] = (X, forces, eps)
        print(f"S = {S} (h = {h:.4f}), {direction}: done")

print()
for direction, exact, paper in [("parallel", exact_par, PAPER_PAR), ("normal", exact_norm, PAPER_NORM)]:
    sizes = sorted({S for (dr, S, _) in results if dr == direction})
    print(f"Translation {direction} to the wall: F / (6 pi mu a U)")
    header = "   d/a        gap " + "".join(f"  S={S:<5d} (paper)  " for S in sizes) + "   exact"
    print(header)
    for k, d in enumerate(d_over_a):
        row = f"{d:9.4f} {a * (d - 1):9.5f} "
        for S in sizes:
            ref = paper.get(S, [np.nan] * len(alphas))[k] if len(alphas) == 7 else np.nan
            row += f"  {results[(direction, S, k)]:9.4f} ({ref:8.4f})"
        print(row + f"  {exact[k]:9.4f}")
    print()

# Flow around the sphere for one case, on the plane y = 0, for ParaView.
axes = [np.linspace(*p[f"grid_{c}"][:2], int(p[f"grid_{c}"][2])) for c in "xyz"]
Z, Y, Xg = np.meshgrid(axes[2], axes[1], axes[0], indexing="ij")
grid = np.c_[Xg.ravel(), Y.ravel(), Z.ravel()]
viz = here / "viz"
viz.mkdir(exist_ok=True)
for direction in ("parallel", "normal"):
    X, forces, eps = vtk_cases[direction]
    u = fl.evaluate(wall, grid, X, forces, eps, mu)
    write_vtp(viz / f"lag_{direction}.vtp", X, {"force": forces})
    write_vtr(viz / f"eul_{direction}.vtr", *axes, {"u": u, "speed": np.linalg.norm(u, axis=1)})
print(f"wrote VTK files to {viz}")
