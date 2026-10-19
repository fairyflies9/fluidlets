"""Resistance matrices and flow field of a sphere in Stokes flow.

Reproduces Sec. IV of Cortez, Fauci & Medovikov, "The method of regularized
Stokeslets in three dimensions: analysis, validation, and application to
helical swimming", Phys. Fluids 17, 031504 (2005). The unit sphere is covered
by a six-patch grid; prescribing a translation U or a rotation Omega at every
point and solving for the regularized forces gives the total force and torque.
Exact values (mu = 1): T33 = 6 pi = 18.850 and R33 = 8 pi = 25.133.

Run from this folder:  python run.py
"""
from pathlib import Path

import numpy as np

import fluidlets as fl
from fluidlets.analytic import translating_sphere_velocity
from fluidlets.geometry import cube_sphere
from fluidlets.io import read_input, write_vtp, write_vtr

here = Path(__file__).parent
p = read_input(here / "input3d")
a, mu = p["radius"], p["mu"]
kernel = fl.Stokeslet3D()
ez = np.array([0.0, 0.0, 1.0])

# Table II of the paper: R33 for eps (rows) and 6 x n x n grids (columns).
PAPER_R33 = {(0.1, 12): 27.09, (0.1, 24): 27.16, (0.1, 36): 27.16, (0.1, 48): 27.16,
             (0.05, 12): 25.53, (0.05, 24): 26.08, (0.05, 36): 26.10, (0.05, 48): 26.11,
             (0.01, 12): 19.62, (0.01, 24): 23.89, (0.01, 36): 24.80, (0.01, 48): 25.09}


def solve(X, velocities, eps):
    forces, info = fl.resistance(kernel, X, velocities, eps, mu, method="gmres",
                                 rtol=p["gmres_tol"], restart=300, maxiter=5)
    return forces


saved = {}
print("Sphere resistance coefficients (exact T33 = 18.850, R33 = 25.133)")
print("   n  points    eps     T33      R33   R33 (paper)")
for n in np.atleast_1d(p["patches_n"]).astype(int):
    X = a * cube_sphere(n)
    for eps in np.atleast_1d(p["eps_sweep"]):
        f_trans = solve(X, np.tile(ez, (len(X), 1)), eps)          # translate with U = e_z
        f_rot = solve(X, np.cross(ez, X), eps)                      # rotate with Omega = e_z
        T33 = f_trans[:, 2].sum()
        R33 = np.cross(X, f_rot)[:, 2].sum()
        paper = PAPER_R33.get((round(float(eps), 3), n), np.nan)
        print(f"{n:4d} {len(X):7d} {eps:6.3f} {T33:8.3f} {R33:8.3f} {paper:9.2f}")
        if n == p["vtk_n"] and np.isclose(eps, p["vtk_eps"]):
            saved = dict(X=X, f_trans=f_trans, f_rot=f_rot, eps=eps)

if not saved:
    raise SystemExit("vtk_n / vtk_eps do not match a computed case")

# Flow field of the translating sphere on the plane y = 0, compared with the
# exact solution; inside the sphere the fluid should move rigidly with U.
X, f, eps = saved["X"], saved["f_trans"], saved["eps"]
axes = [np.linspace(*p[f"grid_{c}"][:2], int(p[f"grid_{c}"][2])) for c in "xyz"]
Z, Y, Xg = np.meshgrid(axes[2], axes[1], axes[0], indexing="ij")
grid = np.c_[Xg.ravel(), Y.ravel(), Z.ravel()]
u = fl.evaluate(kernel, grid, X, f, eps, mu)
exact = translating_sphere_velocity(grid, ez, a)
err = np.linalg.norm(u - exact, axis=1)
r = np.linalg.norm(grid, axis=1)
print(f"\nn = {p['vtk_n']}, eps = {eps}: max velocity error for r > 1.5a: {err[r > 1.5 * a].max():.2e}, "
      f"inside r < 0.8a: {err[r < 0.8 * a].max():.2e}")

# Rigid-body check of Sec. IV: translation + rotation, velocity on z = 0.2 inside.
inside = np.c_[np.random.default_rng(0).uniform(-0.5, 0.5, (200, 2)), np.full(200, 0.2)]
v = fl.evaluate(kernel, inside, X, f + saved["f_rot"], eps, mu)
rigid = ez + np.cross(ez, inside)
print(f"inside the sphere, translating + rotating: max |u - (U + Omega x x)| = {np.abs(v - rigid).max():.2e}")

viz = here / "viz"
viz.mkdir(exist_ok=True)
write_vtp(viz / "lag.vtp", X, {"force_translate": f, "force_rotate": saved["f_rot"]})
write_vtr(viz / "eul.vtr", *axes, {"u": u, "u_exact": exact, "error": err})
print(f"wrote VTK files to {viz}")
