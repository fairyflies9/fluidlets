"""A sphere moving through a Brinkman fluid (a fluid-filled porous medium).

Reproduces Secs. 3.1-3.2 of Cortez, Cummins, Leiderman & Varela, "Computation
of three-dimensional Brinkman flows using regularized methods", J. Comput.
Phys. 229 (2010) 7609. The sphere's surface is covered with regularized
Brinkmanlets; we prescribe a translation or a rotation and solve for the
forces (resistance problem). Exact results (mu = 1, radius a):

    summed force    F = 6 pi a U (1 + alpha a + alpha^2 a^2 / 3)
    torque          T = 8 pi a^3 Omega (1 + alpha a + alpha^2 a^2 / 3) / (1 + alpha a)
    flow, rotation  u = Omega x x  e^{alpha (a - r)} a^3 (1 + alpha r) / (r^3 (1 + alpha a))

Run from this folder:  python run.py
"""
from pathlib import Path

import numpy as np

import fluidlets as fl
from fluidlets.geometry import fibonacci_sphere
from fluidlets.io import read_input, write_vtp, write_vtr

here = Path(__file__).parent
p = read_input(here / "input3d")
a, mu = p["radius"], p["mu"]
alphas = np.atleast_1d(p["alpha_list"]).astype(float)
N = int(p["n_points"])
X = a * fibonacci_sphere(N)
h = np.sqrt(4 * np.pi * a**2 / N)
ez = np.array([0.0, 0.0, 1.0])

# The paper's regularization-1 blob widths for N = 7996, chosen there to
# minimize the drag error (Table 1) and the torque error (Table 2).
PAPER_EPS_DRAG = {0.1: 0.016, 1: 0.018, 5: 0.029, 10: 0.0373, 20: 0.0454}
PAPER_EPS_TORQUE = {0.1: 0.014, 1: 0.0055, 5: 0.0024, 10: 0.0030, 20: 0.00425}
PAPER_EPS_PROFILE = 0.014      # the paper's Fig. 4 flow profiles, all alphas


def blob(al, table):
    """Blob width for this alpha: the paper's value, or eps_ratio * h."""
    if p["eps_mode"] == "paper" and al in table:
        return table[al]
    return p["eps_ratio"] * h


def exact_drag(al):
    return 6 * np.pi * mu * a * (1 + al * a + (al * a) ** 2 / 3)


def exact_torque(al):
    return 8 * np.pi * mu * a**3 * (1 + al * a + (al * a) ** 2 / 3) / (1 + al * a)


def solve(K, velocities, e):
    forces, _ = fl.resistance(K, X, velocities, e, mu, method="gmres", rtol=1e-9, restart=200, maxiter=10)
    return forces


def drag(K, e):
    return solve(K, np.tile(ez, (N, 1)), e)[:, 2].sum()


def tuned_eps(K, al, lo=0.2 * h, hi=1.5 * h, iters=12):
    """Blob width that makes the computed drag exact (bisection; drag grows with eps)."""
    target = exact_drag(al)
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if drag(K, mid) < target else (lo, mid)
    return 0.5 * (lo + hi)


print(f"{N} points on the sphere, spacing h = {h:.4f}, eps_mode = {p['eps_mode']}\n")
print(" alpha  eps(F)    drag      exact     error  | eps(T)    torque    exact    error"
      + ("  |  eps for exact drag (eps/h)   paper eps" if p["tune_eps"] else ""))
profiles = {}
for al in alphas:
    K = fl.Brinkmanlet3D(al)
    eps_F, eps_T = blob(al, PAPER_EPS_DRAG), blob(al, PAPER_EPS_TORQUE)
    F = drag(K, eps_F)
    W = solve(K, np.cross(ez, X), eps_T)
    T = np.cross(X, W)[:, 2].sum()
    line = (f"{al:6.1f} {eps_F:7.4f} {F:9.3f} {exact_drag(al):9.3f} {F / exact_drag(al) - 1:+7.2%}  | "
            f"{eps_T:7.4f} {T:8.3f} {exact_torque(al):8.3f} {T / exact_torque(al) - 1:+7.2%}")
    if p["tune_eps"]:
        e_star = tuned_eps(K, al)
        line += f"  |  {e_star:.4f} ({e_star / h:.2f})            {PAPER_EPS_DRAG.get(al, float('nan')):.4f}"
    print(line, flush=True)
    # angular velocity of the fluid outside the rotating sphere (the paper's Fig. 4).
    # The tiny torque eps leaves gaps between points that spoil the flow just
    # outside the surface, so the profile uses its own (larger) blob width.
    eps_P = PAPER_EPS_PROFILE if p["eps_mode"] == "paper" else eps_T
    W_P = W if eps_P == eps_T else solve(K, np.cross(ez, X), eps_P)
    r = np.linspace(1.05, 2.0, 20) * a
    u = fl.evaluate(K, np.c_[r, np.zeros_like(r), np.zeros_like(r)], X, W_P, eps_P, mu)
    exact = np.exp(al * (a - r)) * a**3 * (1 + al * r) / (r**3 * (1 + al * a))
    profiles[al] = (r, u[:, 1] / r, exact)

viz = here / "viz"
viz.mkdir(exist_ok=True)
with open(viz / "rotating_profiles.csv", "w") as fh:
    fh.write("alpha,r,computed,exact\n")
    for al, (r, comp, ex) in profiles.items():
        for row in zip(r, comp, ex):
            fh.write(f"{al},{row[0]:.5f},{row[1]:.6f},{row[2]:.6f}\n")
print("\nrotating sphere, max error in the fluid's angular velocity for 1.05a <= r <= 2a:")
for al, (r, comp, ex) in profiles.items():
    print(f"  alpha = {al:4.1f}: {np.abs(comp - ex).max():.3f}  (exact value at r = 1.05a: {ex[0]:.3f})")

# Flow past the sphere on the plane y = 0 (the paper's Fig. 1): the sphere is
# held still in a uniform stream U = e_z, so the total flow is U plus the
# disturbance, which decays faster when screening is stronger (larger alpha).
axes = [np.linspace(*p[f"grid_{c}"][:2], int(p[f"grid_{c}"][2])) for c in "xyz"]
Z, Y, Xg = np.meshgrid(axes[2], axes[1], axes[0], indexing="ij")
grid = np.c_[Xg.ravel(), Y.ravel(), Z.ravel()]
for al in np.atleast_1d(p["vtk_alphas"]).astype(float):
    K = fl.Brinkmanlet3D(al)
    eps_F = blob(al, PAPER_EPS_DRAG)
    f = solve(K, -np.tile(ez, (N, 1)), eps_F)          # disturbance cancels the stream on the sphere
    u = fl.evaluate(K, grid, X, f, eps_F, mu) + ez
    u[np.linalg.norm(grid, axis=1) < a] = 0.0           # inside the solid sphere
    write_vtp(viz / f"lag_alpha{al:g}.vtp", X, {"force": f})
    write_vtr(viz / f"eul_alpha{al:g}.vtr", *axes, {"u": u, "speed": np.linalg.norm(u, axis=1)})
print(f"\nwrote rotating_profiles.csv and VTK files for alpha = {p['vtk_alphas']} to {viz}")
