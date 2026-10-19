# A sphere in a porous medium (Brinkman flow)

Some fluids flow through a sparse mesh of fibres: the glycocalyx on blood vessel walls, mucus, biofilms, a gel. Averaged over the fibres, the mesh acts like a drag spread through the fluid, proportional to the local velocity. That is **Brinkman flow**: the Stokes equations plus a resistance term,

μΔ**u** − μα²**u** − ∇p = −**f**,

where 1/α is the *screening length*, roughly the fibre spacing (α² = μ/permeability). With α = 0 there is no mesh and the flow is ordinary Stokes flow. As α grows, the disturbance made by a moving object dies off within a distance of about 1/α instead of spreading far, and the object feels much more drag.

Here a unit sphere is covered with regularized **Brinkmanlets**, the Brinkman version of regularized Stokeslets. As in the other examples, we prescribe the velocity on the surface and solve for the forces (a **resistance problem**), either for a translating sphere or a spinning one.

This reproduces Sections 3.1–3.2 of Cortez, Cummins, Leiderman & Varela (2010), with their "regularization 1" and their blob widths ε for N = 7996 surface points. The exact answers are:

| | exact | our error, α = 0.1 to 20 |
|---|---|---|
| summed force, sphere moving at U | 6πμaU (1 + αa + α²a²/3) | +0.1% to +3.6% |
| torque, sphere spun at Ω | 8πμa³Ω (1 + αa + α²a²/3)/(1 + αa) | +0.3% to +1.0% |
| fluid's angular velocity near the spinning sphere (their Fig. 4) | e^{α(a−r)} a³(1 + αr) / (r³(1 + αa)) | at most 0.012 |

The summed force uses α²a²/3, not the α²a²/9 of the drag on a translating sphere. The regularized forces also hold back the fluid that would fill the sphere's volume against the mesh, which adds α²μ·(4π/3)a³U. That matches the exact column of the paper's Table 1.

**The lesson in ε.** The paper chose a different ε for every α, picked to minimize the error on its own point set. That choice matters more as α grows: when the screening length 1/α approaches the point spacing h, the result swings strongly with ε. With `n_points = 2000` and a fixed `eps_mode = ratio`, the drag error at α = 20 is −66%. Setting `tune_eps = 1` finds the ε that makes the drag exact on your points. It grows from about 0.4h at α = 0.1 to about 1h at α = 20. The rule of thumb: resolve the screening length, h ≲ 1/α, and check ε.

## Run

```bash
python run.py
```

About 2–3 minutes (each of the 15 solves has 24,000 unknowns). It prints the table and writes to `viz/`:

- `rotating_profiles.csv`: computed and exact angular velocity of the fluid around the spinning sphere, for every α.
- `eul_alpha0.1.vtr`, `eul_alpha20.vtr` (with `lag_*.vtp`): flow past the sphere held still in a uniform stream, on the plane y = 0, like the paper's Fig. 1. In ParaView, add a Stream Tracer to each. At α = 0.1 the stream bends around the sphere from far away; at α = 20 it recovers within a fraction of a radius.

## Parameters (`input3d`)

| Key | Meaning |
|---|---|
| `alpha_list` | inverse screening lengths α to run (`Brinkmanlet3D(0)` is the Stokeslet) |
| `radius`, `mu` | sphere radius and viscosity |
| `n_points` | points on the sphere (Fibonacci spiral; the paper used 7996 Voronoi-tessellation points) |
| `eps_mode` | `paper`: the paper's ε for each α (Tables 1–2, Fig. 4); `ratio`: ε = `eps_ratio` · h |
| `eps_ratio` | ε / h when `eps_mode = ratio`, with h = √(4πa²/N) |
| `tune_eps` | 1 = also find the ε that makes the drag exact on these points (slow) |
| `vtk_alphas` | which α to write flow fields for |
| `grid_x`, `grid_y`, `grid_z` | evaluation grid for output: min, max, number of points |

## Reference

Cortez, Cummins, Leiderman & Varela (2010), *Computation of three-dimensional Brinkman flows using regularized methods*, J. Comput. Phys. 229, 7609.
