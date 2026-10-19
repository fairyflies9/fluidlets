# Translating and rotating sphere

A rigid sphere moving through a very viscous fluid is the classic test of any Stokes-flow method, because everything about it is known exactly. A sphere of radius a pulled at speed U feels the drag 6πμaU; one spun at rate Ω feels the torque 8πμa³Ω; and the flow around the translating sphere has a closed-form expression.

Here the sphere is covered with regularized Stokeslets on a six-patch grid (an n × n grid on each face of a cube, pushed out onto the sphere). We prescribe the velocity at every point, either a translation or a rotation, and solve for the forces (a **resistance problem**). Summing the forces gives the drag; summing r × f gives the torque.

This reproduces Section IV of Cortez, Fauci & Medovikov (2005). The rotational coefficient R₃₃ matches the paper's Table II to the two decimals printed there, for every grid and blob width computed:

| n | ε = 0.1 | ε = 0.05 | ε = 0.01 |
|---|---|---|---|
| 12 | 27.09 | 25.53 | 19.62 |
| 24 | 27.16 | 26.08 | 23.89 |

(exact 8π = 25.133). The table shows the two errors of the method pulling in opposite directions: a large blob ε over-smooths the forces (R₃₃ too big), while a blob much smaller than the point spacing leaves gaps between points (R₃₃ too small). Refining the grid fixes the second; shrinking ε fixes the first.

The script also compares the computed flow with the exact solution on the plane y = 0, and checks that the fluid *inside* the sphere moves rigidly. At n = 24 and ε = 0.05 both errors are about 10⁻²; CFM report errors near 4 × 10⁻⁴ for the interior check on a finer 6 × 48 × 48 grid.

## Run

```bash
python run.py
```

About 25 seconds. It prints the table and writes `viz/lag.vtp` (surface points with forces) and `viz/eul.vtr` (computed velocity `u`, exact `u_exact` and `error` on the plane y = 0). Add `36 48` to `patches_n` in `input3d` for the paper's finer columns (these take several minutes).

## Parameters (`input3d`)

| Key | Meaning |
|---|---|
| `radius`, `mu` | sphere radius and viscosity |
| `patches_n` | grid points per cube-face edge; the sphere has 6 n² points |
| `eps` | blob widths to try |
| `gmres_tol` | relative tolerance of the iterative solve |
| `vtk_n`, `vtk_eps` | which case to write to VTK |
| `grid_x`, `grid_y`, `grid_z` | evaluation grid for output: min, max, number of points |

## Reference

Cortez, Fauci & Medovikov (2005), *The method of regularized Stokeslets in three dimensions: analysis, validation, and application to helical swimming*, Phys. Fluids 17, 031504.
