# Sphere moving near a wall

Pull a small sphere through a viscous fluid and it feels Stokes drag, 6πμaU. Bring it close to a wall and the drag goes up, because the fluid between the sphere and the wall has to squeeze out of the way. Moving *toward* the wall is much harder than sliding *along* it: as the gap closes, the drag for head-on approach grows without bound.

Here a sphere of radius 0.1 is covered with rings of regularized Stokeslets and moved either parallel or normal to a no-slip wall at z = 0. The wall is enforced exactly by the image system of Ainley et al. (2008). We solve for the forces that give every surface point the same velocity (a **resistance problem**), add them up, and divide by 6πμaU.

This reproduces Section 5.4 of Ainley, Durkin, Embid, Boindala & Cortez (2008). The script prints, side by side, our result, the paper's result for the same discretization, and the exact value: O'Neill (1964) for parallel motion, and Brenner's (1961) series or Cooley & O'Neill's near-contact formula for normal motion (`fluidlets.analytic`). For d/a ≥ 1.54 our numbers match the paper's tables to all four printed digits, for both discretizations. Closer to the wall they agree to within 0.3%; there the gap is smaller than the point spacing h, the linear system is badly conditioned, and both computations over- or under-shoot the exact drag until the surface is refined.

The wall distances are d/a = cosh α for α = 10, 3, 2, 1, 0.5, 0.3, 0.1, the values in the paper's tables.

## Run

```bash
python run.py
```

About 20 seconds. It prints both tables and writes `viz/lag_parallel.vtp`, `viz/eul_parallel.vtr` and their `normal` counterparts: the surface forces and the flow on the plane y = 0 for d/a = 1.54. Colour the grid by `speed` in ParaView to see the fluid squeezed out of the gap.

For the paper's two finer columns (S = 1486 and 2718), set `n_phi = 19 25 34 46`. Our ring construction gives one extra point at those two sizes (1487 and 2719): the equator ring has 2N_φ + 1 points where the paper evidently used 2N_φ.

## Parameters (`input3d`)

| Key | Meaning |
|---|---|
| `radius`, `mu` | sphere radius and viscosity |
| `kernel`, `wall` | regularized Stokeslet; `wall = z0` adds the wall images |
| `n_phi` | number of latitude rings; the sphere has S points, h = √(4πa²/S) |
| `eps_coef_parallel`, `eps_coef_normal` | blob width ε = coef · h^0.9, as in the paper |
| `alpha` | wall distances d/a = cosh α |
| `gmres_tol` | relative tolerance of the iterative solve |
| `vtk_case` | which distance to write to VTK (index into `alpha`) |
| `grid_x`, `grid_y`, `grid_z` | evaluation grid for output: min, max, number of points |

## References

- Ainley, Durkin, Embid, Boindala & Cortez (2008), *The method of images for regularized Stokeslets*, J. Comput. Phys. 227, 4600.
- Brenner (1961), Chem. Eng. Sci. 16, 242. O'Neill (1964), Mathematika 11, 67. Cooley & O'Neill (1969), Mathematika 16, 37.
