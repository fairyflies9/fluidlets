# Glycocalyx towers in shear flow

The glycocalyx is a brush of proteins and sugars lining blood vessel walls. Here each core protein is a stiff "tower" of regularized Stokeslets: a hexagonal column 10 mm tall and 0.5 mm in radius, made of 21 levels of 7 points. Three towers, 4 mm apart, stand on a no-slip wall at z = 0 in the shear flow u = 0.1 z.

The towers do not move, so this is a **resistance problem**: we know the velocity at every tower point (the flow there must be zero) and solve for the forces the towers exert on the fluid. The wall is handled exactly by the image system of Ainley et al. (2008), so the flow vanishes on z = 0. The middle tower feels less drag than the outer two because its neighbours shield it.

This is the configuration run by the 2011 UNC code in `legacy/sdc_cuda3/run_particles.m`, and the 32³ velocity grid this example writes matches the one that code stored in 2010 to about 10⁻⁷ (single-precision roundoff). See `docs/legacy_verification.md`.

## Run

```bash
python run.py
```

It takes a few seconds and writes `viz/lag.0000.vtp` (tower points with their forces) and `viz/eul.0000.vtr` (velocity on the grid). In ParaView, open both, colour the grid by `speed`, and add a Stream Tracer to see the flow bend around the towers.

## Parameters (`input3d`)

| Key | Meaning |
|---|---|
| `mu` | viscosity |
| `background`, `shear_rate` | imposed shear flow u = shear_rate · z in x |
| `kernel`, `wall` | regularized Stokeslet; `wall = z0` adds the image system for a wall at z = 0 |
| `problem` | `resistance`: velocities known, forces unknown |
| `eps` | blob width (the legacy code's `deltas`) |
| `gmres_tol` | relative tolerance of the iterative solve |
| `grid_x`, `grid_y`, `grid_z` | evaluation grid for output: min, max, number of points |

Geometry: `towers.txt` lists one tower per row as `type x y z height levels` (type 0 is hexagonal). `run.py` builds `glycocalyx_towers.vertex` from it the first time it runs. To try a full hexagonal array, replace the table with `fluidlets.geometry.hex_array_layout(4.0, 10.0)`.

## References

- Miller et al., *Three-dimensional viscous flows through arrays of cylinders with applications to the endothelial surface layer* (glycocalyx paper; source in `legacy/glyco_Aug6/`).
- Cortez, Fauci & Medovikov (2005), Phys. Fluids 17, 031504.
- Ainley, Durkin, Embid, Boindala & Cortez (2008), J. Comput. Phys. 227, 4600.
