# A swimming biflagellate alga

Single-celled algae such as *Chlamydomonas* swim with two flagella at the front of the cell, pulling themselves through the water with something like a breaststroke. At their size, water is so viscous compared with their speed that there is no coasting: the cell moves only while its flagella move, and it carries no net force or torque. Whatever the flagella push back on the fluid, the cell body must balance.

This example simulates such a cell. The body is a rigid ellipsoid; the two flagella are mirror images beating in synchrony. Their shape is set by a tangent angle that changes along the flagellum and in time (Gallagher & Smith 2018, Eqs. 29–30):

ψ(s, t) = −2.5 s − (0.7 + 0.15 sin 2πs) cos(t − 2πs),  0 ≤ s ≤ 1.

We don't tell the cell how to move. At each instant we solve for three things at once: the forces the cell exerts on the fluid, the cell's velocity, and its rate of rotation. They have to satisfy no slip on the whole cell plus zero total force and torque (the **free-swimming problem**). Then we step the cell's position and orientation forward in time.

The method is the nearest-neighbour regularized Stokeslet method of Gallagher & Smith: forces live on a coarse set of points (96 on the body, 40 per flagellum), while the integrals use a finer set (600 and 400), each fine point carrying the force of its nearest coarse point. This keeps the linear system small but accurate.

## Result and comparison with the paper

The cell swims straight forward along +x₂, flagella first, and moves **5.75 × 10⁻² flagellar lengths per beat**.

The paper's Table II reports about 5.52 × 10⁻² for the same setup. The difference is in the flagellum, not the solver. The authors' published shape routine (`ChlamyFromModel.m`) builds the flagellum from the curve between arclength 1/30 and 1, so its flagellum is 29/30 of the intended length. Setting `beat_variant = authors_code` reproduces that shape, and the cell then moves 5.529 × 10⁻² per beat, within the range of their Table II row for 40 force points per flagellum (5.521–5.527 × 10⁻²). The default, `beat_variant = paper`, follows Eqs. 29–30 as printed.

Two smaller differences from the published code: we integrate in time with SciPy's RK45 at relative tolerance 10⁻⁶ (they used MATLAB's `ode113` with default tolerances), and we compute the beat directly rather than from a spline fitted to it.

## Run

```bash
python run.py
```

About 35 seconds for one beat. It writes to `viz/`:

- `lag.pvd` (with `lag.NNNN.vtp`): the swimmer's force points and forces at 21 times over the beat. Open the `.pvd` in ParaView and press play.
- `eul.pvd` (with `eul.NNNN.vtr`): the flow in the plane x₃ = 0 at the start, one-third and two-thirds of the beat. Colour by `speed`, or add a Stream Tracer.
- `trajectory.csv`: the position of the cell centre over time.

## Parameters (`input3d`)

| Key | Meaning |
|---|---|
| `head_axes` | semi-axes of the ellipsoidal cell body (flagellar lengths) |
| `insertion_angle` | angle of each flagellum's base from the forward axis, degrees (30 in the paper's Table II, 36 in its Fig. 1) |
| `n_head`, `n_head_quad` | body force points 6n², body quadrature points 6n² |
| `n_flag`, `n_flag_quad` | force and quadrature points per flagellum |
| `beat_variant` | `paper` (Eqs. 29–30) or `authors_code` (the published code's shorter flagellum) |
| `eps` | blob width: flagellar radius over length, 0.25/20 |
| `beats`, `ode_rtol` | how many beats to simulate; ODE tolerance |
| `frames_per_beat`, `flow_frames` | output times; which frames get a flow field |
| `grid_x`, `grid_y`, `grid_z` | flow-field grid: min, max, number of points |

## Reference

Gallagher & Smith (2018), *Meshfree and efficient modeling of swimming cells*, Phys. Rev. Fluids 3, 053101. Their MATLAB/Octave code is at github.com/djsmithbham/nearestStokesletsSwimmers (GPL-3). The Fluidlets implementation was written from the paper's equations; their code was read only to identify the model parameters and the flagellum-shape detail described above.
