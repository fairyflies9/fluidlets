# Fluidlets — regularized fundamental solutions for fluids and fields

Open-source Python package for the method of regularized Stokeslets, Brinkmanlets, and Oseenlets — plus their scalar electrostatic analogs (regularized Laplace and Yukawa kernels) and an inviscid vortex-sheet solver built on the same regularized-kernel foundation — in 2D and 3D, designed in the spirit of IB2d: readable code, IB2d/IBAMR-compatible input files, VTK output, and a large library of worked examples.

Display name: Fluidlets. Install: `pip install fluidlets`. Import: `import fluidlets`. Repo: `github.com/fairyflies9/fluidlets`.

## Goals

1. One codebase, many kernels. Stokeslet, Brinkmanlet, Oseenlet, Laplace, Yukawa (and later image systems and other fundamental solutions) share a single assembly and solver layer. Kernels may be vector-valued (fluid) or scalar-valued (electrostatics).
2. IB2d lineage. Geometry files (`.vertex`), fiber models (`.spring`, `.beam`, `.target`, `.mass`), and an `input2d`-style parameter file so existing IB2d/IBAMR setups run with minimal changes.
3. Shareable. Pure Python + NumPy + Numba; `pip install fluidlets`; no compiled extensions, no VTK bindings.
4. Teachable. Every example is a folder that runs with one command and opens in ParaView/VisIt.
5. Trustworthy. Analytic validation tests and ε-convergence tests in CI.

## Non-goals (for v1)

- Navier–Stokes / finite-Re flows (Oseen is the limit of what we do).
- Fast multipole / hierarchical matrices (design for them, don't build them yet).
- GPU.
- 3D vortex filaments and vortex rings. The vortex solver is 2D only.
- Nonlinear Poisson–Boltzmann. Only the linearized (Debye–Hückel / Yukawa) form is in scope, since the method relies on superposition.
- Nonlinear viscoelastic constitutive models (Oldroyd-B, FENE-P, Giesekus). These require an Eulerian solve for an advected polymer stress and would make Fluidlets a hybrid grid method. Linear viscoelasticity and generalized Newtonian fluids are in scope (see below); fully nonlinear viscoelasticity belongs in a companion package.

## Backbone: the 2011 UNC glycocalyx code

Fluidlets starts from the regularized Stokeslets code written for the glycocalyx project at UNC (2011; Miller with the Minion/Ortiz/Emmett FRG group; MATLAB + CUDA). It lives in Google Drive at `Glycocalyx/sdc_cuda3/` (folder id `0B4mLG2SPb-tbWVZnRXpPTUM3dG8`), with a possibly later snapshot in `glyco&BMB/glyco_Aug6.zip`. It is copied into the repo as `legacy/sdc_cuda3/` unchanged, with a `legacy/README.md` describing provenance, and is never edited there. Its parts and where they go:

| Legacy file | What it does | Fluidlets home |
|---|---|---|
| `StokesletEval.m`, `comp_stokeslets.cu` | 3D regularized Stokeslet sum, per-source blob width, H₁/H₂ form, CPU + CUDA | `kernels/stokeslet3d.py` + `core/assemble.py` (Numba replaces CUDA) |
| `ImageEval.m`, `ImageEval_CUDA.cu` | plane-wall image system | `kernels/images.py` |
| `StokesSelf.m`, `testGMRES*.m` | matrix-free GMRES resistance solve (velocities → forces) | `core/matvec.py`, `core/solve.py` |
| `particles*.m`, `run_particles.m`, `UpdateStrength*.m` | time-stepped towers in shear flow; SDC-style stepping; force update per step | `time/stepper.py`, `forces/` |
| `build_*_tower.m`, `generate_hex_layout.m`, `towers.txt` | geometry from a text config (type x y z height level) | `geometry/` + `io/readers.py` (towers.txt becomes an example `.vertex` + `input2d`) |
| `testCPUvsGPU.m`, `nvmex.m`, `nvmexopts.bat`, `*.mexw32` | CUDA build/validation; 32-bit Windows binaries | not ported; documented in `legacy/README.md` |
| `*.pvsm`, `velocity_t_xyz_final.raw` | ParaView state + 32³ velocity field | reference output for the VTK writer test |

### Phase 0: verify the legacy code before building on it

Done before any new kernel is written. Everything here lands in `tests/legacy/` and a short `docs/legacy_verification.md`.

1. **Inventory and provenance.** Download `sdc_cuda3/` and unpack `glyco_Aug6.zip`; diff the two; record which is newer; note authorship from file headers. Confirm the CUDA paths are unrunnable here (32-bit Windows mex) and mark them legacy-only.
2. **Kernel formula check.** `StokesletEval.m` uses H₁ = (r²+2ε²)/(2(r²+ε²)^{3/2}), H₂ = 1/(2(r²+ε²)^{3/2}) with an overall 1/(4π): this equals the Cortez–Fauci–Medovikov (2005) regularized Stokeslet (1/8πμ)[f H₁ + (f·x)x H₂] with μ = 1. Write this derivation down, then test numerically: single Stokeslet vs the CFM formula; ε→0 limit vs the singular Oseen tensor; divergence-free check on a grid.
3. **Image system check.** Verify `ImageEval.m` against Ainley et al. (2008): velocity vanishes on the wall to O(ε) at every test point; drag on a sphere approaching a wall vs Brenner's exact result; symmetry of the sphere–wall mobility.
4. **Resistance solve check.** Run `StokesSelf.m` + GMRES on a discretized sphere (resistance problem); compare drag to 6πμaU as a function of ε/spacing. Confirm the matrix the Numba port assembles equals the matvec `StokesSelf.m` applies (random vectors, ‖Ax − StokesSelf(x)‖ ≈ 1e-12).
5. **Time stepping check.** Identify the integrator in `particles*.m` (forward Euler, RK, or SDC sweeps). Verify order of accuracy on a problem with a known solution: a point in a prescribed straight-line velocity field (the commented-out exact solution in `run_particles.m`), and a sedimenting sphere relaxing to terminal velocity.
6. **Regression baseline.** Run the hex-tower shear-flow case (`run_particles.m` with `generate_hex_layout(4,20)`) in Octave; store positions and forces at a few times as `tests/legacy/data/*.npz`. The Python port must reproduce them to roundoff, then to discretization tolerance once anything is deliberately changed.
7. **Known limitations to record.** Dense O(N²) evaluation; no Brinkman/Oseen; wall only via images; forces set by prescribed shear + tower stiffness; no restart; units implicit (mm).

Exit criterion: a written list of what the legacy code gets right, what was wrong or fragile, and a reproducible baseline the port is measured against.

### Phase 1: port the backbone to Python/Numba

- `legacy/` committed as-is. `fluidlets/kernels/stokeslet3d.py` written by transcribing `StokesletEval.m` line for line, then the Numba assembly; the test is bitwise-close agreement with the Octave run of the legacy code on identical inputs.
- `kernels/images.py` from `ImageEval.m`, same test discipline.
- `core/solve.py` reproduces the GMRES resistance solve; add dense LU for small N.
- `geometry/` reproduces `build_*_tower` and `generate_hex_layout`; `towers.txt` becomes `examples/glycocalyx_towers/` with an `input2d` and `.vertex`, which is Fluidlets's first example and its continuity link to the published glycocalyx papers.
- VTK writer validated against the stored ParaView state / raw field.
- Only after Phase 1 passes does the 2D Stokeslet (new), Brinkmanlet, and the rest of the roadmap begin.

## Mathematical scope

All kernels solve a linear Stokes-type problem driven by regularized point forces f_k φ_ε(x − x_k):

- **Stokeslet**: μΔu − ∇p = −Σ f_k φ_ε
- **Brinkmanlet**: μΔu − μα²u − ∇p = −Σ f_k φ_ε (α = inverse screening length)
- **Oseenlet**: μΔu − ρ(U·∇)u − ∇p = −Σ f_k φ_ε (U = uniform background velocity)

with ∇·u = 0 in all cases. Brinkman with α→0 recovers Stokes; Oseen with U→0 recovers Stokes. Both limits are tests.

Each regularized kernel is expressed through a small set of scalar radial functions (H₁, H₂ for velocity; corresponding functions for pressure and stress; D₁, D₂ for doublets used in image systems). The Oseenlet additionally depends on the streamwise coordinate, so its interface carries an anisotropy hook: the kernel receives the full displacement vector, not only r.

### Electrostatic (scalar) kernels

The same blob applied to Laplace-type Green's functions gives regularized point charges q_k φ_ε(x − x_k):

- **Laplace** (regularized point charge): −ε₀ Δψ = Σ q_k φ_ε. Potential ψ, field E = −∇ψ. Analog of the Stokeslet (log r in 2D, 1/r in 3D).
- **Yukawa** (linearized Poisson–Boltzmann / Debye–Hückel): −ε₀ (Δψ − κ²ψ) = Σ q_k φ_ε, κ = inverse Debye length. Analog of the Brinkmanlet (K₀(κr) in 2D, e^{−κr}/r in 3D); κ→0 recovers Laplace.

Problem types mirror the fluid ones: charges → potential (mobility analog) and prescribed potential on a conductor → induced charges (resistance analog, i.e., capacitance problems).

**Electro-hydrodynamic coupling.** A Lagrangian point with charge q_k in field E experiences force q_k E(x_k), which enters the fluid mobility solve as an additional body force alongside springs, beams, and targets. This gives electrophoresis of charged particles and filaments, and (with image systems) electro-osmotic flow near charged walls. Coupling is one-way in v1 (field → force → flow); flow does not advect free charge.

### Inertial rigid bodies (primary pollen model)

A pollen grain is a rigid sphere (circle in 2D) discretized with regularized Stokeslets and no-slip enforced: surface velocity u(x_j) = v + ω × (x_j − x_c). Hydrodynamic force and torque come from the resistance solve, not from a drag law, so the grain disturbs the flow, interacts hydrodynamically with the flower and with other grains, and has a resolved wake. Inertia enters through Newton–Euler on the rigid body:

  m dv/dt = F_hydro + q E_ext(x_c) + (m − m_f) g,   I dω/dt = T_hydro + T_elec.

F_hydro and T_hydro are linear in (v, ω) via the grand resistance matrix R (6×6 in 3D, 3×3 in 2D), assembled from the same kernel matrix, plus an ambient term from the flow past the grain held fixed (background wind, flower disturbance field, other grains). The time step is implicit in the drag:

  (M/Δt + R) [v, ω]^{n+1} = M [v, ω]^n /Δt + F_ambient + F_ext,

which removes the stiffness that a point-particle drag law would introduce at small Stokes number. R is computed by solving the resistance problem for unit translations/rotations of the body in the presence of all other bodies (block solve, one factorization per step).

Electrostatics on a resolved grain: charge lives on the surface points. Two models: (i) insulating grain, fixed surface charge density, total q; (ii) conducting grain, surface potential unknown but uniform, surface charges redistribute in the external field (adds one unknown per body to the scalar solve). The flower is a grounded conductor. Near contact, the regularized kernel smooths over the lubrication gap, so a contact rule is still needed: capture when the gap between grain surface and body surface drops below a fraction of ε, with the grain then removed or frozen onto the body.

The same `RigidBody` object serves sedimenting bodies, rotating swimmers, and any resolved body that should obey Newton–Euler rather than force-free Stokes balance. The inertia-free limit (M→0) recovers the standard force-free/torque-free rigid-body mobility problem.

### Point particles (cheap alternative for sweeps)

For large parameter sweeps, a reduced model: each particle k obeys

  m_k dv_k/dt = 6πμ a_k (u(x_k) − v_k) + q_k E(x_k) + (m_k − m_f) g,   dx_k/dt = v_k,

advected by the fluid with inertia and electrostatics as body forces. No Brownian motion. One-way hydrodynamic coupling (particles feel u, never disturb it); two-way electrostatic coupling (charges are sources in the scalar solve). Capture when within a_k of a body surface. Stiffness at small St handled with an exponential integrator for the drag term. One example compares capture efficiency from the point-particle and resolved rigid-body models.

Diagnostics for both: capture efficiency vs Stokes number St = m U/(6πμ a L), charge, field strength, wind speed. Target application: pollen capture by wind- and insect-pollinated flowers, with the flower a grounded conductor in a background wind (Oseen regime, Re ~ 0.1–10) and pollen carrying charge. Optionally a charged bee as a moving external point charge.

Reference derivations:
- Cortez (2001) SIAM J. Sci. Comput. — 2D regularized Stokeslets
- Cortez, Fauci, Medovikov (2005) Phys. Fluids — 3D regularized Stokeslets
- Cortez, Cummins, Leiderman, Varela (2010) J. Comput. Phys. — regularized Brinkmanlets
- Ainley et al. (2008) J. Comput. Phys. — regularized image systems for a plane wall
- Inertial rigid bodies: grand resistance matrix from regularized Stokeslets; validate the resistance matrix of a sphere/cylinder against exact drag and the Newton–Euler stepping against the analytic velocity relaxation of a sphere released in quiescent fluid (exponential approach with time constant m/(6πμa)).
- Point particles: Maxey–Riley reduced to Stokes drag + gravity (no Basset, no added mass, no Faxén) plus q·E. Validate against the analytic stopping distance of a charged particle in quiescent fluid and against the Stokes-number dependence of capture by a cylinder in Oseen flow.
- Regularized Laplace/Yukawa kernels: derive by convolving the Green's function with the blob (closed form for the standard blobs); document in `docs/theory/electrostatics.md`.
- Vortex sheets: bound-sheet strength for a flat plate at incidence vs thin-airfoil theory; Kelvin circulation conservation; rollup of a sinusoidally perturbed free sheet vs Krasny (1986); flapping rigid plate thrust vs published values; flag flutter onset vs Alben & Shelley (2008).
- Linear viscoelastic regularized Stokeslets: complex-viscosity Stokes; validate against the oscillating sphere in a Maxwell fluid (Stokes' second problem analogs). Document in `docs/theory/viscoelastic.md`.
- Generalized Newtonian via Picard iteration on a volume-source correction; validate against planar Couette/Poiseuille flow of a power-law fluid. Document in `docs/theory/generalized_newtonian.md`.
- Regularized Oseenlets: derive in Fourier space from the blob transform; validate against Lamb/Oseen low-Re drag corrections. Literature is thin; document the derivation in `docs/theory/oseenlet.md`.

## Architecture

```
fluidlets/
  kernels/
    base.py          # Kernel ABC; declares rank (vector/scalar) and output shape
    vector.py        # VectorKernel: velocity, pressure, stress
    scalar.py        # ScalarKernel: potential, field
    blobs.py         # Blob functions φ_ε with known Fourier transforms
    stokeslet2d.py   stokeslet3d.py
    brinkmanlet2d.py brinkmanlet3d.py
    oseenlet2d.py    oseenlet3d.py       # stubs in v1, filled in v2
    laplace2d.py     laplace3d.py        # regularized point charges
    yukawa2d.py      yukawa3d.py         # screened (Debye–Hückel)
    images.py        # plane-wall image systems (v2)
  core/
    assemble.py      # Numba: dense N×N (or M×N evaluation) kernel matrix; real or complex dtype
    sources.py       # Source sets: Lagrangian points and VolumeSource grids, concatenated for the kernel sum
    matvec.py        # Numba: matrix-free u = K f for iterative solves
    solve.py         # dense LU / GMRES; resistance vs. mobility problems
    background.py    # additive background flows (uniform, shear, Poiseuille)
  geometry/
    curves.py        # closed curves, open filaments, uniform arclength spacing
    clouds.py        # 3D surfaces/point clouds
    readers.py       # .vertex / .spring / .beam / .target / .mass (IB2d & IBAMR)
  vortex/
    bound_sheet.py   # Chebyshev collocation for γ(s); leading-edge singularity; Kutta condition
    free_sheet.py    # regularized point vortices, Biot–Savart via laplace2d, insertion/merging
    shedding.py      # trailing-edge circulation release per step
    stepper.py       # implicit plate–fluid coupled step
  rheology/
    linear.py        # complex viscosity models: Newtonian, Maxwell, Kelvin–Voigt, Jeffreys, tabulated
    generalized_newtonian.py   # power-law, Carreau–Yasuda, Cross, Herschel–Bulkley
    picard.py        # fixed-point iteration for generalized Newtonian correction
  bodies/
    plate.py         # implicit inextensible Euler–Bernoulli plate on Chebyshev nodes (vortex solver)
    rigid.py         # RigidBody: point set, center, (v, ω), mass, inertia, surface charge model
    resistance.py    # grand resistance matrix from unit-velocity resistance solves
    newton_euler.py  # implicit-drag time step for inertial rigid bodies
  particles/
    tracers.py       # point particles: drag + q·E + gravity; exponential integrator for drag
    capture.py       # contact detection (gap < fraction of ε) for both bodies and particles; capture log
  forces/
    springs.py beams.py targets.py   # same force models as IB2d
    electric.py      # q·E body force from a scalar-kernel field (electro-hydro coupling)
  io/
    config.py        # input2d-style parameter file parser
    vtk.py           # hand-written VTK XML: .vtp (Lagrangian), .vtr (Eulerian field)
    checkpoint.py    # restart files
  time/
    stepper.py       # forward Euler, RK2, optional semi-implicit for stiff springs
  examples/
    <name>/input2d, <name>.vertex, ..., run.py, README.md
  tests/
  docs/
```

### Kernel interface

```python
class Kernel:
    dim: int
    def radial(self, r): ...           # (H1, H2) for isotropic kernels
    def velocity(self, x, x0, f): ...  # single-source evaluation (reference impl, non-Numba)
    def pressure(self, x, x0, f): ...
    def stress(self, x, x0, f): ...
    # Numba-compatible scalar functions exposed as module-level @njit functions
    # so core/assemble.py can take them as arguments.
```

The hot loops in `core/` are generic: they take the kernel's `@njit` scalar functions as arguments (Numba supports first-class functions) and the kernel's declared output shape (`(dim, dim)` for vector kernels, `(1,)` for scalar kernels). Adding a kernel never touches `core/`.

### Problem types

- **Mobility**: forces known → velocities. Used for fiber/spring/beam dynamics (IB2d-style).
- **Resistance**: velocities known on a body → forces. Used for rigid bodies, swimmers with prescribed kinematics, drag validation.
- **Mixed**: some points force-prescribed, some velocity-prescribed. Single linear system; block structure.

### Input file (`input2d`, IB2d-style)

```
solver      = stokeslet      # stokeslet | brinkmanlet | oseenlet | vortex
kernel      = stokeslet      # (fluid solvers) stokeslet | brinkmanlet | oseenlet
# vortex solver keys
delta       = 0.0            # free-sheet blob width (0 → auto from shed spacing)
n_cheb      = 64             # Chebyshev degree for the bound sheet / plate
plate_stiff = 1.0            # dimensionless bending stiffness
mass_ratio  = 0.1
le_motion   = heave          # none | heave | pitch | heave_pitch; amplitude/frequency keys follow
insert_tol  = 0.02           # free-sheet point insertion spacing
merge_tol   = 0.0            # 0 → no merging
rheology    = newtonian      # newtonian | maxwell | jeffreys | tabulated | powerlaw | carreau | cross | herschel_bulkley
omega       = 0.0            # forcing frequency for linear viscoelastic (frequency-domain) runs
rheo_params = ...            # model-specific: lambda (relaxation time), n, lambda_c, mu_inf, tau_y, ...
picard_tol  = 1e-6           # generalized Newtonian iteration tolerance
picard_relax = 0.5           # under-relaxation factor
efield      = none           # none | laplace | yukawa  (scalar kernel, optional)
kappa       = 0.0            # inverse Debye length for yukawa
permittivity = 1.0
dim         = 2
mu          = 1.0
eps         = 0.0            # 0 → auto: eps_ratio * mean spacing
eps_ratio   = 1.5
alpha       = 0.0            # Brinkman inverse screening length
U_inf       = 0.0 0.0        # Oseen background velocity
dt          = 1e-4
T_final     = 1.0
print_dump  = 100
grid_x      = -2 2 200       # evaluation grid for VTK output (post-processing only)
grid_y      = -2 2 200
springs     = 1
beams       = 1
target_pts  = 1
charges     = 0              # read <name>.charge (point index, q)
rigid_bodies = 0             # read <name>.rigid (body id per vertex, mass, inertia, q, charge_model)
particles   = 0              # read <name>.particles (x, y[, z], vx, vy[, vz], m, a, q)
capture_gap = 0.5            # capture when surface gap < capture_gap * eps
gravity     = 0.0 -9.81      # applied to particles only
background  = none           # none | uniform | shear | poiseuille
```

Document clearly: the evaluation grid affects output only, never the solve.

### Output

- `viz/lag.<step>.vtp` — Lagrangian points: position, velocity, force, connectivity.
- `viz/eul.<step>.vtr` — Eulerian field on the evaluation grid: u, p, vorticity, and when a scalar kernel is active, ψ and E.
- `viz/lag.<step>.vtp` also carries a body id and rigid (v, ω) per body.
- `viz/sheet.<step>.vtp` — free vortex sheet: positions, circulations; plate shape with γ in `lag.<step>.vtp`.
- `viz/part.<step>.vtp` — point-particle positions, velocities, charges, captured flag.
- `capture.csv` — time, particle id, capture location.
- `viz/*.pvd` collection files for time series in ParaView.
- Optional `.npz` dumps for Python post-processing.

## Numerics to decide (see "Open questions")

- Blob choice per kernel; expose ε and allow multiple blobs.
- ε-to-spacing ratio default and documentation of its effect.
- Time integration for stiff springs.
- Dense vs. iterative solver thresholds.

## Validation suite (tests/)

- Translating cylinder in Brinkman medium vs. exact Bessel-K solution; α→0 limit vs. 2D Stokes with far-field handling.
- Translating/rotating sphere in 3D Stokes vs. 6πμaU, 8πμa³Ω.
- Sphere in Brinkman vs. exact solution.
- Oseen: sphere/cylinder drag vs. Lamb/Oseen corrections at small Re.
- Vortex: flat-plate lift vs thin-airfoil theory; Kelvin conservation; Krasny rollup; flapping-plate thrust; flag flutter onset.
- Linear viscoelastic: oscillating sphere/cylinder in a Maxwell fluid vs exact complex-viscosity drag; Newtonian limit λ→0.
- Generalized Newtonian: planar Poiseuille of a power-law fluid between resistance-problem walls vs exact profile; Picard convergence vs n.
- Rigid bodies: resistance matrix of a sphere/cylinder vs exact; velocity relaxation of a released sphere vs exponential; inertia-free limit vs force-free mobility solve.
- Particles: charged particle stopping distance in quiescent fluid vs exact; passive tracer vs streamlines; capture efficiency of a cylinder vs St in Oseen flow (compare to published inertial-impaction curves).
- Laplace: charged conducting disk/sphere → capacitance and surface charge vs exact; dipole field.
- Yukawa: screened charged sphere vs Debye–Hückel exact potential; κ→0 limit vs Laplace.
- Coupled: electrophoretic velocity of a uniformly charged sphere in a uniform field vs the Hückel limit (κa ≪ 1).
- ε-convergence: error vs ε at fixed spacing and vs spacing at fixed ratio.
- IB2d parity: run an IB2d example geometry and compare qualitative dynamics.

## Example library (initial)

1. `cylinder_drag` — validation, all three kernels.
2. `sedimenting_body` — rigid body mobility.
3. `undulating_filament` — sperm-like swimmer with beams.
4. `cilia_array` — prescribed kinematics, resistance problem.
5. `jellyfish_bell` — contracting closed curve with springs/targets, Stokes vs Brinkman.
6. `flow_past_porous_cylinder` — Brinkman screening demonstration.
7. `oseen_wake` — asymmetric wake behind a translating body (once Oseenlet is in).
8. `ib2d_import` — load an unmodified IB2d example and run it.
9. `charged_disk_potential` — pure electrostatics, no flow; ψ and E on the grid.
10. `electrophoresis` — charged flexible filament driven by a uniform field, Stokes vs Brinkman.
11. `electroosmosis_wall` — charged plane wall with image system (v0.4).
12. `cilium_viscoelastic` — single beating cilium, Newtonian vs Maxwell fluid at fixed ω; phase lag in the flow field.
13. `swimmer_shear_thinning` — undulating filament in a Carreau fluid vs Newtonian; shows viscosity field on the grid.
14. `pollen_capture` — grounded stigma/anther geometry in background wind (Oseen), charged inertial pollen as resolved rigid spheres with no-slip, released upstream; capture-efficiency curves vs St, q, wind speed. Optional charged bee.
15. `pollen_capture_pointparticle` — same setup with the reduced point-particle model; compares efficiency curves and shows where the reduced model fails (near-contact interaction, multiple grains).
16. `sedimenting_sphere_inertial` — released sphere relaxing to terminal velocity; validates Newton–Euler stepping.
17. `flapping_rigid_plate` — heaving/pitching rigid plate shedding a vortex sheet; thrust and wake.
18. `flexible_flapping_plate` — same with implicit Euler–Bernoulli plate; stiffness sweep.
19. `flag_flutter` — clamped flexible plate in a free stream; flutter onset vs mass ratio and stiffness.
20. `plate_across_re` — one plate kinematics file run with Stokeslets, Oseenlets, and vortex sheets.

## Tutorials, examples, and outreach

Adoption depends on these as much as on the solver. Each feature ships with its example and tutorial in the same release, not later.

### Three layers

1. **Examples** (`examples/<name>/`): runnable folders as described above. Each has `input2d` (or `input3d`), geometry files, optional `update_*.py` hooks, `run.py`, a `README.md` explaining the physics in a paragraph or two, and a reference image or short animation rendered from the VTK output. `python run.py` works with no arguments and finishes in under a few minutes on a laptop.

2. **Tutorials** (`tutorials/NN_name.ipynb`): Jupyter notebooks that teach the method, not just the package. They build understanding step by step, show the math next to the code, and end with a few exercises. Planned sequence:
   - 01 From blob to regularized Stokeslet: derive H₁, H₂ for one blob and plot the kernel against the singular Stokeslet.
   - 02 Choosing ε: drag on a cylinder/sphere vs ε and point spacing; convergence plots.
   - 03 Mobility, resistance, and mixed problems.
   - 04 The 2D Stokes paradox and how Brinkman screening resolves it.
   - 05 Porous media with Brinkmanlets: screening length and flow past a porous cylinder.
   - 06 Fibers with springs, beams, and targets: porting an IB2d example.
   - 07 Walls and image systems (from the legacy glycocalyx code).
   - 08 Regularized point charges: Laplace and Yukawa kernels, capacitance of a conductor.
   - 09 Inertial rigid bodies and pollen capture.
   - 10 Oseenlets: when inertia breaks the fore–aft symmetry.
   - 11 Vortex sheets: Kutta condition, shedding, and a flapping plate.
   - 12 One plate across Reynolds numbers: Stokes, Oseen, vortex sheets.
   - 13 Non-Newtonian fluids: complex viscosity and shear thinning.

3. **Course modules** (`courses/`): sequences of tutorials and examples packaged for teaching, with learning goals, exercises, solutions kept in a separate instructor folder, and suggested projects.
   - Graduate biofluids / mathematical biology: the full tutorial sequence plus project prompts (swimmers, cilia, pollination, flapping).
   - Undergraduate modules: three or four short notebooks (what is a Stokeslet, sedimentation, a simple swimmer, pollen capture) requiring only calculus.
   - Workshop format: a one-day hands-on session for conferences and summer schools.

### Delivery

- Notebooks run in the browser via Binder and Google Colab badges on each one; no local install required for students.
- Documentation site (mkdocs-material) with an example gallery of thumbnails and animations, the tutorials rendered as pages, and the theory notes in `docs/theory/`.
- Gallery images and animations are generated by a script (`tools/render_gallery.py`) from example VTK output so they never go stale.
- Tutorials are executed in CI (nbmake or similar) so they cannot silently break.
- Short screencasts for the first few tutorials, linked from the docs.

### Rule for every new feature

A feature is not done until it has: a validation test, a runnable example, and either a new tutorial or a section added to an existing one.

## Packaging and community

- `pyproject.toml`, `pip install fluidlets`, Python ≥ 3.10, deps: numpy, numba, scipy (GMRES).
- License: BSD-3 or MIT (decide).
- Docs: mkdocs-material with an example gallery (thumbnails rendered from VTK via a script).
- CI: GitHub Actions running tests on Linux/macOS/Windows.
- Target a JOSS paper after v1.

## Open questions (discuss before/while scaffolding)

1. Blob families: ship one blob per kernel or several with a registry?
2. Time stepping: forward Euler only in v1, or include a semi-implicit option for stiff springs?
3. Solver: dense LU up to what N? GMRES with matrix-free matvec beyond that?
4. Image systems: v1 or v2?
5. 3D kernels: implement Stokeslet3D in v1 so the dimension abstraction is tested, or defer?
6. Nondimensionalization: dimensional inputs (like IB2d) or nondimensional with documented scalings?
7. License and governance.
8. Scalar kernels in v0.1 (cheap, tests the rank abstraction) or v0.2?
9. Frequency-domain only for linear viscoelasticity in v1, or also time-domain hereditary integrals?
10. Reference viscosity μ₀ for the Picard split: zero-shear, infinite-shear, or user-chosen? Affects convergence.
11. Rigid-body resistance matrix: recompute R every step (exact, one factorization per step) or freeze it over several steps when bodies move slowly relative to each other?
11b. Contact rule for resolved grains: freeze onto the body, remove, or add a short-range repulsion to prevent interpenetration before capture is declared?
12. Vortex solver: trailing-edge shedding only (Alben) or also leading-edge separation for high-angle cases?
12b. Free-sheet growth control: merging, far-wake truncation, or neither in v1?
13. Should magnetostatics / other Laplace-type problems reuse the scalar kernels under a different name, or is that out of scope?

## Roadmap

- v0.0 (Phase 0): legacy code verified, baseline data stored, `legacy/` committed.
- v0.1 (Phase 1): 3D Stokeslet + plane-wall images + GMRES resistance ported from legacy with regression tests; `glycocalyx_towers` example; VTK output; IB2d readers.
- v0.1.5: 2D Stokeslet + Brinkmanlet (2D/3D), mobility/mixed solves, force models, 3 more examples, analytic validation tests.
- v0.2: 3D Stokeslet + Brinkmanlet, Laplace/Yukawa scalar kernels, electro-hydro coupling, full force models, time stepping options, example gallery.
- v0.3: Oseenlets 2D/3D with theory doc; inertial rigid bodies (Newton–Euler, implicit drag), point particles, and both `pollen_capture` examples.
- v0.4: image systems, GMRES path, checkpoint/restart.
- v0.5: vortex-sheet solver (`fluidlets.vortex`): bound sheet, Kutta/shedding, free sheet, rigid plate.
- v0.6: implicit flexible plate (Euler–Bernoulli on Chebyshev nodes), flag flutter, cross-Re plate example.
- v0.7: linear viscoelastic (complex μ, frequency domain) and generalized Newtonian (VolumeSource + Picard).
- v0.8: fast summation for volume sources and free vortex sheets (treecode or FMM).
- v1.0: docs site, full tutorial sequence, course modules, Binder/Colab links, JOSS submission.
