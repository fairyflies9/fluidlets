# CLAUDE.md — Fluidlets

Read `DESIGN.md` first. It is the source of truth for scope and architecture.

## What this is

Fluidlets (`import fluidlets`) is a Python package implementing the method of regularized Stokeslets, Brinkmanlets, and Oseenlets in 2D and 3D, with IB2d/IBAMR-compatible input files and VTK output. It is meant to be shared widely and read by students, so clarity beats cleverness.

## Conventions

- Python ≥ 3.10. Dependencies: numpy, numba, scipy. Do not add others without discussion. No VTK bindings, no pyevtk; VTK XML is written by hand in `fluidlets/io/vtk.py`.
- All hot loops live in `fluidlets/core/` and are Numba `@njit` functions that take kernel scalar functions as arguments. Kernel modules never contain assembly loops.
- Kernels are either `VectorKernel` (fluid: velocity/pressure/stress) or `ScalarKernel` (electrostatics: potential/field). Both declare their output shape so `core/` stays rank-agnostic.
- Every kernel module exposes: a `Kernel` subclass, module-level `@njit` radial/scalar functions, and a docstring citing the derivation (paper or `docs/theory/`).
- Reference (non-Numba) implementations are kept for every kernel and used in tests to check the Numba path.
- Kernels and assembly must work for real and complex `mu`; never hard-code `float64` in `core/`. Use the dtype of `mu` to pick the array dtype.
- Non-Newtonian support is limited to linear viscoelastic (complex viscosity, frequency domain) and generalized Newtonian (Picard iteration on a volume-source correction). Do not implement Oldroyd-B or any advected polymer stress; DESIGN.md lists it as a non-goal.
- Arrays are `float64` or `complex128`, shape `(N, dim)`. Index order is `[point, component]` everywhere.
- Units follow IB2d: dimensional inputs, SI-agnostic (the user picks consistent units).
- Electro-hydrodynamic coupling is one-way (field → q·E force → flow) unless DESIGN.md says otherwise. Do not add charge advection.
- Resolved rigid bodies (`bodies/`) enforce no-slip and get force/torque from the resistance solve; inertia via Newton–Euler with the drag term implicit. Never add a drag-law force to a resolved body.
- Point particles (`particles/`) are one-way coupled to the fluid (they feel u, never disturb it) and two-way coupled electrostatically (their charges are sources). No Brownian motion anywhere.
- The vortex solver (`fluidlets/vortex/`) is 2D only and reuses `kernels/laplace2d.py` for Biot–Savart; never duplicate the kernel. The plate uses its own Chebyshev node set; `.vertex` files for plates are output-only. The flexible plate is a separate implicit Euler–Bernoulli solver in `bodies/plate.py`, not the IB2d beam model.
- Input files follow IB2d formats exactly where IB2d has one (`.vertex`, `.spring`, `.beam`, `.target`, `.mass`, `input2d`). Extensions go in new keys, never in changed meanings of existing ones.
- Examples are self-contained folders: `input2d`, geometry files, `run.py`, `README.md`. `python run.py` must work from inside the folder with no arguments.
- Tests use pytest. Every kernel gets an analytic validation test and an ε-convergence test. Tests must run in under ~2 minutes total on a laptop.
- Type hints and NumPy-style docstrings on all public functions.
- Keep functions short. Prefer explicit loops in Numba over clever vectorization when the loop is clearer.

## Legacy code

`legacy/sdc_cuda3/` is the 2011 UNC MATLAB/CUDA regularized Stokeslets code. Never edit it. Phase 0 (verify it) and Phase 1 (port it with regression tests against Octave runs of the legacy code) in DESIGN.md come before any new kernel. Octave is the reference runner for legacy MATLAB; the CUDA/mex files are documentation only.

## Tutorials and examples

- A feature is done only when it has a validation test, a runnable example folder, and a tutorial notebook (new or a new section in an existing one). See "Tutorials, examples, and outreach" in DESIGN.md.
- Notebooks live in `tutorials/NN_name.ipynb`, explain the math alongside the code, end with exercises, and must run top to bottom in CI and on Colab/Binder (no local paths, small problem sizes).
- Example READMEs explain the physics in plain language first, then the parameters.
- Instructor solutions go in `courses/*/instructor/`, never in the student notebooks.

## Workflow

- Make small commits with descriptive messages.
- When adding a kernel, add in this order: reference implementation → Numba scalar functions → validation test → example.
- Before marking a task done, run `pytest` and one example end to end, and open the VTK output to confirm it is well-formed (`python -c "import xml.dom.minidom..."` or a tiny validator in `tools/`).
- Do not implement Oseenlets until the theory note in `docs/theory/oseenlet.md` is written and reviewed.
- Ask before changing the input-file schema or the public API.

## Style

- Comments explain the math, not the Python.
- Name variables after the quantities in the papers (`H1`, `H2`, `eps`, `alpha`, `mu`), not after their types.
