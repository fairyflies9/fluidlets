# Legacy code

Everything in this folder is kept exactly as it was found. Do not edit it. The Python port is checked against it (see `tests/legacy/` and `docs/legacy_verification.md`).

## `sdc_cuda3/`

The regularized Stokeslet code from the UNC glycocalyx project (2009–2011, Miller with the Minion FRG group). MATLAB with CUDA mex kernels.

Provenance:
- Copied from Google Drive `Glycocalyx/sdc_cuda3/` (folder id `0B4mLG2SPb-tbWVZnRXpPTUM3dG8`). The download wrapped it in an extra `sdc_cuda3/` folder, which was removed; no file was changed.
- Comments in the `.cu` files are signed `ksl` and dated 2009–2010 (e.g. `-ksl091129`, `-ksl100312`). The SDC driver (`particles*.m`) and `makesig.m` are Michael Minion's pedagogical SDC code, "Copyright Oct. 2005".
- The CUDA kernels (`*_CUDA.cu`, `comp_stokeslets.cu`) and the 32-bit Windows mex binaries (`*.mexw32`) cannot run on current 64-bit MATLAB. They are documentation only. The MATLAB files `StokesletEval.m` and `ImageEval.m` are the CPU versions of the same kernels and are the reference.
- `velocity_t_xyz_final.raw` (32³ grid, float32, u/v/w interleaved) and the `*.pvsm` ParaView states are stored output from a past run.

What the code does, as it stands:
- `StokesletEval.m`: 3D regularized Stokeslet, Cortez–Fauci–Medovikov (2005) blob, μ = 1.
- `ImageEval.m`: image system for a no-slip wall at z = 0 (Stokeslet, dipole, doublet, rotlet), after Ainley et al. (2008).
- `StokesSelf.m`: matrix-free matvec for MATLAB's `gmres`. It calls the CUDA mex functions; the CPU path below its `return` is unreachable.
- `run_particles.m`: builds towers, solves for the forces that hold them fixed in the shear flow u = 0.1 z, then calls `particlesHex.m`.
- `particlesHex.m`: an SDC time loop in which the tower velocity is set to zero (the GMRES forces already hold them fixed), so tower positions never change. Only a line of tracers moves, by forward Euler. It also writes the 32³ velocity grid.
- `generate_hex_layout.m`: returns early with three hex towers at y = 0, 4, 8; the full layout below the `return` is never reached.

Running it safely: `run_particles.m` and `particlesHex.m` write `towers.txt` and `velocity_t_xyz_final.raw` into the current folder. Always run from a separate work folder with this folder added to the MATLAB path (`tests/legacy/matlab/` does this).

## `glyco_Aug6/`

LaTeX source, figures and PDF of the glycocalyx BMB submission (Miller et al., "Three-dimensional viscous flows through arrays of cylinders with applications to the endothelial surface layer"). It contains no code. It is the reference for the published results the `glycocalyx_towers` example should reproduce.
