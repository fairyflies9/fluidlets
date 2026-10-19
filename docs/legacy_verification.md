# Legacy code verification (Phase 0)

This note records what the 2011 glycocalyx code (`legacy/sdc_cuda3/`) gets right, what is wrong or fragile, and the baseline the Python port is measured against. The checks are run by the MATLAB drivers in `tests/legacy/matlab/` (MATLAB R2024a) and asserted in `tests/legacy/test_phase0.py`.

## How the legacy code is run

The legacy files are never edited. `legacy_setup.m` puts them on the MATLAB path and works in `tests/legacy/matlab/work/`, because `run_particles.m` and `particlesHex.m` write `towers.txt` and `velocity_t_xyz_final.raw` into the current folder and would otherwise overwrite the originals. Three shims in `tests/legacy/matlab/shims/` stand in for things that no longer run:

- `StokesletEval_CUDA.m`, `ImageEval_CUDA.m`: the `.mexw32` binaries are 32-bit and cannot load on 64-bit MATLAB. The shims call `StokesletEval.m` and `ImageEval.m`, the CPU versions of the same sums.
- `quiver3.m`: the legacy `plotme` passes a row vector of positions with a column vector of velocities, which current MATLAB rejects. Plotting is not needed, so the shim draws nothing.

To regenerate the data, from `tests/legacy/matlab/` run `matlab -batch baseline_kernels` (and likewise `check_sphere_drag`, `check_sdc_order`, `baseline_towers`).

## 1. Inventory and provenance

- Only one code snapshot exists. `glyco_Aug6` (the "possibly later snapshot" in DESIGN.md) holds the LaTeX source and figures of the BMB submission and no code; it is kept privately and is not in this repository.
- Authorship: CUDA comments are signed `ksl` and dated 2009–2010; the SDC driver and `makesig.m` are Michael Minion's pedagogical SDC code (2005).
- The CUDA paths are unrunnable here and are treated as documentation.

## 2. Stokeslet kernel — correct

`StokesletEval.m` uses H₁ = 1/(2√(r²+δ²)) + δ²/(2(r²+δ²)^{3/2}) = (r²+2δ²)/(2(r²+δ²)^{3/2}) and H₂ = 1/(2(r²+δ²)^{3/2}), with the matrix H₁I + H₂ xxᵀ divided by 4π. That is (1/8π)[(r²+2δ²)f + (f·x)x]/(r²+δ²)^{3/2}, the Cortez–Fauci–Medovikov (2005) regularized Stokeslet with μ = 1.

Numerically, on 40 random sources with per-source blob widths and 60 targets, the legacy output equals `fluidlets.stokeslet3d_velocity` to 3×10⁻¹⁶ relative error. The starter tests already cover the ε → 0 limit and zero divergence for that formula.

## 3. Image system — correct

`ImageEval.m` adds, at the mirror point of each source, a negative Stokeslet, a dipole, a doublet and a rotlet (Ainley et al. 2008). Checks:

| Check | Result |
|---|---|
| Python transcription vs MATLAB | 7×10⁻¹⁶ relative |
| Velocity on the wall z = 0 | ≤ 3×10⁻¹⁶ (vs 0.43 without images), for ε from 0.4 to 0.025 and all force directions |
| Divergence of Stokeslet + images | ~10⁻⁸ (finite-difference level) |
| Convergence to Blake's (1971) exact wall solution | errors 1.3×10⁻², 3.2×10⁻³, 8.0×10⁻⁴, 2.0×10⁻⁴ for ε = 0.1 to 0.0125: second order |

The wall velocity is zero exactly, not only to O(ε), for this blob.

## 4. Resistance solve — correct

Spheres of radius a = 1 discretized with Fibonacci points, ε = ½ spacing, solved with MATLAB `gmres` using the legacy kernels as the matvec (what `StokesSelf.m` does):

| N | spacing | F / 6πμaU |
|---|---|---|
| 100 | 0.354 | 1.0362 |
| 200 | 0.251 | 1.0243 |
| 400 | 0.177 | 1.0164 |
| 800 | 0.125 | 1.0112 |

The error falls at first order in the spacing, as expected with ε tied to the spacing.

Near the wall (N = 400), drag relative to the same sphere in free space:

| H/a | parallel | Faxén series | perpendicular | Brenner (exact) |
|---|---|---|---|---|
| 1.5 | 1.610 | 1.615 | 3.272 | 3.205 |
| 2 | 1.391 | 1.386 | 2.157 | 2.126 |
| 3 | 1.232 | 1.227 | 1.583 | 1.569 |
| 5 | 1.128 | 1.126 | 1.291 | 1.285 |
| 10 | 1.061 | 1.060 | 1.129 | 1.126 |

Parallel motion agrees with Faxén to within 0.4% at every height. Perpendicular motion is within 2.1% of Brenner's (1961) exact series at H/a = 1.5 and 0.2% at H/a = 10, always slightly high, which matches what Ainley et al. (2008, Table 2) report for coarse discretizations. Brenner's series is taken from Ainley et al., Eq. 22, and implemented in `fluidlets.analytic.brenner_wall_drag`; with d/a = cosh α it reproduces every reference value in their Table 2.

The matrix-equals-matvec check in DESIGN.md (step 4) belongs to Phase 1, when there is an assembled Python matrix to compare.

## 5. Time stepping — first order, not ninth

`particles.m` (deformable triangle towers with springs in shear flow) runs explicit SDC with 9 nodes and 9 sweeps, nominally 9th order. Self-convergence on one tower (T = 0.4, 2 to 32 steps) gives observed orders **0.99, 0.96, 0.97**. Two bugs explain the loss of order:

1. **Lagged forces.** In both the predictor and corrector, `strengths = UpdateStrength(sources, ...)` is called before `sources = reshape(xnew, ...)`, so the velocity at each node uses spring forces from the previous node's positions.
2. **Stale correction term.** `fdiff` is not reset at the start of each sweep, so the first substep of every sweep uses a leftover difference instead of zero.

Separately, the background flow is switched off by step index, with `k0 <= nbigsteps/2` in the predictor but `k0 < nbigsteps/2` in the corrector, so the two disagree on the switching step.

The individual contribution of bugs 1 and 2 has not been isolated. The port should implement SDC correctly and check its order on a problem with a known solution, rather than reproduce these runs.

Other driver issues:
- `particles_image.m` calls `UpdateStrength(sources)` with one argument; `UpdateStrength` needs three, so this driver cannot run.
- `particlesHex.m` (the glycocalyx driver) sets the tower velocity to zero at every node: the GMRES forces already hold the towers fixed, so the SDC loop never moves them. Only the tracers move, by forward Euler.

## 6. Regression baseline — the glycocalyx towers

`baseline_towers.m` runs `run_particles.m` unchanged: three hex towers (radius 0.5, height 10, 21 levels of 7 points, 441 points total) at y = 0, 4, 8, standing on the wall z = 0, held fixed in the shear flow u = 0.1 z with δ = 0.5. It takes 88 s on the CPU kernels.

| Quantity | Result |
|---|---|
| GMRES relative residual | 9.2×10⁻⁷ (tolerance 10⁻⁶) |
| Tower positions after 100 steps | identical to the initial positions |
| Forces at the 21 wall points | ≤ 10⁻¹² (undetermined; GMRES leaves them at its zero guess) |
| x-drag per tower | −25.761, −21.901, −25.761: outer towers mirror each other, the middle one is shielded |
| 32³ velocity grid vs the stored `velocity_t_xyz_final.raw` | 1.4×10⁻⁷ relative difference |

The last row matters most: rerunning the code today on the CPU kernels reproduces the grid that the CUDA code wrote in 2010 to single-precision roundoff (the CUDA kernels work in float32). So the stored file came from exactly this configuration, and the port has two independent references: `tests/legacy/data/towers.mat` (positions, boundary velocities, forces, rerun grid) and the original `.raw` file.

The port must reproduce the velocity field and the forces at the 420 points above the wall. At the wall points it should compare velocities only.

## 7. Known limitations

- Dense O(N²) evaluation; MATLAB loops make even a few hundred points slow.
- No Brinkman or Oseen kernels; the wall only via images; μ fixed at 1.
- `generate_hex_layout.m` returns early with three hard-coded towers (y = 0, 4, 8); the full layout below the `return` is unreachable, and the height/level columns of `towers.txt` are ignored by the tower builders.
- Tower forces in `particles.m` come from springs with a hard-coded constant (300) chosen by trial for stability.
- Bottom tower points sit on the wall (z = 0), where Stokeslet plus image is identically zero, so their forces are not determined by the resistance solve. Regression tests must compare velocities, not forces, at those points.
- No restart; units implicit (mm).
