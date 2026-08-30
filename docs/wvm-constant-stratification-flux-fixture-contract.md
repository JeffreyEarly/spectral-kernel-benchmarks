# WVM constant-stratification nonlinear-flux fixture contract

This document defines the authoritative evidence boundary for the complete issue #20 constant-stratification experiment. It replaces the earlier deterministic coefficient map with the exact Wave–Vortex Model (WVM) phase evolution, coefficient formulas, analytic type-I vertical transforms, pointwise advection, and modal projection.

The timed call begins with retained, vertically truncated `Ap`, `Am`, and `A0` coefficients plus elapsed time. It ends with retained `Fp`, `Fm`, and `F0` nonlinear-flux coefficients. MATLAB dispatch, model-state ownership, time integration, I/O, and diagnostics remain outside the boundary.

## Independent oracle

`tools/exportConstantStratificationFluxFixture.m` constructs the same deterministic antialiased nonhydrostatic state in two WVM backends:

1. the independent MATLAB `WVTransformConstantStratification` implementation; and
2. the compiled `WVTransformConstantStratificationKernel` implementation.

The exporter evaluates `nonlinearFlux` through both backends and refuses the fixture unless their maximum scale-normalized and relative-L2 errors are at most `1e-10`. This cross-implementation audit allows the distinct MATLAB and FFTW execution and accumulation orders observed at production sizes; the observed 256-squared and 512-squared maximum errors were approximately `5.6e-12` and `1.9e-11`, respectively. It is separate from the stricter benchmark gate: every `skbench` implementation must reproduce the compiled WVM oracle within `1e-12`. The fixture exports only physical configuration, logical horizontal mode keys, the three input coefficient arrays, and the three expected flux arrays. It deliberately does not export intermediate velocity, derivative, phase, scale, or projection values. `skbench` must reconstruct those independently from audited WVM formulas.

The v1 preparer accepts only a clean WVM checkout at commit `6ad254fb9756ac918bb72e036020d004879df1f2`, a clean benchmark generator checkout, the audited source hashes, an identity-validated compiled WVM module, exact payload hashes and dimensions, and a complete radial two-thirds mode set. Payloads remain identified by exact `(k,l)` coordinates. The C++ loader resolves equal-radius tie-order differences and reorders them into `skbench`'s provider order outside timing; physical order is not part of correctness.

## Exact mathematical graph

For every retained logical mode `(k,l,j)`, the benchmark reconstructs WVM's dimensional wavenumbers, Coriolis frequency, nonhydrostatic eigenscales, and phase

\[
A_+(t) = A_+ e^{i\omega t}, \qquad
A_-(t) = A_- e^{-i\omega t}, \qquad
A_0(t) = A_0.
\]

It then applies the exact production formulas for `U`, `V`, `W`, and buoyancy, including the inertial, geostrophic, and mean-density-anomaly exceptions. Five inverse triples reconstruct shared `U,V,W` once and the derivative triple for each of four targets. Cosine-grid targets use cosine/cosine/sine derivative families; sine-grid targets use sine/sine/cosine families. Each physical target is

\[
r_q = -(U q_x + V q_y + W q_z).
\]

Four horizontal forward transforms, analytic DCT-I/DST-I projections, WVM's modal normalization, and reference-time phase removal accumulate the retained `Fp`, `Fm`, and `F0` result.

The exporter exposed an important normalization distinction from the earlier synthetic composition. The actual WVM kernel applies no explicit pointwise `1/(Nx Ny)^2` factor. Raw inverse horizontal FFT amplitudes enter the products, and the modal projection applies `1/(Nx Ny)`. The authoritative fixture therefore records pointwise scale one. Applying the development-harness pointwise scale to this graph produces a mathematically incorrect result and is rejected by the complete oracle.

## Control and candidate

The control uses WVM's full horizontal half-spectrum, interleaved complex storage, full horizontal FFTW transforms, and batched in-place FFTW `REDFT00`/`RODFT00` transforms over every half-spectrum row.

The candidate uses the same formulas and physical-space lifetime, but stores only radial retained modes in split-complex form, executes the type-I transforms only on retained rows, and uses the fixed partial-column-pruned tile-16 horizontal transforms. Neither algorithm requires a preservation copy: both take separate caller input and output arrays, preserve the input coefficients, and reuse persistent internal storage.

All application buffers, FFTW plans, phase arrays, mode tables, outputs, and worker pools are created before timing. The component ledger reports phase/reset, coefficient assembly, inverse type-I, horizontal inverse, pointwise work, horizontal forward, forward type-I, and coefficient projection separately. The independently sampled uninstrumented total remains authoritative; components need not sum to it.

## Reference protocol

The reference campaign uses the four F4 workloads from issue #20: `256^2/Nz=129`, `512^2/Nz=257`, `1024^2/Nz=129`, and `512^2/Nz=513`. It freezes one M4 tuple across sizes: type-I internal 16, horizontal outer 12, spatial-static pointwise 8, coefficient workers 2, tile width 16, FFTW `MEASURE | UNALIGNED`, and `VECLIB_MAXIMUM_THREADS=1`.

Three complete rounds are required. Odd rounds time the control first and even rounds time the candidate first to expose measurement-order bias. Two additional rounds run only when the preregistered ratio-spread or decision-boundary triggers fire. Each result retains component samples, uninstrumented totals, correctness metrics, setup/planning, placement, allocation state, explicit persistent and scratch storage, estimated peak, and the common-process observed high-water mark.

The observed high-water mark contains both prepared graphs because the paired benchmark holds both providers for same-oracle comparison. It is valid process-capacity evidence but not an algorithm-specific memory ratio. Algorithm-specific comparisons use the explicit persistent, scratch, and estimated-peak ledgers.

Reference evidence may advance the compact specialization only when the geometric complete-call ratio is at most `0.90`, no workload ratio exceeds `1.03`, the empirical stratified interval excludes a tie, every complete output remains within `1e-12`, the allocation and placement contracts pass, and one tuple is used across all four workloads. Any conclusion is limited to constant stratification and the tested Mac; it does not imply a general-stratification or general-Mac default.
