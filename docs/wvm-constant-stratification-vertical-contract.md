# WVM constant-stratification vertical-transform contract

This document defines the first attributable experiment for issue #20. It reproduces the vertical FFTW call schedule used by the current Wave–Vortex Model (WVM) Float64 nonhydrostatic nonlinear-flux kernel and changes only the number and physical order of horizontal rows presented to those transforms.

The experiment is an isolated vertical component benchmark. It does not reimplement the nonlinear flux, and it cannot by itself establish an end-to-end WVM speedup. Horizontal transforms, coefficient assembly, phase evaluation, spatial products, flux projection arithmetic, and coefficient accumulation remain outside its timed boundary.

## Audited production source

The contract was read from `WVTransformConstantStratificationKernel.cpp` at WVM commit `6ad254fb9756ac918bb72e036020d004879df1f2`. The audited source file has SHA-256 `3e8f618fa813ca274b0c33ed3a34c023fc891ef79b0a062abdc99a967de4a3a9`.

The matching FFT provider creates in-place FFTW 3.3.11 `fftw_plan_guru64_r2r` plans with `FFTW_MEASURE | FFTW_UNALIGNED`. Complex-interleaved storage is treated as two real batches. The transform axis has scalar stride two, while the real and imaginary batches have scalar stride one.

## Exact type-I definitions

For a vertical grid containing `Nz` points, the cosine family is FFTW `REDFT00` of length `Nz`. WVM's forward normalization multiplies every result by `1/(Nz-1)` and multiplies the final vertical mode by an additional factor of one half.

The sine family is FFTW `RODFT00` of length `Nz-2`, applied to the interior values `z=1,...,Nz-2`. WVM sets both endpoints to zero and multiplies every interior result by `1/(Nz-1)`.

On inverse paths, WVM places the required one-half scaling in coefficient production before executing the same unnormalized type-I transforms. Consequently, the raw inverse primitive timing contains the FFTW call but not a separate normalization pass.

Correctness is checked independently against direct cosine and sine sums, including the endpoint rules, and by comparing every retained logical `(k,l)` row with the corresponding row from the full-half-spectrum control. Conjugated stored modes are compared after logical reorientation. Maximum scale-normalized and relative-L2 error must not exceed `1e-12`.

## Production nonlinear-flux transform schedule

One nonhydrostatic four-field nonlinear-flux call executes the following vertical transforms:

1. Reconstruct shared `U,V,W`: two cosine-family and one sine-family complex channels.
2. Reconstruct derivatives for the two cosine-grid targets: two cosine-family and one sine-family complex channels per target.
3. Reconstruct derivatives for the two sine-grid targets: two sine-family and one cosine-family complex channels per target.
4. Project four scalar flux targets: two cosine-family and two sine-family complex channels, including forward normalization.

The complete schedule therefore contains 15 inverse and 4 forward complex vertical channels. The benchmark measures the individual one-channel DCT-I and DST-I calls, forward normalization, and an independently timed execution of this 19-channel schedule.

The schedule timing repeatedly executes the value-independent FFTW plans on reusable arenas. It reproduces the production plan shapes, call counts, strides, placement, and normalization work; it does not claim that the repeatedly transformed arena values constitute a nonlinear-flux result.

## Control and candidate

The control uses the production WVM half-spectrum row count

`H = (Nx/2 + 1) Ny`

with `z` adjacent inside each channel and horizontal row.

The candidate uses only the radial two-thirds retained horizontal modes, stored contiguously in logical retained-mode order. It uses the same interleaved complex representation, FFTW type-I algorithms, vertical lengths, channel batches, normalization, internal worker count, and planning mode. Thus this increment answers only whether omitting discarded horizontal rows from the vertical transforms is worthwhile.

The candidate is expected to reduce the reusable four-channel complex arena from `4 H Nz` values to `4 Nkl Nz` values. Provider-owned plan and thread memory remains opaque and is reported separately from the explicit arena.

## Interpretation and next boundary

A favorable result supports integrating retained-row vertical transforms with the already established compact horizontal algorithm. It does not prove the complete flux is faster because compact horizontal production, coefficient assembly, pointwise work, and any representation boundary may change the total.

The timed benchmark action performs no application allocation and reuses its arenas and plans. FFTW's threaded real-to-real executor may perform provider-owned scheduling allocations during execution; those remain opaque provider activity and are not mislabeled as benchmark-owned storage. The later integration experiment must report this distinction explicitly.

The next production experiment must compose the winning retained horizontal transform with this vertical policy inside the existing WVM nonlinear-flux lifetime. Only that integration can apply issue #20's complete-call adoption gate. Split-complex unit-stride transforms, fused normalization, and explicit even/odd extensions remain later candidates and should be attempted only after this control/candidate result is stable.

## First composed development boundary

The `constant-stratification-flux` kernel is the next bounded experiment. It begins with three ready, radially retained, mode-keyed coefficient arrays and ends with three accumulated mode-keyed coefficient arrays. Inside that boundary it executes:

1. five symmetry-preserving coefficient-assembly passes;
2. the exact 15-channel inverse DCT-I/DST-I family schedule;
3. five horizontal inverse transforms, reconstructing shared `U,V,W` once and one derivative triple per target;
4. four streamed `-(U*qx+V*qy+W*qz)` pointwise expressions;
5. four horizontal forward transforms with radial retention;
6. the exact four-channel forward DCT-I/DST-I schedule and normalization; and
7. four target-accumulation passes into three output coefficient arrays.

The full control clears and assembles a WVM-order half-spectrum before the production-layout horizontal FFTs. The candidate writes compact split rows, uses the fixed tile-16 partial-column-pruned horizontal provider, and never creates a complete type-I half-spectrum. Both paths use the same type-I algorithms, family assignments, horizontal and vertical normalization, pointwise implementation, mode order, coefficient map, and persistent-buffer lifetime.

The development coefficient map is deterministic, Hermitian-symmetry preserving, endpoint aware, and derivative-family aware. It is intentionally not a copy of WVM's `WVCoefficientFormulas.hpp`. Consequently, compact-versus-full agreement establishes that the two benchmark algorithms implement the declared composed mathematical graph, but it is not authoritative validation of WVM's phase evolution, physical coefficient formulas, or complete nonlinear flux. Those formulas must enter through a later WVM-exported fixture or direct WVM integration before any issue #20 adoption decision.

Component timings separately report coefficient assembly, inverse type-I work, horizontal inverse work, pointwise work, horizontal forward work, forward type-I work, coefficient accumulation, and the independently sampled uninstrumented total. Application-owned storage is persistent. FFTW-owned execution allocation and plan/thread memory remain opaque. Explicit algorithm-resident memory includes the reusable type-I arena, seven-real-volume streamed physical lifetime, horizontal scratch, and pointwise worker state; caller coefficient arrays are reported separately.
