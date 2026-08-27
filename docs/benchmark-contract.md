# Benchmark contract

## Mathematical operators

The retained horizontal forward and inverse operators are

$$
T_h^+ = P_h\mathcal{F}_{xy},
\qquad
T_h^- = \mathcal{F}_{xy}^{-1}E_h.
$$

(P_h) selects the retained independent horizontal Fourier modes. (E_h) embeds those values into the Hermitian half-spectrum required by an inverse real transform. Logical values are identified by ((k,l,j,\mathrm{field})); their physical order is not part of the mathematics.

The vertical contract records

$$
N_j = \left\lfloor\frac{2(N_z-1)}{3}\right\rfloor,
$$

although vertical matrix multiplication is intentionally outside the first FFT-only implementation slice.

The contract uses a deterministic truncated orthonormal DCT-II matrix pair as its provider-independent vertical fixture. This fixture is not a model of a particular stratification; it supplies reproducible (N_j\times N_z) and (N_z\times N_j) operators for validating shapes, ordering, modal round trips, and combined horizontal–vertical projections before production vertical matrices are introduced by the GEMM issues.

## Horizontal retention

Production cases use WVM's radial two-thirds rule. For each DFT mode pair ((k,l)), the descriptor keeps one primary member of each conjugate pair, excludes the two-dimensional Nyquist axes, and retains the mode when

$$
\sqrt{K^2+L^2} \le \frac{2K_{\max}}{3}.
$$

Retained modes are ordered stably by radial magnitude, then integer (k), then integer (l), matching WVM's `sortrows([Kh,K,L])` convention. Result bundles include a hash of this logical order. Non-antialiased workloads are correctness diagnostics and do not determine performance decisions.

## Representations

The initial implementation distinguishes these representations:

- A real WVM grid with physical index order (x,y,z,\mathrm{field}), where (x) is contiguous.
- The WVM full Hermitian half-spectrum with physical order (z,\mathrm{field},k_x,k_y), where the (N_z\times\mathrm{fields}) block is contiguous for each stored horizontal mode.
- A plane-major interleaved Hermitian half-spectrum used only for representation checks and comparisons with historical experiments.
- vDSP's packed split-complex real-transform representation, including its special DC and Nyquist boundary packing.
- A compact retained representation with physical order (z,\mathrm{field},\mathrm{radial\ mode}).

FFTW uses guru64 strides to write the WVM full-spectrum representation directly. The vDSP provider operates on its native split representation. Correctness is tested after a mode-keyed mapping; no provider is required to materialize a canonical order before it can be judged correct.

## Correctness

The independent direct DFT uses the same mathematical sign and normalization as FFTW: the forward transform is unnormalized and the inverse transform returns (N_xN_y) times the physical input. The maximum reported error is

$$
\epsilon = \frac{\max_i |a_i-b_i|}{\max(1,\max_i |b_i|)}.
$$

The denominator supplies an explicit absolute-error rule for values near zero. Required correctness tolerance is (epsilon\le10^{-12}).

Focused validation covers impulse, sinusoid, deterministic random, DC, and Nyquist fixtures. It checks full forward provider conformance, inverse round trips, the retained horizontal operator against a directly evaluated mode-keyed oracle, representation round trips, gather/embed round trips, Hermitian boundaries, and permutation invariance.

## Measurement scopes

Every timing record has a scope, stage, direction, state, estimated bytes moved, raw samples, and median. Stage state is one of `executed`, `fused`, `elided`, `setup-only`, or `unsupported`.

The required component ledger is:

1. setup/planning;
2. raw forward FFT;
3. horizontal retention;
4. representation conversion;
5. permutation/packing;
6. raw forward vertical matrix multiplication;
7. modal work;
8. raw inverse vertical matrix multiplication;
9. horizontal embedding;
10. raw inverse FFT;
11. uninstrumented total.

The FFT-only slice marks the vertical and modal stages `unsupported`. FFTW marks representation conversion and packing `elided` because its guru64 plan writes the production WVM layout directly. vDSP marks conversion and permutation/packing `fused` at the ledger level and also publishes the separately measurable adapter components.

Provider-native primitive timing excludes packing, conversion, allocation, planning, and the restoration of destructive inverse inputs. Adapter totals include the transformations needed to accept or return WVM's full-spectrum representations. Retained-operator totals additionally include (P_h) or (E_h) and are measured in separate uninstrumented loops. Component medians need not sum to the total because cache state, fusion, and instrumentation change execution.

All steady-state buffers, vDSP setups, and vDSP worker threads are persistent. No timed loop performs an intentional allocation.

Provider construction reports mutually exclusive setup-only (`otherSeconds`), allocation, and planning durations. `totalSeconds` is their sum; it is not an independently timed fourth component.

## Result bundle

The JSON document conforms to `spectral-kernel-benchmark-v1` and records:

- logical operator, algorithm, representation, mode-order, scheduling, and provider identifiers;
- (H,N_{kl},N_z,N_j), fields, planes, strides, retention rules, grouping, and order hashes;
- machine, OS, compiler, flags, source commit, and dirty-tree metadata;
- planning configuration and cost;
- explicit persistent, scratch, input, full-spectrum, and retained-spectrum bytes;
- component ledger states, timing samples, bytes moved, medians, and correctness metrics.

Each bundle also records the exact WVM history audited for the contract: the production-layout baseline, the historical issue #129 harness, and its recorded decision commit. The benchmark repository commit and dirty state come from the build itself, so an uncommitted development run is identifiable rather than silently presented as a clean reference.

The accompanying CSV contains one row per timing sample and is intended for statistical analysis and `skbench compare`.

## Scope boundary

This repository benchmarks primitives and composed spectral-operator graphs. It does not duplicate WVM's equations, nonlinear flux implementation, time integration, state management, or output system. A later WVM integration benchmark will validate any selected provider/layout tuple inside the production nonlinear calculation.
