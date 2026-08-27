# Benchmark contract

## Mathematical operators

The retained horizontal forward and inverse operators are

$$
T_h^+ = P_h\mathcal{F}_{xy},
\qquad
T_h^- = \mathcal{F}_{xy}^{-1}E_h.
$$

$P_h$ selects the retained independent horizontal Fourier modes. $E_h$ embeds those values into the Hermitian half-spectrum required by an inverse real transform. Logical values are identified by $(k,l,j,\mathrm{field})$; their physical order is not part of the mathematics.

The vertical contract records

$$
N_j = \left\lfloor\frac{2(N_z-1)}{3}\right\rfloor,
$$

although vertical matrix multiplication is intentionally outside the first FFT-only implementation slice.

The contract uses a deterministic truncated orthonormal DCT-II matrix pair as its provider-independent vertical fixture. This fixture is not a model of a particular stratification; it supplies reproducible $N_j\times N_z$ and $N_z\times N_j$ operators for validating shapes, ordering, modal round trips, and combined horizontal–vertical projections before production vertical matrices are introduced by the GEMM issues.

## Horizontal retention

Production cases use WVM's radial two-thirds rule. For each DFT mode pair $(k,l)$, the descriptor keeps one primary member of each conjugate pair, excludes the two-dimensional Nyquist axes, and retains the mode when

$$
\sqrt{K^2+L^2} \le \frac{2K_{\max}}{3}.
$$

Retained modes are ordered stably by radial magnitude, then integer $k$, then integer $l$, matching WVM's `sortrows([Kh,K,L])` convention. Result bundles include a hash of this logical order. Non-antialiased workloads are correctness diagnostics and do not determine performance decisions.

## Representations

The initial implementation distinguishes these representations:

- A real WVM grid with physical index order $(x,y,z,\mathrm{field})$, where $x$ is contiguous.
- The WVM full Hermitian half-spectrum with physical order $(z,\mathrm{field},k_x,k_y)$, where the $N_z\times\mathrm{fields}$ block is contiguous for each stored horizontal mode.
- A plane-major interleaved Hermitian half-spectrum used only for representation checks and comparisons with historical experiments.
- vDSP's packed split-complex real-transform representation, including its special DC and Nyquist boundary packing.
- A compact retained representation with physical order $(z,\mathrm{field},\mathrm{radial\ mode})$.

FFTW uses guru64 strides to write the WVM full-spectrum representation directly. The vDSP provider operates on its native split representation. Correctness is tested after a mode-keyed mapping; no provider is required to materialize a canonical order before it can be judged correct.

The issue #3/#5 baseline matrix has ten named workloads. The historical issue #129 cases are $256^2$, $N_z=65$, fields $=3,4$ and $512^2$, $N_z=129$, fields $=3,4$. The current WVM cases are $256^2$, $N_z=129$, fields $=1,3,4$ and $512^2$, $N_z=257$, fields $=1,3,4$. These cases change the number of independent horizontal planes but do not change the primitive two-dimensional transform.

## Correctness

The independent direct DFT uses the same mathematical sign and normalization as FFTW: the forward transform is unnormalized and the inverse transform returns $N_xN_y$ times the physical input. The maximum reported error is

$$
\epsilon = \frac{\max_i |a_i-b_i|}{\max(1,\max_i |b_i|)}.
$$

The denominator supplies an explicit absolute-error rule for values near zero. The Float64 correctness tolerance is $\epsilon\le10^{-12}$. Float32 experiments must preregister dimension-aware thresholds based on Float32 machine epsilon and operation count; they report raw scale-normalized maximum, relative $L_2$, round-trip, energy, retained-mode, and Hermitian-boundary errors rather than reusing the Float64 threshold.

## Numeric type and execution placement

Every new run declares one numeric type, currently `float64` or `float32`, and its scalar bit width. The initial v1 decision remains Float64-only. Float32 results are a separately identified follow-up evidence chain and cannot silently enter Float64 aggregates.

Every provider and algorithm records distinct forward and inverse execution contracts. Each direction states:

- whether the provider-native operation is in-place, out-of-place, or unsupported;
- whether the complete adapter is in-place, out-of-place, or unsupported;
- whether native execution destroys its input and whether the adapter preserves the caller's input;
- whether repeated execution requires restoration or a preservation copy, and whether that work is included in primitive and adapter timing;
- native and adapter input/output representation identifiers;
- physical extents, element strides, padding, minimum alignment, and aliasing restrictions;
- reusable work-buffer bytes;
- whether native output can feed the opposite transform direction without conversion.

The native vDSP baseline treats `vDSP_fft2d_zripD`, `vDSP_fft2d_zriptD`, `vDSP_fft2d_zropD`, and `vDSP_fft2d_zroptD` as four separate algorithms. They respectively test in-place or out-of-place execution with provider-managed or caller-supplied temporary storage. Native input, output, and caller-supplied work arrays are 64-byte aligned. Caller-supplied work is one split-complex buffer per persistent worker, with each real and imaginary array containing $\max(N_y,N_x/2)$ doubles as required by the Accelerate API. Persistent input/output storage and scratch storage are reported separately.

The issue #4 FFTW strategy screen preserves the WVM guru64 frequency-major representation while independently changing planning effort, alignment assumptions, wisdom state, and parallel topology. Planning modes are `ESTIMATE`, `MEASURE`, `PATIENT`, and `EXHAUSTIVE`. An aligned candidate plans and executes on FFTW-allocated 64-byte-aligned arrays with matching new-array alignment classes; an unaligned candidate adds `FFTW_UNALIGNED`. A generated-import candidate first creates and exports matching wisdom, forgets global wisdom, imports the recorded string, and constructs the execution plans with `FFTW_WISDOM_ONLY`. Wisdom generation, export, import, and final plan construction are reported separately.

An FFTW scheduling topology records internal pthread workers and persistent outer batch shards independently. A single outer shard retains the exact production two-dimensional guru64 batch geometry. Multiple outer shards own disjoint contiguous plane ranges while retaining global WVM output strides, and one persistent worker executes each independently planned shard. Hybrid candidates assign more than one FFTW internal worker to each outer shard. The authoritative primitive interval is the complete scheduled batch wall time; an empty outer dispatch is a non-additive diagnostic. No explicit allocation occurs during steady-state execution.

`PATIENT` and `EXHAUSTIVE` feasibility candidates may specify FFTW's per-plan-call time limit. The result records the requested limit and whether an observed forward or inverse planning call consumed at least 95 percent of it. A budget-exhausted plan is a correct time-bounded candidate, not evidence that an unlimited search completed. Raw execution remains comparable, but the planning label and limitation must remain visible.

The issue #6 batching experiment holds the in-place packed split-complex representation fixed while independently changing decomposition and outer scheduling. `direct-persistent` and `direct-gcd` invoke the native two-dimensional real API once per plane. `separable-persistent` and `separable-gcd` apply `vDSP_fft_zripD` across real rows and `vDSP_fft_zipD` down complex columns, reconstructing the packed DC and Nyquist columns with one reusable split-complex buffer of $N_y$ elements per logical worker. The separable candidate performs no transpose and records that stage as `elided`. Grand Central Dispatch uses a synchronous `dispatch_apply_f` over exactly the requested number of logical chunks on the user-initiated global queue; the operating system retains control of physical thread placement.

Complete native batch wall time is the authoritative primitive measurement for every batching strategy. Separable row and column phases are repeated in separately prepared diagnostic loops, and an empty dispatch with the same chunk topology estimates scheduler overhead. Those diagnostic medians are not subtracted from or added to the primitive wall time: repeated dispatch, cache state, and instrumentation make them non-additive. The implementation allocates no explicit storage in any steady-state execution path; opaque allocation or scheduling inside Accelerate or GCD is not claimed to be observable.

Placement is an algorithm and storage contract, not a mathematical requirement. Primitive timing honors the provider's native contract. Adapter and pipeline timing include preservation, packing, or copying only when the declared caller-data lifetime requires that work. A report must not compare an in-place primitive with an out-of-place adapter as if they perform identical work, and it must not charge a preservation copy when the source is genuinely dead.

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

Provider-native primitive timing excludes packing, conversion, allocation, planning, and any explicitly excluded restoration of destructive inputs. Adapter totals include the transformations needed to accept or return WVM's full-spectrum representations under their recorded caller-data lifetime. Retained-operator totals additionally include $P_h$ or $E_h$ and are measured in separate uninstrumented loops. Component medians need not sum to the total because cache state, fusion, and instrumentation change execution.

All steady-state buffers, vDSP setups, and vDSP worker threads are persistent. No timed loop performs an intentional allocation.

Provider construction reports mutually exclusive setup-only (`otherSeconds`), allocation, and planning durations. `totalSeconds` is their sum; it is not an independently timed fourth component.

## Result bundle

The JSON document conforms to `spectral-kernel-benchmark-v1` and records:

- logical operator, algorithm, representation, mode-order, scheduling, and provider identifiers;
- $H,N_{kl},N_z,N_j$, fields, planes, strides, retention rules, grouping, and order hashes;
- numeric type and forward/inverse execution-placement contracts;
- machine, OS, compiler, flags, source commit, and dirty-tree metadata;
- planning configuration and cost;
- explicit persistent, scratch, input, full-spectrum, and retained-spectrum bytes;
- component ledger states, timing samples, bytes moved, medians, and correctness metrics.

Each bundle also records the exact WVM history audited for the contract: the production-layout baseline, the historical issue #129 harness, and its recorded decision commit. The benchmark repository commit and dirty state come from the build itself, so an uncommitted development run is identifiable rather than silently presented as a clean reference.

The accompanying CSV contains one row per timing sample and is intended for statistical analysis and `skbench compare`.

## Append-only publication

`results/published/catalog.json` is the mutable index of immutable evidence. Every run has a permanent `run.id`, one publication status (`preliminary`, `reference`, `superseded`, or `withdrawn`), issue and experiment associations, a status explanation, and SHA-256 hashes for its JSON and CSV artifacts. Numerical pass/fail status is independent of publication status.

New bundles use `results/published/runs/<run-id>/result.json` and `results/published/runs/<run-id>/samples.csv`. Once published, those files and their catalog artifact paths and hashes cannot change or disappear. Status, explanation, experiment association, and supersession metadata may evolve without hiding the original evidence. Only `reference` runs from clean trees and passing numerical tests may enter adoption statistics.

The initial M4 run is explicitly grandfathered at its original flat artifact paths and remains byte-identical. Its legacy page `/runs/m4-max-quick-20260827.html` remains available alongside its canonical `/runs/20260827T185428Z-lyra/` page.

Stable publication routes are:

- `/runs/<run-id>/` for immutable run records;
- `/experiments/<experiment-id>/` for accumulating issue-level evidence and experiment definitions;
- `/methods/operators-and-representations/` for this methodology;
- `/decisions/v1/` for the evidence synthesis and adoption decision.

Pages generation reads committed artifacts and never executes a benchmark. `python3 tools/validate_publication.py` performs the fast local catalog and immutability checks; `python3 tools/build_site.py --output _site` renders the site independently.

## Scope boundary

This repository benchmarks primitives and composed spectral-operator graphs. It does not duplicate WVM's equations, nonlinear flux implementation, time integration, state management, or output system. A later WVM integration benchmark will validate any selected provider/layout tuple inside the production nonlinear calculation.
