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

The first issue #8 performance increment uses that common immutable matrix pair to compare the primitive products

$$
(N_j\times N_z)(N_z\times K)
\qquad\text{and}\qquad
(N_z\times N_j)(N_j\times K),
$$

where $K=N_{kl}\times\mathrm{fields}$. A column key is $\mathrm{field}+\mathrm{fields}\times\mathrm{mode}$, and the vertical coordinate is contiguous within each column. This order is a controlled input representation for the primitive test, not a mathematical requirement or evidence that its upstream packing cost is negligible.

The grouped issue #8 increment partitions the radially ordered retained modes by exact integer

$$
K^2 = k^2 + l^2
$$

on the square benchmark domains. Equal keys are contiguous in the existing radial order. Each group receives a deterministic dense orthonormal matrix pair produced by pairwise rotations of the common truncated DCT-II rows. This synthetic family changes matrix values without changing dimensions, conditioning, or projection rank. It is a performance and correctness fixture, not a claim about a particular stratification profile.

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
- whether the complete adapter is in-place, out-of-place, an out-of-place provider result exposed through a zero-copy logical view, or unsupported;
- whether native execution destroys its input and whether the adapter preserves the caller's input;
- whether repeated execution requires restoration or a preservation copy, and whether that work is included in primitive and adapter timing;
- native and adapter input/output representation identifiers;
- physical extents, element strides, padding, minimum alignment, and aliasing restrictions;
- reusable work-buffer bytes;
- whether native output can feed the opposite transform direction without conversion.

The native vDSP baseline treats `vDSP_fft2d_zripD`, `vDSP_fft2d_zriptD`, `vDSP_fft2d_zropD`, and `vDSP_fft2d_zroptD` as four separate algorithms. They respectively test in-place or out-of-place execution with provider-managed or caller-supplied temporary storage. Native input, output, and caller-supplied work arrays are 64-byte aligned. Caller-supplied work is one split-complex buffer per persistent worker, with each real and imaginary array containing $\max(N_y,N_x/2)$ doubles as required by the Accelerate API. Persistent input/output storage and scratch storage are reported separately.

The initial issue #4 FFTW strategy screen preserves the WVM guru64 frequency-major representation while independently changing planning effort, alignment assumptions, wisdom state, and parallel topology. The final layout increment also permits a plane-major half-spectrum whose complete $N_y(N_x/2+1)$ output is contiguous for each plane. Interleaved and split complex storage are independent of that spectrum order. Raw provider timing consumes and produces the selected native order. WVM adapter timing separately includes plane-major/frequency-major permutation and, for split storage, split/interleaved conversion. The retained operator selects or embeds mode-keyed coefficients directly from either native order and does not pay for a mathematically unnecessary full-spectrum permutation.

Planning modes are `ESTIMATE`, `MEASURE`, `PATIENT`, and `EXHAUSTIVE`. An aligned candidate plans and executes on FFTW-allocated 64-byte-aligned arrays with matching new-array alignment classes; an unaligned candidate adds `FFTW_UNALIGNED`. A generated-import candidate first creates and exports matching wisdom, forgets global wisdom, imports the recorded string, and constructs the execution plans with `FFTW_WISDOM_ONLY`. Wisdom generation/export, wisdom import, and final plan construction are reported separately.

An FFTW scheduling topology records internal pthread workers and persistent outer batch shards independently. A single outer shard retains the exact production two-dimensional guru64 batch geometry. Multiple outer shards own disjoint contiguous plane ranges while retaining global WVM output strides, and one persistent worker executes each independently planned shard. Hybrid candidates assign more than one FFTW internal worker to each outer shard. The authoritative primitive interval is the complete scheduled batch wall time; an empty outer dispatch is a non-additive diagnostic. A macOS test-side allocator interposer verifies zero `malloc`, `calloc`, `realloc`, `valloc`, `aligned_alloc`, or `posix_memalign` calls across repeated steady-state forward/inverse execution and empty outer dispatch for internal, outer, and hybrid topologies after warmup.

`PATIENT` and `EXHAUSTIVE` feasibility candidates may specify FFTW's per-plan-call time limit. The result records the requested limit and whether an observed forward or inverse planning call consumed at least 95 percent of it. A budget-exhausted plan is a correct time-bounded candidate, not evidence that an unlimited search completed. Raw execution remains comparable, but the planning label and limitation must remain visible.

The bounded issue #4 screen defines its preliminary Pareto frontier independently for each workload. A candidate is eligible only when all correctness checks pass and its planning search does not exhaust the configured feasibility budget. One eligible candidate dominates another when it is no slower in raw forward execution, raw inverse execution, and total setup, and is strictly better in at least one of those objectives. Total setup includes allocation, planning, wisdom generation, and wisdom import. Memory and sample coefficient of variation remain explicit diagnostics; they are not hidden tie-breakers in the nine-sample preliminary screen. Budget-limited and dominated candidates remain in the evidence table.

The reference campaign runs only reviewed screen survivors across six production profiles. It uses three fresh processes per workload/candidate, three warmups, and 21 measured steady-state samples per process. Candidate and profile order rotate by process round. Pareto objectives combine identical workload/candidate signatures by the median of their process medians; individual process samples, process medians, setup costs, variability, and immutable run pages remain visible. Correctness covers the full transform, native-order retained forward and inverse operators, WVM and split conversions where executed, and the normalization-aware round trip at a maximum relative error of $10^{-12}$. The macOS allocation interposer exercises both native spectrum orders after warmup. All FFTW paths are out-of-place; multidimensional guru split in-place remains an explicit unsupported capability rather than an inferred advantage or hidden fallback.

The issue #6 batching experiment holds the in-place packed split-complex representation fixed while independently changing decomposition and outer scheduling. `direct-persistent` and `direct-gcd` invoke the native two-dimensional real API once per plane. `separable-persistent` and `separable-gcd` apply `vDSP_fft_zripD` across real rows and `vDSP_fft_zipD` down complex columns, reconstructing the packed DC and Nyquist columns with one reusable split-complex buffer of $N_y$ elements per logical worker. The separable candidate performs no transpose and records that stage as `elided`. Grand Central Dispatch uses a synchronous `dispatch_apply_f` over exactly the requested number of logical chunks on the user-initiated global queue; the operating system retains control of physical thread placement.

Complete native batch wall time is the authoritative primitive measurement for every batching strategy. Separable row and column phases are repeated in separately prepared diagnostic loops, and an empty dispatch with the same chunk topology estimates scheduler overhead. Those diagnostic medians are not subtracted from or added to the primitive wall time: repeated dispatch, cache state, and instrumentation make them non-additive. The implementation allocates no explicit storage in any steady-state execution path; opaque allocation or scheduling inside Accelerate or GCD is not claimed to be observable.

The bounded issue #8 common-matrix screen compares one `cblas_zgemm` with two `cblas_dgemm` calls. The complex candidate expands the immutable real matrix into complex storage during setup. The split candidate retains one real matrix and separate persistent real and imaginary operands. Both candidates are out-of-place and use prearranged column-major operands. Primitive timing contains only the complete forward or inverse GEMM formulation: packing, split/interleaved conversion, horizontal ordering, allocation, and matrix preparation are excluded. Matrix preparation and explicit persistent memory are still reported separately. The split formulation also exposes its real and imaginary GEMMs as non-additive component diagnostics.

The grouped screen replaces the single large product with one BLAS call per $K^2$ group for complex storage and two real BLAS calls per group for split storage. The serial split baseline remains component-major. Persistent outer candidates pair the real and imaginary calls within each claimed group so one worker completes disjoint output ranges. Weighted-static scheduling constructs contiguous worker ranges during setup using group-column count as the work weight. Dynamic scheduling uses an allocation-free atomic next-group counter. The calling thread participates as worker zero, all other workers persist across samples, and the authoritative primitive interval includes dispatch, synchronization, and every per-group BLAS call. An empty dispatch traverses the same static ranges or dynamic claims without GEMM and is reported as a non-additive diagnostic.

Outer scheduling requires `VECLIB_MAXIMUM_THREADS=1` at process startup. Result metadata distinguishes requested internal and outer workers; thread stacks managed by the C++ runtime and operating system are marked as opaque memory. Persistent scheduler construction and its explicit bookkeeping are setup-only. A test-side allocator interposer verifies zero explicit allocation across warmed-up forward, inverse, component, and empty-dispatch execution.

The installed public Accelerate CBLAS headers expose no variable-size grouped GEMM batch API, so this increment reports that candidate as unsupported rather than silently substituting a loop. Setup also compares every adjacent forward/inverse matrix pair exactly. Adjacent equal pairs could be consolidated without reordering; nonadjacent equality would require moving columns, while unlike matrices require block-diagonal expansion and additional operand storage. Both belong outside a raw primitive comparison unless their movement and memory are accounted explicitly.

Result metadata records group count, minimum, median, and maximum modes and columns per group, matrix-family identity, group-order hash, GEMM calls per direction, scheduling topology, batch capability, and the adjacent-equality result. Matched grouped/common ratios therefore capture both repeated-call overhead and the loss of large-GEMM efficiency. Matched outer/serial ratios isolate scheduling changes within the grouped formulation. Neither includes gathering or permuting columns into group order.

The vertical benchmark also reports an estimated explicit allocation high-water mark. It is the larger of (a) logical matrix-family source storage plus both fully constructed providers plus external physical/modal inputs and (b) both providers plus external inputs plus all four copied outputs used for correctness inspection. The estimate includes aligned provider arrays and scheduler bookkeeping reported by the harness but excludes allocator metadata, C++ thread stacks, opaque Accelerate state, shared libraries, and process/runtime overhead. Before launch, the sweep driver computes a conservative approximation with a small bookkeeping reserve and, by default, rejects a K²-grouped profile above half of physical memory. It records both preflight and result-derived values in the local manifest. The driver also requires a clean source tree and checks the binary's embedded commit and dirty flag against the live tree before accepting a run into its manifest. This is a feasibility guard and accounting estimate, not a sampled resident-set peak.

Accelerate thread-limit candidates run in isolated processes with `VECLIB_MAXIMUM_THREADS` set before program startup. The recorded value is a requested process limit, not a claim about the number of workers Accelerate actually schedules. The initial bounded common, grouped, and outer-scheduling screens use the historical $256^2$, $N_z=65$, fields $=3$ and $512^2$, $N_z=129$, fields $=3$ cases. The finalist portability screen adds every previously missing named issue #8 profile at fields $=1/3/4$ and $N_z=65/129/257$ while holding the serial, static-12, and dynamic-16 candidates fixed. Third-party grouped APIs, blocking, and packing-plus-GEMM remain outside these increments.

## Issue #12 partial-column-pruned horizontal transform

The first issue #12 candidate uses the separability of the two-dimensional DFT. Its forward path applies out-of-place length-$N_x$ real-to-complex transforms to every row and plane, producing plane-major row-spectrum scratch. Only the contiguous nonnegative-$k_x$ band that can intersect the radial two-thirds retained disk receives an in-place length-$N_y$ complex forward transform. The final loop reads the requested stored coordinates, applies the logical conjugation convention for primary negative-$k$ modes, and writes compact retained coefficients keyed by $(k,l,z,\mathrm{field})$.

The inverse path zeros the complete row-spectrum scratch, embeds the compact retained coefficients, repairs the stored $k_x=0$ Hermitian partners, executes inverse complex transforms only for the active $k_x$ band, and applies out-of-place real row transforms. As with FFTW's full inverse, the result carries the unnormalized $N_xN_y$ scale. Correctness compares the retained forward and inverse reconstruction with the independent direct-DFT oracle for impulse, sinusoid, deterministic random, DC, and Nyquist fixtures. The projected round trip is $\mathcal F_{xy}^{-1}E_hP_h\mathcal F_{xy}u$, not the original field when rejected modes are present.

The matched reference uses the same FFTW 3.3.11 build, planning effort, internal worker count, fixture, warmups, samples, and Float64 operator. It separately times the optimized full two-dimensional primitive, radial selection or embedding, and the complete retained operator. The candidate separately times real-row transforms, selected complex-column transforms, direct radial retention or embedding, and its complete retained operator. Component medians are diagnostic and need not sum to the uninstrumented total.

This candidate elides a completed full WVM-order transformed output, but it does not reduce first-pass row work or aggregate scratch capacity: its reusable plane-major buffer contains $N_y(N_x/2+1)$ complex values per plane. It also computes every $k_y$ output inside each active $k_x$ column, even though the radial disk retains fewer values near its edge. Its complete caller contract is out-of-place, while only the private selected-column stage is in-place. Deeper within-column pruning, transform-internal transposition, split storage, and a caller-visible in-place retained operator remain separate algorithms or explicit capability gaps.

The append-only outer-sharding increment partitions the existing plane batch into contiguous plane/field shards. Each persistent worker owns separately planned row and selected-column transforms and a disjoint slice of the one aggregate plane-major scratch allocation. FFTW internal pthreads are fixed at one for this increment. The matched full reference uses the same outer worker count for its two-dimensional plans, mode-keyed radial gather, and inverse zero-fill/embed. Thus the authoritative complete-operator comparison includes identical scheduling topology without multiplying aggregate scratch capacity. Maximum per-shard scratch, aggregate scratch, empty-dispatch time, plan count, and setup time are reported separately.

The full and pruned complete operators both preserve caller inputs and write disjoint outputs. FFTW inverse execution may destroy only provider-private or explicitly supplied full-spectrum scratch. Outer workers operate on disjoint plane indices, and the macOS allocation interposer verifies zero steady-state allocation for full and pruned forward, inverse, adapter, and empty-dispatch execution.

The experiment may promote this candidate when it wins a representative workload or supplies a meaningful latency/memory Pareto point. A slower result rejects only this partial-column decomposition unless a separate feasibility analysis bounds the remaining avoidable work tightly enough to support a broader negative conclusion.

## Issue #7 representation-boundary close-out

The issue #7 close-out holds the Float64 radial two-thirds operator, six production workloads, FFTW 3.3.11 build, `MEASURE` planning, unaligned execution, and one-internal/twelve-outer worker topology fixed. It compares the established plane-major full and partial-column-pruned controls with three representation-boundary algorithms. A persistent view exposes the retained logical modes as immutable indices over a complete plane-major half-spectrum. A fused full-transform path selects and converts directly from plane-major interleaved storage into compact split real and imaginary arrays. A fused pruned path performs the same compact split write from the pruned algorithm's plane-major scratch.

The forward persistent view executes no retention, conversion, packing, or coefficient movement. Its physical result remains a complete provider-order half-spectrum, so its memory and downstream indirect-access contract remain explicit. The inverse view accepts an already zero-padded provider-order half-spectrum whose lifetime ends at the call. FFTW 3.3.11 supplies no input-preserving multidimensional complex-to-real algorithm, so benchmark fixture restoration occurs outside the single-call timing boundary. Creating or maintaining that ready inverse representation is not free; issue #13 must compare it with embedding, packing, or persistent zero-padding policies before the view can enter a composed pipeline.

The compact split candidates preserve the unnormalized issue #7 operator. Their forward selection, logical conjugation, and interleaved-to-split conversion are one pass. Their inverse split-to-interleaved conversion, zero fill, retained embedding, and Hermitian-boundary repair are one pass. Raw FFT or row/column primitive timings remain separate from these movements and from the complete uninstrumented retained-operator totals. All paths are out-of-place at the public boundary and allocate no memory after warmup.

A diagnostic applies the horizontal factor $1/(N_xN_y)$ either inside the fused split selection or in a second compact-array pass. It does not redefine the primary operator and cannot by itself establish a production normalization policy. If an immutable vertical matrix can absorb the factor during setup, that policy performs no horizontal normalization pass and belongs to the issue #13 composed boundary comparison.

The one-round screen uses two warmups and nine samples per candidate/workload process. A new candidate advances to the three-process, three-warmup, 21-sample reference cohort when its geometric ratio to the best matched control is at most 1.05 across the complete workload-direction matrix with at least one win. A distinct split or view representation needed by issue #13 may also advance when its complete-matrix ratio is at most 1.10. Both established controls remain in every reference cohort. Non-advancing algorithms remain permanent preliminary evidence.

## Issue #13 ordering and packing baseline

The first issue #13 increment measures the current MATLAB-style materialization policy without treating it as mathematically required. Its forward adapter reads only retained coefficients from a WVM frequency-major interleaved half-spectrum and writes them directly into the vertical provider's final interleaved or split input. Radial selection, K² grouping, the vertical-contiguous transpose, conjugation of primary negative-$k$ modes, and split conversion where applicable are one fused movement loop. The inverse adapter reads the provider's final vertical output, zeros the complete WVM half-spectrum, scatters the retained coefficients, and writes required $k_x=0$ Hermitian partners.

The component ledger and timing table distinguish raw vertical GEMM, forward gather/pack, inverse scatter/embed, and one-shot movement-plus-GEMM. Reuse sequences at $R=2,4,8$ compare two policies over identical logical work: boundary movement around every vertical use, and one boundary movement around $R$ repeated uses of a persistent compact representation. At $R=1$ the policies are definitionally identical. The measured crossover is the first sampled $R>1$ whose persistent total is lower; no crossover claim is extrapolated beyond the sampled counts.

Forward packing reports one retained-coefficient read and one final-provider-input write. Reverse embedding reports the full-spectrum zero fill, retained-output reads, primary retained writes, and extra Hermitian-boundary writes. Combined and reuse totals report the sum of their declared movement and primitive byte ledgers. Provider arrays, the full caller spectrum, compact retained/modal representations, setup-only matrix storage, and the estimated explicit high-water mark remain separately visible. C++ thread stacks and opaque Accelerate state remain outside explicit memory.

The bounded first sweep covers the historical $256^2$, $N_z=65$, fields $=3$ workload and current $256^2$, $N_z=129$, fields $=1/3/4$ workloads with static-12 and dynamic-16 K²-group scheduling. It excludes raw FFT execution, modal physics, nonlinear flux, 512-class scaling, provider-native fusion, strided/no-reorder kernels, pre-permuted matrix alternatives, and tiled pack-and-GEMM. Those exclusions are explicit capability gaps for later issue #13 increments, not negative performance results.

The next issue #13 increment implements one no-reorder alternative against that immutable baseline. A retained stored frequency already contains a contiguous $N_z\times\mathrm{fields}$ block in WVM order. The direct forward kernel selects the corresponding real K²-dependent matrix and calls complex GEMM with $M=N_j$, $N=\mathrm{fields}$, and $K=N_z$. Its output remains in a full zero-padded frequency-major modal representation with a contiguous $N_j\times\mathrm{fields}$ block per frequency. The inverse uses $M=N_z$, $N=\mathrm{fields}$, and $K=N_j$ and writes directly into full zero-padded WVM storage. Real matrices preserve the conjugation convention for stored negative-$k$ coefficients. The required $k_x=0$ Hermitian partners are repaired inside the timed direct kernel.

The direct provider initializes the zero padding once during setup and performs no allocation, clearing, gather, transpose, split conversion, radial permutation, or reverse scatter during steady-state execution. Its primitive and one-shot totals have the same logical boundary but are sampled separately so primitive GEMM remains independently reportable. Elided movement stages contain no fabricated samples. Reuse totals at $R=2,4,8$ repeat the direct provider-order kernel and are compared with the packed representation's two boundary-movement policies. Result metadata records the direct per-frequency call count, packed group call count, operand byte ledger, immutable matrix preparation, persistent/scratch memory, scheduler setup, correctness, and variability.

The matched comparison holds the mathematical operator, retained mode keys, Float64 precision, matrix family, workload, thread limit, outer schedule, warmups, and samples fixed. It changes only representation and algorithm: many small complex GEMMs in WVM order versus a fused gather-to-split adapter followed by fewer grouped real GEMMs. Raw FFT execution, modal physics, nonlinear flux, provider-native horizontal fusion, tiling, and 512-class scaling remain excluded. Therefore the result can answer whether movement avoidance offsets small-GEMM inefficiency at this boundary; it cannot establish the complete spectral-pipeline winner.

Correctness uses independent scalar products on 17 deterministic columns spanning the operand and full-output equivalence between the complex and split formulations. It records both scale-normalized maximum error and relative $L_2$ error with the Float64 $10^{-12}$ tolerance. A test-side allocator interposer verifies zero explicit allocation across repeated warmed-up forward and inverse primitive calls.

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

The FFT-only slice marks the vertical and modal stages `unsupported`. FFTW marks representation conversion and packing `elided` because its guru64 plan writes the production WVM layout directly. vDSP marks conversion and permutation/packing `fused` at the ledger level and also publishes the separately measurable adapter components. The vertical-GEMM slice marks FFT, horizontal selection, modal work, and the complete total `unsupported`; it marks packing and representation conversion `elided` because the primitive operands are prepared before timing.

Provider-native primitive timing excludes packing, conversion, allocation, planning, and any explicitly excluded restoration of destructive inputs. Adapter totals include the transformations needed to accept or return WVM's full-spectrum representations under their recorded caller-data lifetime. Retained-operator totals additionally include $P_h$ or $E_h$ and are measured in separate uninstrumented loops. Component medians need not sum to the total because cache state, fusion, and instrumentation change execution.

All steady-state buffers, vDSP setups, and vDSP worker threads are persistent. No timed loop performs an intentional allocation.

Provider construction reports mutually exclusive setup-only (`otherSeconds`), allocation, and planning durations. `totalSeconds` is their sum; it is not an independently timed fourth component.

## Cross-Mac portability inference

The issue #11 portability campaign holds the two algorithm graphs, Float64 mathematics, retention, FFTW planning/alignment/wisdom policy, tile width, transpose blocking, physical representations, vertical kernels, placement, fixtures, and allocation policy fixed. It varies only topology-derived scheduling before inference. For performance-core count $P$ and total physical-core count $T$, each graph separately compares horizontal outer sharding at $P$ and $T$ with vertical outer-dynamic scheduling at $T$ and weighted outer-static scheduling at $P$. A deterministic geometric-time rule freezes one topology per graph for every workload on that machine. Calibration results cannot enter adoption aggregates.

Reference timing uses isolated balanced candidate/control processes with three warmups and 21 samples. Three rotated rounds are collected first. Exactly two additional complete rounds are collected when a workload ratio spread exceeds 10%, a workload straddles the 1.03 regression boundary, aggregate round ratios straddle 0.90, or their aggregate median lies from 0.85 through 0.95. The campaign stops after five rounds and reports an unstable result as inconclusive. Ranges bootstrapped from three or five process-pair rounds are empirical intervals, not claims of general statistical confidence.

Memory is collected in separate processes after correctness-only and setup-only storage is released. Timing samples from memory processes do not enter performance aggregates. A case whose estimated explicit peak exceeds the declared safe fraction of physical memory is an explicit capacity exclusion. The runner does not swap deliberately, resize the workload, or substitute a different algorithm. A portability conclusion remains limited to matched feasible profiles and to the named hardware-plus-toolchain configurations; it never creates workload-size dispatch or a general-Mac claim.

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

`results/published/catalog.json` is the mutable index of immutable evidence. Every run has a permanent `run.id`, one publication status (`preliminary`, `reference`, `superseded`, or `withdrawn`), issue and experiment associations, a status explanation, and SHA-256 hashes for its JSON and CSV artifacts. A sweep may also supply a stable `incrementId`; the publisher preserves it so issue-level synthesis can distinguish successive algorithms or scheduling methods while leaving earlier runs intact. Numerical pass/fail status is independent of publication status.

New bundles use `results/published/runs/<run-id>/result.json` and `results/published/runs/<run-id>/samples.csv`. Once published, those files and their catalog artifact paths and hashes cannot change or disappear. Status, explanation, experiment association, and supersession metadata may evolve without hiding the original evidence. Only `reference` runs from clean trees and passing numerical tests may enter adoption statistics.

The initial M4 run is explicitly grandfathered at its original flat artifact paths and remains byte-identical. Its legacy page `/runs/m4-max-quick-20260827.html` remains available alongside its canonical `/runs/20260827T185428Z-lyra/` page.

Stable publication routes are:

- `/runs/<run-id>/` for immutable run records;
- `/experiments/<experiment-id>/` for accumulating issue-level evidence and experiment definitions;
- `/methods/operators-and-representations/` for this methodology;
- `/decisions/v1/` for the evidence synthesis and adoption decision.

Pages generation reads committed artifacts and never executes a benchmark. `python3 tools/validate_publication.py` performs the fast local catalog and immutability checks; `python3 tools/build_site.py --output _site` renders the site independently.

## Implicit and hybrid dealiased convolution screen

Issue #17 is a bounded model-adjacent experiment, not an implementation of WVM's nonlinear flux. Its logical input is four Float64 radially retained Hermitian spectra. A fixed deterministic product map produces either four or twelve quadratic spectral outputs. The explicit oracle embeds those coefficients in a full FFTW half-spectrum, executes four inverse transforms, forms the requested pointwise products, executes one forward transform per output, and retains the radial disk. The FFTW++ candidate accepts the equivalent centered rectangular Hermitian representation with zeros outside the disk and uses the pinned public centered/Hermitian implicit-hybrid implementation.

The authoritative total starts with caller-owned compact coefficients and ends with caller-owned compact coefficients, so embedding, input preservation, and radial output selection are included. The explicit inverse FFT batch, pointwise multiplication, and forward FFT batch remain separately reportable. FFTW++'s transform-multiply-transform stage is reported as fused because its algorithm does not expose honest standalone FFT timings. Planning, provider optimization, fixture construction, correctness comparison, and publication are excluded from execution.

The initial screen fixes one thread, FFTW `MEASURE | UNALIGNED`, the FFTW 3.3.11 build, 256² and 512² grids, four input fields, output counts 4 and 12, and the radial two-thirds disk. The public FFTW++ optimizer crashes when configured directly with more outputs than inputs for this application. The correct twelve-product candidate therefore uses three independently planned four-output applications and includes the necessary input-restoration work. The experiment publishes that capability boundary and does not describe the three-call path as one native fused call.

Correctness is a mode-keyed comparison with the independent explicit oracle using both scale-normalized maximum and relative $L_2$ errors at the Float64 $10^{-12}$ tolerance. The macOS allocator interposer verifies zero application allocations across warmed explicit and FFTW++ executions for both output counts. Placement, native input destruction, caller-input preservation, setup time, provider-resident memory, and the quadratic 1024² capacity projection are published separately. The screen can justify a deeper threaded or WVM-expression experiment; it cannot establish complete-flux performance, Float32 behavior, or a general-Mac recommendation.

## Production-lifetime streamed spectral-flux composition

Issue #19 changes the lifetime and multiplicity of the composed benchmark without claiming to implement the complete nonlinear flux. Its logical boundary starts with 15 ready retained and vertically truncated modal inputs and ends with four ready retained modal targets. Three inputs are shared $U,V,W$ fields. Each target owns three derivative inputs and evaluates $-(Uq_x+Vq_y+Wq_z)$. Shared real volumes are reconstructed once; one three-derivative volume and one target volume are reused across the four targets. Seven real $N_xN_yN_z$ volumes replace the materialized 15-real-volume lifetime used for issue #18 attribution.

The WVM-order control uses full frequency-major interleaved FFTW transforms and direct complex vertical operators. The streaming candidate uses issue #16 partial-column pruning, a fixed 16-plane compact tile, compact radial split storage, and grouped split-real vertical operators. Each graph uses its persistent native modal boundary. The uninstrumented total includes vertical reconstruction, shared and derivative inverse horizontal transforms, pointwise work, four forward horizontal transforms and retention, vertical projection, representation movement, and output writes. Planning, matrix preparation, fixture loading, hash verification, oracle permutation, and correctness evaluation are outside the total and reported separately.

The first increment uses a deterministic synthetic fixture only to validate the provider-independent operator, four-target ordering, buffer reuse, full-volume batching, component ledger, and allocation-free warmed execution. It is preliminary by construction. Reference evidence requires the authoritative export defined in [the WVM spectral-flux fixture contract](wvm-spectral-flux-fixture-contract.md), including the exact WVM commit, generator, payload hashes, mode keys, normalization, derivative conventions, vertical operators, and four expected outputs. Missing authoritative files are an explicit blocker; the benchmark cannot silently fall back to synthetic operators.

Both paths are out-of-place at their public boundary and reuse all application-owned storage. Accelerate may lazily reserve opaque internal state during setup warmup; this remains distinct from application allocation and is covered by the opaque-provider-memory ledger. The allocator interposer verifies that every component and the complete composition perform zero application allocations after that warmup.

## Scope boundary

This repository benchmarks primitives and composed spectral-operator graphs. It does not duplicate WVM's equations, nonlinear flux implementation, time integration, state management, or output system. A later WVM integration benchmark will validate any selected provider/layout tuple inside the production nonlinear calculation.
