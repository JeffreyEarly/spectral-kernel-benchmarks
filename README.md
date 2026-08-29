# Spectral Kernel Benchmarks

This repository is an independent C++20/CMake laboratory for finding the fastest correct antialiased spectral-operator implementation on Apple Silicon. It publishes primitive FFT and matrix-multiplication performance separately from representation adapters, data movement, and complete operator pipelines.

The initial vertical slice compares the production WVM FFTW 3.3.11 two-dimensional real transform with Accelerate/vDSP for the $256 \times 256$, $N_z=65$, fields $=3$ workload. FFTW writes WVM's frequency-major output directly; vDSP uses native packed split-complex storage and reports packing, conversion, permutation, raw transform, and complete retained-operator costs independently. Later append-only evidence expands the exact WVM baseline matrix and tests vDSP placement, scratch, batching, GCD scheduling, and packed-real separable row/column decompositions without rewriting that initial result.

This repository does not reimplement WVM's nonlinear flux calculation. Later composed benchmarks will reproduce the relevant kernel graph, dimensions, reuse, and buffer lifetimes using deterministic synthetic data. Final nonlinear-flux validation remains in WVM.

## Build and test

The default Apple Silicon preset downloads FFTW 3.3.11 from fftw.org, verifies SHA-256 `5630c24cdeb33b131612f7eb4b1a9934234754f9f388ff8617458d0be6f239a1`, and builds it with WVM's NEON/pthreads configuration and `-O3 -mcpu=native` flags.

```sh
cmake --preset macos-release
cmake --build --preset macos-release
ctest --preset macos-release
```

Use `SKBENCH_FFTW_ROOT` at configure time to select an existing compatible FFTW installation instead of the pinned build.

Issue #17 is an optional, non-blocking FFTW++ experiment and is disabled in the ordinary build and CI. Configure with `-DSKBENCH_ENABLE_FFTWPP=ON` to fetch the exact public FFTW++ commit `e685733aba768d77e9234ca02092632f7ccb4c86`, or also set `SKBENCH_FFTWPP_SOURCE_DIR` to use a reviewed local checkout.

## Commands

```sh
build/release/skbench list
build/release/skbench validate --profile smoke
build/release/skbench run --profile quick
build/release/skbench run --profile wvm-historical-256-nz65-f3 --providers fftw --fftw-planning measure --fftw-alignment aligned --fftw-internal-workers 12 --fftw-outer-workers 1
build/release/skbench run --profile wvm-historical-256-nz65-f4 --vdsp-strategy out-of-place-explicit-scratch --workers 12
build/release/skbench run --profile wvm-historical-256-nz65-f3 --vdsp-batch-strategy separable-gcd --workers 12
build/release/skbench run --kernel pruned-horizontal --providers fftw --fftw-planning measure --fftw-internal-workers 12 --profile wvm-historical-256-nz65-f3
build/release/skbench run --kernel pruned-horizontal --providers fftw --fftw-planning measure --fftw-internal-workers 1 --fftw-outer-workers 12 --profile wvm-historical-256-nz65-f3
VECLIB_MAXIMUM_THREADS=12 build/release/skbench run --kernel vertical-gemm --profile wvm-historical-256-nz65-f3
VECLIB_MAXIMUM_THREADS=12 build/release/skbench run --kernel vertical-gemm --vertical-gemm-family k2-grouped --profile wvm-historical-256-nz65-f3
VECLIB_MAXIMUM_THREADS=1 build/release/skbench run --kernel vertical-gemm --vertical-gemm-family k2-grouped --vertical-gemm-schedule outer-dynamic --vertical-gemm-outer-workers 12 --profile wvm-historical-256-nz65-f3
VECLIB_MAXIMUM_THREADS=1 build/release/skbench run --kernel ordering-packing --vertical-gemm-family k2-grouped --vertical-gemm-schedule outer-dynamic --vertical-gemm-outer-workers 16 --profile wvm-historical-256-nz65-f3
build/release/skbench compare --input results/local/<run>.csv
```

`smoke` is a small correctness and contract exercise. `quick` is the first production workload: 195 independent $256 \times 256$ real planes with the exact WVM FFTW guru64 input and output strides. `exhaustive` supplies the historical $512 \times 512$, $N_z=129$, fields $=4$ shape. `skbench list` also exposes the complete issue #3/#5 matrix as named `wvm-historical-*` and `wvm-current-*` profiles.

`validate` checks impulse, sinusoid, deterministic random, DC, and Nyquist fixtures against an independent direct-DFT oracle. It exercises all four native vDSP placement/scratch strategies, direct persistent-pool and GCD scheduling, and the separable packed-real candidates. It also checks full FFT conformance, inverse normalization, retained-mode values, representation round trips, and permutation invariance.

`run` writes a versioned JSON manifest/report and a sample-level CSV file. `--providers fftw` omits the unchanged vDSP provider during FFTW strategy screens. `--fftw-planning`, `--fftw-alignment`, and `--fftw-wisdom` select the planner contract; `--fftw-internal-workers` and `--fftw-outer-workers` distinguish FFTW pthread parallelism from persistent outer batch sharding. `--fftw-spectrum-order wvm|plane-major` independently selects the native Hermitian half-spectrum strides, while `--fftw-layout interleaved|split|paired` selects physical complex storage. Plane-major results report raw native execution, spectrum-order permutation, WVM-compatible adapter, and direct retained-operator totals separately. `--fftw-planning-time-limit` applies FFTW's per-plan-call limit and records whether the observed planning interval exhausted that budget. `--vdsp-strategy` selects `in-place`, `in-place-explicit-scratch`, `out-of-place`, or `out-of-place-explicit-scratch`. `--vdsp-batch-strategy` independently selects `direct-persistent`, `direct-gcd`, `separable-persistent`, or `separable-gcd`; the separable prototype currently supports in-place placement. Scratch runs go to `results/local/` and are ignored. Every new result identifies its numeric type and records the forward and inverse provider-native and adapter execution contracts, including in-place/out-of-place placement, destructive inputs, preservation policy, physical extents, padding, strides, alignment, aliasing, and reusable work memory.

`--kernel pruned-horizontal` selects the issue #12 feasibility candidate. The matched reference performs FFTW's optimized full two-dimensional real transform and then selects the radial two-thirds mode set. The candidate performs all real-row transforms but executes complex column transforms only for the contiguous nonnegative-$k_x$ band that can intersect the retained radial disk. It writes compact mode-keyed output without materializing a completed WVM-order half-spectrum. It still requires one aggregate full-sized plane-major row-spectrum scratch buffer, which is reported explicitly. Forward/inverse row stages, selected-column stages, retention/embedding, empty outer dispatch, setup, memory, and complete uninstrumented retained-operator totals remain separate. `--fftw-internal-workers` and `--fftw-outer-workers` select matched reference/candidate topologies. With multiple outer workers, both algorithms use persistent plane/field shards; each candidate worker owns a disjoint reusable scratch slice. Both complete algorithms remain out-of-place, while the candidate's selected column transforms operate in-place only inside private scratch.

`--kernel vertical-gemm` selects the bounded issue #8 Float64 vertical-projection benchmark. It compares Accelerate complex `zgemm`, with real projection matrices expanded to complex during setup, against two Accelerate real `dgemm` calls per matrix group over persistent split real and imaginary arrays. Both paths are out-of-place. `--vertical-gemm-family common` uses one matrix for every retained horizontal column. `--vertical-gemm-family k2-grouped` assigns one deterministic dense orthonormal matrix pair to each exact integer $K^2=k^2+l^2$ group on the square WVM grids. `--vertical-gemm-schedule` selects the serial baseline, setup-time weighted-static persistent workers, or allocation-free dynamic group claiming; `--vertical-gemm-outer-workers` sets the requested outer worker count. Outer schedules require `VECLIB_MAXIMUM_THREADS=1`, set before process startup, so nested BLAS threading is not silently compared with single-level scheduling. Inputs are already stored as column-major vertical-contiguous matrices in group order, so raw primitive timing excludes packing, horizontal ordering, allocation, and matrix preparation. Those exclusions are explicit experimental boundaries rather than assumptions that the work is free.

The issue #3/#5 sweep driver expands the ten named WVM workloads across all four vDSP strategies and the one-worker, performance-core, and total-core counts. Inspect the commands before starting the deliberately long full matrix:

```sh
python3 tools/run_float64_baseline_sweep.py --dry-run
python3 tools/run_float64_baseline_sweep.py
```

Use `--profiles`, `--strategies`, and `--workers` for a bounded exploratory increment. The driver writes only ignored local bundles and a resumable evidence manifest; publication remains an explicit reviewed step.

The issue #4 diagnostic driver holds the production guru64 representation fixed while screening FFTW planning, alignment, generated/imported wisdom, internal pthreads, persistent outer sharding, and selected hybrid topologies. Its default worker set resolves to 1, 2, 4, 8, performance-core count, and total-core count. `PATIENT` and `EXHAUSTIVE` use explicit per-plan-call budgets, so reaching a budget remains visible rather than silently presenting a truncated search as an unlimited plan:

```sh
python3 tools/run_fftw_strategy_sweep.py --dry-run
python3 tools/run_fftw_strategy_sweep.py
```

Use `--matrix planning` or `--matrix scheduling` for one half of the screen. Wisdom generation, wisdom import, final plan construction, empty outer-dispatch overhead, raw transforms, retained operations, memory, variability, and correctness remain distinct.

The final issue #4 increment adds a materially different provider-native order. Its screen compares WVM frequency-major interleaved plans with plane-major interleaved and split plans while holding the logical transform and retained radial disk fixed. A clean reference campaign then accepts an explicit survivor list, rotates candidate and profile order across three independently planned processes, and collects 21 steady-state samples per process across six production profiles:

```sh
python3 tools/run_fftw_native_order_sweep.py --phase screen --dry-run --allow-dirty-tree
python3 tools/run_fftw_native_order_sweep.py --phase screen
python3 tools/run_fftw_native_order_sweep.py --phase reference --candidate CANDIDATE_ID --dry-run --allow-dirty-tree
```

Reference collection intentionally requires at least one explicit `--candidate`, so the preliminary screen must be reviewed before launching the larger campaign. Every accepted run must come from a clean tree and a binary whose embedded commit and dirty flag match that tree. Raw FFT, order permutation, split conversion, WVM adapter, retained selection/embedding, complete retained operator, setup, placement, memory, and scheduler dispatch remain separately documented.

The issue #6 diagnostic driver compares native two-dimensional calls and packed-real separable row/column transforms under the persistent pool and GCD. Its default matrix uses the representative historical $256^2$ and $512^2$ three-field workloads and workers 1, 2, 4, 8, performance-core count, and total-core count:

```sh
python3 tools/run_vdsp_batch_sweep.py --dry-run
python3 tools/run_vdsp_batch_sweep.py
```

The authoritative primitive measurement is complete batch wall time. Separable row and column phases and an empty-dispatch scheduler diagnostic are also sampled, but they are non-additive because each phase has its own dispatch and cache state. No candidate performs an explicit column transpose; that stage is recorded as `elided`.

The issue #12 driver compares the partial-column-pruned candidate with a same-run full two-dimensional FFTW retained-operator reference. The default matrix covers representative 256-class and 512-class workloads with fields 1, 3, and 4 at one worker and the derived performance-core count. It estimates the harness's explicit peak memory, requires a clean source tree for evidence collection, and verifies that each result's embedded commit and dirty state match the executable source.

```sh
python3 tools/run_pruned_horizontal_sweep.py --dry-run --allow-dirty-tree
python3 tools/run_pruned_horizontal_sweep.py
```

The append-only outer-sharding increment keeps FFTW internal pthreads at one and sweeps persistent outer worker counts `1,4,8,performance`. The full retained reference and partial candidate use the same outer topology, including outer-sharded selection and embedding, so complete-operator ratios do not compare a parallel candidate with a serial adapter. Earlier internally threaded runs remain unchanged.

```bash
python3 tools/run_pruned_outer_sharding_sweep.py --dry-run --allow-dirty-tree
python3 tools/run_pruned_outer_sharding_sweep.py
```

This first candidate prunes only complete high-$k_x$ column transforms. It does not prune individual $k_y$ outputs within an active column, reduce first-pass row work, remove full-sized first-pass scratch, or test transform-internal transposition. A loss therefore rejects this concrete decomposition, not every theoretical subset FFT.

The first issue #8 driver screens the common deterministic DCT-II matrix family at the representative historical $256^2$, $N_z=65$, fields $=3$ and $512^2$, $N_z=129$, fields $=3$ workloads. It runs each Accelerate thread limit in a fresh process because `VECLIB_MAXIMUM_THREADS` is process state:

```sh
python3 tools/run_vertical_gemm_sweep.py --dry-run
python3 tools/run_vertical_gemm_sweep.py
python3 tools/run_vertical_gemm_sweep.py --family k2-grouped --dry-run
python3 tools/run_vertical_gemm_sweep.py --family k2-grouped
python3 tools/run_vertical_gemm_sweep.py --family k2-grouped --thread-limits 1 --schedules serial,outer-static,outer-dynamic --outer-workers 4,8,performance,total --dry-run
python3 tools/run_vertical_gemm_sweep.py --family k2-grouped --thread-limits 1 --topologies serial,outer-static:12,outer-dynamic:16 --profiles wvm-current-256-nz129-f1 wvm-current-512-nz257-f4 --dry-run
```

These bounded increments publish primitive complex and split-real GEMM times, forward and inverse directions, group distributions and call counts, matrix and scheduler setup, explicit persistent memory, opaque thread-stack memory, an estimated explicit allocation high-water mark, empty-dispatch overhead, correctness, confidence intervals, and variability. The grouped/common comparison measures the combined BLAS-call and small-GEMM efficiency penalty. The scheduling comparison measures complete group-loop wall time against a same-commit one-thread serial baseline. `--topologies` selects exact schedule/worker finalists instead of forming a cross-product. Before launching a K²-grouped process, the driver estimates the benchmark's explicit peak from its matrix family, provider buffers, external operands, and inspected outputs; it rejects a case above half of physical memory unless explicitly overridden. Evidence collection also requires a clean source tree and verifies that the binary's embedded commit and dirty state match it; `--allow-dirty-tree` is reserved for explicitly exploratory runs. The installed public Accelerate headers expose no variable-size grouped GEMM batch API, and an exact setup-time equality scan records whether adjacent matrix groups could be consolidated without moving data. None of these issue #8 measurements charges the cost of creating group order. Third-party grouped APIs, blocking, and the packing crossover owned by issue #13 remain open.

`--kernel ordering-packing` began issue #13 with the explicit MATLAB-style baseline. It gathers the retained logical modes directly from WVM frequency-major interleaved half-spectrum storage into the final interleaved or split vertical input, thereby materializing radial/K²-group order and vertical-contiguous columns. The reverse path executes the inverse vertical projection, zeros a full WVM half-spectrum, scatters the retained coefficients, and repairs the stored Hermitian boundary. Movement-only, raw vertical GEMM, one-shot movement-plus-GEMM, and uninstrumented reuse sequences are timed separately. Reuse counts 2, 4, and 8 compare boundary movement on every use with one boundary movement around a persistent compact representation.

The next increment adds a no-reorder competitor to the same run. It reads retained frequency blocks directly from WVM storage, issues one complex GEMM per retained frequency with all fields as columns, keeps modal coefficients in a persistent zero-padded frequency-major representation, and reconstructs directly into persistent zero-padded WVM storage. No radial gather, transpose, split conversion, or reverse scatter is executed. Initial zero fills are setup-only, and Hermitian-boundary repair is fused into the direct kernel. The report preserves separate primitive, movement, one-shot, reuse, setup, memory, byte, and GEMM-call-count records for both algorithms. All timed steady-state paths are out-of-place and allocation-free. Raw FFT execution, modal physics, nonlinear flux, provider-native horizontal fusion, tiled pack-and-GEMM, and 512-class scaling remain excluded from this bounded comparison.

`--kernel spectral-boundary` is the first issue #13 increment that includes horizontal execution. Each isolated run selects one complete representation policy: WVM direct/no-reorder, historical WVM gather-to-split, partial-column-pruned compact interleaved, plane-major fused retained split, or a plane-major retained index view consumed by strided vertical GEMV. The logical forward boundary is horizontal transform and retention followed by the grouped vertical projection; the inverse is the grouped vertical reconstruction followed by horizontal embedding and inverse transformation. Modal work and the nonlinear flux calculation remain excluded.

Raw horizontal transform, raw vertical MM, retention/conversion/packing, inverse zero fill and embedding, setup, memory, and uninstrumented composed totals are all reported separately. Component medians remain diagnostic rather than additive. Multidimensional FFTW inverse transforms may destroy their spectrum input, so WVM-direct and plane-major-view totals rebuild their full zero-padded inverse view on every timed call. This is a real composed-boundary cost; the earlier horizontal-only retained-view result correctly excluded it by taking a ready disposable view as its input. Every policy is out-of-place at the transform boundary and performs zero steady-state allocations.

`--kernel spectral-pipeline` is the bounded issue #9 synthetic round trip. It compares the WVM direct/no-reorder production-layout control with the issue #13 `plane-major-fused-split--outer-dynamic-16` survivor. Each timed total starts with ready real fields and executes horizontal forward and radial retention, vertically truncated forward projection, `real-diagonal-mode-keyed-v1` modal work, inverse vertical reconstruction, horizontal embedding, and inverse FFT. The modal operator precomputes one real bounded weight for every logical `(k,l,j,field)` value during setup and applies the same out-of-place scaling to interleaved WVM modal views or compact split arrays. It preserves Hermitian symmetry, exercises one real downstream modal-data pass, and is not a surrogate implementation of WVM's nonlinear flux.

The pipeline report retains independent raw forward/inverse FFT, raw forward/inverse vertical MM, movement, modal-work, setup, memory, placement, liveness, and authoritative uninstrumented round-trip timings. The component medians are diagnostic and need not sum to the total. Both graphs are out-of-place at the transform boundary, explicitly rebuild any disposable zero-padded inverse spectrum, and allocate no memory in steady state. A canonical compact-interleaved graph supplies the complete mode-keyed oracle, with independent scalar vertical probes, at a Float64 tolerance of `1e-12`.

```sh
python3 tools/run_ordering_packing_sweep.py --dry-run --allow-dirty-tree
python3 tools/run_ordering_packing_sweep.py
python3 tools/run_retained_horizontal_closeout_sweep.py --phase screen --dry-run --allow-dirty-tree
python3 tools/run_retained_horizontal_closeout_sweep.py --phase screen
python3 tools/run_spectral_boundary_sweep.py --phase screen --dry-run --allow-dirty-tree
python3 tools/run_spectral_boundary_sweep.py --phase screen
python3 tools/run_spectral_boundary_sweep.py --phase reference --screen-analysis results/local/<screen>/analysis.json --dry-run
```

The bounded screen crosses the six issue #7 production profiles with dynamic-16 and static-12 issue #8 schedulers. A non-control policy advances only when its complete-matrix geometric ratio to the workload-direction best is at most 1.05 and it wins at least one cell; the best fused-split representation bridge may advance at 1.10. Dynamic-16 WVM direct and packed-split controls remain at reference depth. Because issue #9 consumes a complete bidirectional policy, its reference selection pairs the forward and inverse boundary medians within each workload. It keeps at most three candidates within 3% of the best geometric paired ratio and rejects any candidate whose paired cost is more than 10% behind the best policy for a workload.

The issue #9 driver holds the selected issue #13 topology fixed rather than reopening the provider search:

```sh
python3 tools/run_spectral_pipeline_sweep.py --phase screen --dry-run --allow-dirty-tree
python3 tools/run_spectral_pipeline_sweep.py --phase screen
python3 tools/run_spectral_pipeline_sweep.py --phase reference \
  --screen-analysis results/local/<screen>/analysis.json
```

The one-round screen advances to the three-round reference campaign only when fused split is at least 5% faster geometrically than WVM direct, no workload is more than 10% slower, and every correctness metric passes. Reference-depth M4 adoption statistics require at least 10% geometric improvement, no workload regression above 3%, and a stratified 95% bootstrap interval excluding a tie. Cross-Mac replication remains a separate issue #11 gate.

The append-only nonhydrostatic extension reruns a same-commit fields=4 cohort at `256²/Nz=129`, `512²/Nz=129`, `512²/Nz=257`, and `1024²/Nz=129`. Its screen is a correctness and memory-capability gate rather than a performance filter: if both graphs complete within the 50%-of-physical-memory preflight rule and all metrics remain within `1e-12`, all four workloads advance to three reference rounds even when the screen is a tie or regression.

The fields=4 decision selects one production algorithm rather than dispatching by size. Reference evidence requires at least 10% geometric improvement, a confidence interval excluding a tie, and no workload regression above 5%; the original 3% threshold remains published as a diagnostic. A bounded small-workload exception is documented rather than converted into a second production path.

```sh
python3 tools/run_spectral_pipeline_large_f4_sweep.py --phase screen --dry-run --allow-dirty-tree
python3 tools/run_spectral_pipeline_large_f4_sweep.py --phase screen
python3 tools/run_spectral_pipeline_large_f4_sweep.py --phase reference \
  --screen-analysis results/local/<large-f4-screen>/analysis.json
```

Pipeline memory reports distinguish explicit algorithm-resident storage, benchmark/oracle overhead, the conservative explicit process-peak estimate, and the isolated process high-water measurement. Lower algorithm memory is a result: a workload that only one graph can execute safely is published as a capacity outcome rather than forced through swap. The `512²/Nz=513/fields=4` and `1024²/Nz=257/fields=4` capacity cases remain deferred until benchmark-only storage can be reduced without hiding production-resident buffers.

Issue #16 is a non-blocking follow-up to the M4 decision. Its bounded screen compares the published plane-major fused-split winner with one uniform outer-12 partial-column-pruned implementation that holds a single half-spectrum plane per worker and writes retained values directly to compact radial split storage:

```sh
python3 tools/run_streaming_pruned_pipeline_sweep.py --dry-run --allow-dirty-tree
python3 tools/run_streaming_pruned_pipeline_sweep.py
```

The screen covers only the fields=4 `256²/Nz=129`, `512²/Nz=257`, and `1024²/Nz=129` profiles. It advances only for a correct large-case time improvement or a material algorithm-resident memory reduction within the declared total-time bound. FFTW++ implicit and hybrid convolution work belongs to issue #17 and is not part of this runner.

Issue #17 adds a separate optional kernel for a four-input synthetic quadratic convolution. It does not implement the WVM nonlinear flux. The caller-visible boundary begins and ends with compact radial Hermitian spectra; the explicit FFTW oracle and FFTW++ centered/Hermitian implicit-hybrid candidate therefore both include their required embedding, preservation, and retention work. The initial single-thread screen uses 256² and 512² grids with 4 and 12 products:

```sh
build/issue17/skbench run --kernel dealiased-convolution --profile wvm-current-256-nz129-f4 --convolution-products 4
build/issue17/skbench run --kernel dealiased-convolution --profile wvm-current-512-nz257-f4 --convolution-products 12
```

FFTW++ exposes the transform-multiply-transform stage as one fused algorithm, so the benchmark does not fabricate standalone FFTW++ FFT timings. The explicit oracle continues to report inverse FFT, pointwise multiplication, and forward FFT components. A direct 12-output FFTW++ application is excluded after a reproducible upstream optimizer crash for more outputs than inputs; the correct 12-product candidate uses three persistent four-output applications and includes the input-restoration cost. This result can motivate a later threaded or model-expression experiment, but it cannot support a conclusion about the complete nonlinear flux.

The WVM-derived follow-up keeps that early evidence intact and adds a closer horizontal operator. The nonhydrostatic compiled WVM source reconstructs three advectors and, for each of four targets, evaluates

$$
-(U q_x + V q_y + W q_z).
$$

The benchmark therefore starts with 15 ready compact spectra—$$U$$, $$V$$, $$W$$, and three derivatives for each target—and returns four compact advective-flux spectra. Vertical reconstruction, phase evolution, coefficient projection, and the rest of the nonlinear flux remain excluded. It compares a low-memory serial stream, a 15-input/4-output all-target application, and matched persistent four-target schedules for explicit FFTW and FFTW++:

```sh
build/issue17/skbench run --kernel dealiased-convolution \
  --convolution-map wvm-advection \
  --profile wvm-current-512-nz257-f4

python3 tools/run_dealiased_convolution_wvm_sweep.py --dry-run --allow-dirty-tree
python3 tools/run_dealiased_convolution_wvm_sweep.py
```

The fixed screen spans 256², 512², and 1024² with four target workers. Its matched baseline reconstructs the shared advectors once before dispatching four explicit target calculations. `--convolution-centered-m N` is a feasibility/tuning control for the all-target FFTW++ topology; it is not a workload-size dispatch mechanism. The default all-target policy uses `m=N`, while the low-memory and parallel-target candidates retain optimizer-selected implicit/hybrid parameters. Correctness is checked before timing conclusions, including FFTW++ residue choices that may reorder application inputs.

The reference campaign freezes only the two four-worker finalists and runs each in its own process. `--convolution-candidate explicit-parallel` and `--convolution-candidate fftwpp-parallel` suppress all non-finalist providers; an independent explicit oracle is destroyed and FFTW wisdom is cleared before the selected candidate is planned. Three rounds rotate both candidate and profile order, with three warmups and 21 samples per process:

```sh
python3 tools/run_dealiased_convolution_wvm_reference.py --dry-run --allow-dirty-tree
python3 tools/run_dealiased_convolution_wvm_reference.py
```

The runner first executes the same-commit allocator-interposer test. Reference status records that the complete controlled campaign is suitable for later decisions, independently of which candidate wins. Its separate preregistered adoption gate requires correctness within $$10^{-12}$$, valid out-of-place adapter and input-lifetime contracts, at least 10% geometric time improvement, no profile above 1.03× the control, a stratified paired-bootstrap 95% interval excluding a tie, and at least 20% geometric algorithm-resident-memory reduction. Passing the adoption gate would support carrying the candidate into a composed-operator decision; the campaign still does not measure the complete nonlinear flux or authorize a general-Mac default.

Issue #18 composes the two fixed #17 finalists with vertical reconstruction and projection without implementing the remaining nonlinear-flux calculation. Its input contains 15 ready retained and vertically truncated modal fields. An inverse-only split K²-grouped GEMM reconstructs those fields across physical levels; one reusable level adapter feeds the explicit or FFTW++ four-target horizontal operator; a forward-only split GEMM projects the four outputs to $$N_j = \lfloor 2(N_z - 1)/3 \rfloor$$ modes. Directional vertical providers deliberately omit unused opposite-direction matrices and operands so the memory comparison does not count avoidable benchmark storage.

The first composition increment isolates each candidate at `256²/Nz=129/fields=4`, reports raw inverse and forward vertical GEMM, one-level horizontal execution, all-level movement, the vertically batched horizontal stage, and the authoritative uninstrumented total, and verifies zero steady-state allocations:

```sh
build/issue18/skbench run --kernel vertically-batched-advection \
  --profile wvm-current-256-nz129-f4 \
  --vertical-gemm-family k2-grouped \
  --vertical-gemm-schedule outer-dynamic \
  --vertical-gemm-outer-workers 12 \
  --convolution-candidate explicit-parallel

python3 tools/run_vertically_batched_advection_screen.py --dry-run --allow-dirty-tree
python3 tools/run_vertically_batched_advection_screen.py
```

This preliminary screen recommends reference depth when both paths remain correct and FFTW++ either reaches 0.98× the composed time or retains at most 0.80× the algorithm-resident memory without exceeding 1.05× the time. That is only a continuation gate. The eventual 0.9000 adoption threshold applies to the multi-workload composed reference campaign.

The reference campaign keeps both policies fixed and adds the four-field production matrix: `256²/Nz=129`, `512²/Nz=257`, `512²/Nz=513`, and `1024²/Nz=129`. Each finalist runs alone for three rotated rounds with three warmups and 21 samples. A conservative per-process memory preflight runs before any benchmark; a case that exceeds 75% of physical memory is reported as a capacity exclusion rather than forced through swap.

```sh
python3 tools/run_vertically_batched_advection_reference.py --dry-run --allow-dirty-tree
python3 tools/run_vertically_batched_advection_reference.py
```

The composed adoption gate requires a complete matched matrix, correctness within $$10^{-12}$$, valid out-of-place input-preservation contracts, at least 10% geometric time improvement, no workload above 1.03× the control, a stratified paired-bootstrap 95% interval excluding a tie, and at least 20% geometric algorithm-resident-memory reduction. Primitive vertical GEMM, one-level horizontal execution, level movement, setup, memory, and the authoritative uninstrumented total remain separate. No size-dependent dispatch is permitted, and even a passing M4 result still requires cross-Mac replication and does not constitute a complete nonlinear-flux benchmark.

The append-only locality follow-up keeps one FFT half-spectrum plane per worker but replaces page-strided direct split access with a bounded plane-major compact tile and a 32-mode cache-blocked transpose. It screens fixed tile widths 4, 8, and 16 against both the original tile-1 streaming graph and the same-commit fused-split control:

```sh
python3 tools/run_streaming_pruned_locality_sweep.py --dry-run --allow-dirty-tree
python3 tools/run_streaming_pruned_locality_sweep.py
```

The selection is uniform across all three workloads. A tiled candidate must retain at least a 10% geometric algorithm-resident-memory reduction relative to fused split and materially improve the two-large-case timing relative to direct streaming; the runner never defines a size-dependent dispatch rule. Individual exploratory runs select a tile with `--streaming-tile-width 4|8|16`.

## v1 close-out reference campaigns

The v1 close-out freezes the production FFTW contract, the two vertical-scheduling finalists, and one deep-vertical pipeline robustness case. These are independent reference campaigns rather than one blended benchmark, so raw FFT, raw vertical GEMM, and the complete synthetic pipeline remain separately reusable:

```sh
python3 tools/run_fftw_production_reference.py --dry-run --allow-dirty-tree
python3 tools/run_vertical_gemm_reference.py --dry-run --allow-dirty-tree
python3 tools/run_spectral_pipeline_deep_vertical_reference.py --dry-run --allow-dirty-tree
```

The FFTW campaign covers all ten v1 profiles with the exact WVM-order guru64, `FFTW_MEASURE | FFTW_UNALIGNED`, cold-wisdom, 12-internal-worker contract. The vertical campaign compares fixed outer-dynamic-16 and outer-static-12 K²-grouped schedules for complex zgemm and split dgemm over the same ten profiles. The deep pipeline campaign changes only vertical depth to `512²/Nz=513/fields=4` and compares WVM direct with the already selected uniform streaming tile-16 graph.

Pipeline correctness constructs independent mode-keyed oracle and diagnostic arrays and counts them in the reported setup/high-water capacity evidence. Before the authoritative uninstrumented total, those correctness-only buffers are explicitly released for every compared graph. This prevents the validation harness from creating unequal steady-state memory pressure while preserving an honest record of the complete benchmark process peak.

Issue #13 closes from the accumulated evidence rather than another algorithm screen. The existing tile-16 reference runs are associated with both the streaming experiment and the ordering/packing experiment; their immutable JSON and CSV artifacts are neither copied nor modified. The close-out records direct WVM order, MATLAB gather/radial packing, provider-order matrices, fused conversion, tiled staging, and persistent compact storage as algorithmic alternatives—not mathematical layout requirements.

## Cross-Mac portability campaign

Issue #11 freezes the WVM-order direct graph and the streaming-pruned fixed tile-16 graph, then calibrates only machine-local scheduling. Calibration compares horizontal outer sharding at the performance-core and total-physical-core counts with vertical dynamic-total and weighted-static-performance scheduling. It selects one topology independently for each graph, keeps FFTW internal workers and `VECLIB_MAXIMUM_THREADS` at one, and never contributes calibration samples to reference inference:

```sh
python3 tools/run_cross_mac_spectral_reference.py --phase calibration --dry-run --allow-dirty-tree
python3 tools/run_cross_mac_spectral_reference.py --phase calibration --output results/local/issue11-calibration
python3 tools/run_cross_mac_spectral_reference.py --phase reference --calibration-analysis results/local/issue11-calibration/analysis.json --output results/local/issue11-reference
python3 tools/run_cross_mac_spectral_reference.py --phase combine --machine-analysis results/local/issue11-reference-lyra/analysis.json --machine-analysis results/local/issue11-reference-matilda/analysis.json --output results/local/issue11-cross-mac-synthesis.json
```

The reference phase applies the selected topology uniformly to `256²/Nz=129/F4`, `512²/Nz=257/F4`, `1024²/Nz=129/F4`, and the deep `512²/Nz=513/F4` case. It begins with three balanced rotated rounds, three warmups, and 21 samples per isolated timing process. A preregistered variability or decision-boundary trigger adds exactly two complete rounds; otherwise the campaign stops after three. Memory evidence comes from separate one-sample processes and never enters timing aggregates. A workload above 75% of physical memory is preserved as an explicit capacity exclusion rather than forced through swap or replaced with another size.

Run the same committed source and campaign protocol independently on each machine. Worker counts are topology-derived, so the M4 `outer-12` and dynamic-16 literals become performance-core and total-core policies on other Apple-silicon machines. The resulting evidence is scoped to the machines and toolchains actually tested and does not support size-dependent dispatch or a general-Mac claim.

Only compact reviewed artifacts belong under `results/published/`. New immutable bundles use `results/published/runs/<run-id>/result.json` and `samples.csv`; `results/published/catalog.json` records their hashes, issue-level experiment associations, publication status, and supersession relationships. When a sweep manifest supplies a stable `incrementId`, publication preserves it so an experiment page can separate successive methods without rewriting earlier evidence. The original M4 bundle remains byte-identical at its legacy paths.

## Timing boundaries

Every result distinguishes:

- provider setup and planning;
- raw provider-native forward and inverse FFT or vertical GEMM calls;
- representation packing, conversion, and permutation;
- full-spectrum WVM-compatible adapters;
- horizontal retention and embedding;
- separately measured, uninstrumented retained-operator totals.

The component medians are diagnostic and are not expected to add to the uninstrumented total. Timed steady-state paths reuse all buffers and worker threads.

See [the benchmark contract](docs/benchmark-contract.md) and [the v1 JSON schema](schema/spectral-kernel-benchmark-v1.schema.json) for the exact mathematical and reporting definitions.

## Results dashboard

GitHub Pages presents the compact bundles under `results/published/` as a static dashboard. It shows headline and component timings, correctness and setup details, environment metadata, a run archive, and direct JSON/CSV downloads. The dashboard is generated from committed results; deployment does not execute benchmarks.

Validate the publication catalog without building the site:

```sh
python3 tools/validate_publication.py
```

Build the site locally with:

```sh
python3 tools/build_site.py --output _site
```

The generated archive provides permanent `/runs/<run-id>/` pages, accumulating `/experiments/<experiment-id>/` pages, the shared `/methods/operators-and-representations/` methodology, and the evolving `/decisions/v1/` synthesis. Preliminary, superseded, withdrawn, negative, and unsupported evidence remains visible; only clean, passing `reference` runs contribute to adoption statistics.

The `Publish benchmark dashboard` workflow validates, rebuilds, and deploys the site when dashboard sources or published bundles change on `main`. It only reads committed files and never executes benchmarks. GitHub Pages is configured to use **GitHub Actions** as its source. The project site is [jeffreyearly.github.io/spectral-kernel-benchmarks](https://jeffreyearly.github.io/spectral-kernel-benchmarks/); the accumulating FFTW strategy screen has its own [issue #4 experiment page](https://jeffreyearly.github.io/spectral-kernel-benchmarks/experiments/issue-004-fftw-strategy-sweep/).

## Provenance

The first workload and provider configuration reproduce the FFT portions of [`wave-vortex-model` issue #129](https://github.com/JeffreyEarly/wave-vortex-model/issues/129) while correcting an important measurement ambiguity: the historical screen measured all alternative-provider conversion and worker costs together. This repository preserves that complete adapter result and additionally exposes the raw native FFT call.
