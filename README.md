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

## Commands

```sh
build/release/skbench list
build/release/skbench validate --profile smoke
build/release/skbench run --profile quick
build/release/skbench run --profile wvm-historical-256-nz65-f3 --providers fftw --fftw-planning measure --fftw-alignment aligned --fftw-internal-workers 12 --fftw-outer-workers 1
build/release/skbench run --profile wvm-historical-256-nz65-f4 --vdsp-strategy out-of-place-explicit-scratch --workers 12
build/release/skbench run --profile wvm-historical-256-nz65-f3 --vdsp-batch-strategy separable-gcd --workers 12
build/release/skbench run --kernel pruned-horizontal --providers fftw --fftw-planning measure --fftw-internal-workers 12 --profile wvm-historical-256-nz65-f3
VECLIB_MAXIMUM_THREADS=12 build/release/skbench run --kernel vertical-gemm --profile wvm-historical-256-nz65-f3
VECLIB_MAXIMUM_THREADS=12 build/release/skbench run --kernel vertical-gemm --vertical-gemm-family k2-grouped --profile wvm-historical-256-nz65-f3
VECLIB_MAXIMUM_THREADS=1 build/release/skbench run --kernel vertical-gemm --vertical-gemm-family k2-grouped --vertical-gemm-schedule outer-dynamic --vertical-gemm-outer-workers 12 --profile wvm-historical-256-nz65-f3
VECLIB_MAXIMUM_THREADS=1 build/release/skbench run --kernel ordering-packing --vertical-gemm-family k2-grouped --vertical-gemm-schedule outer-dynamic --vertical-gemm-outer-workers 16 --profile wvm-historical-256-nz65-f3
build/release/skbench compare --input results/local/<run>.csv
```

`smoke` is a small correctness and contract exercise. `quick` is the first production workload: 195 independent $256 \times 256$ real planes with the exact WVM FFTW guru64 input and output strides. `exhaustive` supplies the historical $512 \times 512$, $N_z=129$, fields $=4$ shape. `skbench list` also exposes the complete issue #3/#5 matrix as named `wvm-historical-*` and `wvm-current-*` profiles.

`validate` checks impulse, sinusoid, deterministic random, DC, and Nyquist fixtures against an independent direct-DFT oracle. It exercises all four native vDSP placement/scratch strategies, direct persistent-pool and GCD scheduling, and the separable packed-real candidates. It also checks full FFT conformance, inverse normalization, retained-mode values, representation round trips, and permutation invariance.

`run` writes a versioned JSON manifest/report and a sample-level CSV file. `--providers fftw` omits the unchanged vDSP provider during FFTW strategy screens. `--fftw-planning`, `--fftw-alignment`, and `--fftw-wisdom` select the planner contract; `--fftw-internal-workers` and `--fftw-outer-workers` distinguish FFTW pthread parallelism from persistent outer batch sharding. `--fftw-planning-time-limit` applies FFTW's per-plan-call limit and records whether the observed planning interval exhausted that budget. `--vdsp-strategy` selects `in-place`, `in-place-explicit-scratch`, `out-of-place`, or `out-of-place-explicit-scratch`. `--vdsp-batch-strategy` independently selects `direct-persistent`, `direct-gcd`, `separable-persistent`, or `separable-gcd`; the separable prototype currently supports in-place placement. Scratch runs go to `results/local/` and are ignored. Every new result identifies its numeric type and records the forward and inverse provider-native and adapter execution contracts, including in-place/out-of-place placement, destructive inputs, preservation policy, physical extents, padding, strides, alignment, aliasing, and reusable work memory.

`--kernel pruned-horizontal` selects the first issue #12 feasibility candidate. The matched reference performs FFTW's optimized full two-dimensional real transform and then selects the radial two-thirds mode set. The candidate performs all real-row transforms but executes complex column transforms only for the contiguous nonnegative-$k_x$ band that can intersect the retained radial disk. It writes compact mode-keyed output without materializing a completed WVM-order half-spectrum. It still requires a full-sized plane-major row-spectrum scratch buffer, which is reported explicitly. Forward/inverse row stages, selected-column stages, retention/embedding, setup, memory, and complete uninstrumented retained-operator totals remain separate. Both complete algorithms are out-of-place; the candidate's selected column transforms operate in-place only inside its private scratch.

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

```sh
python3 tools/run_ordering_packing_sweep.py --dry-run --allow-dirty-tree
python3 tools/run_ordering_packing_sweep.py
```

Only compact reviewed artifacts belong under `results/published/`. New immutable bundles use `results/published/runs/<run-id>/result.json` and `samples.csv`; `results/published/catalog.json` records their hashes, issue-level experiment associations, publication status, and supersession relationships. The original M4 bundle remains byte-identical at its legacy paths.

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
