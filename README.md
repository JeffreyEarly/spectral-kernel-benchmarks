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
VECLIB_MAXIMUM_THREADS=12 build/release/skbench run --kernel vertical-gemm --profile wvm-historical-256-nz65-f3
VECLIB_MAXIMUM_THREADS=12 build/release/skbench run --kernel vertical-gemm --vertical-gemm-family k2-grouped --profile wvm-historical-256-nz65-f3
build/release/skbench compare --input results/local/<run>.csv
```

`smoke` is a small correctness and contract exercise. `quick` is the first production workload: 195 independent $256 \times 256$ real planes with the exact WVM FFTW guru64 input and output strides. `exhaustive` supplies the historical $512 \times 512$, $N_z=129$, fields $=4$ shape. `skbench list` also exposes the complete issue #3/#5 matrix as named `wvm-historical-*` and `wvm-current-*` profiles.

`validate` checks impulse, sinusoid, deterministic random, DC, and Nyquist fixtures against an independent direct-DFT oracle. It exercises all four native vDSP placement/scratch strategies, direct persistent-pool and GCD scheduling, and the separable packed-real candidates. It also checks full FFT conformance, inverse normalization, retained-mode values, representation round trips, and permutation invariance.

`run` writes a versioned JSON manifest/report and a sample-level CSV file. `--providers fftw` omits the unchanged vDSP provider during FFTW strategy screens. `--fftw-planning`, `--fftw-alignment`, and `--fftw-wisdom` select the planner contract; `--fftw-internal-workers` and `--fftw-outer-workers` distinguish FFTW pthread parallelism from persistent outer batch sharding. `--fftw-planning-time-limit` applies FFTW's per-plan-call limit and records whether the observed planning interval exhausted that budget. `--vdsp-strategy` selects `in-place`, `in-place-explicit-scratch`, `out-of-place`, or `out-of-place-explicit-scratch`. `--vdsp-batch-strategy` independently selects `direct-persistent`, `direct-gcd`, `separable-persistent`, or `separable-gcd`; the separable prototype currently supports in-place placement. Scratch runs go to `results/local/` and are ignored. Every new result identifies its numeric type and records the forward and inverse provider-native and adapter execution contracts, including in-place/out-of-place placement, destructive inputs, preservation policy, physical extents, padding, strides, alignment, aliasing, and reusable work memory.

`--kernel vertical-gemm` selects the bounded issue #8 Float64 vertical-projection benchmark. It compares Accelerate complex `zgemm`, with real projection matrices expanded to complex during setup, against two Accelerate real `dgemm` loops over persistent split real and imaginary arrays. Both paths are out-of-place. `--vertical-gemm-family common` uses one matrix for every retained horizontal column. `--vertical-gemm-family k2-grouped` assigns one deterministic dense orthonormal matrix pair to each exact integer $K^2=k^2+l^2$ group on the square WVM grids. Inputs are already stored as column-major vertical-contiguous matrices in group order, so raw primitive timing excludes packing, horizontal ordering, allocation, and matrix preparation. Those exclusions are explicit experimental boundaries rather than assumptions that the work is free.

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

The first issue #8 driver screens the common deterministic DCT-II matrix family at the representative historical $256^2$, $N_z=65$, fields $=3$ and $512^2$, $N_z=129$, fields $=3$ workloads. It runs each Accelerate thread limit in a fresh process because `VECLIB_MAXIMUM_THREADS` is process state:

```sh
python3 tools/run_vertical_gemm_sweep.py --dry-run
python3 tools/run_vertical_gemm_sweep.py
python3 tools/run_vertical_gemm_sweep.py --family k2-grouped --dry-run
python3 tools/run_vertical_gemm_sweep.py --family k2-grouped
```

These bounded increments publish primitive complex and split-real GEMM times, forward and inverse directions, group distributions and call counts, matrix setup, explicit persistent memory, correctness, confidence intervals, and variability. The grouped/common comparison measures the combined BLAS-call and small-GEMM efficiency penalty. It does not charge the cost of creating group order. Fields 1/4, $N_z=257$, alternative grouped or batched APIs, blocking, and the packing crossover owned by issue #13 remain open.

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
