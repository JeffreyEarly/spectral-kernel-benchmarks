# Spectral Kernel Benchmarks

This repository is an independent C++20/CMake laboratory for finding the fastest correct antialiased spectral-operator implementation on Apple Silicon. It publishes primitive FFT and matrix-multiplication performance separately from representation adapters, data movement, and complete operator pipelines.

The current vertical slice compares the production WVM FFTW 3.3.11 two-dimensional real transform with Accelerate/vDSP for the $256 \times 256$, $N_z=65$, fields $=3$ workload. FFTW writes WVM's frequency-major output directly; vDSP uses native packed split-complex storage and reports packing, conversion, permutation, raw transform, and complete retained-operator costs independently.

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
build/release/skbench compare --input results/local/<run>.csv
```

`smoke` is a small correctness and contract exercise. `quick` is the first production workload: 195 independent $256 \times 256$ real planes with the exact WVM FFTW guru64 input and output strides. `exhaustive` currently supplies the initial $512 \times 512$, $N_z=129$, fields $=4$ shape; later issues will expand it into a sweep.

`validate` checks impulse, sinusoid, deterministic random, DC, and Nyquist fixtures against an independent direct-DFT oracle. It also checks full FFT conformance, inverse normalization, retained-mode values, representation round trips, and permutation invariance.

`run` writes a versioned JSON manifest/report and a sample-level CSV file. Scratch runs go to `results/local/` and are ignored. Only compact reviewed artifacts belong under `results/published/`.

## Timing boundaries

Every result distinguishes:

- provider setup and planning;
- raw provider-native forward and inverse FFT calls;
- representation packing, conversion, and permutation;
- full-spectrum WVM-compatible adapters;
- horizontal retention and embedding;
- separately measured, uninstrumented retained-operator totals.

The component medians are diagnostic and are not expected to add to the uninstrumented total. Timed steady-state paths reuse all buffers and worker threads.

See [the benchmark contract](docs/benchmark-contract.md) and [the v1 JSON schema](schema/spectral-kernel-benchmark-v1.schema.json) for the exact mathematical and reporting definitions.

## Results dashboard

GitHub Pages presents the compact bundles under `results/published/` as a static dashboard. It shows headline and component timings, correctness and setup details, environment metadata, a run archive, and direct JSON/CSV downloads. The dashboard is generated from committed results; deployment does not execute benchmarks.

Build it locally with:

```sh
python3 tools/build_site.py --output _site
```

The `Publish benchmark dashboard` workflow rebuilds and deploys the site when dashboard sources or published bundles change on `main`. GitHub Pages is configured to use **GitHub Actions** as its source. The project site is `https://jeffreyearly.github.io/spectral-kernel-benchmarks/`.

## Provenance

The first workload and provider configuration reproduce the FFT portions of [`wave-vortex-model` issue #129](https://github.com/JeffreyEarly/wave-vortex-model/issues/129) while correcting an important measurement ambiguity: the historical screen measured all alternative-provider conversion and worker costs together. This repository preserves that complete adapter result and additionally exposes the raw native FFT call.
