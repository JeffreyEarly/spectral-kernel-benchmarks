# WVM compiled-core integration handoff

Issue #23 freezes benchmark implementations and exposes their hardware worker controls. It does not define a production API or establish production-optimal worker settings. The subsequent WaveVortexModel work should preserve the mathematical boundaries below, choose one caller representation explicitly, and retune through the integrated entry point.

## General-stratification boundaries

Both general-stratification implementations consume the same 15 retained, vertically truncated Float64 complex modal inputs and produce the same four retained modal targets. They execute exact WVM F/G vertical operator families, horizontal radial two-thirds retention, four streamed pointwise advection expressions, and the corresponding inverse and forward transforms. Phase and coefficient construction, MATLAB dispatch, WVM state management, time integration, diagnostics, and unrelated nonlinear-flux terms remain outside this boundary.

`wvm-native-optimized-v1` accepts and returns WVM-native frequency-major complex-interleaved family arrays. FFTW plans operate directly on immutable family-strided input views and final family-strided output views. Reconstructed spectra may be destroyed by their single-use inverse transforms; caller-owned modal inputs may not be modified. Persistent state contains FFTW plans, vertical operator matrices and family partitions, retained-mode metadata, thread executors, and output storage. Scratch state contains the streamed seven-real-volume lifetime, vertical reconstruction/projection work, and per-worker horizontal transform storage. No representation bridge or compact caller state is required.

`compact-general-fused-views-v1` accepts and returns compact radial split-complex family arrays. Partial-column-pruned tile-16 horizontal transforms read and write the split F/G family views directly. Vertical projection uses the frozen split-real grouped GEMM implementation. Persistent state contains compact modal arrays, prepared split vertical matrices and family views, retained-mode metadata, FFTW plans, and thread executors. Scratch state contains the same seven-real-volume lifetime plus per-worker half-spectrum planes and compact tiles. A MATLAB caller that retains WVM-native arrays must explicitly pay for a representation bridge; the benchmark does not hide that cost or relabel this path as WVM-native.

The general compiled entry point should behave conceptually as an allocation-free operation of the form

```text
execute_general(inputs, outputs, prepared_operators, workspace, worker_policy)
```

where `inputs` and `outputs` use exactly one declared representation, `prepared_operators` and `workspace` persist across calls, and aliasing is rejected unless a later integration issue proves it safe.

## Constant-stratification boundary

`compact-constant-type1-v1` consumes the retained constant-stratification modal state and produces the exact WVM nonlinear-flux modal targets. It uses compact radial split storage, partial-column-pruned tile-16 horizontal transforms, retained-row complex DCT-I/DST-I channels, the frozen normalization, exact coefficient formulas, and streamed four-target pointwise work. It remains a separate mathematical API from the general 15-to-4 operator and must not share performance scores with it.

Persistent state contains compact modal arrays, type-I plans, coefficient metadata, retained-mode maps, horizontal plans, and thread executors. Scratch state contains the streamed physical volumes, type-I work arrays, and per-worker horizontal transform storage. The production operation should be out-of-place, preserve caller input, and perform zero application allocations after setup.

## Hardware controls and production tuning

Only the following stage-local counts are portable tuning controls:

- horizontal FFT outer workers;
- general vertical outer-dynamic workers;
- constant type-I internal workers; and
- pointwise spatial-static workers.

FFTW internal workers remain one while horizontal outer sharding is active. Accelerate requests one thread while general vertical outer scheduling is active. Nested worker pools may not execute concurrently. One worker tuple must cover every supported size for an implementation; size-dependent dispatch is not part of the frozen contract.

Issue #23 manifests provide benchmark-local provisional defaults and a topology-derived candidate envelope. WVM integration must rerun that bounded envelope through the actual compiled entry point with MATLAB loaded, persistent model buffers in place, production call ordering, and the complete surrounding nonlinear calculation. The production default should be recorded against the WVM source version, compiled-core identity, machine topology, fixture identity, and integrated workload set. If the integrated winner differs from the benchmark-local tuple, the integrated measurement is authoritative.

## Integration acceptance

The later WVM implementation should require mode-keyed output equivalence at the frozen Float64 tolerances, caller-input preservation, zero warmed application allocations, one uniform algorithm per mathematical boundary, and separate component and uninstrumented-total timing. The WVM-native path remains a first-class MATLAB-compatible option even if the compact path is faster. No benchmark result alone authorizes a production default until the integrated boundary passes correctness, performance, memory, and cross-machine validation appropriate to WVM.
