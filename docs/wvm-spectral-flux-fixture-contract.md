# WVM spectral-flux fixture contract

This document defines the evidence boundary between the Wave–Vortex Model (WVM) and experiment `issue-019-production-lifetime-spectral-flux-composition`. It is a data contract, not a second implementation of WVM physics.

## Status classes

An issue #19 run records one fixture status:

- `authoritative-wvm-export`: every required payload was exported by the identified WVM generator and its hashes match the manifest. Only this status may support reference publication.
- `provider-independent-synthetic-development`: `skbench` generated deterministic modal inputs and synthetic vertical operators. This status can validate the harness, representations, buffer lifetime, and preliminary performance, but cannot support adoption or a statement about WVM's complete nonlinear flux.
- `invalid`: a required payload, identifier, dimension, hash, convention, or oracle comparison failed. No timing from the run is interpretable.

The initial implementation deliberately uses `provider-independent-synthetic-development`. Missing authoritative files must stop a requested reference campaign; the runner must not silently substitute its synthetic fixture.

## Logical boundary

The call begins with 15 ready Float64 complex modal fields indexed by logical coordinates `(k,l,j,inputField)` and ends with four ready Float64 complex modal targets indexed by `(k,l,j,target)`. Physical storage order is representation metadata and is not part of the mathematics.

Input fields have the fixed meaning:

| Slots | Meaning |
|---|---|
| `0,1,2` | Shared velocity fields `U,V,W` |
| `3+3*t,4+3*t,5+3*t` | Derivatives `q[t].x,q[t].y,q[t].z` for target `t=0,1,2,3` |

For each target, the physical-space expression is

\[
r_t(x,y,z) = -\left(Uq_{t,x} + Vq_{t,y} + Wq_{t,z}\right).
\]

The streamed lifetime reconstructs the three shared velocity volumes once. It then reconstructs one three-derivative volume, evaluates one target volume, transforms and projects that target, and reuses the derivative and target storage. At most seven real `Nx*Ny*Nz` volumes are live for this part of the graph. The benchmark never requires all 15 real input volumes to coexist.

The boundary excludes phase evolution, construction of the 15 inputs from WVM state coefficients, accumulation of the four outputs into tendencies, eigensystem generation, time integration, model-state ownership, MATLAB dispatch, I/O, and diagnostics.

## Fixture set

One authoritative fixture directory contains a UTF-8 `manifest.json` and immutable little-endian binary payloads. The manifest schema identifier is `spectral-flux-fixture-v1`. A fixture is identified by the SHA-256 digest of the canonical manifest plus the SHA-256 digest recorded for every payload.

The manifest declares:

- `fixtureId`, `schema`, creation time, numeric type, and byte order;
- WVM repository URL, exact source commit, dirty-tree state, generator path, generator commit, and generator command or options;
- `Nx`, `Ny`, `Nz`, retained horizontal count `H`, retained vertical count `Nj`, domain lengths, and the radial and vertical retention rules;
- ordered horizontal mode keys `(k,l)`, ordered vertical mode keys, target order, and every logical array dimension;
- forward and inverse horizontal normalization and sign conventions;
- derivative definitions, units, phase conventions, and the map from the 15 input slots to the four expressions;
- references from each logical input and target to the exact inverse or forward vertical operator it uses;
- each payload path or immutable artifact URL, byte count, element type, logical axes, physical strides, and SHA-256 digest; and
- the expected four modal outputs produced by the WVM reference calculation.

Large payloads do not need to live in Git history. A reference result must nevertheless name an immutable, retrievable artifact and verify its declared byte count and SHA-256 digest before planning or timing. A local unversioned file path is insufficient provenance.

The minimum logical payloads are:

| Payload | Logical shape | Purpose |
|---|---:|---|
| horizontal mode keys | `H x 2` signed integers | Identify retained `(k,l)` coordinates |
| vertical mode keys | `Nj` records | Identify retained vertical coordinates |
| modal inputs | `H x Nj x 15` complex Float64 | Ready input boundary |
| inverse vertical operators | manifest-indexed | Reconstruct the physical `z` samples required by each input field |
| forward vertical operators | manifest-indexed | Project each target from `Nz` samples to `Nj` modes |
| expected modal targets | `H x Nj x 4` complex Float64 | Independent WVM oracle |

Operator payloads may deduplicate identical matrices. Deduplication is represented by stable operator keys and field-to-key mappings; it must not assume that matrices are shared when WVM says they differ. Benchmark setup may permute or split these immutable matrices for a provider, but the prepared representation and its permutation hash remain setup metadata.

The v1 WVM exporter supplies two matrix families, `wave-f` and `wave-g`, indexed by WVM's floating (K^2)-unique groups. Inputs use the fixed family map `[F,F,G,F,F,G,F,F,G,G,G,F,G,G,F]`; targets use `[F,F,G,G]`. Integer (k^2+l^2) values are diagnostic keys, not permission to merge WVM groups: floating construction can produce distinct WVM groups with the same integer key, and the fixture preserves their separate matrices. The benchmark may partition fields into family-contiguous provider calls and permute modes by their exact `(k,l)` keys during setup. Those are physical representation choices, not changes to the logical field or operator map, and they are excluded from primitive and uninstrumented-total timing.

## Validation before timing

Fixture loading and oracle reordering occur outside all timed regions. Validation must reject the fixture before timing if any of the following fails:

1. manifest, payload, byte-count, dimension, or SHA-256 verification;
2. unique and complete horizontal and vertical mode keys;
3. expected radial two-thirds and `Nj=floor(2*(Nz-1)/3)` retention;
4. input-slot, target-slot, operator-key, normalization, or derivative-convention consistency;
5. DC, Nyquist, and Hermitian-boundary requirements; or
6. independent comparison of all four outputs by `(k,l,j,target)`.

Reference Float64 correctness requires maximum scale-normalized error and relative L2 error at most `1e-12`. Oracle permutation, representation conversion used only for comparison, and error evaluation remain outside timing.

## Strict preparation and pilot execution

`skbench` consumes a prepared binary so the timed C++ process does not need a JSON dependency. Preparation is an obligatory validation stage, not an optional converter:

```sh
python3 tools/prepare_spectral_flux_fixture.py \
  --fixture /path/to/export-directory \
  --output /new/path/prepared-fixture.bin
```

The preparer rejects a dirty or non-authoritative WVM export, duplicate or unsafe payload paths, unexpected payloads, malformed metadata, byte-count or SHA-256 mismatches, non-finite values, incorrect mode keys or WVM group maps, normalization differences, and any change to the field/operator or oracle contract. It refuses to overwrite an existing prepared file. The C++ loader independently checks its binary version, byte order, authoritative marker, dimensions, retained mode-key set, WVM group map and diagnostic keys, family map, finite values, DC reality, and absence of trailing bytes. It then reorders inputs and the oracle by `(k,l)` outside timing; floating radial ties need not have the same physical order in WVM and `skbench`.

An explicit `--spectral-flux-fixture prepared-fixture.bin` selects the authoritative path. Omitting the option retains the clearly labeled synthetic development harness; supplying an invalid prepared path fails before planning or timing. The paired pilot runner performs preparation, runs both frozen graphs in isolated processes, requires the same fixture hash and WVM commit in both result bundles, and verifies every target against the exported oracle. Its result classification remains `preliminary` because one 256-squared pilot cannot substitute for the preregistered multi-workload reference campaign.

## Result provenance

Every issue #19 result records the fixture status, schema, WVM repository and commit, generator identity, fixture hash, normalization, mode mapping, derivative convention, authoritative flag, workload and operator dimensions, representation identifiers, and permutation hashes. A clean benchmark commit does not compensate for a non-authoritative fixture: both source provenance and fixture provenance must independently qualify.

Synthetic development results must use publication status `preliminary` and say that they test the harness and algorithmic lifetime only. They cannot enter the issue #19 10% gate or the v1 adoption statistics. Superseding them with WVM-derived evidence preserves their immutable run pages.
