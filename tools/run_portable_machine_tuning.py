#!/usr/bin/env python3
"""Tune frozen spectral implementations without claiming WVM production optimality."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from run_cross_mac_spectral_reference import machine_record
from run_spectral_pipeline_sweep import maximum_correctness_error
from run_vertical_gemm_sweep import git_source_state


EXPERIMENT_ID = "issue-023-portable-tuning-reference-campaign"
INCREMENT_ID = "portable-machine-tuning-calibration-v1"
SCHEMA = "spectral-kernel-machine-tuning-v1"
PUBLICATION_SCHEMA = "spectral-kernel-machine-tuning-publication-v1"
CLASSIFICATION = "preliminary"
INTENDED_USE = "benchmark-local-provisional-default"
PROFILES = (
    "wvm-current-256-nz129-f4",
    "wvm-current-512-nz257-f4",
)
WARMUPS = 2
SAMPLES = 7
TOLERANCE = 1.0e-12
CONSTANT_MAXIMUM_TOLERANCE = 2.0e-12
THREAD_ENVIRONMENT = {"VECLIB_MAXIMUM_THREADS": "1"}
GENERAL_TOTAL_STAGE = (
    "authoritative production-lifetime streamed four-target spectral-flux "
    "composition"
)
CONSTANT_TOTAL_STAGE = (
    "authoritative WVM constant-stratification nonlinear-flux composition"
)


@dataclass(frozen=True)
class Implementation:
    id: str
    boundary: str
    kernel: str
    fixture_kind: str
    provider: str
    representation: str
    mathematical_boundary: str
    knobs: tuple[str, ...]


@dataclass(frozen=True)
class WorkerTuple:
    horizontal: int
    pointwise: int
    general_vertical: int = 1
    constant_type1: int = 1

    def applicable(self, implementation: Implementation) -> dict[str, int]:
        values = asdict(self)
        return {name: values[name] for name in implementation.knobs}


def implementations() -> tuple[Implementation, ...]:
    return (
        Implementation(
            id="wvm-native-optimized-v1",
            boundary="wvm-direct-strided-field-views",
            kernel="production-lifetime-flux",
            fixture_kind="general",
            provider=(
                "pipeline-production-lifetime-wvm-direct-strided-field-views-"
                "authoritative-pointwise-spatial-static"
            ),
            representation="WVM-native persistent complex-interleaved modal state",
            mathematical_boundary=(
                "15 WVM-native modal inputs to four WVM-native modal targets"
            ),
            knobs=("horizontal", "general_vertical", "pointwise"),
        ),
        Implementation(
            id="compact-general-fused-views-v1",
            boundary="streaming-pruned-compact-split-fused-vertical-views",
            kernel="production-lifetime-flux",
            fixture_kind="general",
            provider=(
                "pipeline-production-lifetime-streaming-pruned-tile16-fused-"
                "vertical-views-authoritative-pointwise-spatial-static"
            ),
            representation="compact radial split modal state",
            mathematical_boundary=(
                "15 compact modal inputs to four compact modal targets"
            ),
            knobs=("horizontal", "general_vertical", "pointwise"),
        ),
        Implementation(
            id="compact-constant-type1-v1",
            boundary="streaming-pruned-compact-split",
            kernel="constant-stratification-flux",
            fixture_kind="constant",
            provider=(
                "pipeline-constant-stratification-streaming-pruned-tile16-"
                "authoritative"
            ),
            representation="compact radial split constant-stratification modal state",
            mathematical_boundary=(
                "constant-stratification modal state to exact nonlinear-flux targets"
            ),
            knobs=("horizontal", "constant_type1", "pointwise"),
        ),
    )


def implementation_named(identifier: str) -> Implementation:
    matches = [item for item in implementations() if item.id == identifier]
    if len(matches) != 1:
        raise ValueError(f"unknown implementation: {identifier}")
    return matches[0]


def canonical_hash(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def write_json_atomic(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def load_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def candidate_counts(performance_cores: int, total_cores: int) -> tuple[int, ...]:
    if performance_cores < 1 or total_cores < performance_cores:
        raise ValueError("invalid performance/total physical-core topology")
    values = {1, performance_cores, total_cores}
    value = 2
    while value <= total_cores:
        values.add(value)
        value *= 2
    return tuple(sorted(values))


def portable_seed(performance_cores: int, total_cores: int) -> WorkerTuple:
    pointwise = 1
    while pointwise * 2 <= performance_cores:
        pointwise *= 2
    return WorkerTuple(
        horizontal=performance_cores,
        general_vertical=total_cores,
        constant_type1=total_cores,
        pointwise=pointwise,
    )


def with_knob(worker_tuple: WorkerTuple, knob: str, value: int) -> WorkerTuple:
    values = asdict(worker_tuple)
    values[knob] = value
    return WorkerTuple(**values)


def one_factor_candidates(
    implementation: Implementation, performance_cores: int, total_cores: int,
) -> tuple[WorkerTuple, ...]:
    seed = portable_seed(performance_cores, total_cores)
    counts = candidate_counts(performance_cores, total_cores)
    result = [seed]
    for knob in implementation.knobs:
        result.extend(with_knob(seed, knob, value) for value in counts)
    seen: set[tuple[tuple[str, int], ...]] = set()
    unique: list[WorkerTuple] = []
    for item in result:
        key = tuple(sorted(item.applicable(implementation).items()))
        if key not in seen:
            seen.add(key)
            unique.append(item)
    return tuple(unique)


def geometric_mean(values: list[float]) -> float:
    if not values or any(value <= 0.0 for value in values):
        raise ValueError("geometric mean requires positive observations")
    return math.exp(sum(math.log(value) for value in values) / len(values))


def tuple_identifier(
    implementation: Implementation, worker_tuple: WorkerTuple,
) -> str:
    short = {
        "horizontal": "h",
        "general_vertical": "gv",
        "constant_type1": "ct",
        "pointwise": "p",
    }
    return "--".join(
        f"{short[name]}{value}"
        for name, value in worker_tuple.applicable(implementation).items()
    )


def command_for(
    executable: Path, implementation: Implementation, fixture: Path,
    profile: str, workers: WorkerTuple, warmups: int, samples: int,
    output: Path,
) -> list[str]:
    command = [
        str(executable), "run",
        "--kernel", implementation.kernel,
        "--profile", profile,
        "--fftw-planning", "measure",
        "--fftw-alignment", "unaligned",
        "--fftw-wisdom", "cold",
        "--fftw-outer-workers", str(workers.horizontal),
        "--streaming-tile-width", "16",
        "--pointwise-policy", "spatial-static",
        "--pointwise-workers", str(workers.pointwise),
        "--warmups", str(warmups),
        "--samples", str(samples),
        "--output", str(output),
    ]
    if implementation.fixture_kind == "general":
        command.extend([
            "--boundary-policy", implementation.boundary,
            "--spectral-flux-fixture", str(fixture),
            "--fftw-internal-workers", "1",
            "--vertical-gemm-family", "k2-grouped",
            "--vertical-gemm-schedule", "outer-dynamic",
            "--vertical-gemm-outer-workers", str(workers.general_vertical),
        ])
    else:
        command.extend([
            "--constant-stratification-flux-fixture", str(fixture),
            "--fftw-internal-workers", str(workers.constant_type1),
            "--comparison-order", "candidate-first",
        ])
    return command


def expected_schedule(
    implementation: Implementation, workers: WorkerTuple,
) -> str:
    if implementation.fixture_kind == "general":
        return (
            f"horizontal-outer-{workers.horizontal};vertical-outer-dynamic-"
            f"{workers.general_vertical}-per-operator-family;"
            f"pointwise-spatial-static-{workers.pointwise}"
        )
    return (
        f"vertical-type1-internal-{workers.constant_type1};horizontal-internal-"
        f"1-outer-{workers.horizontal};pointwise-spatial-static-"
        f"{workers.pointwise};comparison-candidate-first"
    )


def total_timing(provider: dict, implementation: Implementation) -> dict:
    expected = (
        GENERAL_TOTAL_STAGE if implementation.fixture_kind == "general"
        else CONSTANT_TOTAL_STAGE
    )
    matches = [
        item for item in provider.get("timings", [])
        if item.get("scope") == "uninstrumented-total"
        and item.get("stage") == expected
    ]
    if len(matches) != 1:
        raise ValueError(
            f"{implementation.id} result lacks one authoritative total timing"
        )
    return matches[0]


def placement_valid(provider: dict) -> bool:
    forward = provider.get("executionContract", {}).get("forward", {})
    return bool(
        forward.get("nativePlacement") == "out-of-place"
        and forward.get("adapterPlacement") == "out-of-place"
        and forward.get("destroysNativeInput") is False
        and forward.get("adapterPreservesCallerInput") is True
        and forward.get("requiresPreservationCopyForRepeatedExecution") is False
    )


def allocation_valid(provider: dict) -> bool:
    return any(
        item.get("state") == "elided"
        and item.get("stage") in {
            "steady-state allocation", "steady-state application allocation",
        }
        for item in provider.get("componentLedger", [])
    )


def correctness_valid(provider: dict, implementation: Implementation) -> tuple[bool, float]:
    metrics = provider.get("correctness", [])
    if not metrics or not all(item.get("passed") is True for item in metrics):
        return False, math.inf
    if implementation.fixture_kind == "general":
        maximum = maximum_correctness_error(provider)
        return bool(math.isfinite(maximum) and maximum <= TOLERANCE), maximum
    authoritative = [
        item for item in metrics
        if str(item.get("name", "")).startswith("complete ")
        and "authoritative WVM oracle" in str(item.get("name", ""))
    ]
    equivalence = [
        item for item in metrics
        if item.get("name") == "complete compact composition versus full-half control"
    ]
    if len(authoritative) != 1 or len(equivalence) != 1:
        return False, math.inf
    bounded = authoritative + equivalence
    maximum = max(float(item["maximumRelativeError"]) for item in bounded)
    relative_l2 = max(float(item["relativeL2Error"]) for item in bounded)
    return bool(
        maximum <= CONSTANT_MAXIMUM_TOLERANCE and relative_l2 <= TOLERANCE
    ), maximum


def result_record(
    result: dict, implementation: Implementation, workers: WorkerTuple,
    profile: str, warmups: int, samples: int,
) -> dict:
    providers = [
        item for item in result.get("providers", [])
        if item.get("id") == implementation.provider
    ]
    if len(providers) != 1:
        raise ValueError(
            f"result lacks provider {implementation.provider}"
        )
    provider = providers[0]
    timing = total_timing(provider, implementation)
    correct, maximum_error = correctness_valid(provider, implementation)
    provenance = result.get("provenance", {}).get("spectralFluxFixture", {})
    expected_schema = (
        "spectral-flux-fixture-v1" if implementation.fixture_kind == "general"
        else "constant-stratification-flux-fixture-v1"
    )
    authoritative_fixture = bool(
        provenance.get("schema") == expected_schema
        and provenance.get("status") == "authoritative-wvm-export"
        and provenance.get("authoritative") is True
        and provenance.get("fixtureHash")
        and provenance.get("waveVortexModelCommit")
    )
    executable_metadata_valid = bool(
        result.get("environment", {}).get("gitCommit") not in {None, "", "unknown"}
        and result.get("environment", {}).get("gitDirty") is False
    )
    valid = bool(
        result.get("status") == "passed"
        and result.get("run", {}).get("profile") == profile
        and result.get("run", {}).get("warmups") == warmups
        and result.get("run", {}).get("samples") == samples
        and len(timing.get("samplesSeconds", [])) == samples
        and provider.get("schedulingId") == expected_schedule(
            implementation, workers,
        )
        and correct and authoritative_fixture and executable_metadata_valid
        and placement_valid(provider) and allocation_valid(provider)
    )
    memory = provider.get("memory", {})
    components = {
        item["stage"]: float(item["medianSeconds"])
        for item in provider.get("timings", [])
        if item.get("medianSeconds") is not None
        and item.get("scope") not in {
            "setup-shared-component", "setup-component", "uninstrumented-total",
        }
    }
    return {
        "runId": result.get("run", {}).get("id"),
        "seconds": float(timing["medianSeconds"]),
        "components": components,
        "maximumCorrectnessError": maximum_error,
        "correctnessValid": correct,
        "authoritativeFixture": authoritative_fixture,
        "fixtureHash": provenance.get("fixtureHash"),
        "waveVortexModelCommit": provenance.get("waveVortexModelCommit"),
        "executableMetadataValid": executable_metadata_valid,
        "placementValid": placement_valid(provider),
        "allocationValid": allocation_valid(provider),
        "schedulingId": provider.get("schedulingId"),
        "memory": {
            key: int(memory.get(key, 0)) for key in (
                "persistentBytes", "scratchBytes", "algorithmResidentBytes",
                "estimatedProcessPeakBytes", "observedProcessHighWaterBytes",
            )
        },
        "valid": valid,
    }


def select_candidate(
    implementation: Implementation, candidates: list[dict],
) -> dict | None:
    eligible = [
        item for item in candidates
        if item.get("complete") is True and item.get("valid") is True
        and item.get("geometricSeconds") is not None
    ]
    if not eligible:
        return None
    fastest = min(float(item["geometricSeconds"]) for item in eligible)
    near = [
        item for item in eligible
        if float(item["geometricSeconds"]) <= 1.01 * fastest
    ]
    return min(
        near,
        key=lambda item: (
            sum(int(value) for value in item["workers"].values()),
            int(item.get("maximumAlgorithmResidentBytes", 0)),
            int(item["candidateOrder"]),
        ),
    )


def analyze_runs(manifest: dict) -> dict:
    runs = manifest.get("runs", [])
    selections: dict[str, dict] = {}
    for contract in manifest["implementations"]:
        implementation = implementation_named(contract["id"])
        rows: list[dict] = []
        candidate_contracts = manifest["candidateMatrix"][implementation.id]
        for candidate_order, candidate in enumerate(candidate_contracts):
            matching = [
                item for item in runs
                if item.get("implementationId") == implementation.id
                and item.get("candidateId") == candidate["id"]
                and item.get("valid") is True
            ]
            by_profile = {item["profile"]: item for item in matching}
            complete = set(by_profile) == set(PROFILES)
            records = [by_profile[profile]["record"] for profile in PROFILES if profile in by_profile]
            valid = bool(complete and all(record.get("valid") is True for record in records))
            rows.append({
                "candidateId": candidate["id"],
                "candidateOrder": candidate_order,
                "workers": candidate["workers"],
                "complete": complete,
                "valid": valid,
                "geometricSeconds": (
                    geometric_mean([float(record["seconds"]) for record in records])
                    if valid else None
                ),
                "maximumAlgorithmResidentBytes": max(
                    (int(record["memory"]["algorithmResidentBytes"]) for record in records),
                    default=0,
                ),
                "profiles": [
                    {
                        "profile": profile,
                        "seconds": by_profile[profile]["record"]["seconds"],
                        "maximumCorrectnessError": by_profile[profile]["record"][
                            "maximumCorrectnessError"
                        ],
                    }
                    for profile in PROFILES if profile in by_profile
                ],
            })
        selected = select_candidate(implementation, rows)
        selections[implementation.id] = {
            "candidates": rows,
            "selected": selected,
            "intendedUse": INTENDED_USE,
            "productionValidated": False,
        }
    return {
        "selectionRule": {
            "score": "geometric mean complete-call time across both calibration profiles",
            "nearFastestFactor": 1.01,
            "tieBreak": ["fewest total workers", "least resident memory", "candidate order"],
        },
        "selections": selections,
    }


def add_combined_neighborhood(manifest: dict) -> bool:
    """Add a bounded second phase after every one-factor cell is valid."""
    changed = False
    analysis = analyze_runs(manifest)
    machine = manifest["machine"]
    seed = portable_seed(
        int(machine["performanceCores"]),
        int(machine["totalPhysicalCores"]),
    )
    for contract in manifest["implementations"]:
        implementation = implementation_named(contract["id"])
        rows = analysis["selections"][implementation.id]["candidates"]
        screen = [
            row for row in rows
            if next(
                candidate for candidate in manifest["candidateMatrix"][implementation.id]
                if candidate["id"] == row["candidateId"]
            )["phase"] == "one-factor-screen"
        ]
        expected = len(one_factor_candidates(
            implementation, int(machine["performanceCores"]),
            int(machine["totalPhysicalCores"]),
        ))
        if len(screen) != expected or not all(
            row["complete"] and row["valid"] for row in screen
        ):
            continue
        combined = seed
        seed_values = seed.applicable(implementation)
        for knob in implementation.knobs:
            eligible = [
                row for row in screen
                if all(
                    name == knob or int(row["workers"][name]) == seed_values[name]
                    for name in implementation.knobs
                )
            ]
            selected = select_candidate(implementation, eligible)
            if selected is None:
                continue
            combined = with_knob(combined, knob, int(selected["workers"][knob]))
        neighborhood = [combined]
        neighborhood.extend(
            with_knob(combined, knob, seed_values[knob])
            for knob in implementation.knobs
        )
        existing = {
            candidate["id"] for candidate in manifest["candidateMatrix"][implementation.id]
        }
        for workers in neighborhood:
            identifier = tuple_identifier(implementation, workers)
            if identifier in existing:
                continue
            manifest["candidateMatrix"][implementation.id].append({
                "id": identifier,
                "phase": "combined-winner-neighborhood",
                "workers": workers.applicable(implementation),
            })
            existing.add(identifier)
            changed = True
    if changed:
        manifest["analysis"] = analyze_runs(manifest)
    return changed


def fixture_assignments(values: list[str], label: str) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"{label} fixtures must use PROFILE=PATH")
        profile, raw = value.split("=", 1)
        if profile not in PROFILES:
            raise ValueError(f"unknown {label} profile: {profile}")
        path = Path(raw).expanduser().resolve()
        if not path.is_file():
            raise ValueError(f"{label} fixture is missing: {path}")
        if profile in result:
            raise ValueError(f"duplicate {label} fixture: {profile}")
        result[profile] = path
    return result


def selected_implementations(values: list[str]) -> tuple[Implementation, ...]:
    if not values or "all" in values:
        return implementations()
    return tuple(implementation_named(value) for value in dict.fromkeys(values))


def contract_record(implementation: Implementation) -> dict:
    contract = asdict(implementation)
    contract["frozen"] = {
        "numericType": "float64",
        "provider": "fftw-3.3.11",
        "tileWidth": 16,
        "placement": "out-of-place-caller-preserving",
        "allocation": "zero-warmed-application-allocations",
        "sizeDependentDispatch": False,
    }
    contract["contractHash"] = canonical_hash(contract)
    return contract


def candidate_matrix(
    selected: tuple[Implementation, ...], machine: dict,
) -> dict[str, list[dict]]:
    result: dict[str, list[dict]] = {}
    for implementation in selected:
        candidates = one_factor_candidates(
            implementation, int(machine["performanceCores"]),
            int(machine["totalPhysicalCores"]),
        )
        result[implementation.id] = [
            {
                "id": tuple_identifier(implementation, workers),
                "phase": "one-factor-screen",
                "workers": workers.applicable(implementation),
            }
            for workers in candidates
        ]
    return result


def worker_tuple_from_record(implementation: Implementation, values: dict) -> WorkerTuple:
    seed = WorkerTuple(1, 1, 1, 1)
    result = seed
    for name in implementation.knobs:
        result = with_knob(result, name, int(values[name]))
    return result


def manifest_template(
    repository: Path, executable: Path, selected: tuple[Implementation, ...],
    fixtures: dict[str, dict[str, Path]], machine: dict, warmups: int,
    samples: int,
) -> dict:
    commit, dirty = git_source_state(repository)
    fixture_records = {
        kind: {
            profile: {"path": str(path), "sha256": file_hash(path)}
            for profile, path in paths.items()
        }
        for kind, paths in fixtures.items()
    }
    contracts = [contract_record(item) for item in selected]
    matrix = candidate_matrix(selected, machine)
    identity = {
        "sourceTreeGitCommit": commit,
        "sourceTreeDirty": dirty,
        "executablePath": str(executable),
        "executableSha256": file_hash(executable),
        "implementationContractHashes": {
            item["id"]: item["contractHash"] for item in contracts
        },
        "fixtureHashes": {
            f"{kind}:{profile}": record["sha256"]
            for kind, profiles in fixture_records.items()
            for profile, record in profiles.items()
        },
    }
    return {
        "schema": SCHEMA,
        "experimentId": EXPERIMENT_ID,
        "incrementId": INCREMENT_ID,
        "classification": CLASSIFICATION,
        "intendedUse": INTENDED_USE,
        "productionValidated": False,
        "createdAtUtc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "question": (
            "Can the frozen implementations expose reproducible benchmark-local "
            "worker defaults on different Apple-silicon machines?"
        ),
        "machine": machine,
        "identity": identity,
        "compatibilityHash": canonical_hash(identity),
        "implementations": contracts,
        "profiles": list(PROFILES),
        "fixtures": fixture_records,
        "candidatePolicy": {
            "counts": "one, powers of two, performance cores, and total physical cores",
            "search": (
                "portable seed plus deterministic one-factor sweeps, followed by "
                "the combined one-factor winners and seed-reversion neighbors"
            ),
            "nestedWorkerPools": False,
            "oneTupleForAllWorkloads": True,
        },
        "capacityPolicy": {
            "physicalMemoryBytes": int(machine.get("physicalMemoryBytes", 0)),
            "calibrationWorkloadsOnly": True,
            "disposition": (
                "record provider-estimated and observed memory for every completed "
                "cell; classify any setup failure before publication rather than "
                "silently resizing or substituting a workload"
            ),
        },
        "candidateMatrix": matrix,
        "threadEnvironment": THREAD_ENVIRONMENT,
        "warmups": warmups,
        "samples": samples,
        "runs": [],
        "analysis": {"selections": {}},
    }


def compatibility_fields(manifest: dict) -> dict:
    return {
        "schema": manifest.get("schema"),
        "experimentId": manifest.get("experimentId"),
        "machine": manifest.get("machine"),
        "identity": manifest.get("identity"),
        "implementations": manifest.get("implementations"),
        "profiles": manifest.get("profiles"),
        "fixtures": manifest.get("fixtures"),
        "candidatePolicy": manifest.get("candidatePolicy"),
        "capacityPolicy": manifest.get("capacityPolicy"),
        "warmups": manifest.get("warmups"),
        "samples": manifest.get("samples"),
    }


def machine_matches(recorded: dict, current: dict) -> bool:
    return all(
        recorded.get(key) == current.get(key)
        for key in (
            "hostname", "cpuBrand", "hardwareModel", "performanceCores",
            "efficiencyCores", "totalPhysicalCores", "physicalMemoryBytes",
        )
    )


def validate_manifest(manifest: dict) -> list[str]:
    errors: list[str] = []
    if manifest.get("schema") != SCHEMA:
        errors.append(f"schema must be {SCHEMA}")
    if manifest.get("experimentId") != EXPERIMENT_ID:
        errors.append(f"experimentId must be {EXPERIMENT_ID}")
    if manifest.get("productionValidated") is not False:
        errors.append("productionValidated must remain false in this benchmark")
    if manifest.get("intendedUse") != INTENDED_USE:
        errors.append(f"intendedUse must be {INTENDED_USE}")
    machine = manifest.get("machine", {})
    if not isinstance(machine, dict) or int(machine.get("totalPhysicalCores", 0)) < 1:
        errors.append("machine topology is missing or invalid")
    identities = manifest.get("identity", {})
    if not isinstance(identities, dict) or not identities.get("executableSha256"):
        errors.append("executable identity is missing")
    elif manifest.get("compatibilityHash") != canonical_hash(identities):
        errors.append("compatibilityHash does not match the recorded identities")
    contract_hashes = identities.get("implementationContractHashes", {})
    for contract in manifest.get("implementations", []):
        expected_hash = contract.get("contractHash")
        unhashed = {key: value for key, value in contract.items() if key != "contractHash"}
        if expected_hash != canonical_hash(unhashed):
            errors.append(f"implementation contract hash changed: {contract.get('id')}")
        if contract_hashes.get(contract.get("id")) != expected_hash:
            errors.append(f"identity lacks implementation contract: {contract.get('id')}")
        try:
            implementation = implementation_named(str(contract.get("id")))
        except ValueError as error:
            errors.append(str(error))
            continue
        candidates = manifest.get("candidateMatrix", {}).get(implementation.id, [])
        seen_candidates: set[str] = set()
        for candidate in candidates:
            identifier = candidate.get("id")
            workers = candidate.get("workers", {})
            if identifier in seen_candidates:
                errors.append(f"duplicate candidate id: {implementation.id}:{identifier}")
            seen_candidates.add(identifier)
            if set(workers) != set(implementation.knobs):
                errors.append(f"candidate changes undeclared knobs: {implementation.id}:{identifier}")
                continue
            worker_tuple = worker_tuple_from_record(implementation, workers)
            if identifier != tuple_identifier(implementation, worker_tuple):
                errors.append(f"candidate id does not match workers: {implementation.id}:{identifier}")
            if candidate.get("phase") not in {
                "one-factor-screen", "combined-winner-neighborhood",
            }:
                errors.append(f"candidate has unknown phase: {implementation.id}:{identifier}")
    run_ids: set[str] = set()
    for run in manifest.get("runs", []):
        identifier = run.get("id")
        if not identifier or identifier in run_ids:
            errors.append(f"duplicate or missing run id: {identifier}")
        run_ids.add(identifier)
    return errors


def plan_records(
    manifest: dict, executable: Path, output: Path,
) -> list[dict]:
    plans: list[dict] = []
    fixture_paths = {
        kind: {profile: Path(record["path"]) for profile, record in profiles.items()}
        for kind, profiles in manifest["fixtures"].items()
    }
    for contract in manifest["implementations"]:
        implementation = implementation_named(contract["id"])
        for candidate in manifest["candidateMatrix"][implementation.id]:
            workers = worker_tuple_from_record(implementation, candidate["workers"])
            for profile in PROFILES:
                identifier = f"{implementation.id}--{candidate['id']}--{profile}"
                path = output / "runs" / f"{identifier}.json"
                plans.append({
                    "id": identifier,
                    "implementationId": implementation.id,
                    "candidateId": candidate["id"],
                    "workers": candidate["workers"],
                    "profile": profile,
                    "fixtureKind": implementation.fixture_kind,
                    "resultPath": path,
                    "command": command_for(
                        executable, implementation,
                        fixture_paths[implementation.fixture_kind][profile],
                        profile, workers, int(manifest["warmups"]),
                        int(manifest["samples"]), path,
                    ),
                })
    return plans


def print_plan(manifest: dict, plans: list[dict]) -> None:
    machine = manifest["machine"]
    print(
        f"machine={machine.get('hostname')} cpu={machine.get('cpuBrand')} "
        f"P={machine.get('performanceCores')} E={machine.get('efficiencyCores')} "
        f"physical={machine.get('totalPhysicalCores')}"
    )
    print(
        f"classification={CLASSIFICATION} intended-use={INTENDED_USE} "
        f"warmups={manifest['warmups']} samples={manifest['samples']} "
        f"physical-memory-bytes={manifest['capacityPolicy']['physicalMemoryBytes']}"
    )
    for contract in manifest["implementations"]:
        candidates = manifest["candidateMatrix"][contract["id"]]
        print(
            f"implementation={contract['id']} boundary={contract['mathematical_boundary']} "
            f"representation={contract['representation']} candidates={len(candidates)}"
        )
        for candidate in candidates:
            print(f"  candidate={candidate['id']} workers={candidate['workers']}")
    print(f"processes={len(plans)} profiles={','.join(PROFILES)}")
    for plan in plans:
        print(f"VECLIB_MAXIMUM_THREADS=1 {' '.join(map(str, plan['command']))}")


def execute_plans(
    repository: Path, output: Path, manifest_path: Path, manifest: dict,
    plans: list[dict], continue_on_error: bool,
) -> bool:
    existing = {item["id"]: item for item in manifest.get("runs", [])}
    failed = False
    for index, plan in enumerate(plans, start=1):
        entry = existing.get(plan["id"])
        if entry is not None and entry.get("valid") is True:
            print(f"[{index}/{len(plans)}] reuse {plan['id']}", flush=True)
            continue
        if entry is not None:
            raise ValueError(
                f"existing incomplete run requires manual inspection: {plan['id']}"
            )
        print(f"[{index}/{len(plans)}] {plan['id']}", flush=True)
        plan["resultPath"].parent.mkdir(parents=True, exist_ok=True)
        log_path = plan["resultPath"].with_suffix(".log")
        samples_path = plan["resultPath"].with_suffix(".csv")
        environment = os.environ.copy()
        environment.update(THREAD_ENVIRONMENT)
        with log_path.open("w", encoding="utf-8") as log:
            completed = subprocess.run(
                plan["command"], cwd=repository, env=environment,
                stdout=log, stderr=subprocess.STDOUT,
            )
        entry = {
            "id": plan["id"],
            "implementationId": plan["implementationId"],
            "candidateId": plan["candidateId"],
            "workers": plan["workers"],
            "profile": plan["profile"],
            "measurementRole": "benchmark-local-calibration",
            "primaryProvider": implementation_named(
                plan["implementationId"]
            ).provider,
            "command": list(map(str, plan["command"])),
            "result": str(plan["resultPath"].relative_to(output)),
            "samples": str(samples_path.relative_to(output)),
            "log": str(log_path.relative_to(output)),
            "exitCode": completed.returncode,
            "valid": False,
        }
        if (
            completed.returncode == 0 and plan["resultPath"].is_file()
            and samples_path.is_file()
        ):
            implementation = implementation_named(plan["implementationId"])
            workers = worker_tuple_from_record(implementation, plan["workers"])
            try:
                record = result_record(
                    load_json(plan["resultPath"]), implementation, workers,
                    plan["profile"], int(manifest["warmups"]),
                    int(manifest["samples"]),
                )
                entry["record"] = record
                entry["runId"] = record["runId"]
                entry["valid"] = record["valid"]
            except (KeyError, TypeError, ValueError) as error:
                entry["validationError"] = str(error)
        manifest["runs"].append(entry)
        manifest["analysis"] = analyze_runs(manifest)
        write_json_atomic(manifest_path, manifest)
        if not entry["valid"]:
            failed = True
            if not continue_on_error:
                break
    manifest["analysis"] = analyze_runs(manifest)
    write_json_atomic(manifest_path, manifest)
    return failed


def add_fixture_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--general-fixture", action="append", default=[])
    parser.add_argument("--constant-fixture", action="append", default=[])


def command_list(arguments: argparse.Namespace) -> int:
    selected = selected_implementations(arguments.implementation)
    machine = machine_record()
    print(json.dumps({
        "schema": SCHEMA,
        "classification": CLASSIFICATION,
        "intendedUse": INTENDED_USE,
        "machine": machine,
        "candidateCounts": candidate_counts(
            int(machine["performanceCores"]),
            int(machine["totalPhysicalCores"]),
        ),
        "implementations": [contract_record(item) for item in selected],
        "profiles": list(PROFILES),
    }, indent=2))
    return 0


def prepare_manifest(arguments: argparse.Namespace) -> tuple[Path, dict, list[dict]]:
    repository = Path(__file__).resolve().parents[1]
    executable = arguments.executable.expanduser().resolve()
    if not executable.is_file():
        raise ValueError(f"benchmark executable is missing: {executable}")
    selected = selected_implementations(arguments.implementation)
    general = fixture_assignments(arguments.general_fixture, "general")
    constant = fixture_assignments(arguments.constant_fixture, "constant")
    required_kinds = {item.fixture_kind for item in selected}
    fixtures = {"general": general, "constant": constant}
    for kind in required_kinds:
        missing = [profile for profile in PROFILES if profile not in fixtures[kind]]
        if missing:
            raise ValueError(
                f"missing {kind} fixture assignment(s): {', '.join(missing)}"
            )
    fixtures = {kind: fixtures[kind] for kind in required_kinds}
    output = arguments.output.expanduser().resolve()
    manifest_path = output / "machine-tuning.json"
    machine = machine_record()
    template = manifest_template(
        repository, executable, selected, fixtures, machine,
        arguments.warmups, arguments.samples,
    )
    if template["identity"]["sourceTreeDirty"] and not arguments.allow_dirty_tree:
        raise ValueError("source tree is dirty; commit before tuning")
    if manifest_path.is_file():
        current = load_json(manifest_path)
        if compatibility_fields(current) != compatibility_fields(template):
            raise ValueError("existing tuning manifest is stale or incompatible")
        manifest = current
    else:
        manifest = template
        if not arguments.dry_run:
            write_json_atomic(manifest_path, manifest)
    return manifest_path, manifest, plan_records(manifest, executable, output)


def command_tune(arguments: argparse.Namespace) -> int:
    try:
        manifest_path, manifest, plans = prepare_manifest(arguments)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        raise SystemExit(str(error)) from error
    print_plan(manifest, plans)
    if arguments.dry_run:
        return 0
    failed = execute_plans(
        Path(__file__).resolve().parents[1], arguments.output.resolve(),
        manifest_path, manifest, plans, arguments.continue_on_error,
    )
    if not failed and add_combined_neighborhood(manifest):
        write_json_atomic(manifest_path, manifest)
        expanded = plan_records(manifest, arguments.executable.resolve(), arguments.output.resolve())
        pending = [
            plan for plan in expanded
            if plan["id"] not in {item["id"] for item in manifest["runs"]}
        ]
        print("combined-winner-neighborhood:")
        print_plan(manifest, pending)
        failed = execute_plans(
            Path(__file__).resolve().parents[1], arguments.output.resolve(),
            manifest_path, manifest, pending, arguments.continue_on_error,
        )
    print(f"manifest={manifest_path}")
    return 1 if failed else 0


def command_resume(arguments: argparse.Namespace) -> int:
    manifest_path = arguments.manifest.expanduser().resolve()
    manifest = load_json(manifest_path)
    errors = validate_manifest(manifest)
    if errors:
        raise SystemExit("; ".join(errors))
    if not machine_matches(manifest["machine"], machine_record()):
        raise SystemExit("tuning manifest belongs to a different machine")
    repository = Path(__file__).resolve().parents[1]
    current_commit, current_dirty = git_source_state(repository)
    if (
        current_commit != manifest["identity"].get("sourceTreeGitCommit")
        or current_dirty != manifest["identity"].get("sourceTreeDirty")
    ):
        raise SystemExit("tuning manifest source tree is stale or incompatible")
    executable = Path(manifest["identity"]["executablePath"])
    if not executable.is_file() or file_hash(executable) != manifest["identity"]["executableSha256"]:
        raise SystemExit("tuning manifest executable is missing or incompatible")
    for kind, profiles in manifest["fixtures"].items():
        for profile, record in profiles.items():
            path = Path(record["path"])
            if not path.is_file() or file_hash(path) != record["sha256"]:
                raise SystemExit(f"tuning fixture is missing or incompatible: {kind}:{profile}")
    output = manifest_path.parent
    plans = plan_records(manifest, executable, output)
    print_plan(manifest, plans)
    if arguments.dry_run:
        return 0
    failed = execute_plans(
        repository, output, manifest_path,
        manifest, plans, arguments.continue_on_error,
    )
    if not failed and add_combined_neighborhood(manifest):
        write_json_atomic(manifest_path, manifest)
        expanded = plan_records(manifest, executable, output)
        pending = [
            plan for plan in expanded
            if plan["id"] not in {item["id"] for item in manifest["runs"]}
        ]
        print("combined-winner-neighborhood:")
        print_plan(manifest, pending)
        failed = execute_plans(
            repository, output, manifest_path,
            manifest, pending, arguments.continue_on_error,
        )
    return 1 if failed else 0


def command_validate(arguments: argparse.Namespace) -> int:
    manifest = load_json(arguments.manifest.expanduser().resolve())
    errors = validate_manifest(manifest)
    if errors:
        for error in errors:
            print(error, file=sys.stderr)
        return 1
    print("portable tuning manifest is valid")
    return 0


def publication_summary(manifest: dict) -> dict:
    errors = validate_manifest(manifest)
    if errors:
        raise ValueError("; ".join(errors))
    selections = manifest.get("analysis", {}).get("selections", {})
    run_lookup = {
        (run.get("implementationId"), run.get("candidateId"), run.get("profile")): run
        for run in manifest.get("runs", [])
    }
    implementation_rows: list[dict] = []
    for contract in manifest.get("implementations", []):
        implementation_id = contract["id"]
        selection = selections.get(implementation_id)
        if not isinstance(selection, dict) or not isinstance(
            selection.get("selected"), dict
        ):
            raise ValueError(f"missing selection for {implementation_id}")
        candidates: list[dict] = []
        for candidate in selection.get("candidates", []):
            profiles: list[dict] = []
            for profile in candidate.get("profiles", []):
                run = run_lookup.get((
                    implementation_id, candidate.get("candidateId"),
                    profile.get("profile"),
                ))
                profiles.append({
                    **profile,
                    **({"runId": run["runId"]} if run and run.get("runId") else {}),
                })
            candidates.append({**candidate, "profiles": profiles})
        selected_id = selection["selected"]["candidateId"]
        selected = next(
            (candidate for candidate in candidates
             if candidate["candidateId"] == selected_id), None,
        )
        if selected is None:
            raise ValueError(f"selected candidate is absent for {implementation_id}")
        seed_id = manifest["candidateMatrix"][implementation_id][0]["id"]
        seed = next(
            (candidate for candidate in candidates
             if candidate["candidateId"] == seed_id), None,
        )
        if seed is None:
            raise ValueError(f"portable seed is absent for {implementation_id}")
        implementation_rows.append({
            "id": implementation_id,
            "mathematicalBoundary": contract["mathematical_boundary"],
            "representation": contract["representation"],
            "knobs": contract["knobs"],
            "portableSeed": seed,
            "selected": selected,
            "seedToSelectedSpeedup": (
                float(seed["geometricSeconds"]) /
                float(selected["geometricSeconds"])
            ),
            "candidateTrace": candidates,
            "intendedUse": selection["intendedUse"],
            "productionValidated": selection["productionValidated"],
        })
    cpu_brand = str(manifest["machine"].get("cpuBrand", "recorded machine"))
    hostname = str(manifest["machine"].get("hostname", "unknown host"))
    return {
        "schema": PUBLICATION_SCHEMA,
        "experimentId": EXPERIMENT_ID,
        "incrementId": manifest.get("incrementId", INCREMENT_ID),
        "publicationStatus": CLASSIFICATION,
        "statusReason": (
            f"Clean {cpu_brand} calibration evidence from {hostname}; provisional "
            "benchmark-local evidence pending cross-machine synthesis."
        ),
        "intendedUse": INTENDED_USE,
        "productionValidated": False,
        "createdAtUtc": manifest.get("createdAtUtc"),
        "machine": manifest["machine"],
        "identity": {
            "sourceTreeGitCommit": manifest["identity"]["sourceTreeGitCommit"],
            "sourceTreeDirty": manifest["identity"]["sourceTreeDirty"],
            "executableSha256": manifest["identity"]["executableSha256"],
            "implementationContractHashes": manifest["identity"][
                "implementationContractHashes"
            ],
            "fixtureHashes": manifest["identity"]["fixtureHashes"],
            "compatibilityHash": manifest["compatibilityHash"],
        },
        "campaign": {
            "profiles": manifest["profiles"],
            "warmups": manifest["warmups"],
            "samples": manifest["samples"],
            "runCount": len(manifest.get("runs", [])),
            "validRunCount": sum(
                run.get("valid") is True for run in manifest.get("runs", [])
            ),
            "candidatePolicy": manifest["candidatePolicy"],
            "selectionRule": manifest["analysis"]["selectionRule"],
            "threadEnvironment": manifest["threadEnvironment"],
        },
        "implementations": implementation_rows,
        "interpretation": (
            "These are benchmark-local provisional worker defaults for this named "
            f"{cpu_brand}. They do not claim production-optimal WVM settings or a "
            "general-Mac default."
        ),
    }


def command_summarize(arguments: argparse.Namespace) -> int:
    manifest = load_json(arguments.manifest.expanduser().resolve())
    summary = publication_summary(manifest)
    write_json_atomic(arguments.output.expanduser().resolve(), summary)
    print(f"publication summary={arguments.output.expanduser().resolve()}")
    return 0


def command_compare(arguments: argparse.Namespace) -> int:
    manifests = [load_json(path.expanduser().resolve()) for path in arguments.manifest]
    for manifest in manifests:
        errors = validate_manifest(manifest)
        if errors:
            raise SystemExit("; ".join(errors))
    print(json.dumps({
        "schema": "spectral-kernel-machine-tuning-comparison-v1",
        "interpretation": (
            "Descriptive benchmark-local comparison; not a production WVM tuning claim."
        ),
        "machines": [
            {
                "machine": manifest["machine"],
                "selections": manifest.get("analysis", {}).get("selections", {}),
            }
            for manifest in manifests
        ],
    }, indent=2))
    return 0


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    subparsers = root.add_subparsers(dest="command", required=True)
    list_parser = subparsers.add_parser("list")
    list_parser.add_argument("--implementation", action="append", default=[])
    list_parser.set_defaults(handler=command_list)

    tune = subparsers.add_parser("tune")
    tune.add_argument("--implementation", action="append", default=[])
    add_fixture_arguments(tune)
    tune.add_argument(
        "--executable", type=Path,
        default=Path(__file__).resolve().parents[1] / "build/release/skbench",
    )
    tune.add_argument("--output", type=Path, required=True)
    tune.add_argument("--warmups", type=int, default=WARMUPS)
    tune.add_argument("--samples", type=int, default=SAMPLES)
    tune.add_argument("--dry-run", action="store_true")
    tune.add_argument("--allow-dirty-tree", action="store_true")
    tune.add_argument("--continue-on-error", action="store_true")
    tune.set_defaults(handler=command_tune)

    resume = subparsers.add_parser("resume")
    resume.add_argument("--manifest", type=Path, required=True)
    resume.add_argument("--dry-run", action="store_true")
    resume.add_argument("--continue-on-error", action="store_true")
    resume.set_defaults(handler=command_resume)

    validate = subparsers.add_parser("validate")
    validate.add_argument("--manifest", type=Path, required=True)
    validate.set_defaults(handler=command_validate)

    summarize = subparsers.add_parser("summarize")
    summarize.add_argument("--manifest", type=Path, required=True)
    summarize.add_argument("--output", type=Path, required=True)
    summarize.set_defaults(handler=command_summarize)

    compare = subparsers.add_parser("compare")
    compare.add_argument("manifest", nargs="+", type=Path)
    compare.set_defaults(handler=command_compare)
    return root


def main() -> int:
    arguments = parser().parse_args()
    return int(arguments.handler(arguments))


if __name__ == "__main__":
    raise SystemExit(main())
