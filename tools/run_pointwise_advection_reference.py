#!/usr/bin/env python3
"""Run the issue #22 matched M4 pointwise worker-topology campaign."""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import subprocess
import sys
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

from run_authoritative_spectral_flux_reference import (
    benchmark_sources_unchanged,
    geometric_mean,
    load_json,
    machine_record,
    percentile_bootstrap,
    stratified_geometric_bootstrap,
    write_json,
)
from run_pointwise_advection_screen import (
    BOUNDARY_POLICY,
    CONTROL_PROVIDER,
    POINTWISE_STAGE,
    TOTAL_STAGE,
    Candidate,
    collect_vectorization_evidence,
)
from run_vertical_gemm_sweep import git_source_state


EXPERIMENT_ID = "issue-022-pointwise-advection-optimization"
INCREMENT_ID = "pointwise-advection-m4-worker-reference-v1"
ANALYSIS_SCHEMA = "spectral-kernel-pointwise-advection-reference-v1"
MANIFEST_SCHEMA = "spectral-kernel-local-sweep-v1"
PROFILES = (
    "wvm-current-256-nz129-f4",
    "wvm-current-512-nz257-f4",
    "wvm-large-1024-nz129-f4",
)
DEEP_PROFILE = "wvm-large-512-nz513-f4"
REFERENCE_ROUNDS = 3
REFERENCE_WARMUPS = 3
REFERENCE_SAMPLES = 21
MEMORY_WARMUPS = 1
MEMORY_SAMPLES = 1
TOLERANCE = 1.0e-12
MEMORY_RATIO_LIMIT = 1.001
NEAR_FASTEST_RATIO = 1.01
FUSION_RESIDUAL_FRACTION = 0.10
HORIZONTAL_WORKERS = 12
VERTICAL_WORKERS = 16
VERTICAL_SCHEDULE = "outer-dynamic"
DEEP_NKL = 45_765
DEEP_NZ = 513


def candidate_matrix() -> list[Candidate]:
    candidates = [Candidate("serial-1", "serial", 1, CONTROL_PROVIDER)]
    candidates.extend(
        Candidate(
            f"spatial-static-{workers}", "spatial-static", workers,
            CONTROL_PROVIDER + "-pointwise-spatial-static",
        )
        for workers in (4, 8, 12, 16)
    )
    return candidates


def fixture_assignments(values: list[str]) -> dict[str, Path]:
    fixtures: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError("fixture assignments must use PROFILE=PREPARED_PATH")
        profile, raw_path = value.split("=", 1)
        if profile not in PROFILES:
            raise ValueError(f"unknown pointwise reference profile: {profile}")
        if profile in fixtures:
            raise ValueError(f"duplicate fixture profile: {profile}")
        path = Path(raw_path).expanduser().resolve()
        if not path.is_file():
            raise ValueError(f"prepared fixture is missing: {path}")
        fixtures[profile] = path
    missing = [profile for profile in PROFILES if profile not in fixtures]
    if missing:
        raise ValueError("missing fixture assignment(s): " + ", ".join(missing))
    return fixtures


def issue19_evidence(path: Path) -> dict:
    evidence = load_json(path)
    if evidence.get("schema") != (
        "spectral-kernel-authoritative-production-lifetime-reference-"
        "publication-v1"
    ):
        raise ValueError("issue #19 evidence has the wrong schema")
    if evidence.get("gate", {}).get(
        "advanceToWvmIntegrationExperiment") is not True:
        raise ValueError("issue #19 evidence did not pass its adoption gate")
    if set(evidence.get("fixtures", {})) != set(PROFILES):
        raise ValueError("issue #19 evidence does not cover all timing fixtures")
    matching = [
        item for item in evidence.get("capacityExclusions", [])
        if item.get("profile") == DEEP_PROFILE
        and item.get("candidateId") ==
            "production-lifetime-streaming-pruned-tile16-authoritative"
    ]
    if len(matching) != 1:
        raise ValueError("issue #19 evidence lacks the deep streaming exclusion")
    return evidence


def screen_evidence(path: Path) -> dict:
    evidence = load_json(path)
    if evidence.get("schema") != (
        "spectral-kernel-pointwise-advection-screen-v1"
    ):
        raise ValueError("issue #22 screen evidence has the wrong schema")
    if evidence.get("completeMatchedMatrix") is not True:
        raise ValueError("issue #22 screen matrix is incomplete")
    if evidence.get("screenGate", {}).get("candidatePassed") is not True:
        raise ValueError("issue #22 screen did not advance a spatial policy")
    return evidence


def machine_matches(reference: dict, current: dict) -> bool:
    expected = reference.get("machine", {})
    return all(
        expected.get(key) == current.get(key)
        for key in ("cpuBrand", "hardwareModel", "physicalMemoryBytes")
    )


def command_for(executable: Path, fixture: Path, profile: str,
                candidate: Candidate, warmups: int, samples: int,
                output: Path) -> list[str]:
    return [
        str(executable), "run",
        "--kernel", "production-lifetime-flux",
        "--boundary-policy", BOUNDARY_POLICY,
        "--spectral-flux-fixture", str(fixture),
        "--profile", profile,
        "--fftw-planning", "measure",
        "--fftw-alignment", "unaligned",
        "--fftw-wisdom", "cold",
        "--fftw-internal-workers", "1",
        "--fftw-outer-workers", str(HORIZONTAL_WORKERS),
        "--streaming-tile-width", "16",
        "--pointwise-policy", candidate.policy,
        "--pointwise-workers", str(candidate.workers),
        "--vertical-gemm-family", "k2-grouped",
        "--vertical-gemm-schedule", VERTICAL_SCHEDULE,
        "--vertical-gemm-outer-workers", str(VERTICAL_WORKERS),
        "--warmups", str(warmups),
        "--samples", str(samples),
        "--output", str(output),
    ]


def expected_scheduling(candidate: Candidate) -> str:
    scheduling = (
        f"horizontal-outer-{HORIZONTAL_WORKERS};"
        f"vertical-{VERTICAL_SCHEDULE}-{VERTICAL_WORKERS}"
        "-per-operator-family"
    )
    if candidate.policy == "spatial-static":
        scheduling += f";pointwise-spatial-static-{candidate.workers}"
    return scheduling


def expected_logical_workers(candidate: Candidate) -> int:
    return HORIZONTAL_WORKERS + VERTICAL_WORKERS + (
        0 if candidate.policy == "serial" else candidate.workers
    )


def provider_record(result: dict, candidate: Candidate) -> dict:
    providers = result.get("providers", [])
    if len(providers) != 1 or providers[0].get("id") != candidate.provider:
        raise ValueError(
            f"{candidate.id} must contain only provider {candidate.provider}"
        )
    return providers[0]


def timing_record(provider: dict, stage: str, scope: str) -> dict:
    matches = [
        item for item in provider.get("timings", [])
        if item.get("stage") == stage and item.get("scope") == scope
    ]
    if len(matches) != 1:
        raise ValueError(
            f"provider must contain one {scope!r} / {stage!r} timing"
        )
    return matches[0]


def maximum_correctness_error(provider: dict) -> float:
    metrics = provider.get("correctness", [])
    if not metrics or not all(item.get("passed") is True for item in metrics):
        return math.inf
    return max(float(item["maximumRelativeError"]) for item in metrics)


def placement_valid(provider: dict) -> bool:
    contract = provider.get("executionContract", {}).get("forward", {})
    return bool(
        contract.get("nativePlacement") == "out-of-place"
        and contract.get("adapterPlacement") == "out-of-place"
        and contract.get("destroysNativeInput") is False
        and contract.get("adapterPreservesCallerInput") is True
        and contract.get("requiresPreservationCopyForRepeatedExecution") is False
    )


def allocation_ledger_valid(provider: dict) -> bool:
    matches = [
        item for item in provider.get("componentLedger", [])
        if item.get("stage") == "steady-state allocation"
    ]
    return bool(
        len(matches) == 1 and matches[0].get("state") == "elided"
        and "persistent" in matches[0].get("detail", "")
    )


def memory_record(provider: dict) -> dict:
    memory = provider.get("memory", {})
    return {
        key: int(memory.get(key, 0))
        for key in (
            "persistentBytes", "scratchBytes", "algorithmResidentBytes",
            "benchmarkHarnessBytes", "estimatedProcessPeakBytes",
            "observedProcessHighWaterBytes",
        )
    }


def collect_record(plan: dict, result: dict, fixture_hash: str,
                   wvm_commit: str, benchmark_commit: str,
                   warmups: int, samples: int) -> dict:
    candidate = plan["candidate"]
    provider = provider_record(result, candidate)
    total = timing_record(provider, TOTAL_STAGE, "uninstrumented-total")
    pointwise = timing_record(provider, POINTWISE_STAGE, "component")
    fixture = result.get("provenance", {}).get("spectralFluxFixture", {})
    error = maximum_correctness_error(provider)
    embedded = result.get("environment", {}).get("gitCommit", "")
    source_matches = bool(
        embedded and embedded != "unknown"
        and benchmark_commit.startswith(embedded)
        and result.get("environment", {}).get("gitDirty") is False
    )
    authoritative = bool(
        fixture.get("schema") == "spectral-flux-fixture-v1"
        and fixture.get("status") == "authoritative-wvm-export"
        and fixture.get("authoritative") is True
        and fixture.get("fixtureHash") == fixture_hash
        and fixture.get("waveVortexModelCommit") == wvm_commit
    )
    valid = bool(
        result.get("status") == "passed"
        and result.get("run", {}).get("profile") == plan["profile"]
        and result.get("run", {}).get("warmups") == warmups
        and result.get("run", {}).get("samples") == samples
        and len(total.get("samplesSeconds", [])) == samples
        and len(pointwise.get("samplesSeconds", [])) == samples
        and math.isfinite(error) and error <= TOLERANCE
        and authoritative and source_matches
        and provider.get("schedulingId") == expected_scheduling(candidate)
        and provider.get("workers") == expected_logical_workers(candidate)
        and placement_valid(provider) and allocation_ledger_valid(provider)
    )
    component_details = {
        item["stage"]: float(item["medianSeconds"])
        for item in provider.get("timings", [])
        if item.get("medianSeconds") is not None
        and item.get("scope") not in {
            "setup-shared-component", "setup-component", "uninstrumented-total",
        }
    }
    return {
        "runId": result.get("run", {}).get("id"),
        "seconds": float(total["medianSeconds"]),
        "pointwiseSeconds": float(pointwise["medianSeconds"]),
        "pointwiseFractionOfTotal": (
            float(pointwise["medianSeconds"]) / float(total["medianSeconds"])
        ),
        "pointwiseBytes": int(pointwise.get("bytesMoved", 0)),
        "effectivePointwiseGigabytesPerSecond": (
            int(pointwise.get("bytesMoved", 0)) /
            float(pointwise["medianSeconds"]) / 1.0e9
        ),
        "components": component_details,
        "memory": memory_record(provider),
        "maximumCorrectnessError": error,
        "authoritativeFixture": authoritative,
        "sourceMetadataMatches": source_matches,
        "schedulingId": provider.get("schedulingId"),
        "totalLogicalWorkers": provider.get("workers"),
        "placementContractValid": placement_valid(provider),
        "allocationLedgerValid": allocation_ledger_valid(provider),
        "valid": valid,
    }


def planned_runs(executable: Path, fixtures: dict[str, Path], output: Path,
                 rounds: list[int], warmups: int, samples: int,
                 role: str) -> list[dict]:
    candidates = candidate_matrix()
    plans: list[dict] = []
    for round_number in rounds:
        profile_offset = (round_number - 1) % len(PROFILES)
        profiles = list(PROFILES[profile_offset:]) + list(PROFILES[:profile_offset])
        for profile_position, profile in enumerate(profiles):
            candidate_offset = (
                round_number - 1 + profile_position
            ) % len(candidates)
            ordered = candidates[candidate_offset:] + candidates[:candidate_offset]
            for candidate in ordered:
                identifier = (
                    f"{role}-round-{round_number}--{profile}--{candidate.id}"
                )
                result_path = output / f"{identifier}.json"
                plans.append({
                    "id": identifier,
                    "round": round_number,
                    "profile": profile,
                    "candidate": candidate,
                    "resultPath": result_path,
                    "command": command_for(
                        executable, fixtures[profile], profile, candidate,
                        warmups, samples, result_path,
                    ),
                })
    return plans


def deep_capacity_exclusions(issue19: dict) -> list[dict]:
    base = next(
        int(item["requiredPhysicalMemoryBytes"])
        for item in issue19["capacityExclusions"]
        if item["profile"] == DEEP_PROFILE
        and item["candidateId"] ==
            "production-lifetime-streaming-pruned-tile16-authoritative"
    )
    direct_view_savings = 64 * DEEP_NKL * DEEP_NZ
    physical = int(issue19["machine"]["physicalMemoryBytes"])
    return [{
        "profile": DEEP_PROFILE,
        "candidateId": candidate.id,
        "status": "capacity-exclusion",
        "requiredPhysicalMemoryBytes": (
            base +
            (0 if candidate.policy == "serial" else 8 * (candidate.workers - 1))
        ),
        "physicalMemoryBytes": physical,
        "directFamilyViewSteadyStateSavingsBytes": direct_view_savings,
        "pointwiseSchedulerExplicitBytes": (
            0 if candidate.policy == "serial" else 8 * (candidate.workers - 1)
        ),
        "sourceEvidence": "issue-019-authoritative-reference-lyra-v1",
        "reason": (
            "the conservative issue #19 vertical-operator construction peak "
            "dominates the deep workload; #21 direct family views elide an "
            "exact four-complex-field steady-state bridge but do not reduce "
            "that setup peak, which still exceeds physical memory"
        ),
    } for candidate in candidate_matrix()]


def manifest_template(phase: str, benchmark_commit: str, runner_commit: str,
                      fixtures: dict[str, Path], warmups: int,
                      samples: int) -> dict:
    return {
        "schema": MANIFEST_SCHEMA,
        "experimentId": EXPERIMENT_ID,
        "incrementId": INCREMENT_ID,
        "phase": phase,
        "classification": "reference",
        "createdAtUtc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "question": (
            "Which one uniform persistent spatial pointwise worker count gives "
            "the fastest correct issue #21 direct-family-view boundary on the M4?"
        ),
        "baseline": (
            f"The clean {benchmark_commit[:12]} issue #21 direct-family-view graph with the "
            "original serial pointwise expression."
        ),
        "controlledVariables": [
            "authoritative WVM fixtures and independent mode-keyed oracle",
            "issue #21 partial-column-pruned tile-16 direct F/G family-view graph",
            "horizontal outer-12 and vertical outer-dynamic-16 topology",
            "Float64 radial and vertical two-thirds antialiasing",
            "three shared, three reusable derivative, and one target real volume",
            "sequential target streaming and zero warmed application allocations",
        ],
        "changedVariables": [
            "serial versus static contiguous pointwise chunks",
            "pointwise logical worker count 4, 8, 12, or 16",
        ],
        "timedOperation": (
            "The isolated four-pass pointwise expression and independently "
            "sampled complete 15-modal-input to four-modal-output issue #21 boundary."
        ),
        "excludedWork": [
            "complete nonlinear flux, MATLAB/MEX, state, timestep, and I/O",
            "fixture load, planning, pool construction, and correctness storage from steady state",
            "pointwise-to-FFT fusion, target concurrency, and size-dependent dispatch",
        ],
        "selectionRule": (
            "Among candidates passing the 0.95 geometric total, 1.03 worst-"
            "profile, interval, correctness, allocation, and memory gates, "
            "select the smallest worker count within 1% of the fastest geometric total."
        ),
        "allocationPolicy": "zero application allocations after persistent setup",
        "profiles": list(PROFILES),
        "capacityProfile": DEEP_PROFILE,
        "fixtures": {profile: str(path) for profile, path in fixtures.items()},
        "candidates": [asdict(candidate) for candidate in candidate_matrix()],
        "benchmarkExecutableCommit": benchmark_commit,
        "runnerSourceTreeGitCommit": runner_commit,
        "sourceTreeDirty": False,
        "warmups": warmups,
        "samples": samples,
        "threadEnvironment": {"VECLIB_MAXIMUM_THREADS": "1"},
        "runs": [],
    }


def run_plans(repository_root: Path, output: Path, manifest: dict,
              plans: list[dict], issue19: dict, benchmark_commit: str,
              continue_on_error: bool) -> tuple[list[dict], bool]:
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "manifest.json"
    if manifest_path.is_file():
        current = load_json(manifest_path)
        for key in (
            "schema", "experimentId", "incrementId", "phase", "profiles",
            "capacityProfile", "fixtures", "candidates",
            "benchmarkExecutableCommit", "runnerSourceTreeGitCommit",
            "sourceTreeDirty", "warmups", "samples",
        ):
            if current.get(key) != manifest.get(key):
                raise ValueError(f"existing manifest disagrees on {key}")
        manifest = current
    else:
        write_json(manifest_path, manifest)
    existing = {item["id"]: item for item in manifest.get("runs", [])}
    records: list[dict] = []
    failed = False
    for index, plan in enumerate(plans, start=1):
        entry = existing.get(plan["id"])
        if entry is not None:
            if entry.get("exitCode") != 0 or not plan["resultPath"].is_file():
                raise ValueError(f"existing run requires recovery: {plan['id']}")
            result = load_json(plan["resultPath"])
            record = collect_record(
                plan, result, issue19["fixtures"][plan["profile"]],
                issue19["waveVortexModelCommit"], benchmark_commit,
                manifest["warmups"], manifest["samples"],
            )
            if not record["valid"]:
                raise ValueError(f"existing run no longer validates: {plan['id']}")
            records.append({
                "candidateId": plan["candidate"].id,
                "profile": plan["profile"], "round": plan["round"],
                "record": record,
            })
            print(f"[{index}/{len(plans)}] reuse {plan['id']}", flush=True)
            continue
        print(f"[{index}/{len(plans)}] {plan['id']}", flush=True)
        log_path = plan["resultPath"].with_suffix(".log")
        environment = os.environ.copy()
        environment["VECLIB_MAXIMUM_THREADS"] = "1"
        with log_path.open("w", encoding="utf-8") as log:
            completed = subprocess.run(
                plan["command"], cwd=repository_root, env=environment,
                stdout=log, stderr=subprocess.STDOUT,
            )
        entry = {
            "id": plan["id"], "round": plan["round"],
            "profile": plan["profile"],
            "candidate": asdict(plan["candidate"]),
            "measurementRole": manifest["phase"],
            "environment": {"VECLIB_MAXIMUM_THREADS": "1"},
            "command": list(map(str, plan["command"])),
            "result": plan["resultPath"].name,
            "log": log_path.name,
            "exitCode": completed.returncode,
        }
        if completed.returncode == 0 and plan["resultPath"].is_file():
            result = load_json(plan["resultPath"])
            record = collect_record(
                plan, result, issue19["fixtures"][plan["profile"]],
                issue19["waveVortexModelCommit"], benchmark_commit,
                manifest["warmups"], manifest["samples"],
            )
            entry.update({
                "runId": record["runId"], "valid": record["valid"],
                "embeddedGitCommit": result.get("environment", {}).get(
                    "gitCommit"),
                "embeddedGitDirty": result.get("environment", {}).get(
                    "gitDirty"),
            })
            if record["valid"]:
                records.append({
                    "candidateId": plan["candidate"].id,
                    "profile": plan["profile"], "round": plan["round"],
                    "record": record,
                })
            else:
                entry["exitCode"] = 1
        manifest["runs"].append(entry)
        write_json(manifest_path, manifest)
        if entry["exitCode"] != 0:
            failed = True
            print(log_path.read_text(encoding="utf-8")[-4000:], file=sys.stderr)
            if not continue_on_error:
                break
    return records, failed


def analyze(timing_records: list[dict], memory_records: list[dict],
            allocation: dict, vectorization: dict, capacity: list[dict],
            benchmark_commit: str, runner_commit: str,
            issue19: dict) -> dict:
    candidates = candidate_matrix()
    candidate_ids = [candidate.id for candidate in candidates]
    timing_cells = {
        (item["profile"], item["candidateId"], item["round"]): item["record"]
        for item in timing_records
    }
    memory_cells = {
        (item["profile"], item["candidateId"]): item["record"]
        for item in memory_records
    }
    expected_timing = {
        (profile, candidate.id, round_number)
        for profile in PROFILES for candidate in candidates
        for round_number in range(1, REFERENCE_ROUNDS + 1)
    }
    expected_memory = {
        (profile, candidate.id)
        for profile in PROFILES for candidate in candidates
    }
    complete = bool(
        set(timing_cells) == expected_timing
        and all(record["valid"] for record in timing_cells.values())
    )
    memory_complete = bool(
        set(memory_cells) == expected_memory
        and all(record["valid"] for record in memory_cells.values())
    )
    maximum_error = max(
        [record["maximumCorrectnessError"] for record in timing_cells.values()]
        + [record["maximumCorrectnessError"] for record in memory_cells.values()]
        + [0.0]
    )
    summaries: list[dict] = []
    for candidate_index, candidate in enumerate(candidates):
        profiles: list[dict] = []
        stratified_total: dict[str, list[float]] = {}
        total_profile_ratios: list[float] = []
        pointwise_profile_ratios: list[float] = []
        residual_fractions: list[float] = []
        memory_profile_ratios: list[float] = []
        for profile_index, profile in enumerate(PROFILES):
            rounds = list(range(1, REFERENCE_ROUNDS + 1))
            if any(
                (profile, candidate.id, round_number) not in timing_cells
                or (profile, "serial-1", round_number) not in timing_cells
                for round_number in rounds
            ):
                continue
            candidate_records = [
                timing_cells[(profile, candidate.id, round_number)]
                for round_number in rounds
            ]
            controls = [
                timing_cells[(profile, "serial-1", round_number)]
                for round_number in rounds
            ]
            total_ratios = [
                candidate_records[index]["seconds"] / controls[index]["seconds"]
                for index in range(len(rounds))
            ]
            pointwise_ratios = [
                candidate_records[index]["pointwiseSeconds"] /
                controls[index]["pointwiseSeconds"]
                for index in range(len(rounds))
            ]
            total_median = statistics.median(total_ratios)
            pointwise_median = statistics.median(pointwise_ratios)
            lower, upper = percentile_bootstrap(
                total_ratios, seed=22000 + 10 * candidate_index + profile_index,
            )
            stratified_total[profile] = total_ratios
            total_profile_ratios.append(total_median)
            pointwise_profile_ratios.append(pointwise_median)
            residual_fractions.append(statistics.median(
                record["pointwiseFractionOfTotal"]
                for record in candidate_records
            ))
            memory_ratio = None
            if (
                (profile, candidate.id) in memory_cells
                and (profile, "serial-1") in memory_cells
            ):
                memory_ratio = (
                    memory_cells[(profile, candidate.id)]["memory"]
                    ["algorithmResidentBytes"] /
                    memory_cells[(profile, "serial-1")]["memory"]
                    ["algorithmResidentBytes"]
                )
                memory_profile_ratios.append(memory_ratio)
            profiles.append({
                "profile": profile,
                "rounds": rounds,
                "controlMedianSeconds": statistics.median(
                    record["seconds"] for record in controls
                ),
                "candidateMedianSeconds": statistics.median(
                    record["seconds"] for record in candidate_records
                ),
                "candidateToSerial": total_median,
                "roundTotalRatios": total_ratios,
                "empiricalPairedRange": {"lower": lower, "upper": upper},
                "pointwiseMedianSeconds": statistics.median(
                    record["pointwiseSeconds"] for record in candidate_records
                ),
                "pointwiseToSerial": pointwise_median,
                "pointwiseFractionOfTotal": residual_fractions[-1],
                "effectivePointwiseGigabytesPerSecond": statistics.median(
                    record["effectivePointwiseGigabytesPerSecond"]
                    for record in candidate_records
                ),
                "algorithmResidentToSerial": memory_ratio,
                "memoryOnly": (
                    memory_cells[(profile, candidate.id)]["memory"]
                    if (profile, candidate.id) in memory_cells else None
                ),
            })
        summary_complete = len(profiles) == len(PROFILES)
        aggregate = (
            geometric_mean(total_profile_ratios) if summary_complete else None
        )
        maximum = max(total_profile_ratios) if summary_complete else None
        interval = None
        if summary_complete:
            lower, upper = stratified_geometric_bootstrap(stratified_total)
            interval = {"lower": lower, "upper": upper}
        pointwise_aggregate = (
            geometric_mean(pointwise_profile_ratios)
            if summary_complete else None
        )
        residual = (
            geometric_mean(residual_fractions) if summary_complete else None
        )
        maximum_memory = (
            max(memory_profile_ratios)
            if len(memory_profile_ratios) == len(PROFILES) else None
        )
        eligible = bool(
            candidate.id != "serial-1" and complete and memory_complete
            and maximum_error <= TOLERANCE
            and allocation.get("exitCode") == 0
            and vectorization.get("passed") is True
            and aggregate is not None and aggregate <= 0.95
            and maximum is not None and maximum <= 1.03
            and interval is not None and interval["upper"] < 1.0
            and maximum_memory is not None
            and maximum_memory <= MEMORY_RATIO_LIMIT
        )
        summaries.append({
            "candidate": asdict(candidate),
            "complete": summary_complete,
            "geometricTotalToSerial": aggregate,
            "maximumProfileTotalToSerial": maximum,
            "empiricalStratifiedPairedRange": interval,
            "geometricPointwiseToSerial": pointwise_aggregate,
            "geometricResidualPointwiseFraction": residual,
            "maximumProfileAlgorithmResidentToSerial": maximum_memory,
            "profiles": profiles,
            "eligible": eligible,
        })
    eligible = [summary for summary in summaries if summary["eligible"]]
    fastest = min(
        eligible, key=lambda item: item["geometricTotalToSerial"],
        default=None,
    )
    near_fastest = []
    selected = None
    if fastest is not None:
        near_fastest = [
            summary for summary in eligible
            if summary["geometricTotalToSerial"] <=
                fastest["geometricTotalToSerial"] * NEAR_FASTEST_RATIO
        ]
        selected = min(
            near_fastest,
            key=lambda item: (
                item["candidate"]["workers"],
                item["geometricTotalToSerial"],
            ),
        )
    capacity_complete = {
        item["candidateId"] for item in capacity
        if item.get("profile") == DEEP_PROFILE
        and item.get("status") == "capacity-exclusion"
        and item.get("requiredPhysicalMemoryBytes", 0) >
            item.get("physicalMemoryBytes", 0)
    } == set(candidate_ids)
    selected_residual = (
        selected["geometricResidualPointwiseFraction"]
        if selected is not None else None
    )
    adoption = bool(
        selected is not None and capacity_complete
        and selected["eligible"]
    )
    return {
        "schema": ANALYSIS_SCHEMA,
        "experimentId": EXPERIMENT_ID,
        "incrementId": INCREMENT_ID,
        "classification": "reference",
        "createdAtUtc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "machine": machine_record(),
        "benchmarkExecutableCommit": benchmark_commit,
        "runnerSourceTreeGitCommit": runner_commit,
        "waveVortexModelCommit": issue19["waveVortexModelCommit"],
        "profiles": list(PROFILES),
        "rounds": REFERENCE_ROUNDS,
        "completeMatchedMatrix": complete,
        "memoryEvidenceComplete": memory_complete,
        "allCorrectWithin1e12": maximum_error <= TOLERANCE,
        "maximumCorrectnessError": maximum_error,
        "candidateSummaries": summaries,
        "fastestEligibleCandidate": fastest,
        "nearFastestCandidates": [
            summary["candidate"]["id"] for summary in near_fastest
        ],
        "selectedCandidate": selected,
        "selectionRule": {
            "nearFastestFactor": NEAR_FASTEST_RATIO,
            "tieBreak": "smallest worker count",
            "sizeDependentDispatchAllowed": False,
        },
        "allocationVerification": allocation,
        "vectorizationEvidence": vectorization,
        "capacityExclusions": capacity,
        "capacityEvidenceComplete": capacity_complete,
        "adoptionGate": {
            "geometricTotalRatioAtMost": 0.95,
            "maximumProfileTotalRatioAtMost": 1.03,
            "empiricalIntervalExcludesTie": bool(
                selected is not None
                and selected["empiricalStratifiedPairedRange"]["upper"] < 1.0
            ),
            "correctnessPassed": maximum_error <= TOLERANCE,
            "allocationVerificationPassed": allocation.get("exitCode") == 0,
            "memoryDoesNotMateriallyRegress": bool(
                selected is not None
                and selected["maximumProfileAlgorithmResidentToSerial"] <=
                    MEMORY_RATIO_LIMIT
            ),
            "capacityEvidenceComplete": capacity_complete,
            "uniformPolicyAcrossSizes": selected is not None,
            "freezeSelectedM4PointwisePolicy": adoption,
        },
        "fusionContinuation": {
            "residualPointwiseFractionThreshold": FUSION_RESIDUAL_FRACTION,
            "selectedResidualPointwiseFraction": selected_residual,
            "activateBoundedPointwiseFftFusion": bool(
                selected is None
                or selected_residual is None
                or selected_residual > FUSION_RESIDUAL_FRACTION
            ),
        },
        "limitations": [
            "Reference status freezes an M4 policy only; it is not a general-Mac default.",
            "The deep workload is a conservative capacity result without a materialized fixture or timing run.",
            "The benchmark remains a synthetic production-shaped spectral boundary, not the complete nonlinear flux or MATLAB/MEX path.",
        ],
    }


def main() -> int:
    repository_root = Path(__file__).resolve().parents[1]
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", action="append", required=True)
    parser.add_argument(
        "--issue19-evidence", type=Path,
        default=repository_root / "results/published/decisions" /
            "issue-019-authoritative-reference-lyra-v1.json",
    )
    parser.add_argument("--screen-analysis", type=Path, required=True)
    parser.add_argument("--benchmark-commit", required=True)
    parser.add_argument(
        "--executable", type=Path,
        default=repository_root / "build/release/skbench",
    )
    parser.add_argument(
        "--test-executable", type=Path,
        default=repository_root / "build/release/skbench_tests",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    arguments = parser.parse_args()
    try:
        fixtures = fixture_assignments(arguments.fixture)
        issue19 = issue19_evidence(arguments.issue19_evidence.resolve())
        screen = screen_evidence(arguments.screen_analysis.resolve())
        benchmark_commit = subprocess.run(
            ["git", "rev-parse", arguments.benchmark_commit],
            cwd=repository_root, check=True, text=True,
            capture_output=True,
        ).stdout.strip()
    except (OSError, ValueError, json.JSONDecodeError,
            subprocess.CalledProcessError) as error:
        parser.error(str(error))
    del screen
    for path, label in (
        (arguments.executable, "benchmark executable"),
        (arguments.test_executable, "test executable"),
    ):
        if not path.is_file():
            parser.error(f"{label} is missing: {path}")
    current_machine = machine_record()
    if not machine_matches(issue19, current_machine):
        parser.error("issue #19 evidence was produced on a different machine")
    runner_commit, dirty = git_source_state(repository_root)
    if dirty:
        parser.error("the runner source tree must be clean before collection")
    if not benchmark_sources_unchanged(
        repository_root, benchmark_commit, runner_commit,
    ):
        parser.error(
            "benchmark source changed after the requested frozen executable commit"
        )

    output = (arguments.output or (
        repository_root / "results/local" /
        f"issue22-pointwise-reference-{timestamp}"
    )).resolve()
    timing_output = output / "timing"
    memory_output = output / "memory"
    timing_plans = planned_runs(
        arguments.executable.resolve(), fixtures, timing_output,
        list(range(1, REFERENCE_ROUNDS + 1)),
        REFERENCE_WARMUPS, REFERENCE_SAMPLES, "reference",
    )
    memory_plans = planned_runs(
        arguments.executable.resolve(), fixtures, memory_output,
        [1], MEMORY_WARMUPS, MEMORY_SAMPLES, "memory",
    )
    capacity = deep_capacity_exclusions(issue19)
    if arguments.dry_run:
        print(arguments.test_executable.resolve())
        for plan in timing_plans:
            print("VECLIB_MAXIMUM_THREADS=1 " + " ".join(plan["command"]))
        for plan in memory_plans:
            print("MEMORY VECLIB_MAXIMUM_THREADS=1 " +
                  " ".join(plan["command"]))
        print(json.dumps({"capacityExclusions": capacity}, indent=2))
        return 0

    output.mkdir(parents=True, exist_ok=True)
    allocation_log = output / "allocation-verification.log"
    with allocation_log.open("w", encoding="utf-8") as log:
        allocation_run = subprocess.run(
            [str(arguments.test_executable.resolve())], cwd=repository_root,
            stdout=log, stderr=subprocess.STDOUT,
        )
    allocation = {
        "command": [str(arguments.test_executable.resolve())],
        "exitCode": allocation_run.returncode,
        "log": allocation_log.name,
        "benchmarkExecutableCommit": benchmark_commit,
    }
    if allocation_run.returncode != 0:
        print(allocation_log.read_text(encoding="utf-8")[-4000:], file=sys.stderr)
        return allocation_run.returncode
    vectorization = collect_vectorization_evidence(repository_root, output)
    if not vectorization["passed"]:
        print((output / vectorization["log"]).read_text(
            encoding="utf-8")[-4000:], file=sys.stderr)
        return 1

    timing_manifest = manifest_template(
        "reference", benchmark_commit, runner_commit, fixtures,
        REFERENCE_WARMUPS, REFERENCE_SAMPLES,
    )
    memory_manifest = manifest_template(
        "memory", benchmark_commit, runner_commit, fixtures,
        MEMORY_WARMUPS, MEMORY_SAMPLES,
    )
    try:
        timing_records, timing_failed = run_plans(
            repository_root, timing_output, timing_manifest, timing_plans,
            issue19, benchmark_commit, arguments.continue_on_error,
        )
        memory_records, memory_failed = run_plans(
            repository_root, memory_output, memory_manifest, memory_plans,
            issue19, benchmark_commit, arguments.continue_on_error,
        )
    except ValueError as error:
        parser.error(str(error))
    result = analyze(
        timing_records, memory_records, allocation, vectorization, capacity,
        benchmark_commit, runner_commit, issue19,
    )
    write_json(output / "analysis.json", result)
    selected = result.get("selectedCandidate")
    if selected is None:
        print("reference result: no spatial policy passed every gate")
    else:
        print(
            "reference result: freeze " + selected["candidate"]["id"] +
            f"; geometric total/serial={selected['geometricTotalToSerial']:.4f}; "
            f"pointwise/serial={selected['geometricPointwiseToSerial']:.4f}"
        )
    print(
        "pointwise/FFT fusion: " +
        ("activate" if result["fusionContinuation"]
         ["activateBoundedPointwiseFftFusion"] else "defer")
    )
    print(f"analysis: {output / 'analysis.json'}")
    return 1 if timing_failed or memory_failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
