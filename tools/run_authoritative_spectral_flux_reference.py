#!/usr/bin/env python3
"""Run the issue #19 authoritative production-lifetime reference campaign."""

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

from run_cross_mac_spectral_reference import (
    ScheduleTopology,
    calibration_matches_machine,
    machine_record,
    percentile_bootstrap,
)
from run_production_lifetime_flux_authoritative_pilot import (
    Candidate,
    component_seconds,
    memory_record,
    provider_record,
    total_seconds,
    candidate_matrix,
)
from run_spectral_pipeline_sweep import (
    geometric_mean,
    maximum_correctness_error,
    stratified_geometric_bootstrap,
)
from run_vertical_gemm_sweep import git_source_state


EXPERIMENT_ID = "issue-019-production-lifetime-spectral-flux-composition"
INCREMENT_ID = "production-lifetime-flux-authoritative-reference-v1"
ANALYSIS_SCHEMA = (
    "spectral-kernel-authoritative-production-lifetime-reference-analysis-v1"
)
CAMPAIGN_SCHEMA = "spectral-kernel-authoritative-reference-campaign-v1"
MANIFEST_SCHEMA = "spectral-kernel-local-sweep-v1"
REFERENCE_ROUNDS = 3
EXTENDED_REFERENCE_ROUNDS = 5
REFERENCE_WARMUPS = 3
REFERENCE_SAMPLES = 21
MEMORY_WARMUPS = 1
MEMORY_SAMPLES = 1
TOLERANCE = 1.0e-12
CONTROL_ID = "production-lifetime-wvm-direct-authoritative"
CANDIDATE_ID = "production-lifetime-streaming-pruned-tile16-authoritative"
PROFILES = (
    "wvm-current-256-nz129-f4",
    "wvm-current-512-nz257-f4",
    "wvm-large-1024-nz129-f4",
    "wvm-large-512-nz513-f4",
)
TIMING_PROFILES = PROFILES[:3]
BENCHMARK_SOURCE_PATHS = (
    "CMakeLists.txt",
    "cmake",
    "include",
    "src",
    "tests/skbench_tests.cpp",
)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def fixture_assignments(values: list[str]) -> dict[str, Path]:
    fixtures: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError("fixture assignments must use PROFILE=PREPARED_PATH")
        profile, raw_path = value.split("=", 1)
        if profile not in TIMING_PROFILES:
            raise ValueError(f"unknown reference fixture profile: {profile}")
        if profile in fixtures:
            raise ValueError(f"duplicate reference fixture profile: {profile}")
        path = Path(raw_path).expanduser().resolve()
        if not path.is_file():
            raise ValueError(f"prepared fixture is missing: {path}")
        fixtures[profile] = path
    missing = [profile for profile in TIMING_PROFILES if profile not in fixtures]
    if missing:
        raise ValueError(
            "missing reference fixture assignment(s): " + ", ".join(missing)
        )
    return fixtures


def selected_topologies(calibration: dict) -> dict[str, ScheduleTopology]:
    if calibration.get("schema") != (
        "spectral-kernel-authoritative-flux-calibration-analysis-v1"
    ):
        raise ValueError("calibration analysis has the wrong schema")
    if calibration.get("topologiesFrozenForReference") is not True:
        raise ValueError("calibration did not freeze reference topologies")
    selected: dict[str, ScheduleTopology] = {}
    for candidate in candidate_matrix():
        raw = calibration.get("selections", {}).get(candidate.id, {}).get(
            "selectedTopology"
        )
        if not isinstance(raw, dict):
            raise ValueError(f"calibration has no topology for {candidate.id}")
        selected[candidate.id] = ScheduleTopology(**raw)
    return selected


def benchmark_sources_unchanged(
    repository_root: Path, benchmark_commit: str, runner_commit: str,
) -> bool:
    ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor", benchmark_commit, runner_commit],
        cwd=repository_root,
    )
    if ancestor.returncode != 0:
        return False
    changed = subprocess.run(
        [
            "git", "diff", "--quiet", f"{benchmark_commit}..{runner_commit}",
            "--", *BENCHMARK_SOURCE_PATHS,
        ],
        cwd=repository_root,
    )
    return changed.returncode == 0


def capacity_and_fixture_evidence(
    calibration: dict, capacity: dict,
) -> tuple[dict[str, dict], list[dict]]:
    if capacity.get("schema") != (
        "spectral-kernel-authoritative-scaleout-capacity-publication-v1"
    ):
        raise ValueError("capacity evidence has the wrong schema")
    wvm_commit = calibration.get("waveVortexModelCommit")
    if not wvm_commit or capacity.get("waveVortexModelCommit") != wvm_commit:
        raise ValueError("calibration and capacity evidence use different WVM commits")
    expected: dict[str, dict] = {
        profile: {
            "fixtureHash": fixture_hash,
            "waveVortexModelCommit": wvm_commit,
        }
        for profile, fixture_hash in calibration.get("fixtureHashes", {}).items()
    }
    exclusions: list[dict] = []
    for workload in capacity.get("workloads", []):
        profile = workload["profile"]
        fixture_hash = workload.get("fixtureHash")
        if fixture_hash is not None:
            existing = expected.get(profile)
            if existing is not None and existing["fixtureHash"] != fixture_hash:
                raise ValueError(f"fixture hash disagreement for {profile}")
            expected[profile] = {
                "fixtureHash": fixture_hash,
                "waveVortexModelCommit": wvm_commit,
            }
        for graph in workload.get("graphs", []):
            if graph.get("status") == "capacity-exclusion":
                exclusions.append({
                    "profile": profile,
                    "candidateId": graph["candidateId"],
                    "requiredPhysicalMemoryBytes": graph[
                        "requiredPhysicalMemoryBytes"
                    ],
                    "physicalMemoryBytes": capacity["machine"][
                        "physicalMemoryBytes"
                    ],
                    "reason": (
                        "authoritative scale-out exact setup-plus-reserve preflight "
                        "exceeds physical memory"
                    ),
                    "sourceEvidence": (
                        "issue-019-authoritative-scaleout-capacity-v1"
                    ),
                })
    if set(expected) != set(TIMING_PROFILES):
        raise ValueError("fixture evidence does not cover all timing profiles")
    deep_exclusions = [
        item for item in exclusions
        if item["profile"] == "wvm-large-512-nz513-f4"
    ]
    if {item["candidateId"] for item in deep_exclusions} != {
        CONTROL_ID, CANDIDATE_ID,
    }:
        raise ValueError("deep workload lacks both frozen-graph capacity exclusions")
    return expected, deep_exclusions


def command_for(
    executable: Path, fixture: Path, profile: str, candidate: Candidate,
    topology: ScheduleTopology, warmups: int, samples: int, output: Path,
) -> list[str]:
    return [
        str(executable), "run",
        "--kernel", "production-lifetime-flux",
        "--profile", profile,
        "--boundary-policy", candidate.policy,
        "--spectral-flux-fixture", str(fixture),
        "--fftw-planning", "measure",
        "--fftw-alignment", "unaligned",
        "--fftw-wisdom", "cold",
        "--fftw-internal-workers", "1",
        "--fftw-outer-workers", str(topology.horizontal_workers),
        "--streaming-tile-width", "16",
        "--vertical-gemm-family", "k2-grouped",
        "--vertical-gemm-schedule", topology.vertical_schedule,
        "--vertical-gemm-outer-workers", str(topology.vertical_workers),
        "--warmups", str(warmups),
        "--samples", str(samples),
        "--output", str(output),
    ]


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


def result_record(
    candidate: Candidate, topology: ScheduleTopology, result: dict,
    expected_fixture: dict, benchmark_commit: str, warmups: int, samples: int,
) -> dict:
    provider = provider_record(candidate, result)
    fixture = result.get("provenance", {}).get("spectralFluxFixture", {})
    total_matches = [
        item for item in provider.get("timings", [])
        if item.get("scope") == "uninstrumented-total"
        and item.get("stage") == (
            "authoritative production-lifetime streamed four-target "
            "spectral-flux composition"
        )
        and item.get("direction") == "forward"
    ]
    error = maximum_correctness_error(provider)
    expected_schedule = (
        f"horizontal-outer-{topology.horizontal_workers};"
        f"vertical-{topology.vertical_schedule}-{topology.vertical_workers}"
        "-per-operator-family"
    )
    embedded_commit = result.get("environment", {}).get("gitCommit", "")
    source_matches = bool(
        embedded_commit and embedded_commit != "unknown"
        and benchmark_commit.startswith(embedded_commit)
        and result.get("environment", {}).get("gitDirty") is False
    )
    authoritative = bool(
        fixture.get("schema") == "spectral-flux-fixture-v1"
        and fixture.get("status") == "authoritative-wvm-export"
        and fixture.get("authoritative") is True
        and fixture.get("fixtureHash") == expected_fixture["fixtureHash"]
        and fixture.get("waveVortexModelCommit") ==
            expected_fixture["waveVortexModelCommit"]
    )
    valid = bool(
        result.get("status") == "passed"
        and result.get("run", {}).get("warmups") == warmups
        and result.get("run", {}).get("samples") == samples
        and len(total_matches) == 1
        and len(total_matches[0].get("samplesSeconds", [])) == samples
        and math.isfinite(error) and error <= TOLERANCE
        and authoritative
        and provider.get("schedulingId") == expected_schedule
        and source_matches
        and placement_valid(provider)
        and allocation_ledger_valid(provider)
    )
    return {
        "runId": result.get("run", {}).get("id"),
        "seconds": total_seconds(provider),
        "components": component_seconds(provider),
        "setup": {
            "totalSeconds": float(provider.get("setup", {}).get("totalSeconds", 0.0)),
            "planningSeconds": float(provider.get("planning", {}).get("seconds", 0.0)),
        },
        "memory": memory_record(provider),
        "maximumCorrectnessError": error,
        "authoritativeFixture": authoritative,
        "fixtureHash": fixture.get("fixtureHash"),
        "waveVortexModelCommit": fixture.get("waveVortexModelCommit"),
        "schedulingId": provider.get("schedulingId"),
        "placementContractValid": placement_valid(provider),
        "allocationLedgerValid": allocation_ledger_valid(provider),
        "sourceMetadataMatches": source_matches,
        "valid": valid,
    }


def planned_runs(
    executable: Path, fixtures: dict[str, Path], output: Path,
    topologies: dict[str, ScheduleTopology], rounds: list[int],
    warmups: int, samples: int, measurement_role: str,
) -> list[dict]:
    candidates = candidate_matrix()
    plans: list[dict] = []
    for round_number in rounds:
        profile_offset = (round_number - 1) % len(TIMING_PROFILES)
        profile_order = (
            list(TIMING_PROFILES[profile_offset:])
            + list(TIMING_PROFILES[:profile_offset])
        )
        candidate_offset = (round_number - 1) % len(candidates)
        candidate_order = candidates[candidate_offset:] + candidates[:candidate_offset]
        for profile in profile_order:
            for candidate in candidate_order:
                topology = topologies[candidate.id]
                identifier = (
                    f"{measurement_role}-round-{round_number}--{profile}--"
                    f"{candidate.id}"
                )
                result_path = output / f"{identifier}.json"
                plans.append({
                    "id": identifier,
                    "round": round_number,
                    "profile": profile,
                    "candidate": candidate,
                    "topology": topology,
                    "resultPath": result_path,
                    "command": command_for(
                        executable, fixtures[profile], profile, candidate,
                        topology, warmups, samples, result_path,
                    ),
                })
    return plans


def conditional_round_decision(records: list[dict]) -> dict:
    cells: dict[tuple[str, str], dict[int, float]] = {}
    for item in records:
        cells.setdefault((item["candidateId"], item["profile"]), {})[
            item["round"]
        ] = item["record"]["seconds"]
    profile_ratios: dict[str, list[float]] = {}
    triggers: list[dict] = []
    for profile in TIMING_PROFILES:
        baseline = cells.get((CONTROL_ID, profile), {})
        candidate = cells.get((CANDIDATE_ID, profile), {})
        rounds = sorted(set(baseline) & set(candidate))
        ratios = [candidate[index] / baseline[index] for index in rounds]
        if len(ratios) != REFERENCE_ROUNDS:
            continue
        profile_ratios[profile] = ratios
        spread = max(ratios) / min(ratios)
        if spread > 1.10:
            triggers.append({
                "id": "profile-ratio-spread", "profile": profile,
                "value": spread, "threshold": 1.10,
            })
        if min(ratios) <= 1.03 <= max(ratios) and min(ratios) < max(ratios):
            triggers.append({
                "id": "profile-straddles-regression-boundary",
                "profile": profile, "lower": min(ratios),
                "upper": max(ratios), "boundary": 1.03,
            })
    aggregate: list[float] = []
    if len(profile_ratios) == len(TIMING_PROFILES):
        aggregate = [
            geometric_mean([
                profile_ratios[profile][round_index]
                for profile in sorted(profile_ratios)
            ])
            for round_index in range(REFERENCE_ROUNDS)
        ]
        if min(aggregate) <= 0.90 <= max(aggregate) and min(aggregate) < max(aggregate):
            triggers.append({
                "id": "aggregate-straddles-improvement-boundary",
                "lower": min(aggregate), "upper": max(aggregate),
                "boundary": 0.90,
            })
        aggregate_median = statistics.median(aggregate)
        if 0.85 <= aggregate_median <= 0.95:
            triggers.append({
                "id": "aggregate-median-near-improvement-boundary",
                "value": aggregate_median, "lower": 0.85, "upper": 0.95,
            })
    complete = len(profile_ratios) == len(TIMING_PROFILES)
    return {
        "completeInitialThreeRoundMatrix": complete,
        "profileRoundRatios": profile_ratios,
        "aggregateRoundRatios": aggregate,
        "triggers": triggers,
        "runAdditionalTwoRounds": complete and bool(triggers),
        "finalRoundCount": (
            EXTENDED_REFERENCE_ROUNDS if complete and triggers
            else REFERENCE_ROUNDS
        ),
    }


def analyze(
    timing_records: list[dict], memory_records: list[dict],
    capacity_exclusions: list[dict], calibration: dict,
    allocation_verification: dict, runner_commit: str,
) -> dict:
    timing_cells: dict[tuple[str, str], dict[int, dict]] = {}
    maximum_error = 0.0
    for item in timing_records:
        record = item["record"]
        timing_cells.setdefault(
            (item["candidateId"], item["profile"]), {}
        )[item["round"]] = record
        maximum_error = max(maximum_error, float(record["maximumCorrectnessError"]))
    memory_cells: dict[tuple[str, str], dict] = {}
    for item in memory_records:
        record = item["record"]
        memory_cells[(item["candidateId"], item["profile"])] = record
        maximum_error = max(maximum_error, float(record["maximumCorrectnessError"]))

    observed_rounds = sorted({
        round_number
        for rounds in timing_cells.values()
        for round_number in rounds
    })
    expected_rounds = list(range(1, len(observed_rounds) + 1))
    protocol_complete = bool(
        observed_rounds == expected_rounds
        and len(observed_rounds) in {REFERENCE_ROUNDS, EXTENDED_REFERENCE_ROUNDS}
    )
    profiles: list[dict] = []
    stratified_ratios: dict[str, list[float]] = {}
    all_valid = True
    for profile_index, profile in enumerate(TIMING_PROFILES):
        baseline = timing_cells.get((CONTROL_ID, profile), {})
        candidate = timing_cells.get((CANDIDATE_ID, profile), {})
        paired_rounds = sorted(set(baseline) & set(candidate))
        if paired_rounds != expected_rounds or not paired_rounds:
            all_valid = False
            continue
        all_valid = all_valid and all(
            baseline[index]["valid"] and candidate[index]["valid"]
            for index in paired_rounds
        )
        ratios = [
            candidate[index]["seconds"] / baseline[index]["seconds"]
            for index in paired_rounds
        ]
        lower, upper = percentile_bootstrap(ratios, seed=19019 + profile_index)
        stratified_ratios[profile] = ratios
        components: dict[str, dict[str, float]] = {}
        component_names = sorted(
            set.intersection(*[
                set(baseline[index]["components"])
                & set(candidate[index]["components"])
                for index in paired_rounds
            ])
        )
        for name in component_names:
            components[name] = {
                "baselineMedianSeconds": statistics.median(
                    baseline[index]["components"][name]
                    for index in paired_rounds
                ),
                "candidateMedianSeconds": statistics.median(
                    candidate[index]["components"][name]
                    for index in paired_rounds
                ),
            }
        memory: dict[str, dict] = {}
        baseline_memory = memory_cells.get((CONTROL_ID, profile))
        candidate_memory = memory_cells.get((CANDIDATE_ID, profile))
        if baseline_memory is not None and candidate_memory is not None:
            memory = {
                CONTROL_ID: baseline_memory["memory"],
                CANDIDATE_ID: candidate_memory["memory"],
            }
        profiles.append({
            "profile": profile,
            "rounds": paired_rounds,
            "baselineMedianSeconds": statistics.median(
                baseline[index]["seconds"] for index in paired_rounds
            ),
            "candidateMedianSeconds": statistics.median(
                candidate[index]["seconds"] for index in paired_rounds
            ),
            "candidateToBaseline": statistics.median(ratios),
            "roundRatios": ratios,
            "empiricalPairedRange": {"lower": lower, "upper": upper},
            "components": components,
            "memoryOnly": memory,
        })

    matched_complete = bool(
        all_valid and {item["profile"] for item in profiles} == set(TIMING_PROFILES)
    )
    memory_complete = bool(
        matched_complete and all(
            set(item["memoryOnly"]) == {CONTROL_ID, CANDIDATE_ID}
            for item in profiles
        )
    )
    capacity_complete = {
        item["candidateId"] for item in capacity_exclusions
        if item["profile"] == "wvm-large-512-nz513-f4"
    } == {CONTROL_ID, CANDIDATE_ID}
    full_workload_disposition = matched_complete and capacity_complete
    geometric_time = (
        geometric_mean([item["candidateToBaseline"] for item in profiles])
        if matched_complete else None
    )
    maximum_time = (
        max(item["candidateToBaseline"] for item in profiles)
        if matched_complete else None
    )
    interval = None
    if matched_complete:
        lower, upper = stratified_geometric_bootstrap(stratified_ratios)
        interval = {"lower": lower, "upper": upper}
    memory_ratios: dict[str, float] = {}
    if memory_complete:
        for key in (
            "algorithmResidentBytes", "scratchBytes",
            "estimatedProcessPeakBytes", "observedProcessHighWaterBytes",
        ):
            memory_ratios[key] = geometric_mean([
                item["memoryOnly"][CANDIDATE_ID][key]
                / item["memoryOnly"][CONTROL_ID][key]
                for item in profiles
            ])

    improvement = bool(geometric_time is not None and geometric_time <= 0.90)
    regression = bool(maximum_time is not None and maximum_time <= 1.03)
    interval_passed = bool(interval is not None and interval["upper"] < 1.0)
    correctness = bool(
        matched_complete and maximum_error <= TOLERANCE
        and all(
            item["record"]["valid"]
            for item in timing_records + memory_records
        )
    )
    allocation = allocation_verification.get("exitCode") == 0
    single_schedule = True
    for candidate in candidate_matrix():
        observed_schedules = {
            item["record"]["schedulingId"]
            for item in timing_records + memory_records
            if item["candidateId"] == candidate.id
        }
        topology = calibration["selections"][candidate.id]["selectedTopology"]
        expected_schedule = (
            f"horizontal-outer-{topology['horizontal_workers']};"
            f"vertical-{topology['vertical_schedule']}-"
            f"{topology['vertical_workers']}-per-operator-family"
        )
        single_schedule = bool(
            single_schedule and observed_schedules == {expected_schedule}
        )
    passed = bool(
        protocol_complete and full_workload_disposition and correctness
        and allocation and memory_complete and improvement and regression
        and interval_passed and single_schedule
    )
    conditional = conditional_round_decision(timing_records)
    if len(observed_rounds) == EXTENDED_REFERENCE_ROUNDS:
        conditional = {
            **conditional,
            "runAdditionalTwoRounds": False,
            "finalRoundCount": EXTENDED_REFERENCE_ROUNDS,
            "additionalRoundsAlreadyCollected": True,
        }
    return {
        "schema": ANALYSIS_SCHEMA,
        "experimentId": EXPERIMENT_ID,
        "incrementId": INCREMENT_ID,
        "phase": "authoritative-reference",
        "classification": "reference",
        "machine": machine_record(),
        "runnerSourceTreeGitCommit": runner_commit,
        "benchmarkExecutableCommit": calibration["sourceTreeGitCommit"],
        "waveVortexModelCommit": calibration["waveVortexModelCommit"],
        "calibrationSelections": calibration["selections"],
        "profilesRequested": list(PROFILES),
        "profilesMatched": [item["profile"] for item in profiles],
        "capacityExclusions": capacity_exclusions,
        "rounds": len(observed_rounds),
        "referenceRoundProtocolComplete": protocol_complete,
        "conditionalRoundDecision": conditional,
        "completeMatchedFeasibleMatrix": matched_complete,
        "completeFullWorkloadDisposition": full_workload_disposition,
        "allCorrectWithin1e12": correctness,
        "maximumCorrectnessError": maximum_error,
        "memoryEvidenceComplete": memory_complete,
        "capacityEvidenceComplete": capacity_complete,
        "singleGraphAndScheduleAcrossSupportedWorkloads": single_schedule,
        "geometricCandidateToBaseline": geometric_time,
        "maximumProfileCandidateToBaseline": maximum_time,
        "empiricalStratifiedPairedRange": interval,
        "memoryOnlyGeometricRatios": memory_ratios,
        "profiles": profiles,
        "allocationVerification": allocation_verification,
        "adoptionGate": {
            "geometricTimeRatioAtMost": 0.90,
            "maximumProfileTimeRatioAtMost": 1.03,
            "improvementPassed": improvement,
            "regressionPassed": regression,
            "empiricalIntervalExcludesTie": interval_passed,
            "correctnessPassed": correctness,
            "referenceRoundProtocolPassed": protocol_complete,
            "allocationVerificationPassed": allocation,
            "memoryEvidenceComplete": memory_complete,
            "capacityEvidenceComplete": capacity_complete,
            "singleGraphAndSchedulePassed": single_schedule,
            "advanceToWvmIntegrationExperiment": passed,
            "completeNonlinearFluxMeasured": False,
            "wvmSourceChangeAuthorized": False,
            "generalMacClaimAllowed": False,
            "sizeDependentDispatchAllowed": False,
        },
    }


def manifest_template(
    phase: str, benchmark_commit: str, runner_commit: str,
    fixtures: dict[str, Path], topologies: dict[str, ScheduleTopology],
    exclusions: list[dict], warmups: int, samples: int,
) -> dict:
    return {
        "schema": MANIFEST_SCHEMA,
        "experimentId": EXPERIMENT_ID,
        "incrementId": INCREMENT_ID,
        "phase": phase,
        "cohortId": (
            f"issue19-authoritative-reference-{phase}-"
            f"{datetime.now(UTC).strftime('%Y%m%d')}"
        ),
        "createdAtUtc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "question": (
            "Does the frozen issue #16 streaming graph improve the exact "
            "production-lifetime WVM spectral-flux boundary by at least 10%?"
        ),
        "baseline": (
            "The same-lifetime WVM-order full-spectrum/direct-complex graph under "
            "its independently calibrated frozen machine-local topology."
        ),
        "controlledVariables": [
            "authoritative same-WVM-commit fixtures and mode-keyed oracle",
            "one frozen graph and topology per candidate across every supported workload",
            "Float64 radial and vertical two-thirds retention",
            "FFTW MEASURE/unaligned/cold/internal-1 and VECLIB_MAXIMUM_THREADS=1",
        ],
        "changedVariables": [
            "WVM-order full-spectrum/direct-complex versus partial-column-pruned tile-16 compact-split graph"
        ],
        "timedOperation": (
            "Fifteen ready retained/truncated modal inputs through streamed shared "
            "and per-target reconstruction, four pointwise expressions, horizontal "
            "forward retention, vertical projection, and four ready modal outputs."
        ),
        "excludedWork": [
            "phase and coefficient assembly/accumulation",
            "complete nonlinear flux, MATLAB dispatch, model state, time integration, I/O, and diagnostics",
            "fixture loading, oracle comparison, and setup from the uninstrumented total",
        ],
        "allocationPolicy": (
            "The same frozen-commit allocator interposer must pass; all application "
            "buffers, plans, schedulers, matrices, and outputs persist after setup."
        ),
        "interpretation": (
            "Timing-only and memory-only workers are separate. Only timing process "
            "medians enter paired inference; calibration and scale-out samples never do."
        ),
        "profiles": list(PROFILES),
        "fixtures": {profile: str(path) for profile, path in fixtures.items()},
        "candidates": [asdict(candidate) for candidate in candidate_matrix()],
        "frozenTopologies": {
            candidate_id: asdict(topology)
            for candidate_id, topology in topologies.items()
        },
        "capacityExclusions": exclusions,
        "benchmarkExecutableCommit": benchmark_commit,
        "runnerSourceTreeGitCommit": runner_commit,
        "sourceTreeGitCommit": benchmark_commit,
        "sourceTreeDirty": False,
        "threadEnvironment": {"VECLIB_MAXIMUM_THREADS": "1"},
        "warmups": warmups,
        "samples": samples,
        "runs": [],
    }


def run_plans(
    repository_root: Path, output: Path, manifest: dict, plans: list[dict],
    expected_fixtures: dict[str, dict], benchmark_commit: str,
    continue_on_error: bool,
) -> tuple[list[dict], bool]:
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "manifest.json"
    if manifest_path.is_file():
        existing_manifest = load_json(manifest_path)
        for key in (
            "schema", "experimentId", "incrementId", "phase", "profiles",
            "fixtures", "candidates", "frozenTopologies", "capacityExclusions",
            "benchmarkExecutableCommit", "runnerSourceTreeGitCommit",
            "sourceTreeGitCommit", "sourceTreeDirty", "warmups", "samples",
        ):
            if existing_manifest.get(key) != manifest.get(key):
                raise ValueError(f"existing {manifest['phase']} manifest disagrees on {key}")
        manifest = existing_manifest
    else:
        write_json(manifest_path, manifest)
    existing_entries: dict[str, dict] = {}
    for entry in manifest.get("runs", []):
        identifier = entry.get("id")
        if not isinstance(identifier, str) or identifier in existing_entries:
            raise ValueError(f"existing {manifest['phase']} manifest has invalid run IDs")
        existing_entries[identifier] = entry

    records: list[dict] = []
    failed = False
    for index, plan in enumerate(plans, start=1):
        existing = existing_entries.get(plan["id"])
        if existing is not None:
            if (
                existing.get("exitCode") != 0
                or existing.get("result") != plan["resultPath"].name
                or existing.get("candidate") != asdict(plan["candidate"])
                or existing.get("topology") != asdict(plan["topology"])
                or not plan["resultPath"].is_file()
            ):
                raise ValueError(f"existing run requires manual recovery: {plan['id']}")
            result = load_json(plan["resultPath"])
            record = result_record(
                plan["candidate"], plan["topology"], result,
                expected_fixtures[plan["profile"]], benchmark_commit,
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
            "topology": asdict(plan["topology"]),
            "primaryProvider": plan["candidate"].primary_provider,
            "measurementRole": manifest["phase"],
            "environment": {"VECLIB_MAXIMUM_THREADS": "1"},
            "command": list(map(str, plan["command"])),
            "exitCode": completed.returncode,
            "log": log_path.name,
            "sourceTreeGitCommit": benchmark_commit,
            "sourceTreeDirty": False,
        }
        if plan["resultPath"].is_file():
            result = load_json(plan["resultPath"])
            record = result_record(
                plan["candidate"], plan["topology"], result,
                expected_fixtures[plan["profile"]], benchmark_commit,
                manifest["warmups"], manifest["samples"],
            )
            entry.update({
                "runId": result.get("run", {}).get("id"),
                "status": result.get("status"),
                "result": plan["resultPath"].name,
                "embeddedGitCommit": result.get("environment", {}).get("gitCommit"),
                "embeddedGitDirty": result.get("environment", {}).get("gitDirty"),
                "sourceMetadataMatches": record["sourceMetadataMatches"],
                "authoritativeFixture": record["authoritativeFixture"],
            })
            if completed.returncode == 0 and record["valid"]:
                records.append({
                    "candidateId": plan["candidate"].id,
                    "profile": plan["profile"], "round": plan["round"],
                    "record": record,
                })
            else:
                completed = subprocess.CompletedProcess(plan["command"], 1)
                entry["exitCode"] = 1
        manifest["runs"].append(entry)
        write_json(manifest_path, manifest)
        if completed.returncode != 0:
            failed = True
            print(log_path.read_text(encoding="utf-8")[-4000:], file=sys.stderr)
            if not continue_on_error:
                break
    return records, failed


def main() -> int:
    repository_root = Path(__file__).resolve().parents[1]
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fixture", action="append", required=True,
        help="Prepared authoritative fixture as PROFILE=PATH; repeat per timing profile",
    )
    parser.add_argument("--calibration-analysis", type=Path, required=True)
    parser.add_argument(
        "--capacity-evidence", type=Path,
        default=(
            repository_root / "results/published/decisions" /
            "issue-019-authoritative-scaleout-capacity-v1.json"
        ),
    )
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
    parser.add_argument("--allow-dirty-runner", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    arguments = parser.parse_args()
    for path, label in (
        (arguments.executable, "benchmark executable"),
        (arguments.test_executable, "test executable"),
        (arguments.calibration_analysis, "calibration analysis"),
        (arguments.capacity_evidence, "capacity evidence"),
    ):
        if not path.is_file():
            parser.error(f"{label} is missing: {path}")
    try:
        fixtures = fixture_assignments(arguments.fixture)
        calibration = load_json(arguments.calibration_analysis.resolve())
        capacity = load_json(arguments.capacity_evidence.resolve())
        topologies = selected_topologies(calibration)
        expected_fixtures, exclusions = capacity_and_fixture_evidence(
            calibration, capacity
        )
    except (OSError, ValueError, json.JSONDecodeError) as error:
        parser.error(str(error))
    current_machine = machine_record()
    if not calibration_matches_machine(calibration, current_machine):
        parser.error("calibration analysis was produced on a different machine")
    runner_commit, runner_dirty = git_source_state(repository_root)
    if runner_dirty and not arguments.allow_dirty_runner:
        parser.error("the runner source tree is dirty; commit before reference collection")
    benchmark_commit = calibration["sourceTreeGitCommit"]
    if calibration.get("sourceTreeDirty") is not False:
        parser.error("reference calibration did not use a clean benchmark executable")
    if not benchmark_sources_unchanged(
        repository_root, benchmark_commit, runner_commit
    ):
        parser.error(
            "benchmark C++/CMake sources changed after calibration; recalibrate a new executable"
        )
    output = (arguments.output or (
        repository_root / "results/local" /
        f"issue19-authoritative-reference-{timestamp}"
    )).resolve()
    timing_output = output / "timing"
    memory_output = output / "memory"
    initial_plans = planned_runs(
        arguments.executable.resolve(), fixtures, timing_output, topologies,
        [1, 2, 3], REFERENCE_WARMUPS, REFERENCE_SAMPLES, "reference",
    )
    memory_plans = planned_runs(
        arguments.executable.resolve(), fixtures, memory_output, topologies,
        [1], MEMORY_WARMUPS, MEMORY_SAMPLES, "memory",
    )
    if arguments.dry_run:
        print(arguments.test_executable.resolve())
        for plan in initial_plans:
            print(f"VECLIB_MAXIMUM_THREADS=1 {' '.join(plan['command'])}")
        for plan in memory_plans:
            print(f"MEMORY VECLIB_MAXIMUM_THREADS=1 {' '.join(plan['command'])}")
        for exclusion in exclusions:
            print(
                f"CAPACITY-EXCLUDED {exclusion['profile']} "
                f"{exclusion['candidateId']} "
                f"{exclusion['requiredPhysicalMemoryBytes']}"
            )
        print(
            f"Planned {len(initial_plans)} initial timing runs, "
            f"{len(memory_plans)} memory-only runs, and conditional rounds 4-5."
        )
        return 0

    output.mkdir(parents=True, exist_ok=True)
    campaign_path = output / "campaign.json"
    campaign = {
        "schema": CAMPAIGN_SCHEMA,
        "experimentId": EXPERIMENT_ID,
        "incrementId": INCREMENT_ID,
        "createdAtUtc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "machine": current_machine,
        "benchmarkExecutableCommit": benchmark_commit,
        "runnerSourceTreeGitCommit": runner_commit,
        "benchmarkSourcesUnchangedSinceCalibration": True,
        "calibrationAnalysis": str(arguments.calibration_analysis.resolve()),
        "capacityEvidence": str(arguments.capacity_evidence.resolve()),
        "fixtures": {profile: str(path) for profile, path in fixtures.items()},
        "expectedFixtureEvidence": expected_fixtures,
        "capacityExclusions": exclusions,
        "frozenTopologies": {
            candidate_id: asdict(topology)
            for candidate_id, topology in topologies.items()
        },
        "allocationVerification": None,
    }
    if campaign_path.is_file():
        existing = load_json(campaign_path)
        for key in (
            "schema", "experimentId", "incrementId", "machine",
            "benchmarkExecutableCommit", "runnerSourceTreeGitCommit",
            "benchmarkSourcesUnchangedSinceCalibration", "calibrationAnalysis",
            "capacityEvidence", "fixtures", "expectedFixtureEvidence",
            "capacityExclusions", "frozenTopologies",
        ):
            if existing.get(key) != campaign.get(key):
                parser.error(f"existing campaign disagrees on {key}")
        campaign = existing
    else:
        write_json(campaign_path, campaign)

    allocation = campaign.get("allocationVerification")
    if not isinstance(allocation, dict) or allocation.get("exitCode") != 0:
        allocation_log = output / "allocation-verification.log"
        with allocation_log.open("w", encoding="utf-8") as log:
            completed = subprocess.run(
                [str(arguments.test_executable.resolve())],
                cwd=repository_root, stdout=log, stderr=subprocess.STDOUT,
            )
        allocation = {
            "command": [str(arguments.test_executable.resolve())],
            "exitCode": completed.returncode,
            "log": allocation_log.name,
            "benchmarkExecutableCommit": benchmark_commit,
        }
        campaign["allocationVerification"] = allocation
        write_json(campaign_path, campaign)
        if completed.returncode != 0:
            print(allocation_log.read_text(encoding="utf-8")[-4000:], file=sys.stderr)
            return completed.returncode

    timing_manifest = manifest_template(
        "reference", benchmark_commit, runner_commit, fixtures, topologies,
        exclusions, REFERENCE_WARMUPS, REFERENCE_SAMPLES,
    )
    try:
        timing_records, failed = run_plans(
            repository_root, timing_output, timing_manifest, initial_plans,
            expected_fixtures, benchmark_commit, arguments.continue_on_error,
        )
    except ValueError as error:
        parser.error(str(error))
    initial_decision = conditional_round_decision(timing_records)
    timing_manifest = load_json(timing_output / "manifest.json")
    timing_manifest["initialConditionalRoundDecision"] = initial_decision
    timing_manifest["initialRounds"] = REFERENCE_ROUNDS
    timing_manifest["maximumRounds"] = EXTENDED_REFERENCE_ROUNDS
    write_json(timing_output / "manifest.json", timing_manifest)
    if not failed and initial_decision["runAdditionalTwoRounds"]:
        extra_plans = planned_runs(
            arguments.executable.resolve(), fixtures, timing_output, topologies,
            [4, 5], REFERENCE_WARMUPS, REFERENCE_SAMPLES, "reference",
        )
        try:
            extra_records, extra_failed = run_plans(
                repository_root, timing_output, timing_manifest, extra_plans,
                expected_fixtures, benchmark_commit,
                arguments.continue_on_error,
            )
        except ValueError as error:
            parser.error(str(error))
        timing_records.extend(extra_records)
        failed = failed or extra_failed

    memory_manifest = manifest_template(
        "memory", benchmark_commit, runner_commit, fixtures, topologies,
        exclusions, MEMORY_WARMUPS, MEMORY_SAMPLES,
    )
    memory_records: list[dict] = []
    if not failed or arguments.continue_on_error:
        try:
            memory_records, memory_failed = run_plans(
                repository_root, memory_output, memory_manifest, memory_plans,
                expected_fixtures, benchmark_commit,
                arguments.continue_on_error,
            )
        except ValueError as error:
            parser.error(str(error))
        failed = failed or memory_failed

    analysis = analyze(
        timing_records, memory_records, exclusions, calibration,
        allocation, runner_commit,
    )
    write_json(output / "analysis.json", analysis)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
