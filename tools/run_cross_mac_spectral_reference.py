#!/usr/bin/env python3
"""Run the issue #11 topology calibration and cross-Mac pipeline reference."""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import random
import statistics
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

from run_spectral_pipeline_sweep import (
    estimated_explicit_peak_bytes as issue9_estimated_peak,
    geometric_mean,
    maximum_correctness_error,
    provider_timing,
    stratified_geometric_bootstrap,
)
from run_streaming_pruned_locality_sweep import (
    LocalityCandidate,
    candidate_provider,
    required_memory,
)
from run_vertical_gemm_sweep import (
    PROFILE_SHAPES,
    gibibytes,
    git_source_state,
    sysctl_integer,
    sysctl_uint64,
)


EXPERIMENT_ID = "issue-009-combined-spectral-pipeline"
INCREMENT_ID = "synthetic-spectral-pipeline-cross-mac-portability-reference-v1"
CALIBRATION_INCREMENT_ID = (
    "synthetic-spectral-pipeline-cross-mac-topology-calibration-v1"
)
REFERENCE_ROUNDS = 3
EXTENDED_REFERENCE_ROUNDS = 5
REFERENCE_WARMUPS = 3
REFERENCE_SAMPLES = 21
CALIBRATION_WARMUPS = 2
CALIBRATION_SAMPLES = 7
MEMORY_WARMUPS = 1
MEMORY_SAMPLES = 1
PROFILES = (
    "wvm-current-256-nz129-f4",
    "wvm-current-512-nz257-f4",
    "wvm-large-1024-nz129-f4",
    "wvm-large-512-nz513-f4",
)
CALIBRATION_PROFILES = (
    "wvm-current-256-nz129-f4",
    "wvm-current-512-nz257-f4",
)
WVM_ID = "wvm-direct"
TILED_ID = "streaming-pruned-tiled-16"


@dataclass(frozen=True)
class AlgorithmGraph:
    id: str
    policy: str
    primary_provider: str
    role: str
    tile_width: int


@dataclass(frozen=True)
class ScheduleTopology:
    id: str
    horizontal_workers: int
    vertical_schedule: str
    vertical_workers: int
    horizontal_worker_class: str
    vertical_worker_class: str


def algorithm_graphs() -> list[AlgorithmGraph]:
    return [
        AlgorithmGraph(
            WVM_ID,
            "wvm-direct",
            "pipeline-wvm-direct",
            "wvm-order-control",
            0,
        ),
        AlgorithmGraph(
            TILED_ID,
            "streaming-pruned-compact-split",
            "pipeline-streaming-pruned-compact-split",
            "persistent-compiled-engine-candidate",
            16,
        ),
    ]


def topology_matrix(performance_cores: int, total_cores: int) -> list[ScheduleTopology]:
    if min(performance_cores, total_cores) < 1:
        raise ValueError("machine topology requires positive performance and total cores")
    if performance_cores > total_cores:
        raise ValueError("performance cores cannot exceed total physical cores")
    topologies: list[ScheduleTopology] = []
    seen: set[tuple[int, str, int]] = set()
    for horizontal_class, horizontal_workers in (
        ("performance", performance_cores),
        ("total", total_cores),
    ):
        for vertical_schedule, vertical_class, vertical_workers in (
            ("outer-dynamic", "total", total_cores),
            ("outer-static", "performance", performance_cores),
        ):
            key = (horizontal_workers, vertical_schedule, vertical_workers)
            if key in seen:
                continue
            seen.add(key)
            topologies.append(ScheduleTopology(
                id=(
                    f"horizontal-{horizontal_class}-{horizontal_workers}--"
                    f"vertical-{vertical_schedule.removeprefix('outer-')}-"
                    f"{vertical_class}-{vertical_workers}"
                ),
                horizontal_workers=horizontal_workers,
                vertical_schedule=vertical_schedule,
                vertical_workers=vertical_workers,
                horizontal_worker_class=horizontal_class,
                vertical_worker_class=vertical_class,
            ))
    return topologies


def machine_topology() -> tuple[int, int, int]:
    total = sysctl_integer("hw.physicalcpu", os.cpu_count() or 1)
    performance = sysctl_integer("hw.perflevel0.physicalcpu", total)
    efficiency = sysctl_integer(
        "hw.perflevel1.physicalcpu", max(0, total - performance),
    )
    return performance, efficiency, total


def _command_output(command: list[str]) -> str:
    try:
        completed = subprocess.run(
            command, check=True, capture_output=True, text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return "unknown"
    return completed.stdout.strip() or "unknown"


def machine_record() -> dict:
    performance, efficiency, total = machine_topology()
    return {
        "hostname": platform.node() or "unknown",
        "cpuBrand": _command_output(["sysctl", "-n", "machdep.cpu.brand_string"]),
        "hardwareModel": _command_output(["sysctl", "-n", "hw.model"]),
        "performanceCores": performance,
        "efficiencyCores": efficiency,
        "totalPhysicalCores": total,
        "physicalMemoryBytes": sysctl_uint64("hw.memsize", 0),
        "macOS": _command_output(["sw_vers", "-productVersion"]),
        "macOSBuild": _command_output(["sw_vers", "-buildVersion"]),
        "compiler": _command_output(["clang", "--version"]),
        "cmake": _command_output(["cmake", "--version"]),
        "power": _command_output(["pmset", "-g", "custom"]),
        "threadEnvironment": {"VECLIB_MAXIMUM_THREADS": "1"},
    }


def calibration_matches_machine(calibration: dict, current: dict) -> bool:
    recorded = calibration.get("machine", {})
    keys = (
        "hostname",
        "cpuBrand",
        "hardwareModel",
        "performanceCores",
        "efficiencyCores",
        "totalPhysicalCores",
    )
    return all(recorded.get(key) == current.get(key) for key in keys)


def command_for(
    executable: Path,
    algorithm: AlgorithmGraph,
    topology: ScheduleTopology,
    profile: str,
    warmups: int,
    samples: int,
    seed: int,
    result_path: Path,
) -> list[str]:
    command = [
        str(executable), "run",
        "--kernel", "spectral-pipeline",
        "--boundary-policy", algorithm.policy,
        "--profile", profile,
        "--fftw-planning", "measure",
        "--fftw-alignment", "unaligned",
        "--fftw-wisdom", "cold",
        "--fftw-internal-workers", "1",
        "--fftw-outer-workers", str(topology.horizontal_workers),
        "--vertical-gemm-family", "k2-grouped",
        "--vertical-gemm-schedule", topology.vertical_schedule,
        "--vertical-gemm-outer-workers", str(topology.vertical_workers),
        "--warmups", str(warmups),
        "--samples", str(samples),
        "--seed", str(seed),
        "--output", str(result_path),
    ]
    if algorithm.tile_width:
        command[4:4] = [
            "--streaming-tile-width", str(algorithm.tile_width),
        ]
    return command


def estimated_explicit_peak_bytes(
    profile: str,
    algorithm: AlgorithmGraph,
    topology: ScheduleTopology,
) -> int:
    if algorithm.id == WVM_ID:
        return issue9_estimated_peak(profile, algorithm.policy)
    nx, _, _, retained_modes, _ = PROFILE_SHAPES[profile]
    baseline = issue9_estimated_peak(profile, "plane-major-fused-split")
    _, nz, fields, _, _ = PROFILE_SHAPES[profile]
    full_spectrum = nx * (nx // 2 + 1) * nz * fields * 16
    worker_fft_scratch = (
        topology.horizontal_workers * nx * (nx // 2 + 1) * 16
    )
    compact_tile = (
        topology.horizontal_workers * algorithm.tile_width
        * retained_modes * 16
    )
    return baseline - full_spectrum + worker_fft_scratch + compact_tile


def _provider(algorithm: AlgorithmGraph, result: dict) -> dict:
    compatible = LocalityCandidate(
        algorithm.id,
        algorithm.policy,
        algorithm.primary_provider,
        algorithm.role,
        algorithm.tile_width,
    )
    return candidate_provider(compatible, result)


def _result_record(algorithm: AlgorithmGraph, result: dict) -> dict:
    provider = _provider(algorithm, result)
    return {
        "seconds": provider_timing(provider),
        "maximumCorrectnessError": maximum_correctness_error(provider),
        "memory": required_memory(provider),
        "environment": result.get("environment", {}),
        "executionContract": provider.get("executionContract", {}),
    }


def calibration_analysis(
    results: list[tuple[AlgorithmGraph, ScheduleTopology, dict]],
    profiles: list[str],
    topologies: list[ScheduleTopology],
) -> dict:
    cells: dict[tuple[str, str, str], dict] = {}
    for algorithm, topology, result in results:
        cells[(algorithm.id, topology.id, result["run"]["profile"])] = (
            _result_record(algorithm, result)
        )

    selections: dict[str, dict] = {}
    rows: list[dict] = []
    all_correct = True
    maximum_error = 0.0
    for algorithm in algorithm_graphs():
        candidates: list[dict] = []
        for topology in topologies:
            topology_cells = [
                cells.get((algorithm.id, topology.id, profile))
                for profile in profiles
            ]
            complete = all(cell is not None for cell in topology_cells)
            valid_cells = [cell for cell in topology_cells if cell is not None]
            errors = [cell["maximumCorrectnessError"] for cell in valid_cells]
            correct = bool(
                complete and errors and all(
                    math.isfinite(error) and error <= 1.0e-12
                    for error in errors
                )
            )
            if errors:
                maximum_error = max(maximum_error, max(errors))
            all_correct = all_correct and (correct if complete else True)
            score = (
                geometric_mean([cell["seconds"] for cell in valid_cells])
                if correct else None
            )
            candidates.append({
                "topology": asdict(topology),
                "complete": complete,
                "correctWithin1e-12": correct,
                "geometricSeconds": score,
                "profiles": [
                    {
                        "profile": profile,
                        "seconds": cell["seconds"],
                        "maximumCorrectnessError": cell["maximumCorrectnessError"],
                    }
                    for profile, cell in zip(profiles, topology_cells)
                    if cell is not None
                ],
            })
        eligible = [
            candidate for candidate in candidates
            if candidate["complete"] and candidate["correctWithin1e-12"]
            and candidate["geometricSeconds"] is not None
        ]
        if not eligible:
            selections[algorithm.id] = {
                "selectedTopology": None,
                "reason": "no complete correct topology across the calibration profiles",
            }
        else:
            fastest = min(candidate["geometricSeconds"] for candidate in eligible)
            near = [
                candidate for candidate in eligible
                if candidate["geometricSeconds"] <= 1.02 * fastest
            ]
            preferred = next((
                candidate for candidate in near
                if candidate["topology"]["horizontal_worker_class"] == "performance"
                and candidate["topology"]["vertical_schedule"] == "outer-dynamic"
                and candidate["topology"]["vertical_worker_class"] == "total"
            ), None)
            selected = preferred or min(
                near,
                key=lambda candidate: (
                    candidate["geometricSeconds"],
                    candidate["topology"]["horizontal_workers"]
                    + candidate["topology"]["vertical_workers"],
                    candidate["topology"]["id"],
                ),
            )
            selections[algorithm.id] = {
                "selectedTopology": selected["topology"],
                "geometricSeconds": selected["geometricSeconds"],
                "fastestGeometricSeconds": fastest,
                "selectedToFastest": selected["geometricSeconds"] / fastest,
                "selectionRule": (
                    "Prefer horizontal-performance/vertical-dynamic-total when "
                    "within 2% of the fastest complete correct topology; otherwise "
                    "select the lowest geometric time with a deterministic worker-count tie break."
                ),
            }
        rows.append({"algorithm": asdict(algorithm), "topologies": candidates})

    return {
        "schema": "spectral-kernel-cross-mac-calibration-analysis-v1",
        "experimentId": EXPERIMENT_ID,
        "incrementId": CALIBRATION_INCREMENT_ID,
        "phase": "calibration",
        "profiles": profiles,
        "algorithms": rows,
        "selections": selections,
        "allCompleteTopologiesCorrectWithin1e-12": all_correct,
        "maximumCorrectnessError": maximum_error,
        "calibrationContributesToReferenceInference": False,
        "sizeDependentDispatchAllowed": False,
    }


def selected_topologies(analysis: dict) -> dict[str, ScheduleTopology]:
    if analysis.get("schema") != "spectral-kernel-cross-mac-calibration-analysis-v1":
        raise ValueError("calibration analysis has the wrong schema")
    selected: dict[str, ScheduleTopology] = {}
    for algorithm in algorithm_graphs():
        record = analysis.get("selections", {}).get(algorithm.id, {})
        topology = record.get("selectedTopology")
        if not isinstance(topology, dict):
            raise ValueError(f"calibration has no selected topology for {algorithm.id}")
        selected[algorithm.id] = ScheduleTopology(**topology)
    return selected


def conditional_round_decision(
    results: list[tuple[AlgorithmGraph, int, dict]],
    profiles: list[str],
) -> dict:
    cells: dict[tuple[str, str], dict[int, float]] = {}
    for algorithm, round_number, result in results:
        cells.setdefault(
            (algorithm.id, result["run"]["profile"]), {},
        )[round_number] = _result_record(algorithm, result)["seconds"]

    profile_ratios: dict[str, list[float]] = {}
    triggers: list[dict] = []
    for profile in profiles:
        baseline = cells.get((WVM_ID, profile), {})
        candidate = cells.get((TILED_ID, profile), {})
        rounds = sorted(set(baseline) & set(candidate))
        ratios = [candidate[round_number] / baseline[round_number] for round_number in rounds]
        if len(ratios) != REFERENCE_ROUNDS:
            continue
        profile_ratios[profile] = ratios
        spread = max(ratios) / min(ratios)
        if spread > 1.10:
            triggers.append({
                "id": "profile-ratio-spread",
                "profile": profile,
                "value": spread,
                "threshold": 1.10,
            })
        if min(ratios) <= 1.03 <= max(ratios) and min(ratios) < max(ratios):
            triggers.append({
                "id": "profile-straddles-regression-boundary",
                "profile": profile,
                "lower": min(ratios),
                "upper": max(ratios),
                "boundary": 1.03,
            })

    aggregate_round_ratios: list[float] = []
    if profile_ratios:
        aggregate_round_ratios = [
            geometric_mean([
                profile_ratios[profile][round_index]
                for profile in sorted(profile_ratios)
            ])
            for round_index in range(REFERENCE_ROUNDS)
        ]
        if (
            min(aggregate_round_ratios) <= 0.90 <= max(aggregate_round_ratios)
            and min(aggregate_round_ratios) < max(aggregate_round_ratios)
        ):
            triggers.append({
                "id": "aggregate-straddles-improvement-boundary",
                "lower": min(aggregate_round_ratios),
                "upper": max(aggregate_round_ratios),
                "boundary": 0.90,
            })
        aggregate_median = statistics.median(aggregate_round_ratios)
        if 0.85 <= aggregate_median <= 0.95:
            triggers.append({
                "id": "aggregate-median-near-improvement-boundary",
                "value": aggregate_median,
                "lower": 0.85,
                "upper": 0.95,
            })

    complete = len(profile_ratios) == len(profiles)
    return {
        "completeInitialThreeRoundMatrix": complete,
        "profileRoundRatios": profile_ratios,
        "aggregateRoundRatios": aggregate_round_ratios,
        "triggers": triggers,
        "runAdditionalTwoRounds": complete and bool(triggers),
        "finalRoundCount": (
            EXTENDED_REFERENCE_ROUNDS
            if complete and triggers else REFERENCE_ROUNDS
        ),
    }


def percentile_bootstrap(
    values: list[float], seed: int = 129, resamples: int = 20_000,
) -> tuple[float, float]:
    if not values:
        raise ValueError("percentile bootstrap requires values")
    generator = random.Random(seed)
    draws = [
        statistics.median(generator.choices(values, k=len(values)))
        for _ in range(resamples)
    ]
    draws.sort()
    return draws[int(0.025 * (len(draws) - 1))], draws[int(0.975 * (len(draws) - 1))]


def reference_analysis(
    timing_results: list[tuple[AlgorithmGraph, int, dict]],
    memory_results: list[tuple[AlgorithmGraph, dict]],
    profiles: list[str],
    exclusions: list[dict],
    calibration: dict,
) -> dict:
    timing_cells: dict[tuple[str, str], dict[int, dict]] = {}
    maximum_error = 0.0
    all_correct = True
    for algorithm, round_number, result in timing_results:
        record = _result_record(algorithm, result)
        maximum_error = max(maximum_error, record["maximumCorrectnessError"])
        all_correct = all_correct and (
            math.isfinite(record["maximumCorrectnessError"])
            and record["maximumCorrectnessError"] <= 1.0e-12
        )
        timing_cells.setdefault(
            (algorithm.id, result["run"]["profile"]), {},
        )[round_number] = record

    memory_cells: dict[tuple[str, str], dict] = {}
    for algorithm, result in memory_results:
        record = _result_record(algorithm, result)
        maximum_error = max(maximum_error, record["maximumCorrectnessError"])
        all_correct = all_correct and (
            math.isfinite(record["maximumCorrectnessError"])
            and record["maximumCorrectnessError"] <= 1.0e-12
        )
        memory_cells[(algorithm.id, result["run"]["profile"])] = record["memory"]

    observed_rounds = sorted({
        round_number
        for rounds in timing_cells.values()
        for round_number in rounds
    })
    expected_rounds = (
        list(range(1, max(observed_rounds) + 1)) if observed_rounds else []
    )
    reference_round_protocol_complete = len(expected_rounds) in {
        REFERENCE_ROUNDS, EXTENDED_REFERENCE_ROUNDS,
    }
    placement_contract_valid = True
    for rounds in timing_cells.values():
        for record in rounds.values():
            execution = record["executionContract"]
            placement_contract_valid = placement_contract_valid and all(
                execution.get(direction, {}).get("nativePlacement") == "out-of-place"
                for direction in ("forward", "inverse")
            )
    profile_rows: list[dict] = []
    stratified_ratios: dict[str, list[float]] = {}
    for profile_index, profile in enumerate(profiles):
        baseline = timing_cells.get((WVM_ID, profile), {})
        candidate = timing_cells.get((TILED_ID, profile), {})
        paired_rounds = sorted(set(baseline) & set(candidate))
        if paired_rounds != expected_rounds or not paired_rounds:
            continue
        ratios = [
            candidate[round_number]["seconds"] / baseline[round_number]["seconds"]
            for round_number in paired_rounds
        ]
        lower, upper = percentile_bootstrap(ratios, seed=129 + profile_index)
        memory = {}
        for key in (WVM_ID, TILED_ID):
            cell = memory_cells.get((key, profile))
            if cell is not None:
                memory[key] = cell
        profile_rows.append({
            "profile": profile,
            "rounds": paired_rounds,
            "baselineMedianSeconds": statistics.median(
                baseline[round_number]["seconds"] for round_number in paired_rounds
            ),
            "candidateMedianSeconds": statistics.median(
                candidate[round_number]["seconds"] for round_number in paired_rounds
            ),
            "candidateToBaseline": statistics.median(ratios),
            "roundRatios": ratios,
            "empiricalPairedRange": {"lower": lower, "upper": upper},
            "memoryOnly": memory,
        })
        stratified_ratios[profile] = ratios

    excluded_profiles = {
        exclusion["profile"] for exclusion in exclusions
    }
    feasible_profiles = [
        profile for profile in profiles if profile not in excluded_profiles
    ]
    complete_matched_feasible_matrix = bool(
        feasible_profiles
        and {row["profile"] for row in profile_rows} == set(feasible_profiles)
    )
    complete_full_matrix = len(profile_rows) == len(profiles)
    geometric_ratio = (
        geometric_mean([row["candidateToBaseline"] for row in profile_rows])
        if profile_rows else None
    )
    maximum_ratio = (
        max(row["candidateToBaseline"] for row in profile_rows)
        if profile_rows else None
    )
    empirical_interval = None
    if complete_matched_feasible_matrix:
        lower, upper = stratified_geometric_bootstrap(stratified_ratios)
        empirical_interval = {"lower": lower, "upper": upper}

    memory_ratios: dict[str, float] = {}
    for memory_key in (
        "algorithmResidentBytes",
        "estimatedProcessPeakBytes",
        "observedProcessHighWaterBytes",
    ):
        values = [
            row["memoryOnly"][TILED_ID][memory_key]
            / row["memoryOnly"][WVM_ID][memory_key]
            for row in profile_rows
            if WVM_ID in row["memoryOnly"] and TILED_ID in row["memoryOnly"]
        ]
        if values:
            memory_ratios[memory_key] = geometric_mean(values)

    improvement_passed = bool(
        geometric_ratio is not None and geometric_ratio <= 0.90
    )
    regression_passed = bool(
        maximum_ratio is not None and maximum_ratio <= 1.03
    )
    interval_excludes_tie = bool(
        empirical_interval is not None and empirical_interval["upper"] < 1.0
    )
    decision_stable = bool(
        reference_round_protocol_complete
        and complete_matched_feasible_matrix and len(profile_rows) >= 2
        and all_correct and placement_contract_valid
        and improvement_passed and regression_passed and interval_excludes_tie
    )

    return {
        "schema": "spectral-kernel-cross-mac-reference-analysis-v1",
        "experimentId": EXPERIMENT_ID,
        "incrementId": INCREMENT_ID,
        "phase": "reference",
        "machine": machine_record(),
        "calibrationSelections": calibration["selections"],
        "profilesRequested": profiles,
        "profilesMatched": [row["profile"] for row in profile_rows],
        "capacityExclusions": exclusions,
        "rounds": len(expected_rounds),
        "referenceRoundProtocolComplete": reference_round_protocol_complete,
        "conditionalRoundDecision": conditional_round_decision(
            timing_results,
            [row["profile"] for row in profile_rows]
            if len(expected_rounds) == REFERENCE_ROUNDS else [],
        ) if len(expected_rounds) == REFERENCE_ROUNDS else {
            "finalRoundCount": len(expected_rounds),
            "additionalRoundsAlreadyCollected": True,
        },
        "completeMatchedFeasibleMatrix": complete_matched_feasible_matrix,
        "completeFullWorkloadMatrix": complete_full_matrix,
        "allCorrectWithin1e-12": all_correct,
        "outOfPlacePlacementContractPassed": placement_contract_valid,
        "maximumCorrectnessError": maximum_error,
        "geometricCandidateToBaseline": geometric_ratio,
        "maximumProfileCandidateToBaseline": maximum_ratio,
        "empiricalStratifiedPairedRange": empirical_interval,
        "memoryOnlyGeometricRatios": memory_ratios,
        "profiles": profile_rows,
        "decisionGate": {
            "geometricRatioAtMost": 0.90,
            "maximumProfileRatioAtMost": 1.03,
            "improvementPassed": improvement_passed,
            "regressionPassed": regression_passed,
            "empiricalIntervalExcludesTie": interval_excludes_tie,
            "correctnessPassed": all_correct,
            "placementContractPassed": placement_contract_valid,
            "referenceRoundProtocolPassed": reference_round_protocol_complete,
            "zeroSteadyStateAllocationRequired": True,
            "portabilityCandidatePassedOnThisMachine": decision_stable,
            "generalMacClaimAllowed": False,
            "sizeDependentDispatchAllowed": False,
        },
    }


def load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def write_json(path: Path, value: dict) -> None:
    with path.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2)
        stream.write("\n")


def source_metadata_matches(
    result: dict, source_commit: str, source_dirty: bool,
) -> bool:
    embedded_commit = result.get("environment", {}).get("gitCommit", "")
    embedded_dirty = result.get("environment", {}).get("gitDirty")
    return bool(
        embedded_commit and embedded_commit != "unknown"
        and source_commit.startswith(embedded_commit)
        and embedded_dirty == source_dirty
    )


def run_commands(
    repository_root: Path,
    output: Path,
    manifest: dict,
    commands: list[dict],
    source_commit: str,
    source_dirty: bool,
    continue_on_error: bool,
) -> tuple[list[tuple], bool]:
    if output.exists():
        if not (output / "manifest.json").is_file():
            raise FileExistsError(
                f"existing output lacks a resumable manifest: {output}"
            )
    else:
        output.mkdir(parents=True, exist_ok=False)
    completed_results: list[tuple] = []
    failed = False
    for index, planned in enumerate(commands, start=1):
        print(f"[{index}/{len(commands)}] {planned['id']}", flush=True)
        log_path = output / f"{planned['id']}.log"
        environment = os.environ.copy()
        environment["VECLIB_MAXIMUM_THREADS"] = "1"
        with log_path.open("w", encoding="utf-8") as log:
            completed = subprocess.run(
                planned["command"], cwd=repository_root, env=environment,
                stdout=log, stderr=subprocess.STDOUT,
            )
        entry = {
            key: value for key, value in planned.items()
            if key not in {"command", "resultPath", "resultTuple"}
        }
        entry.update({
            "command": planned["command"],
            "exitCode": completed.returncode,
            "log": log_path.name,
            "sourceTreeGitCommit": source_commit,
            "sourceTreeDirty": source_dirty,
        })
        result_path = planned["resultPath"]
        if result_path.is_file():
            result = load_json(result_path)
            matches = source_metadata_matches(result, source_commit, source_dirty)
            entry.update({
                "runId": result["run"]["id"],
                "status": result["status"],
                "result": result_path.name,
                "embeddedGitCommit": result["environment"].get("gitCommit", ""),
                "embeddedGitDirty": result["environment"].get("gitDirty"),
                "sourceMetadataMatches": matches,
            })
            if result["status"] == "passed" and matches:
                completed_results.append((*planned["resultTuple"], result))
            else:
                completed = subprocess.CompletedProcess(planned["command"], 1)
                entry["exitCode"] = 1
        manifest["runs"].append(entry)
        write_json(output / "manifest.json", manifest)
        if completed.returncode != 0:
            failed = True
            print(log_path.read_text(encoding="utf-8")[-4000:], file=sys.stderr)
            if not continue_on_error:
                break
    return completed_results, failed


def base_manifest(
    phase: str,
    increment_id: str,
    profiles: list[str],
    source_commit: str,
    source_dirty: bool,
) -> dict:
    return {
        "schema": "spectral-kernel-local-sweep-v1",
        "experimentId": EXPERIMENT_ID,
        "incrementId": increment_id,
        "phase": phase,
        "cohortId": (
            f"issue11-{phase}-{platform.node() or 'unknown'}-"
            f"{datetime.now(UTC).strftime('%Y%m%d')}"
        ),
        "createdAtUtc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "question": (
            "Does the frozen tile-16 persistent-engine graph retain its M4 "
            "advantage on this Apple-silicon machine after topology-derived scheduling?"
        ),
        "baseline": (
            "The same-commit WVM-order direct graph under its independently "
            "calibrated, then frozen, machine-local topology."
        ),
        "changedVariables": [
            "frozen WVM-order direct or streaming-pruned fixed tile-16 algorithm graph",
            "machine-local topology during calibration only",
        ],
        "controlledVariables": [
            "Float64, radial horizontal two-thirds retention, and vertically truncated Nj",
            "FFTW 3.3.11 MEASURE/unaligned/cold with one internal worker",
            "K-squared vertical operators and VECLIB_MAXIMUM_THREADS=1",
            "one schedule per graph across every workload; no size-dependent dispatch",
        ],
        "timedOperation": (
            "One allocation-free ready-real-input to reconstructed-real-output "
            "synthetic antialiased spectral round trip."
        ),
        "excludedWork": [
            "WVM nonlinear flux, phase, coefficient assembly, time integration, MATLAB, and I/O",
            "Float32, GPU, alternate placement contracts, and tile-width tuning",
        ],
        "allocationPolicy": "Zero application allocations in warmed timed execution.",
        "profiles": profiles,
        "algorithms": [asdict(algorithm) for algorithm in algorithm_graphs()],
        "machine": machine_record(),
        "sourceTreeGitCommit": source_commit,
        "sourceTreeDirty": source_dirty,
        "runs": [],
    }


def planned_commands(
    executable: Path,
    output: Path,
    algorithms: list[AlgorithmGraph],
    topologies: dict[str, list[ScheduleTopology]],
    profiles: list[str],
    rounds: list[int],
    warmups: int,
    samples: int,
    seed: int,
    physical_memory: int,
    max_memory_fraction: float,
    measurement_role: str,
) -> tuple[list[dict], list[dict]]:
    commands: list[dict] = []
    exclusions: list[dict] = []
    for round_number in rounds:
        profile_order = profiles[
            (round_number - 1) % len(profiles):
        ] + profiles[:(round_number - 1) % len(profiles)]
        algorithm_order = algorithms[
            (round_number - 1) % len(algorithms):
        ] + algorithms[:(round_number - 1) % len(algorithms)]
        for profile in profile_order:
            for algorithm in algorithm_order:
                for topology in topologies[algorithm.id]:
                    estimate = estimated_explicit_peak_bytes(
                        profile, algorithm, topology,
                    )
                    if (
                        physical_memory > 0
                        and estimate > max_memory_fraction * physical_memory
                    ):
                        exclusions.append({
                            "algorithmId": algorithm.id,
                            "topologyId": topology.id,
                            "profile": profile,
                            "estimatedExplicitPeakBytes": estimate,
                            "physicalMemoryBytes": physical_memory,
                            "maximumMemoryFraction": max_memory_fraction,
                            "reason": "estimated explicit peak exceeds the safe physical-memory fraction",
                        })
                        continue
                    stem = (
                        f"{measurement_role}-round-{round_number}--{profile}--"
                        f"{algorithm.id}--{topology.id}"
                    )
                    result_path = output / f"{stem}.json"
                    commands.append({
                        "id": stem,
                        "round": round_number,
                        "profile": profile,
                        "candidate": asdict(algorithm),
                        "topology": asdict(topology),
                        "primaryProvider": algorithm.primary_provider,
                        "measurementRole": measurement_role,
                        "estimatedExplicitPeakBytes": estimate,
                        "environment": {"VECLIB_MAXIMUM_THREADS": "1"},
                        "command": command_for(
                            executable, algorithm, topology, profile,
                            warmups, samples, seed, result_path,
                        ),
                        "resultPath": result_path,
                        "resultTuple": (
                            (algorithm, topology)
                            if measurement_role == "calibration"
                            else ((algorithm, round_number)
                                  if measurement_role == "timing"
                                  else (algorithm,))
                        ),
                    })
    return commands, exclusions


def main() -> int:
    repository_root = Path(__file__).resolve().parents[1]
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("calibration", "reference"), required=True)
    parser.add_argument(
        "--executable", type=Path,
        default=repository_root / "build/release/skbench",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--calibration-analysis", type=Path)
    parser.add_argument("--profiles", nargs="*")
    parser.add_argument("--max-memory-fraction", type=float, default=0.75)
    parser.add_argument("--allow-dirty-tree", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--seed", type=int, default=129)
    arguments = parser.parse_args()
    if not 0.0 < arguments.max_memory_fraction <= 1.0:
        parser.error("--max-memory-fraction must be in (0, 1]")

    source_commit, source_dirty = git_source_state(repository_root)
    if source_dirty and not arguments.allow_dirty_tree:
        parser.error(
            "the benchmark source tree is dirty; commit and rebuild before evidence collection"
        )
    performance, _, total = machine_topology()
    topologies = topology_matrix(performance, total)
    physical_memory = sysctl_uint64("hw.memsize", 0)
    algorithms = algorithm_graphs()

    if arguments.phase == "calibration":
        profiles = arguments.profiles or list(CALIBRATION_PROFILES)
        unknown = sorted(set(profiles) - set(PROFILES))
        if unknown:
            parser.error(f"unknown profile: {', '.join(unknown)}")
        output = arguments.output or (
            repository_root / "results/local" /
            f"issue11-cross-mac-calibration-{timestamp}"
        )
        command_topologies = {
            algorithm.id: topologies for algorithm in algorithms
        }
        commands, exclusions = planned_commands(
            arguments.executable, output, algorithms, command_topologies,
            profiles, [1], CALIBRATION_WARMUPS, CALIBRATION_SAMPLES,
            arguments.seed, physical_memory, arguments.max_memory_fraction,
            "calibration",
        )
        feasible_profiles = [
            profile for profile in profiles
            if not any(
                exclusion["profile"] == profile for exclusion in exclusions
            )
        ]
        if not feasible_profiles:
            parser.error(
                "no calibration profile is safe for every algorithm/topology candidate"
            )
        commands = [
            planned for planned in commands
            if planned["profile"] in feasible_profiles
        ]
        if arguments.dry_run:
            for planned in commands:
                print(f"VECLIB_MAXIMUM_THREADS=1 {' '.join(planned['command'])}")
            for exclusion in exclusions:
                print(
                    "CAPACITY-EXCLUDED "
                    f"{exclusion['algorithmId']} {exclusion['topologyId']} "
                    f"{exclusion['profile']} "
                    f"{gibibytes(exclusion['estimatedExplicitPeakBytes'])}"
                )
            print(
                f"Planned {len(commands)} calibration run(s) across "
                f"{len(feasible_profiles)} feasible profile(s) and "
                f"{len(topologies)} topology candidate(s)."
            )
            return 0
        manifest = base_manifest(
            "calibration", CALIBRATION_INCREMENT_ID, profiles,
            source_commit, source_dirty,
        )
        manifest.update({
            "warmups": CALIBRATION_WARMUPS,
            "samples": CALIBRATION_SAMPLES,
            "rounds": 1,
            "topologies": [asdict(topology) for topology in topologies],
            "capacityExclusions": exclusions,
            "selectionRule": (
                "Prefer performance-horizontal/dynamic-total when within 2% of "
                "the fastest geometric result; otherwise select the fastest complete "
                "correct topology. Calibration never contributes to reference inference."
            ),
        })
        completed, failed = run_commands(
            repository_root, output, manifest, commands,
            source_commit, source_dirty, arguments.continue_on_error,
        )
        analysis = calibration_analysis(completed, feasible_profiles, topologies)
        analysis["profilesRequested"] = profiles
        analysis["machine"] = machine_record()
        analysis["sourceTreeGitCommit"] = source_commit
        analysis["sourceTreeDirty"] = source_dirty
        analysis["capacityExclusions"] = exclusions
        write_json(output / "analysis.json", analysis)
        return 1 if failed else 0

    if arguments.calibration_analysis is None:
        parser.error("reference phase requires --calibration-analysis")
    try:
        calibration = load_json(arguments.calibration_analysis)
        selected = selected_topologies(calibration)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        parser.error(str(error))
    current_machine = machine_record()
    if not calibration_matches_machine(calibration, current_machine):
        parser.error(
            "calibration analysis was produced on a different machine; "
            "calibrate each host independently"
        )
    if calibration.get("sourceTreeGitCommit") != source_commit:
        parser.error(
            "calibration source commit does not match the reference source commit"
        )
    if calibration.get("sourceTreeDirty") != source_dirty:
        parser.error(
            "calibration dirty state does not match the reference source state"
        )
    profiles = arguments.profiles or list(PROFILES)
    unknown = sorted(set(profiles) - set(PROFILES))
    if unknown:
        parser.error(f"unknown profile: {', '.join(unknown)}")
    output = arguments.output or (
        repository_root / "results/local" /
        f"issue11-cross-mac-reference-{timestamp}"
    )
    timing_output = output / "timing"
    memory_output = output / "memory"
    frozen_topologies = {
        algorithm.id: [selected[algorithm.id]] for algorithm in algorithms
    }

    initial_commands, initial_exclusions = planned_commands(
        arguments.executable, timing_output, algorithms, frozen_topologies,
        profiles, [1, 2, 3], REFERENCE_WARMUPS, REFERENCE_SAMPLES,
        arguments.seed, physical_memory, arguments.max_memory_fraction, "timing",
    )
    paired_profiles = [
        profile for profile in profiles
        if not any(
            exclusion["profile"] == profile
            for exclusion in initial_exclusions
        )
    ]
    memory_commands, memory_exclusions = planned_commands(
        arguments.executable, memory_output, algorithms, frozen_topologies,
        profiles, [1], MEMORY_WARMUPS, MEMORY_SAMPLES,
        arguments.seed, physical_memory, arguments.max_memory_fraction, "memory",
    )
    if arguments.dry_run:
        for planned in initial_commands:
            print(f"VECLIB_MAXIMUM_THREADS=1 {' '.join(planned['command'])}")
        for planned in memory_commands:
            print(f"MEMORY VECLIB_MAXIMUM_THREADS=1 {' '.join(planned['command'])}")
        for exclusion in initial_exclusions:
            print(
                "CAPACITY-EXCLUDED "
                f"{exclusion['algorithmId']} {exclusion['profile']} "
                f"{gibibytes(exclusion['estimatedExplicitPeakBytes'])}"
            )
        print(
            f"Planned {len(initial_commands)} initial timing run(s), "
            f"{len(memory_commands)} memory run(s), and conditional rounds 4-5."
        )
        return 0

    output.mkdir(parents=True, exist_ok=False)
    timing_manifest = base_manifest(
        "reference", INCREMENT_ID, profiles, source_commit, source_dirty,
    )
    timing_manifest.update({
        "measurementRole": "timing-only",
        "warmups": REFERENCE_WARMUPS,
        "samples": REFERENCE_SAMPLES,
        "initialRounds": REFERENCE_ROUNDS,
        "maximumRounds": EXTENDED_REFERENCE_ROUNDS,
        "frozenTopologies": {
            key: asdict(value) for key, value in selected.items()
        },
        "calibrationAnalysis": str(arguments.calibration_analysis),
        "capacityExclusions": initial_exclusions,
    })
    timing_results, failed = run_commands(
        repository_root, timing_output, timing_manifest, initial_commands,
        source_commit, source_dirty, arguments.continue_on_error,
    )
    round_decision = conditional_round_decision(timing_results, paired_profiles)
    timing_manifest["conditionalRoundDecision"] = round_decision
    write_json(timing_output / "manifest.json", timing_manifest)
    if not failed and round_decision["runAdditionalTwoRounds"]:
        extra_commands, extra_exclusions = planned_commands(
            arguments.executable, timing_output, algorithms, frozen_topologies,
            profiles, [4, 5], REFERENCE_WARMUPS, REFERENCE_SAMPLES,
            arguments.seed, physical_memory, arguments.max_memory_fraction, "timing",
        )
        timing_manifest["capacityExclusions"].extend(extra_exclusions)
        extra_results, extra_failed = run_commands(
            repository_root, timing_output, timing_manifest, extra_commands,
            source_commit, source_dirty, arguments.continue_on_error,
        )
        timing_results.extend(extra_results)
        failed = failed or extra_failed

    memory_manifest = base_manifest(
        "memory", INCREMENT_ID, profiles, source_commit, source_dirty,
    )
    memory_manifest.update({
        "measurementRole": "memory-only",
        "warmups": MEMORY_WARMUPS,
        "samples": MEMORY_SAMPLES,
        "rounds": 1,
        "frozenTopologies": {
            key: asdict(value) for key, value in selected.items()
        },
        "calibrationAnalysis": str(arguments.calibration_analysis),
        "capacityExclusions": memory_exclusions,
    })
    memory_results: list[tuple[AlgorithmGraph, dict]] = []
    if not failed or arguments.continue_on_error:
        memory_results, memory_failed = run_commands(
            repository_root, memory_output, memory_manifest, memory_commands,
            source_commit, source_dirty, arguments.continue_on_error,
        )
        failed = failed or memory_failed

    analysis = reference_analysis(
        timing_results, memory_results, profiles,
        initial_exclusions, calibration,
    )
    write_json(output / "analysis.json", analysis)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
