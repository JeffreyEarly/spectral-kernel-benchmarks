#!/usr/bin/env python3
"""Run the issue #21 fused vertical-family-view screen or reference campaign."""

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
    REFERENCE_SAMPLES,
    REFERENCE_WARMUPS,
    TIMING_PROFILES,
    command_for as authoritative_command,
    fixture_assignments,
    geometric_mean,
    load_json,
    machine_record,
    percentile_bootstrap,
    result_record,
    stratified_geometric_bootstrap,
    write_json,
)
from run_cross_mac_spectral_reference import ScheduleTopology
from run_production_lifetime_flux_authoritative_pilot import (
    Candidate,
    provider_record,
)
from run_vertical_gemm_sweep import git_source_state


EXPERIMENT_ID = "issue-021-fused-small-grouped-gemm"
INCREMENT_ID = "fused-vertical-family-views-v1"
SCHEMA = "spectral-kernel-fused-vertical-views-analysis-v1"
MANIFEST_SCHEMA = "spectral-kernel-local-sweep-v1"
SCREEN_WARMUPS = 2
SCREEN_SAMPLES = 7
REFERENCE_ROUNDS = 3
MEMORY_WARMUPS = 1
MEMORY_SAMPLES = 1
TOLERANCE = 1.0e-12
CONTROL_ID = "streaming-tile16-materialized-family-bridge"
CANDIDATE_ID = "streaming-tile16-fused-vertical-family-views"
TOPOLOGY = ScheduleTopology(
    id="horizontal-performance-12--vertical-dynamic-total-16",
    horizontal_workers=12,
    vertical_schedule="outer-dynamic",
    vertical_workers=16,
    horizontal_worker_class="performance",
    vertical_worker_class="total",
)


def candidate_matrix() -> list[Candidate]:
    return [
        Candidate(
            CONTROL_ID,
            "streaming-pruned-compact-split",
            "pipeline-production-lifetime-streaming-pruned-tile16-authoritative",
            "frozen-issue19-materialized-family-bridge-control",
        ),
        Candidate(
            CANDIDATE_ID,
            "streaming-pruned-compact-split-fused-vertical-views",
            "pipeline-production-lifetime-streaming-pruned-tile16-fused-vertical-views-authoritative",
            "direct-strided-split-family-view-candidate",
        ),
    ]


def command_for(executable: Path, fixture: Path, profile: str,
                candidate: Candidate, warmups: int, samples: int,
                output: Path) -> list[str]:
    return authoritative_command(
        executable, fixture, profile, candidate, TOPOLOGY,
        warmups, samples, output,
    )


def issue19_evidence(path: Path) -> dict:
    evidence = load_json(path)
    if evidence.get("schema") != (
        "spectral-kernel-authoritative-production-lifetime-reference-"
        "publication-v1"
    ):
        raise ValueError("issue #19 reference evidence has the wrong schema")
    if evidence.get("gate", {}).get("advanceToWvmIntegrationExperiment") is not True:
        raise ValueError("issue #19 reference evidence did not pass its gate")
    if set(evidence.get("fixtures", {})) != set(TIMING_PROFILES):
        raise ValueError("issue #19 evidence does not cover the timing profiles")
    return evidence


def planned_runs(executable: Path, fixtures: dict[str, Path], output: Path,
                 rounds: list[int], warmups: int, samples: int,
                 role: str) -> list[dict]:
    candidates = candidate_matrix()
    plans: list[dict] = []
    for round_number in rounds:
        profile_offset = (round_number - 1) % len(TIMING_PROFILES)
        profiles = (
            list(TIMING_PROFILES[profile_offset:])
            + list(TIMING_PROFILES[:profile_offset])
        )
        candidate_offset = (round_number - 1) % len(candidates)
        ordered_candidates = (
            candidates[candidate_offset:] + candidates[:candidate_offset]
        )
        for profile in profiles:
            for candidate in ordered_candidates:
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


def expected_fixture_evidence(evidence: dict) -> dict[str, dict]:
    return {
        profile: {
            "fixtureHash": fixture_hash,
            "waveVortexModelCommit": evidence["waveVortexModelCommit"],
        }
        for profile, fixture_hash in evidence["fixtures"].items()
    }


def timing_details(candidate: Candidate, result: dict) -> dict[str, float]:
    provider = provider_record(candidate, result)
    return {
        item["stage"]: float(item["medianSeconds"])
        for item in provider.get("timings", [])
        if item.get("scope") not in {
            "setup-shared-component", "setup-component", "uninstrumented-total",
        }
        and item.get("medianSeconds") is not None
    }


def collect_record(plan: dict, result: dict, expected: dict,
                   benchmark_commit: str, warmups: int,
                   samples: int) -> dict:
    record = result_record(
        plan["candidate"], TOPOLOGY, result,
        expected[plan["profile"]], benchmark_commit, warmups, samples,
    )
    record["timingDetails"] = timing_details(plan["candidate"], result)
    return {
        "candidateId": plan["candidate"].id,
        "profile": plan["profile"],
        "round": plan["round"],
        "record": record,
    }


def manifest_template(phase: str, benchmark_commit: str,
                      fixtures: dict[str, Path], warmups: int,
                      samples: int) -> dict:
    return {
        "schema": MANIFEST_SCHEMA,
        "experimentId": EXPERIMENT_ID,
        "incrementId": INCREMENT_ID,
        "phase": phase,
        "createdAtUtc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "question": (
            "Can direct strided split family views remove the materialized "
            "FFT/vertical bridge and improve the frozen issue #19 production-"
            "lifetime graph by at least 10%?"
        ),
        "baseline": (
            "The frozen issue #19 streaming tile-16 graph with serial "
            "materialization of five derivative/shared triples and four targets."
        ),
        "controlledVariables": [
            "same authoritative WVM fixtures and oracle",
            "same partial-column-pruned FFTW and split dgemm kernels",
            "horizontal performance-12 and vertical dynamic-total-16 topology",
            "same Float64 radial/vertical antialiasing and streamed lifetime",
            "out-of-place, caller-preserving, zero-allocation execution",
        ],
        "changedVariables": [
            "materialized canonical split bridge versus direct strided F/G field views"
        ],
        "timedOperation": (
            "The complete issue #19 fifteen-modal-input to four-modal-target "
            "production-lifetime boundary."
        ),
        "excludedWork": [
            "complete nonlinear flux, phase/coefficient assembly, MATLAB dispatch, model state, and time integration",
            "fixture loading, correctness oracle, setup, and planning from the authoritative total",
            "new GEMM arithmetic or scheduling changes",
        ],
        "allocationPolicy": "zero application allocations after persistent setup",
        "profiles": list(TIMING_PROFILES),
        "fixtures": {profile: str(path) for profile, path in fixtures.items()},
        "candidates": [asdict(candidate) for candidate in candidate_matrix()],
        "frozenTopology": asdict(TOPOLOGY),
        "benchmarkExecutableCommit": benchmark_commit,
        "sourceTreeGitCommit": benchmark_commit,
        "sourceTreeDirty": False,
        "threadEnvironment": {"VECLIB_MAXIMUM_THREADS": "1"},
        "warmups": warmups,
        "samples": samples,
        "runs": [],
    }


def run_plans(repository_root: Path, output: Path, manifest: dict,
              plans: list[dict], expected: dict, benchmark_commit: str,
              continue_on_error: bool) -> tuple[list[dict], bool]:
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "manifest.json"
    if manifest_path.is_file():
        current = load_json(manifest_path)
        for key in (
            "schema", "experimentId", "incrementId", "phase", "profiles",
            "fixtures", "candidates", "frozenTopology",
            "benchmarkExecutableCommit", "sourceTreeGitCommit",
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
                plan, result, expected, benchmark_commit,
                manifest["warmups"], manifest["samples"],
            )
            if not record["record"]["valid"]:
                raise ValueError(f"existing run no longer validates: {plan['id']}")
            records.append(record)
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
            "topology": asdict(TOPOLOGY),
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
            record = collect_record(
                plan, result, expected, benchmark_commit,
                manifest["warmups"], manifest["samples"],
            )
            entry.update({
                "runId": result.get("run", {}).get("id"),
                "status": result.get("status"),
                "result": plan["resultPath"].name,
                "embeddedGitCommit": result.get("environment", {}).get("gitCommit"),
                "embeddedGitDirty": result.get("environment", {}).get("gitDirty"),
                "sourceMetadataMatches": record["record"]["sourceMetadataMatches"],
                "authoritativeFixture": record["record"]["authoritativeFixture"],
            })
            if completed.returncode == 0 and record["record"]["valid"]:
                records.append(record)
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


def analyze(phase: str, timing_records: list[dict], memory_records: list[dict],
            allocation: dict, benchmark_commit: str,
            issue19: dict) -> dict:
    expected_rounds = [1] if phase == "screen" else [1, 2, 3]
    profiles: list[dict] = []
    stratified: dict[str, list[float]] = {}
    maximum_error = 0.0
    complete = True
    for profile_index, profile in enumerate(TIMING_PROFILES):
        cells: dict[str, dict[int, dict]] = {CONTROL_ID: {}, CANDIDATE_ID: {}}
        for item in timing_records:
            if item["profile"] == profile:
                cells[item["candidateId"]][item["round"]] = item["record"]
                maximum_error = max(
                    maximum_error,
                    float(item["record"]["maximumCorrectnessError"]),
                )
        paired = sorted(set(cells[CONTROL_ID]) & set(cells[CANDIDATE_ID]))
        if paired != expected_rounds:
            complete = False
            continue
        ratios = [
            cells[CANDIDATE_ID][round_number]["seconds"]
            / cells[CONTROL_ID][round_number]["seconds"]
            for round_number in paired
        ]
        lower, upper = percentile_bootstrap(ratios, seed=21021 + profile_index)
        stratified[profile] = ratios
        profiles.append({
            "profile": profile,
            "rounds": paired,
            "controlMedianSeconds": statistics.median(
                cells[CONTROL_ID][round_number]["seconds"]
                for round_number in paired
            ),
            "candidateMedianSeconds": statistics.median(
                cells[CANDIDATE_ID][round_number]["seconds"]
                for round_number in paired
            ),
            "candidateToControl": statistics.median(ratios),
            "roundRatios": ratios,
            "empiricalPairedRange": {"lower": lower, "upper": upper},
            "controlComponents": cells[CONTROL_ID][paired[0]]["timingDetails"],
            "candidateComponents": cells[CANDIDATE_ID][paired[0]]["timingDetails"],
        })
    complete = bool(
        complete
        and {item["profile"] for item in profiles} == set(TIMING_PROFILES)
        and all(item["record"]["valid"] for item in timing_records)
    )
    aggregate = (
        geometric_mean([item["candidateToControl"] for item in profiles])
        if complete else None
    )
    maximum = (
        max(item["candidateToControl"] for item in profiles)
        if complete else None
    )
    interval = None
    if complete:
        lower, upper = stratified_geometric_bootstrap(stratified)
        interval = {"lower": lower, "upper": upper}

    memory_cells: dict[tuple[str, str], dict] = {
        (item["profile"], item["candidateId"]): item["record"]["memory"]
        for item in memory_records
        if item["record"]["valid"]
    }
    memory_complete = set(memory_cells) == {
        (profile, candidate_id)
        for profile in TIMING_PROFILES
        for candidate_id in (CONTROL_ID, CANDIDATE_ID)
    }
    memory_ratios: dict[str, float] = {}
    if memory_complete:
        for key in (
            "algorithmResidentBytes", "scratchBytes",
            "estimatedProcessPeakBytes", "observedProcessHighWaterBytes",
        ):
            memory_ratios[key] = geometric_mean([
                memory_cells[(profile, CANDIDATE_ID)][key]
                / memory_cells[(profile, CONTROL_ID)][key]
                for profile in TIMING_PROFILES
            ])

    correctness = bool(complete and maximum_error <= TOLERANCE)
    screen_advance = bool(
        phase == "screen" and complete and correctness
        and aggregate is not None and aggregate <= 0.90
        and maximum is not None and maximum <= 1.03
    )
    reference_gate = bool(
        phase == "reference" and complete and correctness and memory_complete
        and allocation.get("exitCode") == 0
        and aggregate is not None and aggregate <= 0.90
        and maximum is not None and maximum <= 1.03
        and interval is not None and interval["upper"] < 1.0
        and memory_ratios["algorithmResidentBytes"] <= 1.0
    )
    return {
        "schema": SCHEMA,
        "experimentId": EXPERIMENT_ID,
        "incrementId": INCREMENT_ID,
        "phase": phase,
        "classification": "reference" if phase == "reference" else "preliminary",
        "machine": machine_record(),
        "benchmarkExecutableCommit": benchmark_commit,
        "waveVortexModelCommit": issue19["waveVortexModelCommit"],
        "frozenTopology": asdict(TOPOLOGY),
        "rounds": len(expected_rounds),
        "completeMatchedMatrix": complete,
        "allCorrectWithin1e12": correctness,
        "maximumCorrectnessError": maximum_error,
        "geometricCandidateToControl": aggregate,
        "maximumProfileCandidateToControl": maximum,
        "empiricalStratifiedPairedRange": interval,
        "memoryEvidenceComplete": memory_complete,
        "memoryOnlyGeometricRatios": memory_ratios,
        "profiles": profiles,
        "allocationVerification": allocation,
        "screenAdvanceToReference": screen_advance,
        "adoptionGate": {
            "geometricTimeRatioAtMost": 0.90,
            "maximumProfileTimeRatioAtMost": 1.03,
            "empiricalIntervalExcludesTie": bool(
                interval is not None and interval["upper"] < 1.0
            ),
            "correctnessPassed": correctness,
            "allocationVerificationPassed": allocation.get("exitCode") == 0,
            "memoryDoesNotRegress": bool(
                memory_complete
                and memory_ratios["algorithmResidentBytes"] <= 1.0
            ),
            "advanceFusedViewsToWvmIntegration": reference_gate,
            "newGemmArithmeticMeasured": False,
            "completeNonlinearFluxMeasured": False,
            "wvmSourceChangeAuthorized": False,
        },
    }


def main() -> int:
    repository_root = Path(__file__).resolve().parents[1]
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("screen", "reference"), default="screen")
    parser.add_argument("--fixture", action="append", required=True)
    parser.add_argument(
        "--issue19-evidence", type=Path,
        default=repository_root / "results/published/decisions" /
            "issue-019-authoritative-reference-lyra-v1.json",
    )
    parser.add_argument("--screen-analysis", type=Path)
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
    parser.add_argument("--allow-dirty-tree", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    arguments = parser.parse_args()
    try:
        fixtures = fixture_assignments(arguments.fixture)
        issue19 = issue19_evidence(arguments.issue19_evidence.resolve())
    except (OSError, ValueError, json.JSONDecodeError) as error:
        parser.error(str(error))
    for path, label in (
        (arguments.executable, "benchmark executable"),
        (arguments.test_executable, "test executable"),
    ):
        if not path.is_file():
            parser.error(f"{label} is missing: {path}")
    current_machine = machine_record()
    for key in ("cpuBrand", "hardwareModel", "physicalMemoryBytes"):
        if current_machine.get(key) != issue19.get("machine", {}).get(key):
            parser.error("issue #19 reference evidence was produced on a different machine")
    benchmark_commit, dirty = git_source_state(repository_root)
    if dirty and not arguments.allow_dirty_tree:
        parser.error("the source tree is dirty; commit and rebuild before collection")
    if arguments.phase == "reference":
        if arguments.screen_analysis is None or not arguments.screen_analysis.is_file():
            parser.error("reference phase requires --screen-analysis")
        screen = load_json(arguments.screen_analysis.resolve())
        if (
            screen.get("schema") != SCHEMA
            or screen.get("phase") != "screen"
            or screen.get("screenAdvanceToReference") is not True
            or screen.get("benchmarkExecutableCommit") != benchmark_commit
        ):
            parser.error("screen evidence does not authorize this reference phase")

    phase = arguments.phase
    rounds = [1] if phase == "screen" else [1, 2, 3]
    warmups = SCREEN_WARMUPS if phase == "screen" else REFERENCE_WARMUPS
    samples = SCREEN_SAMPLES if phase == "screen" else REFERENCE_SAMPLES
    output = (arguments.output or (
        repository_root / "results/local" /
        f"issue21-fused-vertical-views-{phase}-{timestamp}"
    )).resolve()
    timing_output = output / "timing"
    memory_output = output / "memory"
    timing_plans = planned_runs(
        arguments.executable.resolve(), fixtures, timing_output,
        rounds, warmups, samples, phase,
    )
    memory_plans = planned_runs(
        arguments.executable.resolve(), fixtures, memory_output,
        [1], MEMORY_WARMUPS, MEMORY_SAMPLES, "memory",
    )
    if arguments.dry_run:
        print(arguments.test_executable.resolve())
        for plan in timing_plans:
            print(f"VECLIB_MAXIMUM_THREADS=1 {' '.join(plan['command'])}")
        for plan in memory_plans:
            print(f"MEMORY VECLIB_MAXIMUM_THREADS=1 {' '.join(plan['command'])}")
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

    expected = expected_fixture_evidence(issue19)
    timing_manifest = manifest_template(
        phase, benchmark_commit, fixtures, warmups, samples,
    )
    memory_manifest = manifest_template(
        "memory", benchmark_commit, fixtures, MEMORY_WARMUPS, MEMORY_SAMPLES,
    )
    try:
        timing_records, timing_failed = run_plans(
            repository_root, timing_output, timing_manifest, timing_plans,
            expected, benchmark_commit, arguments.continue_on_error,
        )
        memory_records, memory_failed = run_plans(
            repository_root, memory_output, memory_manifest, memory_plans,
            expected, benchmark_commit, arguments.continue_on_error,
        )
    except ValueError as error:
        parser.error(str(error))
    result = analyze(
        phase, timing_records, memory_records, allocation,
        benchmark_commit, issue19,
    )
    write_json(output / "analysis.json", result)
    return 1 if timing_failed or memory_failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
