#!/usr/bin/env python3
"""Run the issue #25 WVM-native layout-optimization campaign."""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

from run_authoritative_spectral_flux_reference import (
    fixture_assignments,
    load_json,
    write_json,
)
from run_cross_mac_spectral_reference import (
    machine_record,
    percentile_bootstrap,
)
from run_spectral_pipeline_sweep import (
    geometric_mean,
    maximum_correctness_error,
    stratified_geometric_bootstrap,
)
from run_vertical_gemm_sweep import git_source_state


EXPERIMENT_ID = "issue-025-wvm-native-layout-optimization"
INCREMENT_ID = "wvm-native-strided-field-views-v1"
ANALYSIS_SCHEMA = "spectral-kernel-wvm-native-layout-optimization-analysis-v1"
MANIFEST_SCHEMA = "spectral-kernel-local-sweep-v1"
PROFILES = (
    "wvm-current-256-nz129-f4",
    "wvm-current-512-nz257-f4",
    "wvm-large-1024-nz129-f4",
)
DEEP_PROFILE = "wvm-large-512-nz513-f4"
TOTAL_STAGE = (
    "authoritative production-lifetime streamed four-target spectral-flux "
    "composition"
)
MOVEMENT_STAGE = "WVM-order triple extraction and target scatter"
SCREEN_WARMUPS = 2
SCREEN_SAMPLES = 7
REFERENCE_WARMUPS = 3
REFERENCE_SAMPLES = 21
REFERENCE_ROUNDS = 3
MEMORY_WARMUPS = 1
MEMORY_SAMPLES = 1
HORIZONTAL_WORKERS = 12
VERTICAL_WORKERS = 16
POINTWISE_WORKERS = 8
TOLERANCE = 1.0e-12


@dataclass(frozen=True)
class Candidate:
    id: str
    boundary_policy: str
    pointwise_policy: str
    pointwise_workers: int
    provider: str
    role: str


def candidate_matrix() -> list[Candidate]:
    base = "pipeline-production-lifetime-wvm-direct-authoritative"
    direct = (
        "pipeline-production-lifetime-wvm-direct-strided-field-views-"
        "authoritative"
    )
    return [
        Candidate(
            "wvm-native-issue19-control", "wvm-direct", "serial", 1,
            base, "frozen issue #19 WVM-native control",
        ),
        Candidate(
            "wvm-native-pointwise-only", "wvm-direct", "spatial-static",
            POINTWISE_WORKERS, base + "-pointwise-spatial-static",
            "issue #22 pointwise policy without bridge elision",
        ),
        Candidate(
            "wvm-native-strided-views-only",
            "wvm-direct-strided-field-views", "serial", 1, direct,
            "native family-stride FFTW views without pointwise threading",
        ),
        Candidate(
            "wvm-native-optimized-v1",
            "wvm-direct-strided-field-views", "spatial-static",
            POINTWISE_WORKERS, direct + "-pointwise-spatial-static",
            "uniform WVM-native finalist",
        ),
    ]


def issue19_evidence(path: Path) -> dict:
    evidence = load_json(path)
    expected_schema = (
        "spectral-kernel-authoritative-production-lifetime-reference-"
        "publication-v1"
    )
    if evidence.get("schema") != expected_schema:
        raise ValueError("issue #19 reference evidence has the wrong schema")
    if set(evidence.get("fixtures", {})) != set(PROFILES):
        raise ValueError("issue #19 evidence does not cover the timing profiles")
    if not evidence.get("waveVortexModelCommit"):
        raise ValueError("issue #19 evidence lacks the WVM commit")
    return evidence


def command_for(executable: Path, fixture: Path, profile: str,
                candidate: Candidate, warmups: int, samples: int,
                output: Path) -> list[str]:
    return [
        str(executable), "run",
        "--kernel", "production-lifetime-flux",
        "--profile", profile,
        "--boundary-policy", candidate.boundary_policy,
        "--spectral-flux-fixture", str(fixture),
        "--fftw-planning", "measure",
        "--fftw-alignment", "unaligned",
        "--fftw-wisdom", "cold",
        "--fftw-internal-workers", "1",
        "--fftw-outer-workers", str(HORIZONTAL_WORKERS),
        "--vertical-gemm-family", "k2-grouped",
        "--vertical-gemm-schedule", "outer-dynamic",
        "--vertical-gemm-outer-workers", str(VERTICAL_WORKERS),
        "--pointwise-policy", candidate.pointwise_policy,
        "--pointwise-workers", str(candidate.pointwise_workers),
        "--warmups", str(warmups),
        "--samples", str(samples),
        "--output", str(output),
    ]


def expected_schedule(candidate: Candidate) -> str:
    value = (
        f"horizontal-outer-{HORIZONTAL_WORKERS};"
        f"vertical-outer-dynamic-{VERTICAL_WORKERS}-per-operator-family"
    )
    if candidate.pointwise_policy == "spatial-static":
        value += f";pointwise-spatial-static-{candidate.pointwise_workers}"
    return value


def timing(provider: dict, scope: str, stage: str) -> dict:
    matches = [
        item for item in provider.get("timings", [])
        if item.get("scope") == scope and item.get("stage") == stage
    ]
    if len(matches) != 1:
        raise ValueError(f"provider lacks one {scope} / {stage} timing")
    return matches[0]


def collect_record(result: dict, candidate: Candidate, profile: str,
                   expected_fixture_hash: str, wvm_commit: str,
                   benchmark_commit: str, warmups: int, samples: int) -> dict:
    providers = result.get("providers", [])
    if len(providers) != 1 or providers[0].get("id") != candidate.provider:
        raise ValueError(f"{candidate.id} has the wrong provider identity")
    provider = providers[0]
    total = timing(provider, "uninstrumented-total", TOTAL_STAGE)
    movement = timing(provider, "adapter-component", MOVEMENT_STAGE)
    fixture = result.get("provenance", {}).get("spectralFluxFixture", {})
    error = maximum_correctness_error(provider)
    embedded = result.get("environment", {}).get("gitCommit", "")
    source_matches = bool(
        embedded and embedded != "unknown"
        and benchmark_commit.startswith(embedded)
        and result.get("environment", {}).get("gitDirty") is False
    )
    ledger = {
        item.get("stage"): item for item in provider.get("componentLedger", [])
    }
    allocation_valid = bool(
        ledger.get("steady-state allocation", {}).get("state") == "elided"
    )
    movement_elided = candidate.boundary_policy.endswith("strided-field-views")
    movement_valid = bool(
        (movement.get("state") == "elided"
         and not movement.get("samplesSeconds")
         and int(movement.get("bytesMoved", 0)) == 0)
        if movement_elided else
        (movement.get("state") == "executed"
         and len(movement.get("samplesSeconds", [])) == samples
         and int(movement.get("bytesMoved", 0)) > 0)
    )
    expected_workers = HORIZONTAL_WORKERS + VERTICAL_WORKERS + (
        0 if candidate.pointwise_policy == "serial"
        else candidate.pointwise_workers
    )
    valid = bool(
        result.get("status") == "passed"
        and result.get("run", {}).get("profile") == profile
        and result.get("run", {}).get("warmups") == warmups
        and result.get("run", {}).get("samples") == samples
        and len(total.get("samplesSeconds", [])) == samples
        and math.isfinite(error) and error <= TOLERANCE
        and fixture.get("schema") == "spectral-flux-fixture-v1"
        and fixture.get("status") == "authoritative-wvm-export"
        and fixture.get("authoritative") is True
        and fixture.get("fixtureHash") == expected_fixture_hash
        and fixture.get("waveVortexModelCommit") == wvm_commit
        and provider.get("schedulingId") == expected_schedule(candidate)
        and int(provider.get("workers", 0)) == expected_workers
        and allocation_valid and movement_valid and source_matches
    )
    components = {
        item["stage"]: float(item["medianSeconds"])
        for item in provider.get("timings", [])
        if item.get("medianSeconds") is not None
        and item.get("scope") not in {
            "setup-shared-component", "setup-component", "uninstrumented-total",
        }
    }
    memory = provider.get("memory", {})
    return {
        "runId": result.get("run", {}).get("id"),
        "seconds": float(total["medianSeconds"]),
        "components": components,
        "memory": {
            key: int(memory.get(key, 0)) for key in (
                "persistentBytes", "scratchBytes", "algorithmResidentBytes",
                "estimatedProcessPeakBytes", "observedProcessHighWaterBytes",
            )
        },
        "maximumCorrectnessError": error,
        "sourceMetadataMatches": source_matches,
        "allocationLedgerValid": allocation_valid,
        "movementContractValid": movement_valid,
        "valid": valid,
    }


def plans(executable: Path, fixtures: dict[str, Path], output: Path,
          rounds: list[int], warmups: int, samples: int,
          role: str) -> list[dict]:
    candidates = candidate_matrix()
    result = []
    for round_number in rounds:
        profile_offset = (round_number - 1) % len(PROFILES)
        profiles = list(PROFILES[profile_offset:]) + list(PROFILES[:profile_offset])
        for profile_index, profile in enumerate(profiles):
            offset = (round_number - 1 + profile_index) % len(candidates)
            ordered = candidates[offset:] + candidates[:offset]
            for candidate in ordered:
                identifier = (
                    f"{role}-round-{round_number}--{profile}--{candidate.id}"
                )
                path = output / f"{identifier}.json"
                result.append({
                    "id": identifier, "round": round_number,
                    "profile": profile, "candidate": candidate,
                    "resultPath": path,
                    "command": command_for(
                        executable, fixtures[profile], profile, candidate,
                        warmups, samples, path,
                    ),
                })
    return result


def manifest(phase: str, fixtures: dict[str, Path], commit: str,
             warmups: int, samples: int) -> dict:
    return {
        "schema": MANIFEST_SCHEMA,
        "experimentId": EXPERIMENT_ID,
        "incrementId": INCREMENT_ID,
        "phase": phase,
        "classification": "reference" if phase != "screen" else "preliminary",
        "createdAtUtc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "question": (
            "How much of the compact pipeline advantage can be recovered while "
            "preserving WVM-native persistent modal input and output storage?"
        ),
        "baseline": "The frozen issue #19 wvm-direct graph.",
        "controlledVariables": [
            "authoritative WVM fixtures, mathematical operator, normalization, and oracle",
            "WVM-native persistent complex-interleaved modal inputs and outputs",
            "Float64 radial and vertical two-thirds retention",
            "horizontal outer-12 and vertical outer-dynamic-16 topology",
            "one graph across every workload and zero warmed application allocations",
        ],
        "changedVariables": [
            "materialized WVM-order FFT bridges versus direct native-family FFTW field views",
            "serial pointwise execution versus the frozen spatial-static-8 policy",
        ],
        "timedOperation": TOTAL_STAGE,
        "excludedWork": [
            "complete WVM nonlinear flux, phase/coefficient assembly, MATLAB dispatch, model state, and time integration",
            "fixture load, planning, correctness comparison, and hardware worker tuning",
            "compact persistent caller state and constant-stratification specialization",
        ],
        "allocationPolicy": "zero application allocations after persistent setup",
        "profiles": list(PROFILES),
        "capacityProfile": DEEP_PROFILE,
        "fixtures": {key: str(value) for key, value in fixtures.items()},
        "candidates": [asdict(item) for item in candidate_matrix()],
        "benchmarkExecutableCommit": commit,
        "sourceTreeGitCommit": commit,
        "sourceTreeDirty": False,
        "threadEnvironment": {"VECLIB_MAXIMUM_THREADS": "1"},
        "warmups": warmups,
        "samples": samples,
        "runs": [],
    }


def run_plans(repository: Path, output: Path, template: dict,
              run_plans: list[dict], evidence: dict, commit: str,
              continue_on_error: bool) -> tuple[list[dict], bool]:
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "manifest.json"
    if manifest_path.is_file():
        current = load_json(manifest_path)
        for key in (
            "schema", "experimentId", "incrementId", "phase", "profiles",
            "fixtures", "candidates", "benchmarkExecutableCommit", "warmups",
            "samples",
        ):
            if current.get(key) != template.get(key):
                raise ValueError(f"existing manifest disagrees on {key}")
        template = current
    else:
        write_json(manifest_path, template)
    existing = {item["id"]: item for item in template.get("runs", [])}
    records = []
    failed = False
    for index, plan in enumerate(run_plans, start=1):
        print(f"[{index}/{len(run_plans)}] {plan['id']}", flush=True)
        entry = existing.get(plan["id"])
        if entry is None:
            log_path = plan["resultPath"].with_suffix(".log")
            environment = os.environ.copy()
            environment["VECLIB_MAXIMUM_THREADS"] = "1"
            with log_path.open("w", encoding="utf-8") as log:
                completed = subprocess.run(
                    plan["command"], cwd=repository, env=environment,
                    stdout=log, stderr=subprocess.STDOUT,
                )
            entry = {
                "id": plan["id"], "round": plan["round"],
                "profile": plan["profile"],
                "candidate": asdict(plan["candidate"]),
                "measurementRole": template["phase"],
                "environment": {"VECLIB_MAXIMUM_THREADS": "1"},
                "command": list(map(str, plan["command"])),
                "result": plan["resultPath"].name,
                "log": log_path.name,
                "exitCode": completed.returncode,
                "primaryProvider": plan["candidate"].provider,
            }
            template["runs"].append(entry)
            write_json(manifest_path, template)
        if entry.get("exitCode") != 0 or not plan["resultPath"].is_file():
            failed = True
            if not continue_on_error:
                break
            continue
        result = load_json(plan["resultPath"])
        record = collect_record(
            result, plan["candidate"], plan["profile"],
            evidence["fixtures"][plan["profile"]],
            evidence["waveVortexModelCommit"], commit,
            template["warmups"], template["samples"],
        )
        entry.update({
            "runId": record["runId"],
            "valid": record["valid"],
            "embeddedGitCommit": result.get("environment", {}).get("gitCommit"),
            "embeddedGitDirty": result.get("environment", {}).get("gitDirty"),
        })
        if not record["valid"]:
            entry["exitCode"] = 1
            write_json(manifest_path, template)
            failed = True
            if not continue_on_error:
                break
            continue
        records.append({
            "profile": plan["profile"], "round": plan["round"],
            "candidateId": plan["candidate"].id, "record": record,
        })
    return records, failed


def ratio_summary(records: list[dict], numerator: str,
                  denominator: str) -> dict:
    by_key = {
        (item["profile"], item["candidateId"], item["round"]): item["record"]
        for item in records
    }
    profile_rows = []
    stratified = {}
    for profile_index, profile in enumerate(PROFILES):
        rounds = sorted({
            round_number for candidate_profile, candidate, round_number in by_key
            if candidate_profile == profile and candidate == numerator
            and (profile, denominator, round_number) in by_key
        })
        ratios = [
            by_key[(profile, numerator, round_number)]["seconds"] /
            by_key[(profile, denominator, round_number)]["seconds"]
            for round_number in rounds
        ]
        if not ratios:
            continue
        lower, upper = percentile_bootstrap(
            ratios, seed=25000 + profile_index,
        )
        stratified[profile] = ratios
        profile_rows.append({
            "profile": profile, "rounds": rounds,
            "numeratorMedianSeconds": statistics.median(
                by_key[(profile, numerator, value)]["seconds"] for value in rounds
            ),
            "denominatorMedianSeconds": statistics.median(
                by_key[(profile, denominator, value)]["seconds"] for value in rounds
            ),
            "medianRatio": statistics.median(ratios),
            "roundRatios": ratios,
            "empiricalPairedRange": {"lower": lower, "upper": upper},
        })
    complete = len(profile_rows) == len(PROFILES)
    interval = None
    if complete:
        lower, upper = stratified_geometric_bootstrap(stratified)
        interval = {"lower": lower, "upper": upper}
    return {
        "numerator": numerator, "denominator": denominator,
        "profiles": profile_rows,
        "geometricRatio": geometric_mean(
            [item["medianRatio"] for item in profile_rows]
        ) if complete else None,
        "maximumProfileRatio": max(
            item["medianRatio"] for item in profile_rows
        ) if complete else None,
        "empiricalStratifiedPairedRange": interval,
        "complete": complete,
    }


def deep_capacity(evidence: dict) -> list[dict]:
    controls = [
        item for item in evidence.get("capacityExclusions", [])
        if item.get("profile") == DEEP_PROFILE
        and item.get("candidateId") ==
            "production-lifetime-wvm-direct-authoritative"
    ]
    if len(controls) != 1:
        raise ValueError("issue #19 lacks the deep WVM-direct exclusion")
    control = controls[0]
    half_rows = (512 // 2 + 1) * 512
    bridge_savings = 64 * half_rows * 513
    result = []
    for candidate in candidate_matrix():
        direct = candidate.boundary_policy.endswith("strided-field-views")
        required = int(control["requiredPhysicalMemoryBytes"]) - (
            bridge_savings if direct else 0
        )
        result.append({
            "profile": DEEP_PROFILE, "candidateId": candidate.id,
            "status": "capacity-exclusion",
            "requiredPhysicalMemoryBytes": required,
            "physicalMemoryBytes": int(control["physicalMemoryBytes"]),
            "elidedBridgeBytes": bridge_savings if direct else 0,
            "reason": (
                "the conservative issue #19 vertical-operator construction peak "
                "still exceeds physical memory after charging the exact direct-view "
                "bridge saving"
            ),
        })
    return result


def analyze(phase: str, timing_records: list[dict], memory_records: list[dict],
            allocation: dict, evidence: dict, commit: str) -> dict:
    control = "wvm-native-issue19-control"
    finalist = "wvm-native-optimized-v1"
    comparisons = {
        "pointwiseOnlyVsControl": ratio_summary(
            timing_records, "wvm-native-pointwise-only", control),
        "stridedViewsOnlyVsControl": ratio_summary(
            timing_records, "wvm-native-strided-views-only", control),
        "optimizedVsControl": ratio_summary(timing_records, finalist, control),
        "pointwiseMarginalAfterViews": ratio_summary(
            timing_records, finalist, "wvm-native-strided-views-only"),
        "viewsMarginalAfterPointwise": ratio_summary(
            timing_records, finalist, "wvm-native-pointwise-only"),
    }
    expected_rounds = 1 if phase == "screen" else REFERENCE_ROUNDS
    expected_cells = len(PROFILES) * len(candidate_matrix()) * expected_rounds
    complete = len(timing_records) == expected_cells and all(
        item["record"]["valid"] for item in timing_records
    )
    memory_by_key = {
        (item["profile"], item["candidateId"]): item["record"]["memory"]
        for item in memory_records if item["record"]["valid"]
    }
    memory_complete = len(memory_by_key) == len(PROFILES) * len(candidate_matrix())
    memory_ratios = {}
    if memory_complete:
        for key in (
            "persistentBytes", "scratchBytes", "algorithmResidentBytes",
            "estimatedProcessPeakBytes", "observedProcessHighWaterBytes",
        ):
            memory_ratios[key] = geometric_mean([
                memory_by_key[(profile, finalist)][key] /
                memory_by_key[(profile, control)][key]
                for profile in PROFILES
            ])
    maximum_error = max(
        [item["record"]["maximumCorrectnessError"] for item in timing_records]
        + [item["record"]["maximumCorrectnessError"] for item in memory_records]
        + [0.0]
    )
    final = comparisons["optimizedVsControl"]
    correct = maximum_error <= TOLERANCE
    common_gate = bool(
        complete and correct and final["complete"]
        and final["geometricRatio"] <= 0.90
        and final["maximumProfileRatio"] <= 1.03
    )
    screen_advance = phase == "screen" and common_gate
    reference_gate = bool(
        phase == "reference" and common_gate and memory_complete
        and allocation.get("exitCode") == 0
        and final["empiricalStratifiedPairedRange"]["upper"] < 1.0
        and memory_ratios["algorithmResidentBytes"] <= 1.0
    )
    return {
        "schema": ANALYSIS_SCHEMA,
        "experimentId": EXPERIMENT_ID,
        "incrementId": INCREMENT_ID,
        "phase": phase,
        "classification": "reference" if phase == "reference" else "preliminary",
        "machine": machine_record(),
        "benchmarkExecutableCommit": commit,
        "waveVortexModelCommit": evidence["waveVortexModelCommit"],
        "frozenTopology": {
            "horizontalWorkers": HORIZONTAL_WORKERS,
            "verticalSchedule": "outer-dynamic",
            "verticalWorkers": VERTICAL_WORKERS,
            "pointwiseWorkers": POINTWISE_WORKERS,
        },
        "completeMatchedMatrix": complete,
        "allCorrectWithin1e12": correct,
        "maximumCorrectnessError": maximum_error,
        "comparisons": comparisons,
        "memoryEvidenceComplete": memory_complete,
        "memoryFinalistToControl": memory_ratios,
        "deepCapacityExclusions": deep_capacity(evidence),
        "allocationVerification": allocation,
        "screenAdvanceToReference": screen_advance,
        "decision": {
            "finalist": finalist if reference_gate else None,
            "freezeWvmNativeOptimizedV1": reference_gate,
            "onePolicyAcrossSizes": True,
            "sizeDependentDispatch": False,
            "nativePrunedDisposition": (
                "not a layout-neutral increment: all validated pruned providers "
                "require compact split staging; a new native-interleaved pruned "
                "provider is deferred rather than silently charged as direct views"
            ),
            "completeNonlinearFluxMeasured": False,
            "hardwareWorkerTuningDeferredToIssue23": True,
        },
    }


def main() -> int:
    repository = Path(__file__).resolve().parents[1]
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("screen", "reference"), default="screen")
    parser.add_argument("--fixture", action="append", required=True)
    parser.add_argument(
        "--issue19-evidence", type=Path,
        default=repository / "results/published/decisions" /
            "issue-019-authoritative-reference-lyra-v1.json",
    )
    parser.add_argument("--screen-analysis", type=Path)
    parser.add_argument(
        "--executable", type=Path,
        default=repository / "build/release/skbench",
    )
    parser.add_argument(
        "--test-executable", type=Path,
        default=repository / "build/release/skbench_tests",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--allow-dirty-tree", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    arguments = parser.parse_args()
    try:
        fixtures = fixture_assignments(arguments.fixture)
        evidence = issue19_evidence(arguments.issue19_evidence.resolve())
    except (OSError, ValueError, json.JSONDecodeError) as error:
        parser.error(str(error))
    for path in (arguments.executable, arguments.test_executable):
        if not path.is_file():
            parser.error(f"required executable is missing: {path}")
    current_machine = machine_record()
    for key in ("cpuBrand", "hardwareModel", "physicalMemoryBytes"):
        if current_machine.get(key) != evidence.get("machine", {}).get(key):
            parser.error("issue #19 evidence was produced on a different machine")
    commit, dirty = git_source_state(repository)
    if dirty and not arguments.allow_dirty_tree:
        parser.error("commit and rebuild the benchmark before collection")
    if arguments.phase == "reference":
        if arguments.screen_analysis is None or not arguments.screen_analysis.is_file():
            parser.error("reference phase requires --screen-analysis")
        screen = load_json(arguments.screen_analysis.resolve())
        if (
            screen.get("schema") != ANALYSIS_SCHEMA
            or screen.get("phase") != "screen"
            or screen.get("screenAdvanceToReference") is not True
            or screen.get("benchmarkExecutableCommit") != commit
        ):
            parser.error("screen evidence does not authorize this reference phase")
    phase = arguments.phase
    rounds = [1] if phase == "screen" else list(range(1, REFERENCE_ROUNDS + 1))
    warmups = SCREEN_WARMUPS if phase == "screen" else REFERENCE_WARMUPS
    samples = SCREEN_SAMPLES if phase == "screen" else REFERENCE_SAMPLES
    output = (arguments.output or repository / "results/local" /
              f"issue25-wvm-native-{phase}-{timestamp}").resolve()
    timing_output = output / "timing"
    memory_output = output / "memory"
    timing_plans = plans(
        arguments.executable.resolve(), fixtures, timing_output,
        rounds, warmups, samples, phase,
    )
    memory_plans = plans(
        arguments.executable.resolve(), fixtures, memory_output,
        [1], MEMORY_WARMUPS, MEMORY_SAMPLES, "memory",
    )
    if arguments.dry_run:
        for plan in timing_plans + memory_plans:
            print("VECLIB_MAXIMUM_THREADS=1 " + " ".join(plan["command"]))
        return 0
    output.mkdir(parents=True, exist_ok=True)
    allocation_log = output / "allocation-verification.log"
    with allocation_log.open("w", encoding="utf-8") as log:
        allocation_run = subprocess.run(
            [str(arguments.test_executable.resolve())], cwd=repository,
            stdout=log, stderr=subprocess.STDOUT,
        )
    allocation = {
        "command": [str(arguments.test_executable.resolve())],
        "exitCode": allocation_run.returncode,
        "log": allocation_log.name,
        "benchmarkExecutableCommit": commit,
    }
    if allocation_run.returncode != 0:
        print(allocation_log.read_text(encoding="utf-8")[-4000:], file=sys.stderr)
        return allocation_run.returncode
    try:
        timing_records, timing_failed = run_plans(
            repository, timing_output,
            manifest(phase, fixtures, commit, warmups, samples),
            timing_plans, evidence, commit, arguments.continue_on_error,
        )
        memory_records, memory_failed = run_plans(
            repository, memory_output,
            manifest("memory", fixtures, commit, MEMORY_WARMUPS, MEMORY_SAMPLES),
            memory_plans, evidence, commit, arguments.continue_on_error,
        )
        result = analyze(
            phase, timing_records, memory_records, allocation, evidence, commit,
        )
    except ValueError as error:
        parser.error(str(error))
    write_json(output / "analysis.json", result)
    return 1 if timing_failed or memory_failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
