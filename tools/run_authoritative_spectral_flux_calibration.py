#!/usr/bin/env python3
"""Calibrate issue #19 authoritative graph scheduling without reference inference."""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

from run_cross_mac_spectral_reference import (
    ScheduleTopology,
    machine_record,
    machine_topology,
    topology_matrix,
)
from run_production_lifetime_flux_authoritative_pilot import (
    Candidate,
    candidate_matrix,
    provider_record,
    total_seconds,
)
from run_spectral_pipeline_sweep import maximum_correctness_error
from run_vertical_gemm_sweep import git_source_state


EXPERIMENT_ID = "issue-019-production-lifetime-spectral-flux-composition"
INCREMENT_ID = "production-lifetime-flux-authoritative-calibration-v1"
MANIFEST_SCHEMA = "spectral-kernel-local-sweep-v1"
ANALYSIS_SCHEMA = "spectral-kernel-authoritative-flux-calibration-analysis-v1"
PROFILES = (
    "wvm-current-256-nz129-f4",
    "wvm-current-512-nz257-f4",
)
WARMUPS = 2
SAMPLES = 7
TOLERANCE = 1.0e-12


def geometric_mean(values: list[float]) -> float:
    if not values or any(not math.isfinite(value) or value <= 0.0 for value in values):
        raise ValueError("geometric mean requires positive finite values")
    return math.exp(math.fsum(math.log(value) for value in values) / len(values))


def fixture_assignments(values: list[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError("fixture assignments must use PROFILE=PREPARED_PATH")
        profile, raw_path = value.split("=", 1)
        if profile not in PROFILES:
            raise ValueError(f"unknown calibration fixture profile: {profile}")
        if profile in result:
            raise ValueError(f"duplicate calibration fixture profile: {profile}")
        path = Path(raw_path).expanduser().resolve()
        if not path.is_file():
            raise ValueError(f"prepared fixture is missing: {path}")
        result[profile] = path
    missing = [profile for profile in PROFILES if profile not in result]
    if missing:
        raise ValueError(
            "missing calibration fixture assignment(s): " + ", ".join(missing)
        )
    return result


def command_for(
    executable: Path,
    fixture: Path,
    profile: str,
    candidate: Candidate,
    topology: ScheduleTopology,
    output: Path,
) -> list[str]:
    return [
        str(executable),
        "run",
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
        "--warmups", str(WARMUPS),
        "--samples", str(SAMPLES),
        "--output", str(output),
    ]


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


def result_record(
    candidate: Candidate, topology: ScheduleTopology, result: dict,
) -> dict:
    provider = provider_record(candidate, result)
    error = maximum_correctness_error(provider)
    fixture = result.get("provenance", {}).get("spectralFluxFixture", {})
    expected_schedule = (
        f"horizontal-outer-{topology.horizontal_workers};"
        f"vertical-{topology.vertical_schedule}-{topology.vertical_workers}"
        "-per-operator-family"
    )
    total_matches = [
        item for item in provider.get("timings", [])
        if item.get("scope") == "uninstrumented-total"
    ]
    sample_count = (
        len(total_matches[0].get("samplesSeconds", []))
        if len(total_matches) == 1 else 0
    )
    valid = bool(
        result.get("status") == "passed"
        and result.get("run", {}).get("profile") in PROFILES
        and math.isfinite(error)
        and error <= TOLERANCE
        and fixture.get("status") == "authoritative-wvm-export"
        and fixture.get("authoritative") is True
        and fixture.get("schema") == "spectral-flux-fixture-v1"
        and provider.get("schedulingId") == expected_schedule
        and sample_count == SAMPLES
    )
    return {
        "runId": result.get("run", {}).get("id"),
        "seconds": total_seconds(provider),
        "maximumCorrectnessError": error,
        "correctWithin1e12": valid,
        "fixtureHash": fixture.get("fixtureHash"),
        "waveVortexModelCommit": fixture.get("waveVortexModelCommit"),
        "schedulingId": provider.get("schedulingId"),
        "samples": sample_count,
    }


def choose_topology(candidates: list[dict]) -> dict:
    eligible = [
        candidate for candidate in candidates
        if candidate["complete"] and candidate["correctWithin1e12"]
        and candidate["geometricSeconds"] is not None
    ]
    if not eligible:
        return {
            "selectedTopology": None,
            "reason": "no complete correct topology across both calibration profiles",
        }
    fastest = min(float(candidate["geometricSeconds"]) for candidate in eligible)
    near = [
        candidate for candidate in eligible
        if float(candidate["geometricSeconds"]) <= 1.02 * fastest
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
            float(candidate["geometricSeconds"]),
            int(candidate["topology"]["horizontal_workers"])
                + int(candidate["topology"]["vertical_workers"]),
            candidate["topology"]["id"],
        ),
    )
    return {
        "selectedTopology": selected["topology"],
        "geometricSeconds": selected["geometricSeconds"],
        "fastestGeometricSeconds": fastest,
        "selectedToFastest": float(selected["geometricSeconds"]) / fastest,
        "selectionRule": (
            "Prefer horizontal-performance/vertical-dynamic-total when within "
            "2% of the fastest complete correct topology; otherwise select the "
            "lowest geometric time with a deterministic worker-count tie break."
        ),
    }


def analyze(
    results: list[tuple[Candidate, ScheduleTopology, dict]],
    topologies: list[ScheduleTopology],
    machine: dict,
    source_commit: str,
) -> dict:
    cells: dict[tuple[str, str, str], dict] = {}
    fixture_hashes: dict[str, set[str]] = {profile: set() for profile in PROFILES}
    wvm_commits: set[str] = set()
    maximum_error = 0.0
    for candidate, topology, result in results:
        profile = result.get("run", {}).get("profile")
        record = result_record(candidate, topology, result)
        cells[(candidate.id, topology.id, profile)] = record
        fixture_hashes[profile].add(str(record["fixtureHash"]))
        wvm_commits.add(str(record["waveVortexModelCommit"]))
        maximum_error = max(maximum_error, float(record["maximumCorrectnessError"]))

    selections: dict[str, dict] = {}
    algorithms: list[dict] = []
    all_complete_correct = True
    for candidate in candidate_matrix():
        topology_rows: list[dict] = []
        for topology in topologies:
            profile_cells = [
                cells.get((candidate.id, topology.id, profile))
                for profile in PROFILES
            ]
            complete = all(cell is not None for cell in profile_cells)
            valid_cells = [cell for cell in profile_cells if cell is not None]
            correct = bool(
                complete and all(cell["correctWithin1e12"] for cell in valid_cells)
            )
            all_complete_correct = all_complete_correct and correct
            topology_rows.append({
                "topology": asdict(topology),
                "complete": complete,
                "correctWithin1e12": correct,
                "geometricSeconds": (
                    geometric_mean([float(cell["seconds"]) for cell in valid_cells])
                    if correct else None
                ),
                "profiles": [
                    {"profile": profile, **cell}
                    for profile, cell in zip(PROFILES, profile_cells)
                    if cell is not None
                ],
            })
        selections[candidate.id] = choose_topology(topology_rows)
        algorithms.append({
            "candidate": asdict(candidate),
            "topologies": topology_rows,
        })

    fixtures_consistent = bool(
        all(len(hashes) == 1 for hashes in fixture_hashes.values())
        and len(wvm_commits) == 1
        and "None" not in wvm_commits
    )
    frozen = bool(
        fixtures_consistent and all_complete_correct
        and all(
            selection.get("selectedTopology") is not None
            for selection in selections.values()
        )
    )
    return {
        "schema": ANALYSIS_SCHEMA,
        "experimentId": EXPERIMENT_ID,
        "incrementId": INCREMENT_ID,
        "phase": "authoritative-topology-calibration",
        "classification": "timing-only-calibration",
        "profiles": list(PROFILES),
        "machine": machine,
        "sourceTreeGitCommit": source_commit,
        "sourceTreeDirty": False,
        "warmups": WARMUPS,
        "samples": SAMPLES,
        "algorithms": algorithms,
        "selections": selections,
        "fixtureHashes": {
            profile: next(iter(hashes)) if len(hashes) == 1 else None
            for profile, hashes in fixture_hashes.items()
        },
        "waveVortexModelCommit": (
            next(iter(wvm_commits)) if len(wvm_commits) == 1 else None
        ),
        "allCompleteTopologiesCorrectWithin1e12": all_complete_correct,
        "maximumCorrectnessError": maximum_error,
        "singleWvmCommitAcrossCalibrationFixtures": fixtures_consistent,
        "topologiesFrozenForReference": frozen,
        "calibrationContributesToReferenceInference": False,
        "adoptionGateEvaluated": False,
        "sizeDependentDispatchAllowed": False,
        "interpretation": (
            "Calibration selects one machine-local topology per frozen graph. "
            "Its samples never enter candidate/control reference ratios, empirical "
            "intervals, regression checks, or the 0.90 adoption gate."
        ),
    }


def planned_runs(
    executable: Path,
    fixtures: dict[str, Path],
    output: Path,
    topologies: list[ScheduleTopology],
) -> list[dict]:
    plans: list[dict] = []
    candidates = candidate_matrix()
    for profile_index, profile in enumerate(PROFILES):
        rotated_topologies = (
            topologies[profile_index:] + topologies[:profile_index]
        )
        for topology_index, topology in enumerate(rotated_topologies):
            rotated_candidates = (
                candidates[topology_index % len(candidates):]
                + candidates[:topology_index % len(candidates)]
            )
            for candidate in rotated_candidates:
                identifier = f"{profile}--{candidate.id}--{topology.id}"
                result_path = output / f"{identifier}.json"
                plans.append({
                    "id": identifier,
                    "profile": profile,
                    "candidate": candidate,
                    "topology": topology,
                    "resultPath": result_path,
                    "command": command_for(
                        executable, fixtures[profile], profile,
                        candidate, topology, result_path,
                    ),
                })
    return plans


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    repository_root = Path(__file__).resolve().parents[1]
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fixture", action="append", required=True,
        help="Prepared authoritative fixture as PROFILE=PATH; repeat per profile",
    )
    parser.add_argument(
        "--executable", type=Path,
        default=repository_root / "build/release/skbench",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--performance-cores", type=int)
    parser.add_argument("--total-cores", type=int)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--allow-dirty-tree", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    arguments = parser.parse_args()
    if not arguments.executable.is_file():
        parser.error(f"benchmark executable is missing: {arguments.executable}")
    try:
        fixtures = fixture_assignments(arguments.fixture)
    except ValueError as error:
        parser.error(str(error))
    if (arguments.performance_cores is None) != (arguments.total_cores is None):
        parser.error("--performance-cores and --total-cores must be supplied together")
    if arguments.performance_cores is None:
        performance_cores, _, total_cores = machine_topology()
    else:
        performance_cores = arguments.performance_cores
        total_cores = arguments.total_cores
    try:
        topologies = topology_matrix(performance_cores, total_cores)
    except ValueError as error:
        parser.error(str(error))
    source_commit, source_dirty = git_source_state(repository_root)
    if source_dirty and not arguments.allow_dirty_tree:
        parser.error(
            "the benchmark source tree is dirty; commit and rebuild before calibration"
        )
    output = arguments.output or (
        repository_root / "results/local" /
        f"issue19-authoritative-calibration-{timestamp}"
    )
    plans = planned_runs(
        arguments.executable.resolve(), fixtures, output, topologies
    )
    if arguments.dry_run:
        for plan in plans:
            print(f"VECLIB_MAXIMUM_THREADS=1 {' '.join(plan['command'])}")
        print(
            f"Planned {len(plans)} calibration runs across {len(PROFILES)} "
            f"profiles, {len(candidate_matrix())} graphs, and "
            f"{len(topologies)} topologies."
        )
        return 0
    resuming = output.exists()
    if resuming and not (output / "manifest.json").is_file():
        parser.error(
            f"existing calibration output lacks a resumable manifest: {output}"
        )
    if not resuming:
        output.mkdir(parents=True)
    machine = machine_record()
    proposed_manifest = {
        "schema": MANIFEST_SCHEMA,
        "experimentId": EXPERIMENT_ID,
        "incrementId": INCREMENT_ID,
        "phase": "authoritative-topology-calibration",
        "cohortId": (
            f"issue19-authoritative-calibration-{machine['hostname']}-"
            f"{datetime.now(UTC).strftime('%Y%m%d')}"
        ),
        "createdAtUtc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "question": (
            "Which one topology per frozen issue #19 graph minimizes the "
            "authoritative production-lifetime boundary across both calibration "
            "workloads on this machine?"
        ),
        "baseline": (
            "The established issue #11 P-core/total-core topology matrix, applied "
            "separately to each authoritative issue #19 graph."
        ),
        "controlledVariables": [
            "authoritative WVM fixtures, logical 15-to-4 operator, normalization, and mode keys",
            "Float64, radial and vertical two-thirds retention, and fixed graph implementations",
            "FFTW 3.3.11 MEASURE/unaligned/cold with one internal worker",
            "VECLIB_MAXIMUM_THREADS=1 and zero warmed application allocations",
        ],
        "changedVariables": [
            "horizontal outer sharding at performance-core or total-core count",
            "vertical outer-dynamic total-core or weighted outer-static performance-core work",
        ],
        "timedOperation": (
            "Fifteen ready modal inputs through exact vertical reconstruction, "
            "streamed four-target physical work, horizontal transforms and "
            "retention, exact vertical projection, and four ready modal targets."
        ),
        "excludedWork": [
            "fixture preparation/loading, setup, oracle evaluation, and calibration planning",
            "reference inference, candidate/control adoption ratios, and the 0.90 gate",
            "complete nonlinear flux, phase, coefficient assembly, MATLAB, state, time integration, and I/O",
        ],
        "allocationPolicy": (
            "Timing-only calibration processes use the frozen seven-real-volume "
            "lifetime; memory-only reference workers remain a later phase."
        ),
        "interpretation": (
            "Select one topology per graph across both profiles. Calibration never "
            "contributes to reference inference and cannot choose a graph winner."
        ),
        "profiles": list(PROFILES),
        "fixtures": {profile: str(path) for profile, path in fixtures.items()},
        "candidates": [asdict(candidate) for candidate in candidate_matrix()],
        "topologies": [asdict(topology) for topology in topologies],
        "selectionRule": (
            "Prefer horizontal-performance/vertical-dynamic-total when within 2% "
            "of the fastest correct geometric result across both profiles; "
            "otherwise select the fastest with a deterministic tie break."
        ),
        "machine": machine,
        "threadEnvironment": {"VECLIB_MAXIMUM_THREADS": "1"},
        "sourceTreeGitCommit": source_commit,
        "sourceTreeDirty": source_dirty,
        "rounds": 1,
        "warmups": WARMUPS,
        "samples": SAMPLES,
        "runs": [],
    }
    if resuming:
        manifest = json.loads(
            (output / "manifest.json").read_text(encoding="utf-8")
        )
        for key in (
            "schema", "experimentId", "incrementId", "phase", "profiles",
            "fixtures", "candidates", "topologies", "selectionRule",
            "sourceTreeGitCommit", "sourceTreeDirty", "warmups", "samples",
        ):
            if manifest.get(key) != proposed_manifest.get(key):
                parser.error(
                    f"existing calibration manifest disagrees on {key}"
                )
        for key in (
            "hostname", "performanceCores", "efficiencyCores",
            "totalPhysicalCores", "physicalMemoryBytes",
        ):
            if manifest.get("machine", {}).get(key) != machine.get(key):
                parser.error(
                    f"existing calibration manifest belongs to a different {key}"
                )
    else:
        manifest = proposed_manifest
        write_json(output / "manifest.json", manifest)

    existing_entries: dict[str, dict] = {}
    for entry in manifest.get("runs", []):
        identifier = entry.get("id")
        if not isinstance(identifier, str) or identifier in existing_entries:
            parser.error("existing calibration manifest has invalid run identifiers")
        existing_entries[identifier] = entry

    completed_results: list[tuple[Candidate, ScheduleTopology, dict]] = []
    failed = False
    for index, plan in enumerate(plans, start=1):
        existing = existing_entries.get(plan["id"])
        if existing is not None:
            if (
                existing.get("exitCode") != 0
                or existing.get("candidate") != asdict(plan["candidate"])
                or existing.get("topology") != asdict(plan["topology"])
                or existing.get("result") != plan["resultPath"].name
                or not plan["resultPath"].is_file()
            ):
                parser.error(
                    f"existing calibration run requires manual recovery: {plan['id']}"
                )
            result = json.loads(
                plan["resultPath"].read_text(encoding="utf-8")
            )
            source_matches = source_metadata_matches(
                result, source_commit, source_dirty
            )
            record = result_record(
                plan["candidate"], plan["topology"], result
            )
            if not source_matches or not record["correctWithin1e12"]:
                parser.error(
                    f"existing calibration run no longer validates: {plan['id']}"
                )
            completed_results.append(
                (plan["candidate"], plan["topology"], result)
            )
            print(
                f"[{index}/{len(plans)}] reuse {plan['id']}", flush=True
            )
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
            "id": plan["id"],
            "round": 1,
            "profile": plan["profile"],
            "candidate": asdict(plan["candidate"]),
            "topology": asdict(plan["topology"]),
            "primaryProvider": plan["candidate"].primary_provider,
            "environment": {"VECLIB_MAXIMUM_THREADS": "1"},
            "command": list(map(str, plan["command"])),
            "exitCode": completed.returncode,
            "log": log_path.name,
            "sourceTreeGitCommit": source_commit,
            "sourceTreeDirty": source_dirty,
        }
        if plan["resultPath"].is_file():
            result = json.loads(plan["resultPath"].read_text(encoding="utf-8"))
            source_matches = source_metadata_matches(
                result, source_commit, source_dirty
            )
            record = result_record(
                plan["candidate"], plan["topology"], result
            )
            entry.update({
                "runId": result.get("run", {}).get("id"),
                "status": result.get("status"),
                "result": plan["resultPath"].name,
                "embeddedGitCommit": result.get("environment", {}).get("gitCommit"),
                "embeddedGitDirty": result.get("environment", {}).get("gitDirty"),
                "sourceMetadataMatches": source_matches,
                "authoritativeFixture": bool(record["correctWithin1e12"]),
            })
            if completed.returncode == 0 and source_matches and record["correctWithin1e12"]:
                completed_results.append(
                    (plan["candidate"], plan["topology"], result)
                )
            else:
                completed = subprocess.CompletedProcess(plan["command"], 1)
        manifest["runs"].append(entry)
        write_json(output / "manifest.json", manifest)
        if completed.returncode != 0:
            failed = True
            print(log_path.read_text(encoding="utf-8")[-4000:], file=sys.stderr)
            if not arguments.continue_on_error:
                break

    if completed_results:
        analysis = analyze(
            completed_results, topologies, machine, source_commit
        )
        write_json(output / "analysis.json", analysis)
        if not analysis["topologiesFrozenForReference"]:
            failed = True
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
