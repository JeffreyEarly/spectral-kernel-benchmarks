#!/usr/bin/env python3
"""Screen issue #22 pointwise policies on the frozen direct-view pipeline."""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

from run_vertical_gemm_sweep import git_source_state


EXPERIMENT_ID = "issue-022-pointwise-advection-optimization"
INCREMENT_ID = "pointwise-advection-first-screen-v1"
MANIFEST_SCHEMA = "spectral-kernel-local-sweep-v1"
ANALYSIS_SCHEMA = "spectral-kernel-pointwise-advection-screen-v1"
PROFILES = (
    "wvm-current-256-nz129-f4",
    "wvm-current-512-nz257-f4",
)
BOUNDARY_POLICY = "streaming-pruned-compact-split-fused-vertical-views"
CONTROL_PROVIDER = (
    "pipeline-production-lifetime-streaming-pruned-tile16-"
    "fused-vertical-views-authoritative"
)
POINTWISE_STAGE = "four streamed pointwise advection expressions"
TOTAL_STAGE = (
    "authoritative production-lifetime streamed four-target "
    "spectral-flux composition"
)
TOLERANCE = 1.0e-12
MEMORY_RATIO_LIMIT = 1.001


@dataclass(frozen=True)
class Candidate:
    id: str
    policy: str
    workers: int
    provider: str


def candidate_matrix(spatial_workers: list[int]) -> list[Candidate]:
    candidates = [
        Candidate("serial-1", "serial", 1, CONTROL_PROVIDER),
        Candidate(
            "vector-serial-1", "vector-serial", 1,
            CONTROL_PROVIDER + "-pointwise-vector-serial",
        ),
    ]
    candidates.extend(
        Candidate(
            f"spatial-static-{workers}", "spatial-static", workers,
            CONTROL_PROVIDER + "-pointwise-spatial-static",
        )
        for workers in spatial_workers
    )
    return candidates


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def geometric_mean(values: list[float]) -> float:
    if not values or any(value <= 0.0 for value in values):
        raise ValueError("geometric mean requires positive values")
    return math.exp(sum(math.log(value) for value in values) / len(values))


def collect_vectorization_evidence(repository_root: Path,
                                   output: Path) -> dict:
    log_path = output / "vectorization-evidence.log"
    with tempfile.TemporaryDirectory(prefix="skbench-pointwise-") as temporary:
        command = [
            "/usr/bin/xcrun", "--sdk", "macosx", "clang++",
            "-Iinclude", "-Isrc", "-std=gnu++20", "-O3", "-mcpu=native",
            "-Rpass=loop-vectorize", "-c", "src/pointwise_advection.cpp",
            "-o", str(Path(temporary) / "pointwise_advection.o"),
        ]
        with log_path.open("w", encoding="utf-8") as log:
            completed = subprocess.run(
                command, cwd=repository_root, stdout=log,
                stderr=subprocess.STDOUT,
            )
    contents = log_path.read_text(encoding="utf-8")
    remarks = [
        line.strip() for line in contents.splitlines()
        if "pointwise_advection.cpp" in line
        and "remark: vectorized loop" in line
    ]
    return {
        "command": command,
        "exitCode": completed.returncode,
        "log": log_path.name,
        "vectorizedLoopRemarks": remarks,
        "passed": completed.returncode == 0 and len(remarks) >= 2,
    }


def fixture_assignments(values: list[str]) -> dict[str, Path]:
    fixtures: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError("fixture assignments must use PROFILE=PREPARED_PATH")
        profile, raw_path = value.split("=", 1)
        if profile not in PROFILES:
            raise ValueError(f"unknown pointwise profile: {profile}")
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
        "--fftw-outer-workers", "12",
        "--streaming-tile-width", "16",
        "--pointwise-policy", candidate.policy,
        "--pointwise-workers", str(candidate.workers),
        "--vertical-gemm-family", "k2-grouped",
        "--vertical-gemm-schedule", "outer-dynamic",
        "--vertical-gemm-outer-workers", "16",
        "--warmups", str(warmups),
        "--samples", str(samples),
        "--output", str(output),
    ]


def provider(result: dict, expected: Candidate) -> dict:
    providers = result.get("providers", [])
    if len(providers) != 1 or providers[0].get("id") != expected.provider:
        raise ValueError(
            f"{expected.id} must contain only provider {expected.provider}"
        )
    return providers[0]


def timing_seconds(record: dict, stage: str) -> float:
    values = [
        float(item["medianSeconds"])
        for item in record.get("timings", [])
        if item.get("stage") == stage and item.get("medianSeconds") is not None
    ]
    if len(values) != 1:
        raise ValueError(f"provider must contain exactly one {stage!r} timing")
    return values[0]


def maximum_correctness_error(record: dict) -> float:
    metrics = record.get("correctness", [])
    if not metrics or not all(item.get("passed") is True for item in metrics):
        return math.inf
    return max(float(item["maximumRelativeError"]) for item in metrics)


def result_record(result: dict, candidate: Candidate, profile: str,
                  commit: str, dirty: bool) -> dict:
    record = provider(result, candidate)
    workload = result.get("workload", {})
    pointwise = timing_seconds(record, POINTWISE_STAGE)
    total = timing_seconds(record, TOTAL_STAGE)
    volume = (
        int(workload.get("Nx", 0)) * int(workload.get("Ny", 0)) *
        int(workload.get("Nz", 0))
    )
    effective_bytes = 28 * volume * 8
    environment = result.get("environment", {})
    error = maximum_correctness_error(record)
    valid = bool(
        result.get("status") == "passed"
        and result.get("run", {}).get("profile") == profile
        and commit.startswith(environment.get("gitCommit", ""))
        and environment.get("gitDirty") is dirty
        and result.get("provenance", {}).get(
            "spectralFluxFixture", {}).get("authoritative") is True
        and error <= TOLERANCE
    )
    return {
        "runId": result.get("run", {}).get("id"),
        "profile": profile,
        "candidateId": candidate.id,
        "pointwiseSeconds": pointwise,
        "totalSeconds": total,
        "pointwiseFractionOfTotal": pointwise / total,
        "effectiveBytes": effective_bytes,
        "effectiveGigabytesPerSecond": effective_bytes / pointwise / 1.0e9,
        "schedulerSeconds": next((
            float(item["medianSeconds"])
            for item in record.get("timings", [])
            if item.get("stage") == "pointwise empty persistent dispatch"
        ), 0.0),
        "algorithmResidentBytes": int(record.get("memory", {}).get(
            "algorithmResidentBytes", 0)),
        "scratchBytes": int(record.get("memory", {}).get("scratchBytes", 0)),
        "observedProcessHighWaterBytes": int(record.get("memory", {}).get(
            "observedProcessHighWaterBytes", 0)),
        "maximumCorrectnessError": error,
        "valid": valid,
    }


def analyze(records: list[dict], candidates: list[Candidate], commit: str,
            dirty: bool) -> dict:
    cells = {
        (record["profile"], record["candidateId"]): record
        for record in records
    }
    expected = {
        (profile, candidate.id)
        for profile in PROFILES for candidate in candidates
    }
    complete = set(cells) == expected and all(record["valid"] for record in records)
    summaries: list[dict] = []
    control = {
        profile: cells.get((profile, "serial-1")) for profile in PROFILES
    }
    for candidate in candidates:
        candidate_records = [
            cells.get((profile, candidate.id)) for profile in PROFILES
        ]
        if any(record is None for record in candidate_records) or any(
            control[profile] is None for profile in PROFILES
        ):
            continue
        pointwise_ratios = [
            candidate_records[index]["pointwiseSeconds"] /
            control[profile]["pointwiseSeconds"]
            for index, profile in enumerate(PROFILES)
        ]
        total_ratios = [
            candidate_records[index]["totalSeconds"] /
            control[profile]["totalSeconds"]
            for index, profile in enumerate(PROFILES)
        ]
        memory_ratios = [
            candidate_records[index]["algorithmResidentBytes"] /
            control[profile]["algorithmResidentBytes"]
            for index, profile in enumerate(PROFILES)
        ]
        summaries.append({
            "candidate": asdict(candidate),
            "geometricPointwiseToSerial": geometric_mean(pointwise_ratios),
            "geometricTotalToSerial": geometric_mean(total_ratios),
            "maximumProfileTotalToSerial": max(total_ratios),
            "geometricAlgorithmResidentToSerial": geometric_mean(memory_ratios),
            "maximumProfileAlgorithmResidentToSerial": max(memory_ratios),
            "maximumAdditionalAlgorithmResidentBytes": max(
                candidate_records[index]["algorithmResidentBytes"] -
                control[profile]["algorithmResidentBytes"]
                for index, profile in enumerate(PROFILES)
            ),
            "geometricResidualPointwiseFraction": geometric_mean([
                record["pointwiseFractionOfTotal"]
                for record in candidate_records
            ]),
            "profiles": candidate_records,
        })
    eligible = [
        summary for summary in summaries
        if summary["candidate"]["id"] != "serial-1"
        and summary["geometricTotalToSerial"] <= 0.95
        and summary["maximumProfileTotalToSerial"] <= 1.03
        and summary["maximumProfileAlgorithmResidentToSerial"] <=
            MEMORY_RATIO_LIMIT
        and all(
            profile["maximumCorrectnessError"] <= TOLERANCE
            for profile in summary["profiles"]
        )
    ]
    selected = min(
        eligible, key=lambda item: (
            item["geometricTotalToSerial"],
            item["candidate"]["workers"], item["candidate"]["id"],
        ), default=None,
    )
    fusion_recommended = bool(
        selected is None
        or selected["geometricResidualPointwiseFraction"] > 0.10
    )
    return {
        "schema": ANALYSIS_SCHEMA,
        "experimentId": EXPERIMENT_ID,
        "incrementId": INCREMENT_ID,
        "classification": "preliminary",
        "createdAtUtc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "benchmarkExecutableCommit": commit,
        "sourceTreeDirty": dirty,
        "profiles": list(PROFILES),
        "completeMatchedMatrix": complete,
        "allCorrectWithin1e12": bool(
            complete and all(
                record["maximumCorrectnessError"] <= TOLERANCE
                for record in records
            )
        ),
        "candidateSummaries": summaries,
        "selectedCandidate": selected,
        "screenGate": {
            "geometricTotalRatioAtMost": 0.95,
            "maximumProfileTotalRatioAtMost": 1.03,
            "maximumProfileAlgorithmResidentRatioAtMost": MEMORY_RATIO_LIMIT,
            "residualPointwiseFractionAtMostBeforeSkippingFusion": 0.10,
            "candidatePassed": selected is not None,
            "activatePointwiseFftFusion": fusion_recommended,
        },
        "limitations": (
            "One preliminary process per cell with no empirical interval; "
            "the screen cannot support adoption or publication as reference evidence."
        ),
    }


def main() -> int:
    repository_root = Path(__file__).resolve().parents[1]
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", action="append", required=True)
    parser.add_argument(
        "--spatial-workers", default="1,4,8,12,16",
        help="Comma-separated persistent spatial worker counts.",
    )
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--samples", type=int, default=7)
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
        spatial_workers = sorted({
            int(value) for value in arguments.spatial_workers.split(",")
        })
        if not spatial_workers or spatial_workers[0] <= 0:
            raise ValueError("spatial worker counts must be positive")
    except (OSError, ValueError, json.JSONDecodeError) as error:
        parser.error(str(error))
    if arguments.warmups < 0 or arguments.samples <= 0:
        parser.error("warmups must be nonnegative and samples must be positive")
    for path, label in (
        (arguments.executable, "benchmark executable"),
        (arguments.test_executable, "test executable"),
    ):
        if not path.is_file():
            parser.error(f"{label} is missing: {path}")
    commit, dirty = git_source_state(repository_root)
    if dirty and not arguments.allow_dirty_tree:
        parser.error("the source tree is dirty; commit and rebuild before collection")

    candidates = candidate_matrix(spatial_workers)
    output = (arguments.output or (
        repository_root / "results/local" /
        f"issue22-pointwise-first-screen-{timestamp}"
    )).resolve()
    plans: list[dict] = []
    for profile_index, profile in enumerate(PROFILES):
        ordered = candidates if profile_index % 2 == 0 else list(reversed(candidates))
        for candidate in ordered:
            identifier = f"{profile}--{candidate.id}"
            result_path = output / f"{identifier}.json"
            plans.append({
                "id": identifier,
                "profile": profile,
                "candidate": candidate,
                "resultPath": result_path,
                "command": command_for(
                    arguments.executable.resolve(), fixtures[profile], profile,
                    candidate, arguments.warmups, arguments.samples, result_path,
                ),
            })
    if arguments.dry_run:
        print(arguments.test_executable.resolve())
        for plan in plans:
            print("VECLIB_MAXIMUM_THREADS=1 " + " ".join(plan["command"]))
        return 0

    output.mkdir(parents=True, exist_ok=True)
    allocation_log = output / "allocation-verification.log"
    with allocation_log.open("w", encoding="utf-8") as log:
        allocation_run = subprocess.run(
            [str(arguments.test_executable.resolve())], cwd=repository_root,
            stdout=log, stderr=subprocess.STDOUT,
        )
    if allocation_run.returncode != 0:
        print(allocation_log.read_text(encoding="utf-8")[-4000:], file=sys.stderr)
        return allocation_run.returncode
    vectorization = collect_vectorization_evidence(repository_root, output)
    if not vectorization["passed"]:
        print((output / vectorization["log"]).read_text(
            encoding="utf-8")[-4000:], file=sys.stderr)
        return 1

    manifest_path = output / "manifest.json"
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "experimentId": EXPERIMENT_ID,
        "incrementId": INCREMENT_ID,
        "classification": "preliminary",
        "createdAtUtc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "question": (
            "Can a memory-neutral vector or spatially static pointwise policy "
            "improve the issue #21 direct-family-view complete boundary by 5%?"
        ),
        "profiles": list(PROFILES),
        "fixtures": {profile: str(path) for profile, path in fixtures.items()},
        "candidates": [asdict(candidate) for candidate in candidates],
        "controlledVariables": [
            "authoritative WVM fixtures and oracle",
            "issue #21 direct-family-view algorithm and tile width 16",
            "horizontal outer-12 and vertical dynamic-16 scheduling",
            "Float64 antialiasing, streamed target lifetime, and seven real volumes",
        ],
        "changedVariables": [
            "pointwise loop vector contract",
            "pointwise persistent spatial worker count",
        ],
        "timedOperation": (
            "Ready physical arrays through four pointwise expressions and the "
            "complete 15-modal-input to four-modal-output issue #21 boundary."
        ),
        "excludedWork": [
            "complete nonlinear flux, MATLAB/MEX, state, timestep, and I/O",
            "fixture loading, planning, correctness storage, and pool setup from steady state",
            "pointwise-to-FFT fusion and target concurrency",
        ],
        "allocationPolicy": "zero application allocations after persistent setup",
        "benchmarkExecutableCommit": commit,
        "sourceTreeDirty": dirty,
        "warmups": arguments.warmups,
        "samples": arguments.samples,
        "threadEnvironment": {"VECLIB_MAXIMUM_THREADS": "1"},
        "vectorizationEvidence": vectorization,
        "runs": [],
    }
    if manifest_path.is_file():
        current = load_json(manifest_path)
        for key in (
            "schema", "experimentId", "incrementId", "profiles", "fixtures",
            "candidates", "benchmarkExecutableCommit", "sourceTreeDirty",
            "warmups", "samples",
        ):
            if current.get(key) != manifest.get(key):
                parser.error(f"existing manifest disagrees on {key}")
        manifest = current
        manifest["vectorizationEvidence"] = vectorization
        write_json(manifest_path, manifest)
    else:
        write_json(manifest_path, manifest)

    existing = {item["id"]: item for item in manifest.get("runs", [])}
    records: list[dict] = []
    failed = False
    for index, plan in enumerate(plans, start=1):
        entry = existing.get(plan["id"])
        if entry is not None:
            if entry.get("exitCode") != 0 or not plan["resultPath"].is_file():
                parser.error(f"existing run requires recovery: {plan['id']}")
            result = load_json(plan["resultPath"])
            records.append(result_record(
                result, plan["candidate"], plan["profile"], commit, dirty,
            ))
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
            "id": plan["id"],
            "profile": plan["profile"],
            "candidate": asdict(plan["candidate"]),
            "command": list(map(str, plan["command"])),
            "result": plan["resultPath"].name,
            "log": log_path.name,
            "exitCode": completed.returncode,
        }
        if completed.returncode == 0 and plan["resultPath"].is_file():
            result = load_json(plan["resultPath"])
            record = result_record(
                result, plan["candidate"], plan["profile"], commit, dirty,
            )
            entry.update({
                "runId": record["runId"],
                "valid": record["valid"],
            })
            if record["valid"]:
                records.append(record)
            else:
                entry["exitCode"] = 1
        manifest["runs"].append(entry)
        write_json(manifest_path, manifest)
        if entry["exitCode"] != 0:
            failed = True
            print(log_path.read_text(encoding="utf-8")[-4000:], file=sys.stderr)
            if not arguments.continue_on_error:
                break

    analysis = analyze(records, candidates, commit, dirty)
    analysis["allocationVerification"] = {
        "command": [str(arguments.test_executable.resolve())],
        "exitCode": allocation_run.returncode,
        "log": allocation_log.name,
    }
    analysis["vectorizationEvidence"] = vectorization
    write_json(output / "analysis.json", analysis)
    if analysis["selectedCandidate"] is None:
        print("screen result: no policy passed the complete-total gate")
    else:
        selected = analysis["selectedCandidate"]
        print(
            "screen result: selected " + selected["candidate"]["id"] +
            f"; geometric total/serial={selected['geometricTotalToSerial']:.4f}; "
            f"pointwise/serial={selected['geometricPointwiseToSerial']:.4f}"
        )
    print(
        "pointwise/FFT fusion gate: " +
        ("activate" if analysis["screenGate"]["activatePointwiseFftFusion"]
         else "defer")
    )
    print(f"analysis: {output / 'analysis.json'}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
