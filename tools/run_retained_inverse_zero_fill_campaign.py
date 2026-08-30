#!/usr/bin/env python3
"""Screen retained-inverse preparation policies for issue #24."""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

from run_vertical_gemm_sweep import git_source_state


EXPERIMENT_ID = "issue-024-retained-inverse-zero-fill"
INCREMENT_ID = "retained-inverse-zero-fill-screen-v1"
MANIFEST_SCHEMA = "spectral-kernel-local-sweep-v1"
ANALYSIS_SCHEMA = "spectral-kernel-retained-inverse-zero-fill-screen-v1"
PROFILES = (
    "wvm-current-256-nz129-f4",
    "wvm-current-512-nz257-f4",
    "wvm-large-1024-nz129-f4",
)
BOUNDARY_POLICY = (
    "streaming-pruned-compact-split-fused-vertical-views"
)
BASE_PROVIDER = (
    "pipeline-production-lifetime-streaming-pruned-tile16-"
    "fused-vertical-views-authoritative"
)
TOTAL_STAGE = (
    "authoritative production-lifetime streamed four-target "
    "spectral-flux composition"
)
SHARED_INVERSE_STAGE = "shared U,V,W horizontal reconstruction"
DERIVATIVE_INVERSE_STAGE = "per-target derivative horizontal reconstruction"
COMBINED_PREPARATION_STAGE = "fused inverse family-view load and embedding"
TILE_LOAD_STAGE = "inverse compact tile load from split family views"
SCATTER_STAGE = (
    "inverse retained and Hermitian-boundary scatter from compact tile"
)
FULL_CLEAR_STAGE = "inverse full half-spectrum zero fill"
ACTIVE_CLEAR_STAGE = "inverse active-column reset"
TOLERANCE = 1.0e-12
INVERSE_SCREEN_RATIO = 0.95
TOTAL_SCREEN_RATIO = 0.97
MAXIMUM_PROFILE_RATIO = 1.03
MEMORY_RATIO_LIMIT = 1.01


@dataclass(frozen=True)
class Candidate:
    id: str
    policy: str
    provider: str
    role: str


def candidate_matrix() -> list[Candidate]:
    return [
        Candidate(
            "full-zero-control", "full-zero",
            BASE_PROVIDER + "-pointwise-spatial-static",
            "frozen-issue21-and-issue22-control",
        ),
        Candidate(
            "active-column-reset", "active-reset",
            BASE_PROVIDER +
            "-inverse-active-reset-pointwise-spatial-static",
            "preserved-row-input-with-strided-active-column-clear",
        ),
        Candidate(
            "compact-preserved-input", "compact-preserved",
            BASE_PROVIDER +
            "-inverse-compact-preserved-pointwise-spatial-static",
            "preserved-compact-column-input-and-out-of-place-column-fft",
        ),
        Candidate(
            "full-stride-preserved-input", "full-preserved",
            BASE_PROVIDER +
            "-inverse-full-preserved-pointwise-spatial-static",
            "preserved-full-stride-column-input-and-out-of-place-column-fft",
        ),
    ]


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def geometric_mean(values: list[float]) -> float:
    if not values or any(value <= 0.0 for value in values):
        raise ValueError("geometric mean requires positive values")
    return math.exp(sum(math.log(value) for value in values) / len(values))


def fixture_assignments(values: list[str]) -> dict[str, Path]:
    fixtures: dict[str, Path] = {}
    for value in values:
        profile, separator, raw_path = value.partition("=")
        if not separator:
            raise ValueError("fixture assignments must use PROFILE=PREPARED_PATH")
        if profile not in PROFILES:
            raise ValueError(f"unknown inverse-preparation profile: {profile}")
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


def command_for(
    executable: Path, fixture: Path, profile: str, candidate: Candidate,
    warmups: int, samples: int, output: Path,
) -> list[str]:
    return [
        str(executable), "run",
        "--kernel", "production-lifetime-flux",
        "--profile", profile,
        "--boundary-policy", BOUNDARY_POLICY,
        "--spectral-flux-fixture", str(fixture),
        "--fftw-planning", "measure",
        "--fftw-alignment", "unaligned",
        "--fftw-wisdom", "cold",
        "--fftw-internal-workers", "1",
        "--fftw-outer-workers", "12",
        "--streaming-tile-width", "16",
        "--streaming-inverse-policy", candidate.policy,
        "--vertical-gemm-family", "k2-grouped",
        "--vertical-gemm-schedule", "outer-dynamic",
        "--vertical-gemm-outer-workers", "16",
        "--pointwise-policy", "spatial-static",
        "--pointwise-workers", "8",
        "--warmups", str(warmups),
        "--samples", str(samples),
        "--output", str(output),
    ]


def provider_record(result: dict, candidate: Candidate) -> dict:
    providers = result.get("providers", [])
    if len(providers) != 1 or providers[0].get("id") != candidate.provider:
        raise ValueError(
            f"{candidate.id} must contain only provider {candidate.provider}"
        )
    return providers[0]


def timing(record: dict, stage: str) -> dict:
    matches = [
        item for item in record.get("timings", [])
        if item.get("stage") == stage
    ]
    if len(matches) != 1:
        raise ValueError(f"provider must contain exactly one {stage!r} timing")
    return matches[0]


def optional_timing(record: dict, stages: tuple[str, ...]) -> dict | None:
    matches = [
        item for item in record.get("timings", [])
        if item.get("stage") in stages
    ]
    if len(matches) > 1:
        raise ValueError("provider contains multiple inverse-clear timings")
    return matches[0] if matches else None


def maximum_correctness_error(record: dict) -> float:
    metrics = record.get("correctness", [])
    if not metrics or not all(item.get("passed") is True for item in metrics):
        return math.inf
    return max(float(item["maximumRelativeError"]) for item in metrics)


def allocation_ledger_valid(record: dict) -> bool:
    matches = [
        item for item in record.get("componentLedger", [])
        if item.get("stage") == "steady-state allocation"
    ]
    return bool(
        len(matches) == 1 and matches[0].get("state") == "elided"
        and "persistent" in matches[0].get("detail", "")
    )


def collect_record(
    result: dict, candidate: Candidate, profile: str,
    commit: str, warmups: int, samples: int,
) -> dict:
    record = provider_record(result, candidate)
    environment = result.get("environment", {})
    fixture = result.get("provenance", {}).get("spectralFluxFixture", {})
    maximum_error = maximum_correctness_error(record)
    total = timing(record, TOTAL_STAGE)
    shared_inverse = timing(record, SHARED_INVERSE_STAGE)
    derivative_inverse = timing(record, DERIVATIVE_INVERSE_STAGE)
    preparation = timing(record, COMBINED_PREPARATION_STAGE)
    tile_load = timing(record, TILE_LOAD_STAGE)
    scatter = timing(record, SCATTER_STAGE)
    clear = optional_timing(record, (FULL_CLEAR_STAGE, ACTIVE_CLEAR_STAGE))
    inverse_seconds = (
        float(shared_inverse["medianSeconds"])
        + float(derivative_inverse["medianSeconds"])
    )
    memory = record.get("memory", {})
    source_matches = bool(
        environment.get("gitCommit")
        and commit.startswith(environment["gitCommit"])
        and environment.get("gitDirty") is False
    )
    valid = bool(
        result.get("status") == "passed"
        and result.get("run", {}).get("profile") == profile
        and result.get("run", {}).get("warmups") == warmups
        and result.get("run", {}).get("samples") == samples
        and len(total.get("samplesSeconds", [])) == samples
        and fixture.get("authoritative") is True
        and source_matches
        and maximum_error <= TOLERANCE
        and allocation_ledger_valid(record)
    )
    return {
        "runId": result.get("run", {}).get("id"),
        "profile": profile,
        "candidateId": candidate.id,
        "totalSeconds": float(total["medianSeconds"]),
        "inverseBoundarySeconds": inverse_seconds,
        "inversePreparationSeconds": float(preparation["medianSeconds"]),
        "tileLoadSeconds": float(tile_load["medianSeconds"]),
        "clearSeconds": (
            float(clear["medianSeconds"]) if clear is not None else 0.0
        ),
        "clearState": clear.get("state") if clear is not None else "elided",
        "scatterSeconds": float(scatter["medianSeconds"]),
        "bytes": {
            "combinedPreparation": int(preparation.get("bytesMoved", 0)),
            "tileLoad": int(tile_load.get("bytesMoved", 0)),
            "clear": int(clear.get("bytesMoved", 0)) if clear is not None else 0,
            "scatter": int(scatter.get("bytesMoved", 0)),
        },
        "memory": {
            "algorithmResidentBytes": int(memory.get("algorithmResidentBytes", 0)),
            "scratchBytes": int(memory.get("scratchBytes", 0)),
            "estimatedProcessPeakBytes": int(
                memory.get("estimatedProcessPeakBytes", 0)
            ),
            "observedProcessHighWaterBytes": int(
                memory.get("observedProcessHighWaterBytes", 0)
            ),
        },
        "maximumCorrectnessError": maximum_error,
        "allocationLedgerValid": allocation_ledger_valid(record),
        "sourceMetadataMatches": source_matches,
        "valid": valid,
    }


def analyze(records: list[dict], candidates: list[Candidate], commit: str) -> dict:
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
    control_id = candidates[0].id
    for candidate in candidates:
        profiles = [cells.get((profile, candidate.id)) for profile in PROFILES]
        controls = [cells.get((profile, control_id)) for profile in PROFILES]
        if any(item is None for item in profiles + controls):
            continue
        total_ratios = [
            profiles[index]["totalSeconds"] / controls[index]["totalSeconds"]
            for index in range(len(PROFILES))
        ]
        inverse_ratios = [
            profiles[index]["inverseBoundarySeconds"] /
            controls[index]["inverseBoundarySeconds"]
            for index in range(len(PROFILES))
        ]
        memory_ratios = [
            profiles[index]["memory"]["algorithmResidentBytes"] /
            controls[index]["memory"]["algorithmResidentBytes"]
            for index in range(len(PROFILES))
        ]
        summary = {
            "candidate": asdict(candidate),
            "geometricTotalToControl": geometric_mean(total_ratios),
            "geometricInverseBoundaryToControl": geometric_mean(inverse_ratios),
            "maximumProfileTotalToControl": max(total_ratios),
            "maximumProfileInverseBoundaryToControl": max(inverse_ratios),
            "maximumProfileAlgorithmResidentToControl": max(memory_ratios),
            "maximumAdditionalAlgorithmResidentBytes": max(
                profiles[index]["memory"]["algorithmResidentBytes"] -
                controls[index]["memory"]["algorithmResidentBytes"]
                for index in range(len(PROFILES))
            ),
            "profileTotalRatios": dict(zip(PROFILES, total_ratios)),
            "profileInverseBoundaryRatios": dict(zip(PROFILES, inverse_ratios)),
            "profiles": profiles,
        }
        summary["passesReferenceAdvanceGate"] = bool(
            candidate.id != control_id
            and summary["geometricInverseBoundaryToControl"] <=
                INVERSE_SCREEN_RATIO
            and summary["geometricTotalToControl"] <= TOTAL_SCREEN_RATIO
            and summary["maximumProfileTotalToControl"] <=
                MAXIMUM_PROFILE_RATIO
            and summary["maximumProfileAlgorithmResidentToControl"] <=
                MEMORY_RATIO_LIMIT
            and all(item["maximumCorrectnessError"] <= TOLERANCE for item in profiles)
        )
        summaries.append(summary)
    advancing = [
        item["candidate"]["id"] for item in summaries
        if item["passesReferenceAdvanceGate"]
    ]
    return {
        "schema": ANALYSIS_SCHEMA,
        "experimentId": EXPERIMENT_ID,
        "incrementId": INCREMENT_ID,
        "classification": "preliminary",
        "createdAtUtc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "benchmarkExecutableCommit": commit,
        "profiles": list(PROFILES),
        "completeMatchedMatrix": complete,
        "allCorrectWithin1e12": bool(
            complete and all(
                record["maximumCorrectnessError"] <= TOLERANCE
                for record in records
            )
        ),
        "candidateSummaries": summaries,
        "referenceCandidateIds": advancing,
        "screenGate": {
            "geometricInverseBoundaryRatioAtMost": INVERSE_SCREEN_RATIO,
            "geometricTotalRatioAtMost": TOTAL_SCREEN_RATIO,
            "maximumProfileTotalRatioAtMost": MAXIMUM_PROFILE_RATIO,
            "maximumProfileAlgorithmResidentRatioAtMost": MEMORY_RATIO_LIMIT,
            "candidatePassed": bool(advancing),
        },
        "disposition": (
            "advance survivors to repeated reference depth"
            if advancing else
            "retain the frozen full-zero policy; no candidate improved the "
            "inverse boundary enough to justify reference depth"
        ),
        "limitations": (
            "One rotated preliminary process per cell with no empirical "
            "confidence interval. The screen can reject candidates that miss "
            "the preregistered component and total gates, but cannot authorize "
            "adoption."
        ),
    }


def main() -> int:
    repository_root = Path(__file__).resolve().parents[1]
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", action="append", required=True)
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

    candidates = candidate_matrix()
    output = (arguments.output or (
        repository_root / "results/local" /
        f"issue24-retained-inverse-zero-fill-screen-{timestamp}"
    )).resolve()
    plans: list[dict] = []
    for profile_index, profile in enumerate(PROFILES):
        offset = profile_index % len(candidates)
        ordered = candidates[offset:] + candidates[:offset]
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

    manifest_path = output / "manifest.json"
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "experimentId": EXPERIMENT_ID,
        "incrementId": INCREMENT_ID,
        "phase": "screen",
        "classification": "preliminary",
        "createdAtUtc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "question": (
            "Can the frozen tile-16 direct-family-view inverse reduce repeated "
            "full half-spectrum clearing without losing FFTW locality?"
        ),
        "baseline": (
            "The frozen issue #21 direct-family-view graph, issue #22 "
            "spatial-static-8 pointwise policy, and one contiguous full "
            "half-spectrum clear before each inverse plane."
        ),
        "profiles": list(PROFILES),
        "fixtures": {profile: str(path) for profile, path in fixtures.items()},
        "candidates": [asdict(candidate) for candidate in candidates],
        "controlledVariables": [
            "authoritative WVM fixtures, mode-keyed oracle, and Float64 arithmetic",
            "FFTW 3.3.11 MEASURE/UNALIGNED with one internal worker",
            "horizontal outer-12, vertical dynamic-16, and pointwise static-8",
            "tile width 16, direct F/G split family views, and exact vertical matrices",
            "four-field production multiplicity: fifteen inverse and four forward transforms",
        ],
        "changedVariables": [
            "full contiguous clear versus active-column clear",
            "in-place column inverse versus preserved compact or full-stride input",
            "inverse scratch required by preserved-input candidates",
        ],
        "timedOperation": (
            "Separately time inverse tile load, clear, retained/Hermitian "
            "scatter, complete fifteen-transform inverse boundary, and the "
            "uninstrumented fifteen-input to four-output composition."
        ),
        "excludedWork": [
            "complete nonlinear flux, coefficient assembly, MATLAB/MEX, model state, and time integration",
            "fixture loading, correctness oracle, planning, and setup from steady-state totals",
            "size-dependent dispatch and custom sparse inverse FFT butterflies",
        ],
        "allocationPolicy": "zero application allocations after persistent setup",
        "benchmarkExecutableCommit": commit,
        "sourceTreeDirty": dirty,
        "warmups": arguments.warmups,
        "samples": arguments.samples,
        "threadEnvironment": {"VECLIB_MAXIMUM_THREADS": "1"},
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
            records.append(collect_record(
                result, plan["candidate"], plan["profile"], commit,
                arguments.warmups, arguments.samples,
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
            "primaryProvider": plan["candidate"].provider,
            "command": list(map(str, plan["command"])),
            "result": plan["resultPath"].name,
            "log": log_path.name,
            "exitCode": completed.returncode,
        }
        if completed.returncode == 0 and plan["resultPath"].is_file():
            result = load_json(plan["resultPath"])
            record = collect_record(
                result, plan["candidate"], plan["profile"], commit,
                arguments.warmups, arguments.samples,
            )
            entry.update({"runId": record["runId"], "valid": record["valid"]})
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

    analysis = analyze(records, candidates, commit)
    analysis["allocationVerification"] = {
        "command": [str(arguments.test_executable.resolve())],
        "exitCode": allocation_run.returncode,
        "log": allocation_log.name,
    }
    write_json(output / "analysis.json", analysis)
    if analysis["referenceCandidateIds"]:
        print(
            "screen result: advance " +
            ", ".join(analysis["referenceCandidateIds"])
        )
    else:
        print("screen result: no inverse-preparation candidate advances")
    print(f"manifest={manifest_path}")
    print(f"analysis={output / 'analysis.json'}")
    return 1 if failed or not analysis["completeMatchedMatrix"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
