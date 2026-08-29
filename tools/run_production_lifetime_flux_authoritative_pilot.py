#!/usr/bin/env python3
"""Run the bounded authoritative issue #19 256-squared pilot pair."""

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

from prepare_spectral_flux_fixture import prepare
from run_spectral_pipeline_sweep import maximum_correctness_error
from run_vertical_gemm_sweep import git_source_state


EXPERIMENT_ID = "issue-019-production-lifetime-spectral-flux-composition"
INCREMENT_ID = "production-lifetime-flux-authoritative-pilot-v1"
COHORT_ID = "issue19-authoritative-wvm-pilot-256-v1"
PROFILE = "wvm-current-256-nz129-f4"
TOTAL_STAGE = (
    "authoritative production-lifetime streamed four-target spectral-flux composition"
)


@dataclass(frozen=True)
class Candidate:
    id: str
    policy: str
    primary_provider: str
    role: str


def candidate_matrix() -> list[Candidate]:
    return [
        Candidate(
            "production-lifetime-wvm-direct-authoritative",
            "wvm-direct",
            "pipeline-production-lifetime-wvm-direct-authoritative",
            "same-lifetime-wvm-order-control",
        ),
        Candidate(
            "production-lifetime-streaming-pruned-tile16-authoritative",
            "streaming-pruned-compact-split",
            "pipeline-production-lifetime-streaming-pruned-tile16-authoritative",
            "issue16-fixed-tile16-candidate",
        ),
    ]


def command_for(executable: Path, prepared_fixture: Path, candidate: Candidate,
                warmups: int, samples: int, result_path: Path) -> list[str]:
    return [
        str(executable),
        "run",
        "--kernel",
        "production-lifetime-flux",
        "--boundary-policy",
        candidate.policy,
        "--spectral-flux-fixture",
        str(prepared_fixture),
        "--profile",
        PROFILE,
        "--fftw-planning",
        "measure",
        "--fftw-alignment",
        "unaligned",
        "--fftw-wisdom",
        "cold",
        "--fftw-internal-workers",
        "1",
        "--fftw-outer-workers",
        "12",
        "--streaming-tile-width",
        "16",
        "--vertical-gemm-family",
        "k2-grouped",
        "--vertical-gemm-schedule",
        "outer-dynamic",
        "--vertical-gemm-outer-workers",
        "16",
        "--warmups",
        str(warmups),
        "--samples",
        str(samples),
        "--output",
        str(result_path),
    ]


def provider_record(candidate: Candidate, result: dict) -> dict:
    providers = result.get("providers", [])
    if len(providers) != 1 or providers[0].get("id") != candidate.primary_provider:
        raise ValueError(
            f"{candidate.id} must contain only {candidate.primary_provider}"
        )
    return providers[0]


def total_seconds(provider: dict) -> float:
    matches = [
        timing
        for timing in provider.get("timings", [])
        if timing.get("scope") == "uninstrumented-total"
        and timing.get("stage") == TOTAL_STAGE
        and timing.get("direction") == "forward"
    ]
    if len(matches) != 1:
        raise ValueError(f"provider {provider.get('id')} lacks one authoritative total")
    return float(matches[0]["medianSeconds"])


def component_seconds(provider: dict) -> dict[str, float]:
    return {
        timing["stage"]: float(timing["medianSeconds"])
        for timing in provider.get("timings", [])
        if timing.get("scope") in {
            "primitive", "component", "retained-operator-total",
            "adapter-component",
        }
    }


def memory_record(provider: dict) -> dict[str, int]:
    memory = provider.get("memory", {})
    keys = (
        "algorithmResidentBytes", "scratchBytes", "estimatedProcessPeakBytes",
        "observedProcessHighWaterBytes",
    )
    if any(int(memory.get(key, 0)) <= 0 for key in keys):
        raise ValueError(f"provider {provider.get('id')} lacks memory evidence")
    return {key: int(memory[key]) for key in keys}


def analyze(results: list[tuple[Candidate, dict]], prepared: dict) -> dict:
    rows: dict[str, dict] = {}
    maximum_error = 0.0
    fixture_hashes: set[str] = set()
    wvm_commits: set[str] = set()
    all_correct = True
    all_authoritative = True
    for candidate, result in results:
        provider = provider_record(candidate, result)
        fixture = result.get("provenance", {}).get("spectralFluxFixture", {})
        error = maximum_correctness_error(provider)
        authoritative = bool(
            fixture.get("status") == "authoritative-wvm-export"
            and fixture.get("authoritative") is True
            and fixture.get("schema") == "spectral-flux-fixture-v1"
            and fixture.get("fixtureHash") == prepared["fixtureHash"]
            and fixture.get("waveVortexModelCommit") ==
                prepared["waveVortexModelCommit"]
        )
        maximum_error = max(maximum_error, error)
        all_correct = all_correct and math.isfinite(error) and error <= 1.0e-12
        all_authoritative = all_authoritative and authoritative
        fixture_hashes.add(str(fixture.get("fixtureHash")))
        wvm_commits.add(str(fixture.get("waveVortexModelCommit")))
        rows[candidate.id] = {
            "providerId": provider["id"],
            "seconds": total_seconds(provider),
            "components": component_seconds(provider),
            "memory": memory_record(provider),
            "maximumCorrectnessError": error,
            "fixtureStatus": fixture.get("status"),
            "fixtureHash": fixture.get("fixtureHash"),
            "waveVortexModelCommit": fixture.get("waveVortexModelCommit"),
        }
    expected = {candidate.id for candidate in candidate_matrix()}
    complete = set(rows) == expected
    control = rows.get("production-lifetime-wvm-direct-authoritative")
    candidate = rows.get(
        "production-lifetime-streaming-pruned-tile16-authoritative"
    )
    time_ratio = None
    memory_ratio = None
    if control is not None and candidate is not None:
        time_ratio = candidate["seconds"] / control["seconds"]
        memory_ratio = (
            candidate["memory"]["algorithmResidentBytes"] /
            control["memory"]["algorithmResidentBytes"]
        )
    return {
        "schema": "spectral-kernel-production-lifetime-flux-analysis-v1",
        "phase": "authoritative-pilot",
        "cohortId": COHORT_ID,
        "profile": PROFILE,
        "completePair": complete,
        "allCorrectWithin1e-12": all_correct,
        "maximumCorrectnessError": maximum_error,
        "allFixturesAuthoritative": all_authoritative,
        "singleFixtureHash": len(fixture_hashes) == 1,
        "singleWaveVortexModelCommit": len(wvm_commits) == 1,
        "candidateToControl": {
            "time": time_ratio,
            "algorithmResidentBytes": memory_ratio,
        },
        "candidates": rows,
        "interpretation": {
            "classification": "authoritative-single-workload-pilot",
            "eligibleForReference": False,
            "adoptionGateEvaluated": False,
            "referenceBlocker": (
                "This increment validates one 256-squared fixture and the exact "
                "bridge only. The preregistered multi-workload, repeated-round, "
                "capacity, and cross-Mac reference campaign has not run."
            ),
        },
    }


def main() -> int:
    repository_root = Path(__file__).resolve().parents[1]
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", required=True, type=Path)
    parser.add_argument(
        "--executable", type=Path,
        default=repository_root / "build/release/skbench",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--samples", type=int, default=9)
    parser.add_argument("--allow-dirty-tree", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    arguments = parser.parse_args()
    if min(arguments.warmups, arguments.samples) < 1:
        parser.error("--warmups and --samples must be positive")
    if not arguments.executable.is_file():
        parser.error(f"benchmark executable is missing: {arguments.executable}")
    if not (arguments.fixture / "manifest.json").is_file():
        parser.error(f"fixture manifest is missing: {arguments.fixture / 'manifest.json'}")

    source_commit, source_dirty = git_source_state(repository_root)
    if source_dirty and not arguments.allow_dirty_tree:
        parser.error(
            "the benchmark source tree is dirty; commit and rebuild for pilot "
            "collection or use --allow-dirty-tree for an exploratory run"
        )
    output = arguments.output or (
        repository_root / "results/local" /
        f"issue19-production-lifetime-authoritative-pilot-{timestamp}"
    )
    output.mkdir(parents=True, exist_ok=False)
    prepared_path = output / "prepared-fixture.bin"
    try:
        prepared = prepare(arguments.fixture.resolve(), prepared_path.resolve())
    except (ValueError, OSError) as error:
        parser.error(str(error))

    candidates = candidate_matrix()
    manifest = {
        "schema": "spectral-kernel-local-sweep-v1",
        "experimentId": EXPERIMENT_ID,
        "incrementId": INCREMENT_ID,
        "phase": "authoritative-pilot",
        "cohortId": COHORT_ID,
        "createdAtUtc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "question": (
            "Do both frozen issue #19 graphs reproduce one exact WVM-exported "
            "256-squared 15-to-4 spectral-flux boundary within 1e-12?"
        ),
        "baseline": (
            "The matched WVM-order full-spectrum graph is the pilot denominator; "
            "the earlier synthetic pair tested only harness lifetime."
        ),
        "controlledVariables": [
            "one SHA-256-verified spectral-flux-fixture-v1 export and WVM commit",
            "exact wave-f/wave-g K-squared matrices and field mappings",
            "four targets, normalization, mode keys, Float64, and 1e-12 oracle",
            "FFTW 3.3.11 MEASURE/unaligned/cold, internal-1/outer-12",
            "K-squared outer-dynamic-16 with VECLIB_MAXIMUM_THREADS=1",
        ],
        "changedVariables": [
            "full WVM-order FFTW versus partial-column-pruned tile-16 FFTW",
            "direct complex versus compact split-real exact F/G vertical providers",
            "full frequency-major versus radial compact split representation",
        ],
        "timedOperation": (
            "Fifteen ready modal inputs through exact vertical reconstruction, "
            "streamed physical products, horizontal transforms and retention, "
            "exact vertical projection, and four ready modal targets."
        ),
        "excludedWork": [
            "fixture preparation, loading, hashing, and oracle comparison",
            "phase evolution and construction of the 15 inputs from model state",
            "coefficient accumulation, complete nonlinear flux, MATLAB dispatch, "
            "time integration, state, I/O, and diagnostics",
            "the multi-workload reference gate and cross-Mac inference",
        ],
        "allocationPolicy": (
            "Seven real volumes and all spectral/provider buffers persist; the "
            "fixture and correctness-only copies are released before total timing."
        ),
        "interpretation": (
            "This is authoritative operator correctness for one workload and a "
            "descriptive pilot ratio, not the preregistered reference campaign."
        ),
        "profiles": [PROFILE],
        "candidates": [asdict(candidate) for candidate in candidates],
        "fixture": {
            "fixtureId": prepared["fixtureId"],
            "fixtureHash": prepared["fixtureHash"],
            "waveVortexModelCommit": prepared["waveVortexModelCommit"],
            "preparedBytes": prepared["preparedBytes"],
        },
        "threadEnvironment": {"VECLIB_MAXIMUM_THREADS": "1"},
        "sourceTreeGitCommit": source_commit,
        "sourceTreeDirty": source_dirty,
        "rounds": 1,
        "warmups": arguments.warmups,
        "samples": arguments.samples,
        "runs": [],
    }

    failed = False
    completed_results: list[tuple[Candidate, dict]] = []
    for index, candidate in enumerate(candidates, start=1):
        result_path = output / f"{PROFILE}--{candidate.id}.json"
        command = command_for(
            arguments.executable.resolve(), prepared_path.resolve(), candidate,
            arguments.warmups, arguments.samples, result_path,
        )
        print(f"[{index}/{len(candidates)}] {result_path.stem}", flush=True)
        log_path = result_path.with_suffix(".log")
        environment = os.environ.copy()
        environment["VECLIB_MAXIMUM_THREADS"] = "1"
        with log_path.open("w", encoding="utf-8") as log:
            completed = subprocess.run(
                command, cwd=repository_root, env=environment,
                stdout=log, stderr=subprocess.STDOUT,
            )
        entry = {
            "id": result_path.stem,
            "round": 1,
            "profile": PROFILE,
            "candidate": asdict(candidate),
            "primaryProvider": candidate.primary_provider,
            "environment": {"VECLIB_MAXIMUM_THREADS": "1"},
            "command": list(map(str, command)),
            "exitCode": completed.returncode,
            "log": log_path.name,
            "sourceTreeGitCommit": source_commit,
            "sourceTreeDirty": source_dirty,
        }
        if result_path.is_file():
            result = json.loads(result_path.read_text(encoding="utf-8"))
            embedded_commit = result.get("environment", {}).get("gitCommit", "")
            embedded_dirty = result.get("environment", {}).get("gitDirty")
            metadata_matches = bool(
                embedded_commit and embedded_commit != "unknown"
                and source_commit.startswith(embedded_commit)
                and embedded_dirty == source_dirty
            )
            fixture_record = result.get("provenance", {}).get(
                "spectralFluxFixture", {}
            )
            fixture_matches = bool(
                fixture_record.get("authoritative") is True
                and fixture_record.get("fixtureHash") == prepared["fixtureHash"]
                and fixture_record.get("waveVortexModelCommit") ==
                    prepared["waveVortexModelCommit"]
            )
            entry.update({
                "runId": result.get("run", {}).get("id"),
                "status": result.get("status"),
                "result": result_path.name,
                "embeddedGitCommit": embedded_commit,
                "embeddedGitDirty": embedded_dirty,
                "sourceMetadataMatches": metadata_matches,
                "authoritativeFixtureMatches": fixture_matches,
            })
            if (completed.returncode == 0 and result.get("status") == "passed"
                    and metadata_matches and fixture_matches):
                completed_results.append((candidate, result))
            else:
                completed = subprocess.CompletedProcess(command, 1)
        manifest["runs"].append(entry)
        (output / "manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
        if completed.returncode != 0:
            failed = True
            print(log_path.read_text(encoding="utf-8")[-4000:], file=sys.stderr)
            if not arguments.continue_on_error:
                break

    if completed_results:
        analysis = analyze(completed_results, prepared)
        (output / "analysis.json").write_text(
            json.dumps(analysis, indent=2) + "\n", encoding="utf-8"
        )
        if not (
            analysis["completePair"] and analysis["allCorrectWithin1e-12"]
            and analysis["allFixturesAuthoritative"]
            and analysis["singleFixtureHash"]
            and analysis["singleWaveVortexModelCommit"]
        ):
            failed = True
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
