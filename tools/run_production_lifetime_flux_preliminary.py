#!/usr/bin/env python3
"""Run the preliminary issue #19 production-lifetime flux harness pair."""

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

from run_spectral_pipeline_sweep import maximum_correctness_error
from run_vertical_gemm_sweep import git_source_state


EXPERIMENT_ID = "issue-019-production-lifetime-spectral-flux-composition"
INCREMENT_ID = "production-lifetime-flux-preliminary-harness-v1"
COHORT_ID = "issue19-synthetic-development-256-v1"
PROFILE = "wvm-current-256-nz129-f4"
TOTAL_STAGE = "production-lifetime streamed four-target spectral-flux composition"


@dataclass(frozen=True)
class Candidate:
    id: str
    policy: str
    primary_provider: str
    role: str


def candidate_matrix() -> list[Candidate]:
    return [
        Candidate(
            "production-lifetime-wvm-direct",
            "wvm-direct",
            "pipeline-production-lifetime-wvm-direct",
            "same-lifetime-wvm-order-control",
        ),
        Candidate(
            "production-lifetime-streaming-pruned-tile16",
            "streaming-pruned-compact-split",
            "pipeline-production-lifetime-streaming-pruned-tile16",
            "issue16-fixed-tile16-candidate",
        ),
    ]


def command_for(
    executable: Path,
    candidate: Candidate,
    warmups: int,
    samples: int,
    seed: int,
    result_path: Path,
) -> list[str]:
    return [
        str(executable),
        "run",
        "--kernel",
        "production-lifetime-flux",
        "--boundary-policy",
        candidate.policy,
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
        "--seed",
        str(seed),
        "--output",
        str(result_path),
    ]


def provider_record(candidate: Candidate, result: dict) -> dict:
    providers = [
        provider
        for provider in result.get("providers", [])
        if provider.get("id") == candidate.primary_provider
    ]
    if len(providers) != 1 or len(result.get("providers", [])) != 1:
        raise ValueError(
            f"{candidate.id} result must contain exactly its one provider"
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
        raise ValueError(f"provider {provider.get('id')} lacks one total timing")
    return float(matches[0]["medianSeconds"])


def component_seconds(provider: dict) -> dict[str, float]:
    components: dict[str, float] = {}
    for timing in provider.get("timings", []):
        if timing.get("scope") in {
            "primitive",
            "component",
            "retained-operator-total",
            "adapter-component",
        }:
            components[timing["stage"]] = float(timing["medianSeconds"])
    return components


def memory_record(provider: dict) -> dict[str, int]:
    memory = provider.get("memory", {})
    keys = (
        "algorithmResidentBytes",
        "scratchBytes",
        "estimatedProcessPeakBytes",
        "observedProcessHighWaterBytes",
    )
    if any(int(memory.get(key, 0)) <= 0 for key in keys):
        raise ValueError(f"provider {provider.get('id')} lacks memory evidence")
    return {key: int(memory[key]) for key in keys}


def analyze(results: list[tuple[Candidate, dict]]) -> dict:
    rows: dict[str, dict] = {}
    maximum_error = 0.0
    all_correct = True
    all_synthetic = True
    for candidate, result in results:
        provider = provider_record(candidate, result)
        error = maximum_correctness_error(provider)
        fixture = result.get("provenance", {}).get("spectralFluxFixture", {})
        synthetic = bool(
            fixture.get("status") ==
                "provider-independent-synthetic-development"
            and fixture.get("authoritative") is False
            and not fixture.get("waveVortexModelCommit")
        )
        maximum_error = max(maximum_error, error)
        all_correct = all_correct and math.isfinite(error) and error <= 1.0e-12
        all_synthetic = all_synthetic and synthetic
        rows[candidate.id] = {
            "providerId": provider["id"],
            "seconds": total_seconds(provider),
            "components": component_seconds(provider),
            "memory": memory_record(provider),
            "maximumCorrectnessError": error,
            "fixtureStatus": fixture.get("status"),
        }

    expected = {candidate.id for candidate in candidate_matrix()}
    complete = set(rows) == expected
    control = rows.get("production-lifetime-wvm-direct")
    candidate = rows.get("production-lifetime-streaming-pruned-tile16")
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
        "phase": "preliminary-harness",
        "cohortId": COHORT_ID,
        "profile": PROFILE,
        "completePair": complete,
        "allCorrectWithin1e-12": all_correct,
        "maximumCorrectnessError": maximum_error,
        "allFixturesSyntheticDevelopment": all_synthetic,
        "candidateToControl": {
            "time": time_ratio,
            "algorithmResidentBytes": memory_ratio,
        },
        "candidates": rows,
        "interpretation": {
            "classification": "preliminary-harness-only",
            "eligibleForReference": False,
            "adoptionGateEvaluated": False,
            "referenceBlocker": (
                "Authoritative versioned WVM fixtures are not yet loaded; "
                "synthetic development evidence cannot enter the 0.90 gate."
            ),
        },
    }


def main() -> int:
    repository_root = Path(__file__).resolve().parents[1]
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--executable",
        type=Path,
        default=repository_root / "build/release/skbench",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--samples", type=int, default=9)
    parser.add_argument("--seed", type=int, default=129)
    parser.add_argument("--allow-dirty-tree", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    arguments = parser.parse_args()
    if min(arguments.warmups, arguments.samples) < 1:
        parser.error("--warmups and --samples must be positive")
    if not arguments.executable.is_file():
        parser.error(f"benchmark executable is missing: {arguments.executable}")

    source_commit, source_dirty = git_source_state(repository_root)
    if source_dirty and not arguments.allow_dirty_tree:
        parser.error(
            "the benchmark source tree is dirty; commit and rebuild for evidence "
            "collection or use --allow-dirty-tree for an exploratory run"
        )

    output = arguments.output or (
        repository_root
        / "results/local"
        / f"issue19-production-lifetime-preliminary-{timestamp}"
    )
    candidates = candidate_matrix()
    commands: list[tuple[Candidate, list[str], Path]] = []
    for candidate in candidates:
        result_path = output / f"{PROFILE}--{candidate.id}.json"
        commands.append(
            (
                candidate,
                command_for(
                    arguments.executable,
                    candidate,
                    arguments.warmups,
                    arguments.samples,
                    arguments.seed,
                    result_path,
                ),
                result_path,
            )
        )

    if arguments.dry_run:
        for _, command, _ in commands:
            print(f"VECLIB_MAXIMUM_THREADS=1 {' '.join(map(str, command))}")
        print("Planned two isolated synthetic-development runs; no gate is evaluated.")
        return 0

    output.mkdir(parents=True, exist_ok=False)
    manifest = {
        "schema": "spectral-kernel-local-sweep-v1",
        "experimentId": EXPERIMENT_ID,
        "incrementId": INCREMENT_ID,
        "phase": "preliminary-harness",
        "cohortId": COHORT_ID,
        "createdAtUtc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "question": (
            "Do the two frozen issue #19 graphs implement the same allocation-free "
            "15-to-4 streamed lifetime with correct full-volume batching?"
        ),
        "baseline": (
            "The matched WVM-order control uses the same seven-real-volume lifetime; "
            "issue #18 is attribution evidence and is not the timing denominator."
        ),
        "controlledVariables": [
            "provider-independent deterministic synthetic modal fixture",
            "four target expressions, target order, normalization, and 1e-12 oracle",
            "Float64 radial two-thirds and Nj=floor(2*(Nz-1)/3) retention",
            "FFTW 3.3.11 MEASURE/unaligned/cold, internal-1/outer-12",
            "K-squared outer-dynamic-16 vertical schedule with VECLIB_MAXIMUM_THREADS=1",
        ],
        "changedVariables": [
            "full WVM-order interleaved FFTW versus partial-column-pruned tile-16 FFTW",
            "direct complex vertical operators versus compact split grouped real operators",
            "frequency-major full spectra versus persistent radial compact split spectra",
        ],
        "timedOperation": (
            "Fifteen ready native modal inputs through streamed vertical and horizontal "
            "reconstruction, four pointwise expressions, retention, projection, and "
            "four ready native modal outputs."
        ),
        "excludedWork": [
            "phase evolution and construction of the 15 inputs from WVM state",
            "coefficient-space accumulation into tendencies",
            "complete nonlinear flux, MATLAB dispatch, time integration, state, I/O, and diagnostics",
            "authoritative WVM fixture evidence and adoption inference",
        ],
        "allocationPolicy": (
            "All application buffers and schedulers persist after setup; an allocator "
            "interposer verifies zero application allocations after opaque provider warmup."
        ),
        "interpretation": (
            "This pair is preliminary harness evidence only. Its ratio is descriptive, "
            "does not evaluate the 0.90 gate, and cannot enter adoption statistics."
        ),
        "profiles": [PROFILE],
        "candidates": [asdict(candidate) for candidate in candidates],
        "threadEnvironment": {"VECLIB_MAXIMUM_THREADS": "1"},
        "sourceTreeGitCommit": source_commit,
        "sourceTreeDirty": source_dirty,
        "rounds": 1,
        "warmups": arguments.warmups,
        "samples": arguments.samples,
        "seed": arguments.seed,
        "runs": [],
    }

    failed = False
    completed_results: list[tuple[Candidate, dict]] = []
    for index, (candidate, command, result_path) in enumerate(commands, start=1):
        stem = result_path.stem
        print(f"[{index}/{len(commands)}] {stem}", flush=True)
        log_path = output / f"{stem}.log"
        environment = os.environ.copy()
        environment["VECLIB_MAXIMUM_THREADS"] = "1"
        with log_path.open("w", encoding="utf-8") as log:
            completed = subprocess.run(
                command,
                cwd=repository_root,
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
        entry = {
            "id": stem,
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
            with result_path.open(encoding="utf-8") as stream:
                result = json.load(stream)
            embedded_commit = result.get("environment", {}).get("gitCommit", "")
            embedded_dirty = result.get("environment", {}).get("gitDirty")
            metadata_matches = bool(
                embedded_commit
                and embedded_commit != "unknown"
                and source_commit.startswith(embedded_commit)
                and embedded_dirty == source_dirty
            )
            fixture = result.get("provenance", {}).get(
                "spectralFluxFixture", {}
            )
            synthetic_fixture = bool(
                fixture.get("status") ==
                    "provider-independent-synthetic-development"
                and fixture.get("authoritative") is False
            )
            entry.update(
                {
                    "runId": result.get("run", {}).get("id"),
                    "status": result.get("status"),
                    "result": result_path.name,
                    "embeddedGitCommit": embedded_commit,
                    "embeddedGitDirty": embedded_dirty,
                    "sourceMetadataMatches": metadata_matches,
                    "syntheticDevelopmentFixture": synthetic_fixture,
                }
            )
            if (
                completed.returncode == 0
                and result.get("status") == "passed"
                and metadata_matches
                and synthetic_fixture
            ):
                completed_results.append((candidate, result))
            else:
                completed = subprocess.CompletedProcess(command, 1)
        manifest["runs"].append(entry)
        with (output / "manifest.json").open("w", encoding="utf-8") as stream:
            json.dump(manifest, stream, indent=2)
            stream.write("\n")
        if completed.returncode != 0:
            failed = True
            print(log_path.read_text(encoding="utf-8")[-4000:], file=sys.stderr)
            if not arguments.continue_on_error:
                break

    if completed_results:
        with (output / "analysis.json").open("w", encoding="utf-8") as stream:
            json.dump(analyze(completed_results), stream, indent=2)
            stream.write("\n")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
