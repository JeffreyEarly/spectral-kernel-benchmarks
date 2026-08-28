#!/usr/bin/env python3
"""Run the bounded issue #16 streaming-pruned pipeline screen."""

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

from run_spectral_pipeline_sweep import (
    Candidate,
    estimated_explicit_peak_bytes as issue9_estimated_peak,
    geometric_mean,
    maximum_correctness_error,
    provider_record,
    provider_timing,
)
from run_vertical_gemm_sweep import (
    PROFILE_SHAPES,
    gibibytes,
    git_source_state,
    sysctl_uint64,
)


EXPERIMENT_ID = "issue-016-streaming-pruned-compact-split"
INCREMENT_ID = "streaming-pruned-compact-split-screen-v1"
COHORT_ID = "issue16-three-profile-f4-v1"
BASELINE_ID = "plane-major-fused-split--outer-dynamic-16"
CANDIDATE_ID = "streaming-pruned-compact-split--outer-dynamic-16"
PROFILES = (
    "wvm-current-256-nz129-f4",
    "wvm-current-512-nz257-f4",
    "wvm-large-1024-nz129-f4",
)
LARGE_PROFILES = (
    "wvm-current-512-nz257-f4",
    "wvm-large-1024-nz129-f4",
)


@dataclass(frozen=True)
class ScreenCandidate:
    id: str
    policy: str
    primary_provider: str
    role: str


def candidate_matrix() -> list[ScreenCandidate]:
    return [
        ScreenCandidate(
            BASELINE_ID,
            "plane-major-fused-split",
            "pipeline-plane-major-fused-split",
            "published-issue9-single-policy-control",
        ),
        ScreenCandidate(
            CANDIDATE_ID,
            "streaming-pruned-compact-split",
            "pipeline-streaming-pruned-compact-split",
            "issue16-streaming-pruned-candidate",
        ),
    ]


def command_for(
    executable: Path,
    candidate: ScreenCandidate,
    profile: str,
    warmups: int,
    samples: int,
    seed: int,
    result_path: Path,
) -> list[str]:
    return [
        str(executable), "run",
        "--kernel", "spectral-pipeline",
        "--boundary-policy", candidate.policy,
        "--profile", profile,
        "--fftw-planning", "measure",
        "--fftw-alignment", "unaligned",
        "--fftw-wisdom", "cold",
        "--fftw-internal-workers", "1",
        "--fftw-outer-workers", "12",
        "--vertical-gemm-family", "k2-grouped",
        "--vertical-gemm-schedule", "outer-dynamic",
        "--vertical-gemm-outer-workers", "16",
        "--warmups", str(warmups),
        "--samples", str(samples),
        "--seed", str(seed),
        "--output", str(result_path),
    ]


def estimated_explicit_peak_bytes(profile: str, policy: str) -> int:
    baseline = issue9_estimated_peak(profile, "plane-major-fused-split")
    if policy == "plane-major-fused-split":
        return baseline
    if policy != "streaming-pruned-compact-split":
        raise ValueError(f"unknown issue #16 policy: {policy}")
    nx, nz, fields, _, _ = PROFILE_SHAPES[profile]
    full_spectrum = nx * (nx // 2 + 1) * nz * fields * 16
    worker_local_scratch = 12 * nx * (nx // 2 + 1) * 16
    return baseline - full_spectrum + worker_local_scratch


def required_memory(provider: dict) -> dict[str, int]:
    keys = (
        "algorithmResidentBytes",
        "benchmarkHarnessBytes",
        "estimatedProcessPeakBytes",
        "observedProcessHighWaterBytes",
    )
    memory = provider.get("memory", {})
    missing = [key for key in keys if int(memory.get(key, 0)) <= 0]
    if missing:
        raise ValueError(
            f"provider {provider['id']} lacks positive memory fields: "
            f"{', '.join(missing)}"
        )
    return {key: int(memory[key]) for key in keys}


def candidate_provider(candidate: ScreenCandidate, result: dict) -> dict:
    compatible = Candidate(
        candidate.id, candidate.policy, candidate.primary_provider, candidate.role,
    )
    return provider_record(compatible, result)


def analyze(results: list[tuple[ScreenCandidate, dict]]) -> dict:
    cells: dict[tuple[str, str], tuple[float, dict[str, int]]] = {}
    maximum_error = 0.0
    all_correct = True
    for candidate, result in results:
        provider = candidate_provider(candidate, result)
        profile = result["run"]["profile"]
        cells[(candidate.id, profile)] = (
            provider_timing(provider), required_memory(provider),
        )
        error = maximum_correctness_error(provider)
        maximum_error = max(maximum_error, error)
        all_correct = all_correct and math.isfinite(error) and error <= 1.0e-12

    complete = set(cells) == {
        (candidate.id, profile)
        for candidate in candidate_matrix()
        for profile in PROFILES
    }
    rows: list[dict] = []
    for profile in PROFILES:
        baseline = cells.get((BASELINE_ID, profile))
        candidate = cells.get((CANDIDATE_ID, profile))
        if baseline is None or candidate is None:
            continue
        memory: dict[str, dict[str, float]] = {}
        for key in (
            "algorithmResidentBytes",
            "benchmarkHarnessBytes",
            "estimatedProcessPeakBytes",
            "observedProcessHighWaterBytes",
        ):
            memory[key] = {
                "baselineBytes": baseline[1][key],
                "candidateBytes": candidate[1][key],
                "candidateToBaseline": candidate[1][key] / baseline[1][key],
            }
        rows.append({
            "profile": profile,
            "baselineSeconds": baseline[0],
            "candidateSeconds": candidate[0],
            "candidateToBaseline": candidate[0] / baseline[0],
            "memory": memory,
        })

    timing_ratios = [row["candidateToBaseline"] for row in rows]
    resident_ratios = [
        row["memory"]["algorithmResidentBytes"]["candidateToBaseline"]
        for row in rows
    ]
    large_ratios = [
        row["candidateToBaseline"] for row in rows
        if row["profile"] in LARGE_PROFILES
    ]
    geometric_ratio = geometric_mean(timing_ratios) if timing_ratios else None
    large_geometric_ratio = geometric_mean(large_ratios) if large_ratios else None
    resident_ratio = geometric_mean(resident_ratios) if resident_ratios else None
    time_gate = bool(
        complete and all_correct and large_geometric_ratio is not None
        and large_geometric_ratio <= 0.97
    )
    memory_gate = bool(
        complete and all_correct and resident_ratio is not None
        and geometric_ratio is not None and resident_ratio <= 0.90
        and geometric_ratio <= 1.02
    )
    advances = time_gate or memory_gate
    if not complete or not all_correct:
        classification = "correctness-or-capability-failure"
    elif time_gate:
        classification = "advances-on-large-case-time"
    elif memory_gate:
        classification = "advances-on-memory-within-time-bound"
    else:
        classification = "negative-preliminary-screen"

    return {
        "schema": "spectral-kernel-streaming-pruned-analysis-v1",
        "phase": "screen",
        "cohortId": COHORT_ID,
        "baselineCandidateId": BASELINE_ID,
        "optimizationCandidateId": CANDIDATE_ID,
        "completeProductionMatrix": complete,
        "allCorrectWithin1e-12": all_correct,
        "maximumCorrectnessError": maximum_error,
        "geometricCandidateToBaseline": geometric_ratio,
        "largeCaseGeometricCandidateToBaseline": large_geometric_ratio,
        "geometricAlgorithmResidentCandidateToBaseline": resident_ratio,
        "profiles": rows,
        "screenGate": {
            "largeCaseRatioAtMost": 0.97,
            "algorithmResidentRatioAtMost": 0.90,
            "memoryPathOverallTimeRatioAtMost": 1.02,
            "largeCaseTimePassed": time_gate,
            "memoryWithinTimePassed": memory_gate,
            "advanceToReference": advances,
            "classification": classification,
            "sizeDependentDispatchAllowed": False,
        },
    }


def main() -> int:
    repository_root = Path(__file__).resolve().parents[1]
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--executable", type=Path,
        default=repository_root / "build/release/skbench",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--seed", type=int, default=129)
    parser.add_argument("--max-memory-fraction", type=float, default=0.5)
    parser.add_argument("--allow-memory-risk", action="store_true")
    parser.add_argument("--allow-dirty-tree", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    arguments = parser.parse_args()
    if min(arguments.warmups, arguments.samples) < 1:
        parser.error("--warmups and --samples must be positive")
    if not 0.0 < arguments.max_memory_fraction <= 1.0:
        parser.error("--max-memory-fraction must be in (0, 1]")

    source_commit, source_dirty = git_source_state(repository_root)
    if source_dirty and not arguments.allow_dirty_tree:
        parser.error(
            "the benchmark source tree is dirty; commit and rebuild for evidence "
            "collection or use --allow-dirty-tree for an exploratory run"
        )

    candidates = candidate_matrix()
    estimates = {
        candidate.id: {
            profile: estimated_explicit_peak_bytes(profile, candidate.policy)
            for profile in PROFILES
        }
        for candidate in candidates
    }
    physical_memory = sysctl_uint64("hw.memsize", 0)
    if physical_memory > 0 and not arguments.allow_memory_risk:
        rejected = [
            (candidate_id, profile, estimate)
            for candidate_id, profile_estimates in estimates.items()
            for profile, estimate in profile_estimates.items()
            if estimate > arguments.max_memory_fraction * physical_memory
        ]
        if rejected:
            details = ", ".join(
                f"{candidate}/{profile} ({gibibytes(estimate)})"
                for candidate, profile, estimate in rejected
            )
            parser.error(
                f"estimated explicit peak exceeds {arguments.max_memory_fraction:.0%} "
                f"of {gibibytes(physical_memory)} physical memory: {details}"
            )

    output = arguments.output or (
        repository_root / "results/local" /
        f"issue16-streaming-pruned-screen-{timestamp}"
    )
    commands: list[
        tuple[str, str, ScreenCandidate, list[str], Path]
    ] = []
    for profile_index, profile in enumerate(PROFILES):
        ordered = candidates if profile_index % 2 == 0 else list(reversed(candidates))
        for candidate in ordered:
            stem = f"{profile}--{candidate.id}"
            result_path = output / f"{stem}.json"
            commands.append((
                stem, profile, candidate,
                command_for(
                    arguments.executable, candidate, profile,
                    arguments.warmups, arguments.samples,
                    arguments.seed, result_path,
                ),
                result_path,
            ))

    if arguments.dry_run:
        for _, _, _, command, _ in commands:
            print(f"VECLIB_MAXIMUM_THREADS=1 {' '.join(command)}")
        print(
            f"Planned {len(commands)} isolated preliminary run(s) across "
            f"{len(PROFILES)} four-field profiles."
        )
        return 0

    output.mkdir(parents=True, exist_ok=False)
    manifest = {
        "schema": "spectral-kernel-local-sweep-v1",
        "experimentId": EXPERIMENT_ID,
        "incrementId": INCREMENT_ID,
        "phase": "screen",
        "cohortId": COHORT_ID,
        "createdAtUtc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "question": (
            "Can outer-12 partial-column pruning stream directly into compact radial "
            "split storage, reduce full-spectrum memory, and improve the complete "
            "four-field pipeline relative to the issue #9 winner?"
        ),
        "baseline": (
            "The issue #9 plane-major-fused-split--outer-dynamic-16 winner, "
            "rerun on the same clean commit in every workload."
        ),
        "changedVariables": [
            "batch-sized full plane-major spectrum versus per-worker single-plane scratch",
            "full FFT plus fused split selection versus partial-column-pruned direct split streaming",
        ],
        "controlledVariables": [
            "one uniform algorithm across all sizes; no size-dependent dispatch",
            "Float64 radial horizontal two-thirds retention and Nj=floor(2*(Nz-1)/3)",
            "FFTW 3.3.11 MEASURE/unaligned/cold internal-1/outer-12",
            "K-squared outer-dynamic-16 vertical GEMM with VECLIB_MAXIMUM_THREADS=1",
            "real-diagonal-mode-keyed-v1 modal work and the independent 1e-12 oracle",
        ],
        "timedOperation": (
            "One allocation-free ready-real-input to reconstructed-real-output "
            "synthetic antialiased spectral round trip."
        ),
        "componentLedger": [
            "forward/inverse real-row and selected-column FFT work",
            "direct split write and inverse embed/zero fill",
            "complete retained horizontal forward/inverse operators",
            "raw forward/inverse vertical MM and mode-keyed modal work",
            "authoritative uninstrumented total",
            "algorithm-resident, benchmark-only, estimated-peak, and observed-high-water memory",
        ],
        "excludedWork": [
            "deeper within-column pruning and custom FFT butterflies",
            "FFTW++, implicit/hybrid dealiasing, and issue #17",
            "WVM nonlinear flux, time integration, state management, I/O, and full-model claims",
        ],
        "allocationPolicy": "Zero allocations in every timed steady-state graph.",
        "screenGate": (
            "Advance only if correctness and capability pass and either the geometric "
            "complete-pipeline ratio over the two large decision cases is at most 0.97, "
            "or algorithm-resident memory is at most 0.90 geometrically while the "
            "three-workload total-time ratio remains at most 1.02."
        ),
        "profiles": list(PROFILES),
        "largeDecisionProfiles": list(LARGE_PROFILES),
        "candidates": [asdict(candidate) for candidate in candidates],
        "candidateOrderAlternatesByProfile": True,
        "threadEnvironment": {"VECLIB_MAXIMUM_THREADS": "1"},
        "physicalMemoryBytes": physical_memory,
        "maximumMemoryFraction": arguments.max_memory_fraction,
        "estimatedExplicitPeakBytes": estimates,
        "sourceTreeGitCommit": source_commit,
        "sourceTreeDirty": source_dirty,
        "rounds": 1,
        "warmups": arguments.warmups,
        "samples": arguments.samples,
        "seed": arguments.seed,
        "runs": [],
    }

    failed = False
    completed_results: list[tuple[ScreenCandidate, dict]] = []
    for index, (stem, profile, candidate, command, result_path) in enumerate(
        commands, start=1,
    ):
        print(f"[{index}/{len(commands)}] {stem}", flush=True)
        log_path = output / f"{stem}.log"
        environment = os.environ.copy()
        environment["VECLIB_MAXIMUM_THREADS"] = "1"
        with log_path.open("w", encoding="utf-8") as log:
            completed = subprocess.run(
                command, cwd=repository_root, env=environment,
                stdout=log, stderr=subprocess.STDOUT,
            )
        entry = {
            "id": stem,
            "round": 1,
            "profile": profile,
            "candidate": asdict(candidate),
            "primaryProvider": candidate.primary_provider,
            "estimatedExplicitPeakBytes": estimates[candidate.id][profile],
            "environment": {"VECLIB_MAXIMUM_THREADS": "1"},
            "command": command,
            "exitCode": completed.returncode,
            "log": log_path.name,
            "sourceTreeGitCommit": source_commit,
            "sourceTreeDirty": source_dirty,
        }
        if result_path.is_file():
            with result_path.open(encoding="utf-8") as stream:
                result = json.load(stream)
            embedded_commit = result["environment"].get("gitCommit", "")
            embedded_dirty = result["environment"].get("gitDirty")
            metadata_matches = (
                bool(embedded_commit) and embedded_commit != "unknown"
                and source_commit.startswith(embedded_commit)
                and embedded_dirty == source_dirty
            )
            entry.update({
                "runId": result["run"]["id"],
                "status": result["status"],
                "result": result_path.name,
                "embeddedGitCommit": embedded_commit,
                "embeddedGitDirty": embedded_dirty,
                "sourceMetadataMatches": metadata_matches,
            })
            if result["status"] == "passed" and metadata_matches:
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
