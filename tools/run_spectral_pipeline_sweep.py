#!/usr/bin/env python3
"""Run the bounded issue #9 synthetic antialiased spectral pipeline campaign."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import statistics
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

from run_fftw_native_order_sweep import REFERENCE_PROFILES, rotated
from run_spectral_boundary_sweep import estimated_explicit_peak_bytes as boundary_peak
from run_vertical_gemm_sweep import (
    PROFILE_SHAPES,
    gibibytes,
    git_source_state,
    sysctl_uint64,
)


BASELINE_ID = "wvm-direct--outer-dynamic-16"
CANDIDATE_ID = "plane-major-fused-split--outer-dynamic-16"


@dataclass(frozen=True)
class Candidate:
    id: str
    policy: str
    primary_provider: str
    role: str


def candidate_matrix() -> list[Candidate]:
    return [
        Candidate(
            BASELINE_ID,
            "wvm-direct",
            "pipeline-wvm-direct",
            "production-layout-control",
        ),
        Candidate(
            CANDIDATE_ID,
            "plane-major-fused-split",
            "pipeline-plane-major-fused-split",
            "selected-issue13-optimization-candidate",
        ),
    ]


def load_screen_analysis(path: Path | None) -> dict | None:
    if path is None:
        return None
    with path.open(encoding="utf-8") as stream:
        analysis = json.load(stream)
    if analysis.get("schema") != "spectral-kernel-pipeline-analysis-v1":
        raise ValueError("--screen-analysis has the wrong schema")
    if analysis.get("phase") != "screen":
        raise ValueError("--screen-analysis must describe the screen phase")
    return analysis


def select_candidates(phase: str, screen_analysis: dict | None) -> list[Candidate]:
    if phase == "reference":
        if screen_analysis is None:
            raise ValueError("reference phase requires --screen-analysis")
        if not screen_analysis.get("advanceToReference", False):
            raise ValueError(
                "the selected optimization candidate did not satisfy the preregistered screen gate"
            )
    return candidate_matrix()


def provider_record(candidate: Candidate, result: dict) -> dict:
    return next(
        provider for provider in result["providers"]
        if provider["id"] == candidate.primary_provider
    )


def provider_timing(provider: dict) -> float:
    matches = [
        timing for timing in provider["timings"]
        if timing["scope"] == "uninstrumented-total"
        and timing["stage"] == "synthetic antialiased spectral pipeline"
        and timing["direction"] == "round-trip"
    ]
    if len(matches) != 1:
        raise ValueError(
            f"provider {provider['id']} lacks one synthetic-pipeline round-trip timing"
        )
    return float(matches[0]["medianSeconds"])


def maximum_correctness_error(provider: dict) -> float:
    metrics = provider.get("correctness", [])
    if not metrics or not all(metric.get("passed", False) for metric in metrics):
        return math.inf
    return max(float(metric["maximumRelativeError"]) for metric in metrics)


def geometric_mean(values: list[float]) -> float:
    if not values or any(value <= 0.0 for value in values):
        raise ValueError("geometric mean requires positive values")
    return math.exp(sum(math.log(value) for value in values) / len(values))


def percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def stratified_geometric_bootstrap(
    profile_ratios: dict[str, list[float]],
    seed: int = 129,
    resamples: int = 20000,
) -> tuple[float, float]:
    generator = random.Random(seed)
    profiles = sorted(profile_ratios)
    draws: list[float] = []
    for _ in range(resamples):
        sampled = [
            generator.choice(profile_ratios[profile])
            for profile in profiles
        ]
        draws.append(geometric_mean(sampled))
    return percentile(draws, 0.025), percentile(draws, 0.975)


def analyze(
    results: list[tuple[Candidate, int, dict]],
    phase: str = "screen",
) -> dict:
    cells: dict[tuple[str, str], list[tuple[int, float]]] = {}
    maximum_error = 0.0
    all_correct = True
    for candidate, round_number, result in results:
        provider = provider_record(candidate, result)
        profile = result["run"]["profile"]
        cells.setdefault((candidate.id, profile), []).append(
            (round_number, provider_timing(provider))
        )
        error = maximum_correctness_error(provider)
        maximum_error = max(maximum_error, error)
        all_correct = all_correct and math.isfinite(error) and error <= 1.0e-12

    profiles = sorted({profile for _, profile in cells})
    profile_rows: list[dict] = []
    profile_round_ratios: dict[str, list[float]] = {}
    complete = True
    for profile in profiles:
        baseline = dict(cells.get((BASELINE_ID, profile), []))
        candidate = dict(cells.get((CANDIDATE_ID, profile), []))
        shared_rounds = sorted(set(baseline) & set(candidate))
        if not baseline or not candidate or len(shared_rounds) != len(baseline) or len(shared_rounds) != len(candidate):
            complete = False
            continue
        round_ratios = [candidate[round_number] / baseline[round_number]
                        for round_number in shared_rounds]
        profile_round_ratios[profile] = round_ratios
        profile_rows.append({
            "profile": profile,
            "baselineSeconds": statistics.median(baseline.values()),
            "candidateSeconds": statistics.median(candidate.values()),
            "candidateToBaseline": (
                statistics.median(candidate.values()) /
                statistics.median(baseline.values())
            ),
            "roundRatios": round_ratios,
        })

    complete = complete and len(profiles) == len(REFERENCE_PROFILES)
    ratios = [row["candidateToBaseline"] for row in profile_rows]
    geometric_ratio = geometric_mean(ratios) if ratios else None
    maximum_ratio = max(ratios) if ratios else None
    screen_improvement = bool(
        complete and geometric_ratio is not None and geometric_ratio <= 0.95
    )
    screen_regression = bool(
        complete and maximum_ratio is not None and maximum_ratio <= 1.10
    )
    advance = bool(screen_improvement and screen_regression and all_correct)

    confidence_interval = None
    improvement_gate = None
    regression_gate = None
    confidence_gate = None
    adoption_gate = None
    if phase == "reference" and complete and profile_round_ratios:
        lower, upper = stratified_geometric_bootstrap(profile_round_ratios)
        confidence_interval = {"lower": lower, "upper": upper}
        improvement_gate = geometric_ratio <= 0.90
        regression_gate = maximum_ratio <= 1.03
        confidence_gate = upper < 1.0
        adoption_gate = bool(
            improvement_gate and regression_gate and confidence_gate and all_correct
        )

    return {
        "schema": "spectral-kernel-pipeline-analysis-v1",
        "phase": phase,
        "baselineCandidateId": BASELINE_ID,
        "optimizationCandidateId": CANDIDATE_ID,
        "completeProductionMatrix": complete,
        "allCorrectWithin1e-12": all_correct,
        "maximumCorrectnessError": maximum_error,
        "geometricCandidateToBaseline": geometric_ratio,
        "maximumProfileCandidateToBaseline": maximum_ratio,
        "profiles": profile_rows,
        "screenGate": {
            "geometricRatioAtMost": 0.95,
            "maximumProfileRatioAtMost": 1.10,
            "improvementPassed": screen_improvement,
            "regressionPassed": screen_regression,
            "correctnessPassed": all_correct,
        },
        "advanceToReference": advance,
        "referenceGate": {
            "geometricRatioAtMost": 0.90,
            "maximumProfileRatioAtMost": 1.03,
            "confidenceInterval": confidence_interval,
            "improvementPassed": improvement_gate,
            "regressionPassed": regression_gate,
            "confidenceExcludesTie": confidence_gate,
            "m4AdoptionStatisticsPassed": adoption_gate,
            "crossMacReplicationStillRequired": True,
        },
    }


def estimated_explicit_peak_bytes(profile: str, policy: str) -> int:
    nx, nz, fields, nkl, _ = PROFILE_SHAPES[profile]
    nj = 2 * (nz - 1) // 3
    physical = nz * nkl * fields * 16
    modal = nj * nkl * fields * 16
    weights = nj * nkl * fields * 8
    correctness_buffers = 2 * physical + 3 * modal + weights
    return boundary_peak(profile, policy) + correctness_buffers + nx * nx * nz * fields * 8


def command_for(
    executable: Path,
    candidate: Candidate,
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


def main() -> int:
    repository_root = Path(__file__).resolve().parents[1]
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("screen", "reference"), default="screen")
    parser.add_argument(
        "--executable", type=Path,
        default=repository_root / "build/release/skbench",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--profiles", nargs="*")
    parser.add_argument("--screen-analysis", type=Path)
    parser.add_argument("--rounds", type=int)
    parser.add_argument("--warmups", type=int)
    parser.add_argument("--samples", type=int)
    parser.add_argument("--seed", type=int, default=129)
    parser.add_argument("--max-memory-fraction", type=float, default=0.5)
    parser.add_argument("--allow-memory-risk", action="store_true")
    parser.add_argument("--allow-dirty-tree", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    arguments = parser.parse_args()

    profiles = arguments.profiles or list(REFERENCE_PROFILES)
    unknown_profiles = sorted(set(profiles) - set(REFERENCE_PROFILES))
    if unknown_profiles:
        parser.error(f"unknown profile: {', '.join(unknown_profiles)}")
    try:
        screen_analysis = load_screen_analysis(arguments.screen_analysis)
        candidates = select_candidates(arguments.phase, screen_analysis)
    except (OSError, ValueError) as error:
        parser.error(str(error))

    if arguments.phase == "screen":
        rounds = arguments.rounds or 1
        warmups = arguments.warmups or 2
        samples = arguments.samples or 9
        increment_id = "synthetic-spectral-pipeline-screen-v1"
    else:
        rounds = arguments.rounds or 3
        warmups = arguments.warmups or 3
        samples = arguments.samples or 21
        increment_id = "synthetic-spectral-pipeline-reference-v1"
    if min(rounds, warmups, samples) < 1:
        parser.error("--rounds, --warmups, and --samples must be positive")
    if not 0.0 < arguments.max_memory_fraction <= 1.0:
        parser.error("--max-memory-fraction must be in (0, 1]")

    source_commit, source_dirty = git_source_state(repository_root)
    if source_dirty and not arguments.allow_dirty_tree:
        parser.error(
            "the benchmark source tree is dirty; commit and rebuild for evidence collection "
            "or use --allow-dirty-tree for an explicitly exploratory run"
        )

    physical_memory = sysctl_uint64("hw.memsize", 0)
    estimated_peaks = {
        candidate.id: {
            profile: estimated_explicit_peak_bytes(profile, candidate.policy)
            for profile in profiles
        }
        for candidate in candidates
    }
    if physical_memory > 0 and not arguments.allow_memory_risk:
        rejected = [
            (candidate_id, profile, estimate)
            for candidate_id, profile_estimates in estimated_peaks.items()
            for profile, estimate in profile_estimates.items()
            if estimate > arguments.max_memory_fraction * physical_memory
        ]
        if rejected:
            details = ", ".join(
                f"{candidate_id}/{profile} ({gibibytes(estimate)})"
                for candidate_id, profile, estimate in rejected
            )
            parser.error(
                f"estimated explicit peak exceeds {arguments.max_memory_fraction:.0%} of "
                f"{gibibytes(physical_memory)} physical memory: {details}; "
                "use --allow-memory-risk to override"
            )

    output = arguments.output or (
        repository_root / "results/local" /
        f"issue9-spectral-pipeline-{arguments.phase}-{timestamp}"
    )
    commands: list[tuple[str, int, str, Candidate, list[str], Path]] = []
    for round_index in range(rounds):
        round_candidates = rotated(candidates, round_index)
        round_profiles = (
            profiles[round_index % len(profiles):]
            + profiles[:round_index % len(profiles)]
        )
        for profile in round_profiles:
            for candidate in round_candidates:
                stem = f"round-{round_index + 1}--{profile}--{candidate.id}"
                result_path = output / f"{stem}.json"
                commands.append((
                    stem, round_index + 1, profile, candidate,
                    command_for(
                        arguments.executable, candidate, profile, warmups,
                        samples, arguments.seed, result_path,
                    ),
                    result_path,
                ))

    if arguments.dry_run:
        for _, _, _, _, command, _ in commands:
            print(f"VECLIB_MAXIMUM_THREADS=1 {' '.join(command)}")
        print(
            f"Planned {len(commands)} isolated run(s): {rounds} round(s), "
            f"{len(profiles)} profile(s), {len(candidates)} candidate(s)."
        )
        return 0

    output.mkdir(parents=True, exist_ok=False)
    manifest = {
        "schema": "spectral-kernel-local-sweep-v1",
        "experimentId": "issue-009-combined-spectral-pipeline",
        "incrementId": increment_id,
        "phase": arguments.phase,
        "createdAtUtc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "question": (
            "Does the issue #13 plane-major fused-split representation remain faster "
            "than the WVM direct/no-reorder control through one complete synthetic "
            "antialiased spectral round trip with deterministic modal work?"
        ),
        "baseline": (
            "WVM frequency-major interleaved FFTW, direct per-frequency complex vertical "
            "projection, provider-order modal view, explicit inverse-spectrum rebuild, "
            "and the same mode-keyed real diagonal modal operator."
        ),
        "changedVariables": [
            "WVM frequency-major interleaved versus plane-major compact split representation",
            "direct per-frequency zgemm versus fused retention and grouped split dgemm",
            "elided provider-order access versus fused selection, split conversion, and embedding",
        ],
        "controlledVariables": [
            "Float64 radial horizontal two-thirds retention and Nj=floor(2*(Nz-1)/3)",
            "deterministic real-diagonal-mode-keyed-v1 modal work",
            "FFTW 3.3.11 MEASURE, unaligned, cold, internal-1/outer-12",
            "K-squared matrices, outer-dynamic-16, VECLIB_MAXIMUM_THREADS=1, fixtures, seed, and workloads",
        ],
        "timedOperation": (
            "One uninstrumented ready-real-input to reconstructed-real-output invocation: "
            "horizontal forward and retention, vertical forward, modal work, vertical inverse, "
            "horizontal embedding, and inverse FFT."
        ),
        "componentLedger": [
            "raw forward and inverse FFT",
            "retention, representation conversion, packing, zero fill, and embedding",
            "raw forward and inverse vertical MM",
            "mode-keyed modal work",
            "setup, allocation, memory, placement, and liveness",
        ],
        "excludedWork": [
            "WVM nonlinear flux, time integration, state management, I/O, and full-model claims",
            "fixture generation, correctness oracle, planning, and allocation from steady-state totals",
        ],
        "allocationPolicy": "Zero allocations in every timed steady-state graph.",
        "screenGate": (
            "Advance both graphs to reference depth only if the fused-split candidate is at "
            "least 5% faster geometrically, no workload is more than 10% slower, and all "
            "correctness metrics pass within 1e-12."
        ),
        "referenceGate": (
            "M4 adoption statistics pass only if the candidate is at least 10% faster "
            "geometrically, no workload regresses by more than 3%, the stratified 95% "
            "bootstrap interval excludes a tie, and correctness remains within 1e-12. "
            "Cross-Mac replication remains separately required."
        ),
        "referenceProtocol": {
            "independentlyPlannedProcesses": rounds,
            "candidateOrderRotatedByRound": True,
            "profileOrderRotatedByRound": True,
            "warmupsPerProcess": warmups,
            "samplesPerProcess": samples,
        },
        "profiles": profiles,
        "candidates": [asdict(candidate) for candidate in candidates],
        "threadEnvironment": {"VECLIB_MAXIMUM_THREADS": "1"},
        "physicalMemoryBytes": physical_memory,
        "estimatedExplicitPeakBytes": estimated_peaks,
        "sourceTreeGitCommit": source_commit,
        "sourceTreeDirty": source_dirty,
        "rounds": rounds,
        "warmups": warmups,
        "samples": samples,
        "seed": arguments.seed,
        "runs": [],
    }

    failed = False
    completed_results: list[tuple[Candidate, int, dict]] = []
    for index, (stem, round_number, profile, candidate, command, result_path) in enumerate(
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
            "round": round_number,
            "profile": profile,
            "candidate": asdict(candidate),
            "primaryProvider": candidate.primary_provider,
            "estimatedExplicitPeakBytes": estimated_peaks[candidate.id][profile],
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
                bool(embedded_commit)
                and embedded_commit != "unknown"
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
                completed_results.append((candidate, round_number, result))
            else:
                completed = subprocess.CompletedProcess(command, 1)
                print(
                    f"invalid evidence for {stem}: status={result['status']}, "
                    f"source={source_commit} dirty={source_dirty}, "
                    f"binary={embedded_commit} dirty={embedded_dirty}",
                    file=sys.stderr,
                )
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
            json.dump(analyze(completed_results, arguments.phase), stream, indent=2)
            stream.write("\n")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
