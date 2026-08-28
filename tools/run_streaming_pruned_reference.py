#!/usr/bin/env python3
"""Run the fixed-policy issue #16 compact-tile reference campaign."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import statistics
import subprocess
import sys
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

from run_fftw_native_order_sweep import rotated
from run_spectral_pipeline_sweep import (
    geometric_mean,
    maximum_correctness_error,
    percentile,
    provider_timing,
    stratified_geometric_bootstrap,
)
from run_streaming_pruned_locality_sweep import (
    BASELINE_ID,
    EXPERIMENT_ID,
    PROFILES,
    LocalityCandidate,
    candidate_matrix,
    candidate_provider,
    command_for,
    estimated_explicit_peak_bytes,
    required_memory,
)
from run_vertical_gemm_sweep import gibibytes, git_source_state, sysctl_uint64


INCREMENT_ID = "streaming-pruned-compact-split-reference-v1"
COHORT_ID = "issue16-fixed-tile16-three-profile-f4-reference-v1"
TILED_ID = "streaming-pruned-tiled-16--outer-dynamic-16"
REFERENCE_ROUNDS = 3


def reference_candidates() -> list[LocalityCandidate]:
    by_id = {candidate.id: candidate for candidate in candidate_matrix()}
    return [by_id[BASELINE_ID], by_id[TILED_ID]]


def percentile_bootstrap(
    values: list[float], seed: int = 129, resamples: int = 20000,
) -> tuple[float, float]:
    if not values:
        raise ValueError("percentile bootstrap requires values")
    generator = random.Random(seed)
    draws = [
        statistics.median(generator.choices(values, k=len(values)))
        for _ in range(resamples)
    ]
    return percentile(draws, 0.025), percentile(draws, 0.975)


def stage_seconds(
    provider: dict, scope: str, stage: str, direction: str,
) -> float | None:
    timing = next(
        (
            item for item in provider.get("timings", [])
            if item.get("scope") == scope
            and item.get("stage") == stage
            and item.get("direction") == direction
        ),
        None,
    )
    return None if timing is None else float(timing["medianSeconds"])


def component_seconds(candidate: LocalityCandidate, provider: dict) -> dict[str, float]:
    horizontal_stage = (
        "full FFT fused compact split horizontal operator"
        if candidate.id == BASELINE_ID
        else "streaming pruned compact split horizontal operator"
    )
    requested = {
        "horizontalForward": (
            "retained-operator-total", horizontal_stage, "forward",
        ),
        "horizontalInverse": (
            "retained-operator-total", horizontal_stage, "inverse",
        ),
        "verticalForward": ("primitive", "raw vertical MM", "forward"),
        "modalWork": ("component", "mode-keyed modal work", "modal"),
        "verticalInverse": ("primitive", "raw vertical MM", "inverse"),
        "uninstrumentedTotal": (
            "uninstrumented-total",
            "synthetic antialiased spectral pipeline",
            "round-trip",
        ),
    }
    components: dict[str, float] = {}
    for key, timing_key in requested.items():
        value = stage_seconds(provider, *timing_key)
        if value is None:
            raise ValueError(
                f"provider {provider['id']} lacks required component {key}"
            )
        components[key] = value
    return components


def analyze(
    results: list[tuple[LocalityCandidate, int, dict]],
) -> dict:
    cells: dict[tuple[str, str], list[dict]] = {}
    maximum_error = 0.0
    all_correct = True
    for candidate, round_number, result in results:
        provider = candidate_provider(candidate, result)
        error = maximum_correctness_error(provider)
        maximum_error = max(maximum_error, error)
        all_correct = all_correct and math.isfinite(error) and error <= 1.0e-12
        cells.setdefault((candidate.id, result["run"]["profile"]), []).append({
            "round": round_number,
            "seconds": provider_timing(provider),
            "components": component_seconds(candidate, provider),
            "memory": required_memory(provider),
            "maximumCorrectnessError": error,
        })

    profiles: list[dict] = []
    profile_round_ratios: dict[str, list[float]] = {}
    complete = True
    for profile_index, profile in enumerate(PROFILES):
        baseline = {
            entry["round"]: entry
            for entry in cells.get((BASELINE_ID, profile), [])
        }
        candidate = {
            entry["round"]: entry
            for entry in cells.get((TILED_ID, profile), [])
        }
        rounds = sorted(set(baseline) & set(candidate))
        if (
            rounds != list(range(1, REFERENCE_ROUNDS + 1))
            or len(baseline) != REFERENCE_ROUNDS
            or len(candidate) != REFERENCE_ROUNDS
        ):
            complete = False
            continue
        baseline_times = [baseline[round_number]["seconds"] for round_number in rounds]
        candidate_times = [candidate[round_number]["seconds"] for round_number in rounds]
        round_ratios = [
            candidate[round_number]["seconds"] / baseline[round_number]["seconds"]
            for round_number in rounds
        ]
        profile_round_ratios[profile] = round_ratios
        lower, upper = percentile_bootstrap(
            round_ratios, seed=129 + profile_index,
        )

        component_rows: dict[str, dict[str, float]] = {}
        for key in baseline[rounds[0]]["components"]:
            baseline_value = statistics.median(
                baseline[round_number]["components"][key]
                for round_number in rounds
            )
            candidate_value = statistics.median(
                candidate[round_number]["components"][key]
                for round_number in rounds
            )
            component_rows[key] = {
                "baselineSeconds": baseline_value,
                "candidateSeconds": candidate_value,
                "candidateToBaseline": candidate_value / baseline_value,
            }

        memory_rows: dict[str, dict[str, float]] = {}
        for key in baseline[rounds[0]]["memory"]:
            baseline_value = statistics.median(
                baseline[round_number]["memory"][key]
                for round_number in rounds
            )
            candidate_value = statistics.median(
                candidate[round_number]["memory"][key]
                for round_number in rounds
            )
            memory_rows[key] = {
                "baselineBytes": baseline_value,
                "candidateBytes": candidate_value,
                "candidateToBaseline": candidate_value / baseline_value,
            }

        baseline_median = statistics.median(baseline_times)
        candidate_median = statistics.median(candidate_times)
        profiles.append({
            "profile": profile,
            "baselineSeconds": baseline_median,
            "candidateSeconds": candidate_median,
            "candidateToBaseline": candidate_median / baseline_median,
            "roundRatios": round_ratios,
            "pairedBootstrap95": {"lower": lower, "upper": upper},
            "components": component_rows,
            "memory": memory_rows,
        })

    complete = complete and len(profiles) == len(PROFILES)
    timing_ratios = [row["candidateToBaseline"] for row in profiles]
    resident_ratios = [
        row["memory"]["algorithmResidentBytes"]["candidateToBaseline"]
        for row in profiles
    ]
    observed_ratios = [
        row["memory"]["observedProcessHighWaterBytes"]["candidateToBaseline"]
        for row in profiles
    ]
    geometric_ratio = geometric_mean(timing_ratios) if timing_ratios else None
    maximum_ratio = max(timing_ratios) if timing_ratios else None
    geometric_resident = geometric_mean(resident_ratios) if resident_ratios else None
    geometric_observed = geometric_mean(observed_ratios) if observed_ratios else None
    confidence_interval = None
    if complete and profile_round_ratios:
        lower, upper = stratified_geometric_bootstrap(profile_round_ratios)
        confidence_interval = {"lower": lower, "upper": upper}

    improvement_passed = bool(
        complete and geometric_ratio is not None and geometric_ratio <= 0.90
    )
    regression_passed = bool(
        complete and maximum_ratio is not None and maximum_ratio <= 1.03
    )
    confidence_passed = bool(
        confidence_interval is not None and confidence_interval["upper"] < 1.0
    )
    memory_passed = bool(
        complete and geometric_resident is not None and geometric_resident <= 0.90
    )
    decision_passed = bool(
        improvement_passed and regression_passed and confidence_passed
        and memory_passed and all_correct
    )

    return {
        "schema": "spectral-kernel-streaming-pruned-reference-analysis-v1",
        "phase": "reference",
        "cohortId": COHORT_ID,
        "baselineCandidateId": BASELINE_ID,
        "optimizationCandidateId": TILED_ID,
        "completeProductionMatrix": complete,
        "allCorrectWithin1e12": all_correct,
        "maximumCorrectnessError": maximum_error,
        "geometricCandidateToBaseline": geometric_ratio,
        "maximumProfileCandidateToBaseline": maximum_ratio,
        "geometricAlgorithmResidentCandidateToBaseline": geometric_resident,
        "geometricObservedHighWaterCandidateToBaseline": geometric_observed,
        "stratifiedPairedBootstrap95": confidence_interval,
        "profiles": profiles,
        "referenceGate": {
            "geometricTimeRatioAtMost": 0.90,
            "maximumProfileTimeRatioAtMost": 1.03,
            "geometricAlgorithmResidentRatioAtMost": 0.90,
            "improvementPassed": improvement_passed,
            "regressionPassed": regression_passed,
            "confidenceExcludesTie": confidence_passed,
            "memoryReductionPassed": memory_passed,
            "correctnessPassed": all_correct,
            "zeroSteadyStateAllocationRequired": True,
            "allocationVerification": "skbench-unit macOS allocator interposer",
            "singleUniformPolicyPassed": decision_passed,
            "sizeDependentDispatchAllowed": False,
            "crossMacReplicationStillRequired": True,
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
    parser.add_argument("--rounds", type=int, default=REFERENCE_ROUNDS)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--samples", type=int, default=21)
    parser.add_argument("--seed", type=int, default=129)
    parser.add_argument("--max-memory-fraction", type=float, default=0.5)
    parser.add_argument("--allow-memory-risk", action="store_true")
    parser.add_argument("--allow-dirty-tree", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    arguments = parser.parse_args()
    if arguments.rounds != REFERENCE_ROUNDS:
        parser.error(f"the preregistered reference campaign requires {REFERENCE_ROUNDS} rounds")
    if arguments.warmups != 3 or arguments.samples != 21:
        parser.error("the preregistered reference campaign requires 3 warmups and 21 samples")
    if not 0.0 < arguments.max_memory_fraction <= 1.0:
        parser.error("--max-memory-fraction must be in (0, 1]")

    source_commit, source_dirty = git_source_state(repository_root)
    if source_dirty and not arguments.allow_dirty_tree:
        parser.error(
            "the benchmark source tree is dirty; commit and rebuild for reference "
            "evidence or use --allow-dirty-tree for an exploratory run"
        )

    candidates = reference_candidates()
    estimates = {
        candidate.id: {
            profile: estimated_explicit_peak_bytes(profile, candidate)
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
        f"issue16-streaming-pruned-reference-{timestamp}"
    )
    commands: list[
        tuple[str, int, str, LocalityCandidate, list[str], Path]
    ] = []
    profiles = list(PROFILES)
    for round_index in range(REFERENCE_ROUNDS):
        round_candidates = rotated(candidates, round_index)
        round_profiles = (
            profiles[round_index % len(profiles):]
            + profiles[:round_index % len(profiles)]
        )
        for profile in round_profiles:
            for candidate in round_candidates:
                stem = (
                    f"round-{round_index + 1}--{profile}--{candidate.id}"
                )
                result_path = output / f"{stem}.json"
                commands.append((
                    stem, round_index + 1, profile, candidate,
                    command_for(
                        arguments.executable, candidate, profile,
                        arguments.warmups, arguments.samples,
                        arguments.seed, result_path,
                    ),
                    result_path,
                ))

    if arguments.dry_run:
        for _, _, _, _, command, _ in commands:
            print(f"VECLIB_MAXIMUM_THREADS=1 {' '.join(command)}")
        print(
            f"Planned {len(commands)} isolated reference run(s): "
            f"{REFERENCE_ROUNDS} rotated round(s), {len(PROFILES)} profile(s), "
            f"{len(candidates)} fixed candidate(s)."
        )
        return 0

    output.mkdir(parents=True, exist_ok=False)
    manifest = {
        "schema": "spectral-kernel-local-sweep-v1",
        "experimentId": EXPERIMENT_ID,
        "incrementId": INCREMENT_ID,
        "phase": "reference",
        "cohortId": COHORT_ID,
        "createdAtUtc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "question": (
            "Does one fixed tile-16 streaming pruned-to-compact-split policy "
            "reliably beat fused split while reducing algorithm-resident memory?"
        ),
        "baseline": (
            "The same-commit issue #9 plane-major fused-split M4 winner, "
            "rerun in every profile and process round."
        ),
        "changedVariables": [
            "full 2-D FFT plus fused compact split versus partial-column FFT with fixed tile-16 compact staging",
        ],
        "controlledVariables": [
            "one uniform tile width of 16 for every workload; no selector or size dispatch",
            "Float64 radial horizontal two-thirds retention and vertically truncated Nj",
            "FFTW MEASURE/unaligned/cold internal-1/outer-12",
            "K-squared vertical matrices, outer-dynamic-16, VECLIB_MAXIMUM_THREADS=1, modal work, and oracle",
        ],
        "timedOperation": (
            "One allocation-free ready-real-input to reconstructed-real-output "
            "synthetic antialiased spectral round trip."
        ),
        "componentLedger": [
            "retained horizontal forward and inverse totals",
            "row FFTs, selected-column FFTs, compact staging, transpose, embed, and zero fill where separable",
            "raw forward and inverse vertical MM, modal work, and authoritative uninstrumented total",
            "scratch, algorithm-resident, benchmark-only, explicit-peak, and observed-high-water memory",
        ],
        "excludedWork": [
            "tile selection, size-dependent dispatch, and additional locality tuning",
            "WVM nonlinear flux, time integration, I/O, and issue #17",
        ],
        "allocationPolicy": (
            "Zero allocations in every timed steady-state graph; verified by "
            "the macOS allocator-interposer unit test on the same source commit."
        ),
        "referenceGate": (
            "Require a complete matched three-round matrix, correctness within 1e-12, "
            "at least 10% geometric time and algorithm-resident-memory improvement, "
            "no profile regression above 3%, and a stratified paired-bootstrap "
            "95% interval excluding a tie."
        ),
        "profiles": list(PROFILES),
        "candidates": [asdict(candidate) for candidate in candidates],
        "candidateOrderRotatedByRound": True,
        "profileOrderRotatedByRound": True,
        "threadEnvironment": {"VECLIB_MAXIMUM_THREADS": "1"},
        "physicalMemoryBytes": physical_memory,
        "maximumMemoryFraction": arguments.max_memory_fraction,
        "estimatedExplicitPeakBytes": estimates,
        "sourceTreeGitCommit": source_commit,
        "sourceTreeDirty": source_dirty,
        "rounds": REFERENCE_ROUNDS,
        "warmups": arguments.warmups,
        "samples": arguments.samples,
        "seed": arguments.seed,
        "runs": [],
    }

    failed = False
    completed_results: list[tuple[LocalityCandidate, int, dict]] = []
    for index, (
        stem, round_number, profile, candidate, command, result_path,
    ) in enumerate(commands, start=1):
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
                completed_results.append((candidate, round_number, result))
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
