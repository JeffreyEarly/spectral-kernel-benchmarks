#!/usr/bin/env python3
"""Run the memory-aware issue #9 large four-field pipeline campaign."""

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

from run_fftw_native_order_sweep import rotated
from run_spectral_pipeline_sweep import (
    BASELINE_ID,
    CANDIDATE_ID,
    Candidate,
    candidate_matrix,
    command_for,
    estimated_explicit_peak_bytes,
    geometric_mean,
    maximum_correctness_error,
    provider_record,
    provider_timing,
    stratified_geometric_bootstrap,
)
from run_vertical_gemm_sweep import gibibytes, git_source_state, sysctl_uint64


COHORT_ID = "large-f4-v1"
PROFILES = (
    "wvm-current-256-nz129-f4",
    "wvm-historical-512-nz129-f4",
    "wvm-current-512-nz257-f4",
    "wvm-large-1024-nz129-f4",
)


def load_screen_analysis(path: Path | None) -> dict | None:
    if path is None:
        return None
    with path.open(encoding="utf-8") as stream:
        analysis = json.load(stream)
    if analysis.get("schema") != "spectral-kernel-pipeline-analysis-v1":
        raise ValueError("--screen-analysis has the wrong schema")
    if analysis.get("phase") != "screen" or analysis.get("cohortId") != COHORT_ID:
        raise ValueError("--screen-analysis must describe the large-f4-v1 screen")
    return analysis


def select_candidates(phase: str, screen_analysis: dict | None) -> list[Candidate]:
    if phase == "reference":
        if screen_analysis is None:
            raise ValueError("reference phase requires --screen-analysis")
        if not screen_analysis.get("advanceToReference", False):
            raise ValueError(
                "the four-field screen did not satisfy the correctness/capability gate"
            )
    return candidate_matrix()


def required_memory(provider: dict) -> dict[str, int]:
    memory = provider.get("memory", {})
    keys = (
        "algorithmResidentBytes",
        "benchmarkHarnessBytes",
        "estimatedProcessPeakBytes",
        "observedProcessHighWaterBytes",
    )
    missing = [key for key in keys if int(memory.get(key, 0)) <= 0]
    if missing:
        raise ValueError(
            f"provider {provider['id']} lacks positive memory fields: {', '.join(missing)}"
        )
    return {key: int(memory[key]) for key in keys}


def analyze(
    results: list[tuple[Candidate, int, dict]],
    phase: str = "screen",
) -> dict:
    cells: dict[tuple[str, str], list[tuple[int, float, dict[str, int]]]] = {}
    maximum_error = 0.0
    all_correct = True
    for candidate, round_number, result in results:
        provider = provider_record(candidate, result)
        profile = result["run"]["profile"]
        cells.setdefault((candidate.id, profile), []).append(
            (round_number, provider_timing(provider), required_memory(provider))
        )
        error = maximum_correctness_error(provider)
        maximum_error = max(maximum_error, error)
        all_correct = all_correct and math.isfinite(error) and error <= 1.0e-12

    profiles = sorted({profile for _, profile in cells})
    profile_rows: list[dict] = []
    profile_round_ratios: dict[str, list[float]] = {}
    complete = set(profiles) == set(PROFILES)
    for profile in profiles:
        baseline_entries = {entry[0]: entry[1:] for entry in cells.get((BASELINE_ID, profile), [])}
        candidate_entries = {entry[0]: entry[1:] for entry in cells.get((CANDIDATE_ID, profile), [])}
        shared_rounds = sorted(set(baseline_entries) & set(candidate_entries))
        if (
            not baseline_entries
            or not candidate_entries
            or len(shared_rounds) != len(baseline_entries)
            or len(shared_rounds) != len(candidate_entries)
        ):
            complete = False
            continue
        baseline_times = [baseline_entries[round_number][0] for round_number in shared_rounds]
        candidate_times = [candidate_entries[round_number][0] for round_number in shared_rounds]
        round_ratios = [
            candidate_entries[round_number][0] / baseline_entries[round_number][0]
            for round_number in shared_rounds
        ]
        profile_round_ratios[profile] = round_ratios
        memory_row: dict[str, dict[str, float]] = {}
        for key in (
            "algorithmResidentBytes",
            "benchmarkHarnessBytes",
            "estimatedProcessPeakBytes",
            "observedProcessHighWaterBytes",
        ):
            baseline_value = statistics.median(
                baseline_entries[round_number][1][key] for round_number in shared_rounds
            )
            candidate_value = statistics.median(
                candidate_entries[round_number][1][key] for round_number in shared_rounds
            )
            memory_row[key] = {
                "baselineBytes": baseline_value,
                "candidateBytes": candidate_value,
                "candidateToBaseline": candidate_value / baseline_value,
            }
        profile_rows.append({
            "profile": profile,
            "baselineSeconds": statistics.median(baseline_times),
            "candidateSeconds": statistics.median(candidate_times),
            "candidateToBaseline": (
                statistics.median(candidate_times) / statistics.median(baseline_times)
            ),
            "roundRatios": round_ratios,
            "memory": memory_row,
        })

    timing_ratios = [row["candidateToBaseline"] for row in profile_rows]
    geometric_ratio = geometric_mean(timing_ratios) if timing_ratios else None
    maximum_ratio = max(timing_ratios) if timing_ratios else None
    resident_ratios = [
        row["memory"]["algorithmResidentBytes"]["candidateToBaseline"]
        for row in profile_rows
    ]
    observed_ratios = [
        row["memory"]["observedProcessHighWaterBytes"]["candidateToBaseline"]
        for row in profile_rows
    ]
    geometric_resident_ratio = geometric_mean(resident_ratios) if resident_ratios else None
    geometric_observed_ratio = geometric_mean(observed_ratios) if observed_ratios else None

    screen_correctness = bool(complete and all_correct)
    advance = screen_correctness
    confidence_interval = None
    improvement_gate = None
    regression_gate = None
    confidence_gate = None
    adoption_gate = None
    classification = None
    if phase == "reference" and complete and profile_round_ratios:
        lower, upper = stratified_geometric_bootstrap(profile_round_ratios)
        confidence_interval = {"lower": lower, "upper": upper}
        improvement_gate = geometric_ratio <= 0.90
        regression_gate = maximum_ratio <= 1.03
        confidence_gate = upper < 1.0
        adoption_gate = bool(
            improvement_gate and regression_gate and confidence_gate and all_correct
        )
        timing_tie = bool(
            (0.95 <= geometric_ratio <= 1.05) or (lower <= 1.0 <= upper)
        )
        memory_advantage = bool(
            geometric_resident_ratio is not None and geometric_resident_ratio <= 0.95
        )
        if adoption_gate:
            classification = "fused-split-performance-win"
        elif timing_tie and memory_advantage:
            classification = "tie-with-memory-advantage"
        else:
            classification = "size-specific-dispatch"

    return {
        "schema": "spectral-kernel-pipeline-analysis-v1",
        "phase": phase,
        "cohortId": COHORT_ID,
        "baselineCandidateId": BASELINE_ID,
        "optimizationCandidateId": CANDIDATE_ID,
        "completeProductionMatrix": complete,
        "allCorrectWithin1e-12": all_correct,
        "maximumCorrectnessError": maximum_error,
        "geometricCandidateToBaseline": geometric_ratio,
        "maximumProfileCandidateToBaseline": maximum_ratio,
        "geometricAlgorithmResidentCandidateToBaseline": geometric_resident_ratio,
        "geometricObservedHighWaterCandidateToBaseline": geometric_observed_ratio,
        "profiles": profile_rows,
        "screenGate": {
            "policy": "correctness-and-capability-only",
            "completeFourFieldCohort": complete,
            "correctnessPassed": screen_correctness,
            "performanceDoesNotGateReferenceCollection": True,
        },
        "advanceToReference": advance,
        "referenceGate": {
            "geometricRatioAtMost": 0.90,
            "maximumProfileRatioAtMost": 1.03,
            "confidenceInterval": confidence_interval,
            "improvementPassed": improvement_gate,
            "regressionPassed": regression_gate,
            "confidenceExcludesTie": confidence_gate,
            "m4NonhydrostaticAdoptionStatisticsPassed": adoption_gate,
            "classification": classification,
            "crossMacReplicationStillRequired": True,
        },
    }


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

    try:
        screen_analysis = load_screen_analysis(arguments.screen_analysis)
        candidates = select_candidates(arguments.phase, screen_analysis)
    except (OSError, ValueError) as error:
        parser.error(str(error))

    if arguments.phase == "screen":
        rounds = arguments.rounds or 1
        warmups = arguments.warmups or 1
        samples = arguments.samples or 5
        increment_id = "synthetic-spectral-pipeline-large-f4-screen-v1"
    else:
        rounds = arguments.rounds or 3
        warmups = arguments.warmups or 3
        samples = arguments.samples or 21
        increment_id = "synthetic-spectral-pipeline-large-f4-reference-v1"
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
            for profile in PROFILES
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
                f"{gibibytes(physical_memory)} physical memory: {details}"
            )

    output = arguments.output or (
        repository_root / "results/local" /
        f"issue9-spectral-pipeline-large-f4-{arguments.phase}-{timestamp}"
    )
    commands: list[tuple[str, int, str, Candidate, list[str], Path]] = []
    profiles = list(PROFILES)
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
            f"{len(profiles)} four-field profile(s), {len(candidates)} candidate(s)."
        )
        return 0

    output.mkdir(parents=True, exist_ok=False)
    manifest = {
        "schema": "spectral-kernel-local-sweep-v1",
        "experimentId": "issue-009-combined-spectral-pipeline",
        "incrementId": increment_id,
        "phase": arguments.phase,
        "cohortId": COHORT_ID,
        "createdAtUtc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "question": (
            "Does the fused compact-split graph remain faster or provide a material "
            "memory advantage for nonhydrostatic four-field workloads from 256^2 to 1024^2?"
        ),
        "baseline": (
            "The same issue #9 WVM direct/no-reorder production-layout graph, rerun in "
            "every round and workload on the same clean commit."
        ),
        "changedVariables": [
            "WVM full frequency-major interleaved versus compact plane-major split representation",
            "horizontal and vertical resolution within an all-fields=4 cohort",
        ],
        "controlledVariables": [
            "Float64 radial horizontal two-thirds retention and Nj=floor(2*(Nz-1)/3)",
            "real-diagonal-mode-keyed-v1 modal work",
            "FFTW MEASURE/unaligned/cold internal-1/outer-12",
            "K-squared vertical matrices, outer-dynamic-16, and VECLIB_MAXIMUM_THREADS=1",
        ],
        "timedOperation": (
            "One uninstrumented ready-real-input to reconstructed-real-output synthetic "
            "antialiased spectral round trip."
        ),
        "componentLedger": [
            "raw forward/inverse FFT",
            "retention, conversion, packing, zero fill, and embedding",
            "raw forward/inverse vertical MM",
            "mode-keyed modal work",
            "algorithm-resident, benchmark-only, estimated-peak, and observed-high-water memory",
        ],
        "excludedWork": [
            "WVM nonlinear flux, time integration, state management, I/O, and full-model claims",
            "512^2 x Nz=513 x fields=4 and 1024^2 x Nz=257 x fields=4 capacity cases",
        ],
        "allocationPolicy": "Zero allocations in every timed steady-state graph.",
        "screenGate": (
            "Advance all four profiles to reference depth when both graphs complete, all "
            "correctness metrics pass within 1e-12, and preflight respects the 50% memory rule; "
            "screen performance does not gate reference collection."
        ),
        "referenceGate": (
            "Classify the four-field cohort as a fused-split performance win, tie with memory "
            "advantage, or size-specific dispatch using the existing 10% geometric, 3% "
            "maximum-regression, confidence, correctness, and memory evidence."
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
        "maximumMemoryFraction": arguments.max_memory_fraction,
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
