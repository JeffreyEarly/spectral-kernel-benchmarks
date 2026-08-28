#!/usr/bin/env python3
"""Run the final three-way M4 pipeline and WVM-native control bridge."""

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
    BASELINE_ID as WVM_ID,
    CANDIDATE_ID as FUSED_ID,
    estimated_explicit_peak_bytes as issue9_estimated_peak,
    geometric_mean,
    maximum_correctness_error,
    provider_timing,
    stratified_geometric_bootstrap,
)
from run_streaming_pruned_locality_sweep import (
    LocalityCandidate,
    candidate_provider,
    command_for,
    estimated_explicit_peak_bytes as streaming_estimated_peak,
    required_memory,
)
from run_streaming_pruned_reference import (
    PROFILES,
    REFERENCE_ROUNDS,
    TILED_ID,
    percentile_bootstrap,
)
from run_vertical_gemm_sweep import gibibytes, git_source_state, sysctl_uint64


EXPERIMENT_ID = "issue-009-combined-spectral-pipeline"
INCREMENT_ID = "synthetic-spectral-pipeline-three-way-native-control-reference-v1"
COHORT_ID = "m4-three-way-native-control-three-profile-f4-v1"


def candidate_matrix() -> list[LocalityCandidate]:
    return [
        LocalityCandidate(
            WVM_ID,
            "wvm-direct",
            "pipeline-wvm-direct",
            "matlab-native-frequency-major-interleaved-control",
            0,
        ),
        LocalityCandidate(
            FUSED_ID,
            "plane-major-fused-split",
            "pipeline-plane-major-fused-split",
            "previous-persistent-engine-winner",
            0,
        ),
        LocalityCandidate(
            TILED_ID,
            "streaming-pruned-compact-split",
            "pipeline-streaming-pruned-compact-split",
            "fixed-tile16-persistent-engine-candidate",
            16,
        ),
    ]


def estimated_explicit_peak_bytes(
    profile: str, candidate: LocalityCandidate,
) -> int:
    if candidate.id == TILED_ID:
        return streaming_estimated_peak(profile, candidate)
    return issue9_estimated_peak(profile, candidate.policy)


def setup_record(provider: dict) -> dict[str, float]:
    setup = provider.get("setup", {})
    planning = provider.get("planning", {})
    return {
        "totalSeconds": float(setup.get("totalSeconds", 0.0)),
        "allocationSeconds": float(setup.get("allocationSeconds", 0.0)),
        "planningSeconds": float(planning.get("seconds", 0.0)),
        "planningTemporaryBytes": float(planning.get("temporaryBytes", 0.0)),
    }


def pair_key(numerator: str, denominator: str) -> str:
    return f"{numerator}--to--{denominator}"


def analyze(
    results: list[tuple[LocalityCandidate, int, dict]],
) -> dict:
    candidates = candidate_matrix()
    candidate_ids = [candidate.id for candidate in candidates]
    cells: dict[tuple[str, str], dict[int, dict]] = {}
    maximum_error = 0.0
    all_correct = True
    for candidate, round_number, result in results:
        provider = candidate_provider(candidate, result)
        error = maximum_correctness_error(provider)
        maximum_error = max(maximum_error, error)
        all_correct = all_correct and math.isfinite(error) and error <= 1.0e-12
        cells.setdefault(
            (candidate.id, result["run"]["profile"]), {},
        )[round_number] = {
            "seconds": provider_timing(provider),
            "memory": required_memory(provider),
            "setup": setup_record(provider),
            "executionContract": provider.get("executionContract", {}),
            "maximumCorrectnessError": error,
        }

    complete = True
    profile_rows: list[dict] = []
    global_pair_ratios: dict[str, dict[str, list[float]]] = {
        pair_key(FUSED_ID, WVM_ID): {},
        pair_key(TILED_ID, WVM_ID): {},
        pair_key(TILED_ID, FUSED_ID): {},
    }
    for profile_index, profile in enumerate(PROFILES):
        by_candidate = {
            candidate_id: cells.get((candidate_id, profile), {})
            for candidate_id in candidate_ids
        }
        required_rounds = list(range(1, REFERENCE_ROUNDS + 1))
        if any(
            sorted(entries) != required_rounds
            for entries in by_candidate.values()
        ):
            complete = False
            continue

        candidate_rows: list[dict] = []
        for candidate_id in candidate_ids:
            entries = by_candidate[candidate_id]
            rounds = [entries[round_number] for round_number in required_rounds]
            memory: dict[str, float] = {}
            for key in rounds[0]["memory"]:
                memory[key] = statistics.median(
                    entry["memory"][key] for entry in rounds
                )
            setup: dict[str, float] = {}
            for key in rounds[0]["setup"]:
                setup[key] = statistics.median(
                    entry["setup"][key] for entry in rounds
                )
            candidate_rows.append({
                "candidateId": candidate_id,
                "medianSeconds": statistics.median(
                    entry["seconds"] for entry in rounds
                ),
                "roundSeconds": [entry["seconds"] for entry in rounds],
                "memory": memory,
                "setup": setup,
                "executionContract": rounds[0]["executionContract"],
                "maximumCorrectnessError": max(
                    entry["maximumCorrectnessError"] for entry in rounds
                ),
            })

        row_by_id = {row["candidateId"]: row for row in candidate_rows}
        comparisons: list[dict] = []
        for numerator, denominator in (
            (FUSED_ID, WVM_ID),
            (TILED_ID, WVM_ID),
            (TILED_ID, FUSED_ID),
        ):
            numerator_row = row_by_id[numerator]
            denominator_row = row_by_id[denominator]
            round_ratios = [
                numerator_seconds / denominator_seconds
                for numerator_seconds, denominator_seconds in zip(
                    numerator_row["roundSeconds"],
                    denominator_row["roundSeconds"],
                )
            ]
            lower, upper = percentile_bootstrap(
                round_ratios, seed=129 + profile_index,
            )
            key = pair_key(numerator, denominator)
            global_pair_ratios[key][profile] = round_ratios
            comparisons.append({
                "id": key,
                "numeratorCandidateId": numerator,
                "denominatorCandidateId": denominator,
                "ratio": (
                    numerator_row["medianSeconds"]
                    / denominator_row["medianSeconds"]
                ),
                "roundRatios": round_ratios,
                "pairedBootstrap95": {"lower": lower, "upper": upper},
            })
        profile_rows.append({
            "profile": profile,
            "candidates": candidate_rows,
            "comparisons": comparisons,
        })

    complete = complete and len(profile_rows) == len(PROFILES)
    pairwise_summaries: list[dict] = []
    for numerator, denominator in (
        (FUSED_ID, WVM_ID),
        (TILED_ID, WVM_ID),
        (TILED_ID, FUSED_ID),
    ):
        key = pair_key(numerator, denominator)
        comparisons = [
            next(item for item in row["comparisons"] if item["id"] == key)
            for row in profile_rows
        ]
        ratios = [item["ratio"] for item in comparisons]
        interval = None
        if complete:
            lower, upper = stratified_geometric_bootstrap(
                global_pair_ratios[key],
            )
            interval = {"lower": lower, "upper": upper}

        numerator_resident = []
        denominator_resident = []
        numerator_observed = []
        denominator_observed = []
        for row in profile_rows:
            candidates_by_id = {
                item["candidateId"]: item for item in row["candidates"]
            }
            numerator_resident.append(
                candidates_by_id[numerator]["memory"]["algorithmResidentBytes"]
            )
            denominator_resident.append(
                candidates_by_id[denominator]["memory"]["algorithmResidentBytes"]
            )
            numerator_observed.append(
                candidates_by_id[numerator]["memory"]["observedProcessHighWaterBytes"]
            )
            denominator_observed.append(
                candidates_by_id[denominator]["memory"]["observedProcessHighWaterBytes"]
            )
        pairwise_summaries.append({
            "id": key,
            "numeratorCandidateId": numerator,
            "denominatorCandidateId": denominator,
            "geometricTimeRatio": geometric_mean(ratios) if ratios else None,
            "maximumProfileTimeRatio": max(ratios) if ratios else None,
            "stratifiedPairedBootstrap95": interval,
            "geometricAlgorithmResidentRatio": geometric_mean([
                numerator_value / denominator_value
                for numerator_value, denominator_value in zip(
                    numerator_resident, denominator_resident,
                )
            ]) if ratios else None,
            "geometricObservedHighWaterRatio": geometric_mean([
                numerator_value / denominator_value
                for numerator_value, denominator_value in zip(
                    numerator_observed, denominator_observed,
                )
            ]) if ratios else None,
        })

    tile_to_wvm = next(
        item for item in pairwise_summaries
        if item["numeratorCandidateId"] == TILED_ID
        and item["denominatorCandidateId"] == WVM_ID
    )
    interval = tile_to_wvm["stratifiedPairedBootstrap95"]
    engine_gate = bool(
        complete and all_correct
        and tile_to_wvm["geometricTimeRatio"] <= 0.90
        and tile_to_wvm["maximumProfileTimeRatio"] <= 1.03
        and interval is not None and interval["upper"] < 1.0
    )

    return {
        "schema": "spectral-kernel-native-bridge-analysis-v1",
        "phase": "reference",
        "cohortId": COHORT_ID,
        "completeProductionMatrix": complete,
        "allCorrectWithin1e12": all_correct,
        "maximumCorrectnessError": maximum_error,
        "profiles": profile_rows,
        "pairwiseSummaries": pairwise_summaries,
        "deploymentDecisions": {
            "persistentCompiledEngine": {
                "eligibleCandidateIds": candidate_ids,
                "selectedCandidateId": TILED_ID if engine_gate else None,
                "selectionGatePassed": engine_gate,
                "boundary": "ready real input to reconstructed real output; internal representation persists inside the compiled operator",
            },
            "matlabOwnedWvmNativeSpectralBoundary": {
                "eligibleCandidateIds": [WVM_ID],
                "selectedCandidateId": WVM_ID,
                "selectionGatePassed": complete and all_correct,
                "boundary": "MATLAB-visible WVM frequency-major interleaved spectral arrays remain authoritative between compiled calls",
                "scope": "best measured eligible native-layout algorithm, not the fastest unrestricted pipeline",
            },
        },
        "referenceGate": {
            "geometricTileToWvmAtMost": 0.90,
            "maximumProfileTileToWvmAtMost": 1.03,
            "confidenceExcludesTie": bool(
                interval is not None and interval["upper"] < 1.0
            ),
            "correctnessPassed": all_correct,
            "zeroSteadyStateAllocationRequired": True,
            "allocationVerification": "skbench-unit macOS allocator interposer",
            "singlePolicyPersistentEnginePassed": engine_gate,
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
        parser.error("the preregistered campaign requires exactly 3 rounds")
    if arguments.warmups != 3 or arguments.samples != 21:
        parser.error("the preregistered campaign requires 3 warmups and 21 samples")
    if not 0.0 < arguments.max_memory_fraction <= 1.0:
        parser.error("--max-memory-fraction must be in (0, 1]")

    source_commit, source_dirty = git_source_state(repository_root)
    if source_dirty and not arguments.allow_dirty_tree:
        parser.error(
            "the benchmark source tree is dirty; commit and rebuild for reference evidence"
        )

    candidates = candidate_matrix()
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
        f"issue9-native-bridge-reference-{timestamp}"
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
                stem = f"round-{round_index + 1}--{profile}--{candidate.id}"
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
            f"3 rotated rounds, {len(PROFILES)} profiles, 3 frozen candidates."
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
            "What is the exact same-commit relationship among the best WVM-native, "
            "previous fused-split, and fixed tile-16 persistent-engine pipelines?"
        ),
        "baseline": (
            "The established WVM frequency-major interleaved direct/no-reorder "
            "outer-dynamic-16 algorithm, rerun in every round and workload."
        ),
        "changedVariables": [
            "WVM-native full interleaved, fused compact split, or streaming fixed tile-16 compact split representation graph",
        ],
        "controlledVariables": [
            "three frozen algorithms; no new tuning or workload-size selector",
            "Float64 radial horizontal two-thirds retention and vertically truncated Nj",
            "FFTW MEASURE/unaligned/cold internal-1/outer-12",
            "K-squared vertical matrices, outer-dynamic-16, VECLIB_MAXIMUM_THREADS=1, modal work, oracle, and allocation policy",
        ],
        "timedOperation": (
            "One ready-real-input to reconstructed-real-output synthetic antialiased "
            "spectral round trip."
        ),
        "componentLedger": [
            "raw FFT or pruned row/selected-column components",
            "retention, representation movement, rebuild, and embedding",
            "raw forward/inverse vertical MM, modal work, and authoritative uninstrumented total",
            "setup/planning, execution placement, scratch, algorithm-resident, benchmark-only, explicit-peak, and observed memory",
        ],
        "excludedWork": [
            "new algorithm tuning, size-dependent dispatch, and implicit/hybrid dealiasing",
            "WVM nonlinear flux, time integration, I/O, and full-model speed claims",
        ],
        "allocationPolicy": (
            "Zero allocations in every timed steady-state graph; verified by the "
            "macOS allocator-interposer unit test on the benchmarked commit."
        ),
        "deploymentContracts": {
            "persistentCompiledEngine": (
                "The compiled operator owns internal representation from ready real "
                "input through reconstructed real output."
            ),
            "matlabOwnedWvmNativeSpectralBoundary": (
                "MATLAB retains authoritative WVM frequency-major interleaved spectral "
                "arrays between compiled calls; only the WVM-native candidate is eligible."
            ),
        },
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
