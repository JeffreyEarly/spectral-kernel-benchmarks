#!/usr/bin/env python3
"""Run the issue #9 deep-vertical robustness reference for the selected policy."""

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

from run_spectral_pipeline_sweep import (
    estimated_explicit_peak_bytes as issue9_estimated_peak,
    maximum_correctness_error,
    provider_timing,
)
from run_streaming_pruned_locality_sweep import (
    LocalityCandidate,
    command_for,
    estimated_explicit_peak_bytes as streaming_estimated_peak,
    required_memory,
)
from run_streaming_pruned_reference import percentile_bootstrap
from run_vertical_gemm_sweep import gibibytes, git_source_state, sysctl_uint64


EXPERIMENT_ID = "issue-009-combined-spectral-pipeline"
INCREMENT_ID = "synthetic-spectral-pipeline-deep-vertical-reference-v1"
COHORT_ID = "issue9-512-nz513-f4-deep-vertical-v1"
PROFILE = "wvm-large-512-nz513-f4"
BASELINE_ID = "wvm-direct--outer-dynamic-16"
CANDIDATE_ID = "streaming-pruned-tiled-16--outer-dynamic-16"
REFERENCE_ROUNDS = 3


def candidate_matrix() -> list[LocalityCandidate]:
    return [
        LocalityCandidate(
            BASELINE_ID,
            "wvm-direct",
            "pipeline-wvm-direct",
            "matlab-visible-wvm-native-control",
            0,
        ),
        LocalityCandidate(
            CANDIDATE_ID,
            "streaming-pruned-compact-split",
            "pipeline-streaming-pruned-compact-split",
            "selected-persistent-compiled-engine-policy",
            16,
        ),
    ]


def rotated(values: list, offset: int) -> list:
    if not values:
        return []
    shift = offset % len(values)
    return values[shift:] + values[:shift]


def candidate_provider(candidate: LocalityCandidate, result: dict) -> dict:
    return next(
        provider for provider in result["providers"]
        if provider["id"] == candidate.primary_provider
    )


def stage_seconds(
    provider: dict, scope: str, stage: str, direction: str,
) -> float | None:
    matches = [
        item for item in provider.get("timings", [])
        if item.get("scope") == scope
        and item.get("stage") == stage
        and item.get("direction") == direction
    ]
    if not matches:
        return None
    if len(matches) != 1:
        raise ValueError(
            f"provider {provider['id']} has multiple {scope}/{stage}/{direction} timings"
        )
    return float(matches[0]["medianSeconds"])


def components(candidate: LocalityCandidate, provider: dict) -> dict[str, float]:
    if candidate.id == BASELINE_ID:
        horizontal_scope = "primitive"
        horizontal_stage = "raw FFT"
        movement_stages = (
            "logical retained provider-order view",
            "rebuild zero-padded inverse spectrum",
        )
    else:
        horizontal_scope = "retained-operator-total"
        horizontal_stage = "streaming pruned compact split horizontal operator"
        movement_stages = (
            "plane-major compact staging and blocked split transpose",
            "blocked split load, compact staging, embed, and zero fill",
        )
    requested = {
        "horizontalForward": (horizontal_scope, horizontal_stage, "forward"),
        "horizontalInverse": (horizontal_scope, horizontal_stage, "inverse"),
        "verticalForward": ("primitive", "raw vertical MM", "forward"),
        "verticalInverse": ("primitive", "raw vertical MM", "inverse"),
        "modalWork": ("component", "mode-keyed modal work", "modal"),
        "movementForward": ("adapter-component", movement_stages[0], "forward"),
        "movementInverse": ("adapter-component", movement_stages[1], "inverse"),
    }
    values: dict[str, float] = {}
    for key, timing_key in requested.items():
        value = stage_seconds(provider, *timing_key)
        if value is None:
            raise ValueError(
                f"provider {provider['id']} lacks required deep-vertical component {key}"
            )
        values[key] = value
    values["uninstrumentedTotal"] = provider_timing(provider)
    return values


def analyze(results: list[tuple[LocalityCandidate, int, dict]]) -> dict:
    cells: dict[str, dict[int, dict]] = {}
    maximum_error = 0.0
    all_correct = True
    placement_valid = True
    for candidate, round_number, result in results:
        provider = candidate_provider(candidate, result)
        error = maximum_correctness_error(provider)
        maximum_error = max(maximum_error, error)
        all_correct = all_correct and math.isfinite(error) and error <= 1.0e-12
        execution = provider["executionContract"]
        placement_valid = placement_valid and (
            execution["forward"]["nativePlacement"] == "out-of-place"
            and execution["inverse"]["nativePlacement"] == "out-of-place"
            and execution["forward"]["adapterPreservesCallerInput"]
            and execution["inverse"]["adapterPreservesCallerInput"]
        )
        cells.setdefault(candidate.id, {})[round_number] = {
            "seconds": provider_timing(provider),
            "components": components(candidate, provider),
            "memory": required_memory(provider),
        }

    baseline = cells.get(BASELINE_ID, {})
    candidate = cells.get(CANDIDATE_ID, {})
    rounds = sorted(set(baseline) & set(candidate))
    complete = (
        rounds == list(range(1, REFERENCE_ROUNDS + 1))
        and len(baseline) == REFERENCE_ROUNDS
        and len(candidate) == REFERENCE_ROUNDS
    )
    if not complete:
        return {
            "schema": "spectral-kernel-pipeline-deep-vertical-analysis-v1",
            "experimentId": EXPERIMENT_ID,
            "incrementId": INCREMENT_ID,
            "phase": "reference",
            "completeProductionMatrix": False,
            "allCorrectWithin1e-12": all_correct,
            "maximumCorrectnessError": maximum_error,
        }

    baseline_seconds = statistics.median(baseline[index]["seconds"] for index in rounds)
    candidate_seconds = statistics.median(candidate[index]["seconds"] for index in rounds)
    ratios = [candidate[index]["seconds"] / baseline[index]["seconds"] for index in rounds]
    lower, upper = percentile_bootstrap(ratios)

    component_rows: dict[str, dict[str, float]] = {}
    for key in baseline[rounds[0]]["components"]:
        baseline_value = statistics.median(
            baseline[index]["components"][key] for index in rounds
        )
        candidate_value = statistics.median(
            candidate[index]["components"][key] for index in rounds
        )
        component_rows[key] = {
            "baselineSeconds": baseline_value,
            "candidateSeconds": candidate_value,
            "candidateToBaseline": candidate_value / baseline_value,
        }

    memory_rows: dict[str, dict[str, float]] = {}
    for key in baseline[rounds[0]]["memory"]:
        baseline_value = statistics.median(
            baseline[index]["memory"][key] for index in rounds
        )
        candidate_value = statistics.median(
            candidate[index]["memory"][key] for index in rounds
        )
        memory_rows[key] = {
            "baselineBytes": baseline_value,
            "candidateBytes": candidate_value,
            "candidateToBaseline": candidate_value / baseline_value,
        }

    ratio = candidate_seconds / baseline_seconds
    return {
        "schema": "spectral-kernel-pipeline-deep-vertical-analysis-v1",
        "experimentId": EXPERIMENT_ID,
        "incrementId": INCREMENT_ID,
        "cohortId": COHORT_ID,
        "phase": "reference",
        "profile": PROFILE,
        "baselineCandidateId": BASELINE_ID,
        "optimizationCandidateId": CANDIDATE_ID,
        "completeProductionMatrix": complete,
        "allCorrectWithin1e-12": all_correct,
        "allPlacementContractsValid": placement_valid,
        "maximumCorrectnessError": maximum_error,
        "baselineSeconds": baseline_seconds,
        "candidateSeconds": candidate_seconds,
        "candidateToBaseline": ratio,
        "roundRatios": ratios,
        "pairedBootstrap95": {"lower": lower, "upper": upper},
        "components": component_rows,
        "memory": memory_rows,
        "robustnessGate": {
            "maximumRatioAtMost": 1.03,
            "regressionPassed": ratio <= 1.03,
            "confidenceExcludesMaterialRegression": upper <= 1.03,
            "correctnessPassed": all_correct,
            "placementPassed": placement_valid,
            "zeroSteadyStateAllocationRequired": True,
            "allocationVerification": "skbench-unit macOS allocator interposer",
            "singleUniformPolicyRetained": True,
            "sizeDependentDispatchAllowed": False,
            "crossMacReplicationStillRequired": True,
        },
    }


def estimated_peak(profile: str, candidate: LocalityCandidate) -> int:
    if candidate.id == BASELINE_ID:
        return issue9_estimated_peak(profile, candidate.policy)
    return streaming_estimated_peak(profile, candidate)


def main() -> int:
    repository_root = Path(__file__).resolve().parents[1]
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--executable", type=Path,
        default=repository_root / "build/release/skbench",
    )
    parser.add_argument(
        "--output", type=Path,
        default=repository_root / "results/local" / f"issue9-deep-vertical-reference-{timestamp}",
    )
    parser.add_argument("--rounds", type=int, default=REFERENCE_ROUNDS)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--samples", type=int, default=21)
    parser.add_argument("--seed", type=int, default=129)
    parser.add_argument("--max-memory-fraction", type=float, default=0.75)
    parser.add_argument("--allow-memory-risk", action="store_true")
    parser.add_argument("--allow-dirty-tree", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    arguments = parser.parse_args()

    if min(arguments.rounds, arguments.warmups, arguments.samples) < 1:
        parser.error("rounds, warmups, and samples must be positive")
    if not 0.0 < arguments.max_memory_fraction <= 1.0:
        parser.error("--max-memory-fraction must be in (0, 1]")
    source_commit, source_dirty = git_source_state(repository_root)
    if source_dirty and not arguments.allow_dirty_tree:
        parser.error(
            "the benchmark source tree is dirty; commit and rebuild for evidence collection "
            "or use --allow-dirty-tree for an exploratory run"
        )

    candidates = candidate_matrix()
    physical_memory = sysctl_uint64("hw.memsize", 0)
    estimated_peaks = {
        candidate.id: estimated_peak(PROFILE, candidate) for candidate in candidates
    }
    if physical_memory and not arguments.allow_memory_risk:
        rejected = [
            candidate_id for candidate_id, estimate in estimated_peaks.items()
            if estimate > arguments.max_memory_fraction * physical_memory
        ]
        if rejected:
            details = ", ".join(
                f"{candidate_id} ({gibibytes(estimated_peaks[candidate_id])})"
                for candidate_id in rejected
            )
            parser.error(
                f"estimated explicit peak exceeds {arguments.max_memory_fraction:.0%} of "
                f"{gibibytes(physical_memory)} physical memory: {details}"
            )

    commands: list[tuple[str, int, LocalityCandidate, list[str], Path]] = []
    for round_index in range(arguments.rounds):
        for candidate in rotated(candidates, round_index):
            stem = f"round-{round_index + 1}--{PROFILE}--{candidate.id}"
            result_path = arguments.output / f"{stem}.json"
            commands.append((
                stem,
                round_index + 1,
                candidate,
                command_for(
                    arguments.executable, candidate, PROFILE, arguments.warmups,
                    arguments.samples, arguments.seed, result_path,
                ),
                result_path,
            ))

    if arguments.dry_run:
        for candidate_id, estimate in estimated_peaks.items():
            print(f"estimated explicit peak {candidate_id}: {gibibytes(estimate)}")
        for _, _, _, command, _ in commands:
            print(f"VECLIB_MAXIMUM_THREADS=1 {' '.join(command)}")
        print(f"Planned {len(commands)} isolated deep-vertical reference run(s).")
        return 0

    arguments.output.mkdir(parents=True, exist_ok=False)
    manifest = {
        "schema": "spectral-kernel-local-sweep-v1",
        "experimentId": EXPERIMENT_ID,
        "incrementId": INCREMENT_ID,
        "phase": "reference",
        "cohortId": COHORT_ID,
        "createdAtUtc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "question": "Does the selected uniform tile-16 streaming policy remain correct, capacity-safe, and non-regressing when vertical resolution increases to Nz=513?",
        "baseline": "The same-commit WVM direct/native-layout control used by the final issue #9 three-way campaign.",
        "changedVariables": ["vertical resolution increased to Nz=513 at fixed 512 squared and fields=4"],
        "controlledVariables": [
            "the already-frozen WVM-direct and streaming tile-16 policies",
            "Float64 radial and vertical two-thirds retention, modal work, FFTW, vertical kernels, scheduling, and oracle",
            "out-of-place placement, zero steady-state allocation, and one policy across sizes",
        ],
        "timedOperation": "One uninstrumented ready-real-input to reconstructed-real-output synthetic antialiased spectral round trip.",
        "excludedWork": [
            "new tuning, size-dependent dispatch, nonlinear flux, time integration, I/O, and full-model claims",
        ],
        "referenceProtocol": {
            "independentlyPlannedProcesses": arguments.rounds,
            "candidateOrderRotatedByRound": True,
            "warmupsPerProcess": arguments.warmups,
            "samplesPerProcess": arguments.samples,
        },
        "profiles": [PROFILE],
        "candidates": [asdict(candidate) for candidate in candidates],
        "threadEnvironment": {"VECLIB_MAXIMUM_THREADS": "1"},
        "physicalMemoryBytes": physical_memory,
        "maximumMemoryFraction": arguments.max_memory_fraction,
        "estimatedExplicitPeakBytes": estimated_peaks,
        "sourceTreeGitCommit": source_commit,
        "sourceTreeDirty": source_dirty,
        "runs": [],
    }

    failed = False
    completed_results: list[tuple[LocalityCandidate, int, dict]] = []
    for index, (stem, round_number, candidate, command, result_path) in enumerate(commands, start=1):
        print(f"[{index}/{len(commands)}] {stem}", flush=True)
        log_path = arguments.output / f"{stem}.log"
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
            "profile": PROFILE,
            "candidate": asdict(candidate),
            "primaryProvider": candidate.primary_provider,
            "estimatedExplicitPeakBytes": estimated_peaks[candidate.id],
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
                "sourceMetadataMatches": metadata_matches,
            })
            if result["status"] == "passed" and metadata_matches:
                completed_results.append((candidate, round_number, result))
            else:
                completed = subprocess.CompletedProcess(command, 1)
        manifest["runs"].append(entry)
        with (arguments.output / "manifest.json").open("w", encoding="utf-8") as stream:
            json.dump(manifest, stream, indent=2)
            stream.write("\n")
        if completed.returncode != 0:
            failed = True
            print(log_path.read_text(encoding="utf-8")[-4000:], file=sys.stderr)
            if not arguments.continue_on_error:
                break

    if completed_results:
        with (arguments.output / "analysis.json").open("w", encoding="utf-8") as stream:
            json.dump(analyze(completed_results), stream, indent=2)
            stream.write("\n")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
