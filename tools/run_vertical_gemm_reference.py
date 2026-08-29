#!/usr/bin/env python3
"""Run the frozen issue #8 K-squared grouped vertical-GEMM reference campaign."""

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

from run_float64_baseline_sweep import PROFILES
from run_vertical_gemm_sweep import (
    estimated_k2_explicit_peak_bytes,
    gibibytes,
    git_source_state,
    sysctl_uint64,
)


EXPERIMENT_ID = "issue-008-vertical-projection-gemm"
INCREMENT_ID = "vertical-k2-grouped-finalists-reference-v1"
REFERENCE_ROUNDS = 3
PROVIDER_IDS = ("accelerate-zgemm", "accelerate-split-dgemm")
DIRECTIONS = ("forward", "inverse")


@dataclass(frozen=True)
class Candidate:
    id: str
    schedule: str
    outer_workers: int


def candidate_matrix() -> list[Candidate]:
    return [
        Candidate("outer-dynamic-16", "outer-dynamic", 16),
        Candidate("outer-static-12", "outer-static", 12),
    ]


def rotated(values: list, offset: int) -> list:
    if not values:
        return []
    shift = offset % len(values)
    return values[shift:] + values[:shift]


def primitive_seconds(provider: dict, direction: str) -> float:
    matches = [
        item for item in provider.get("timings", [])
        if item.get("scope") == "primitive"
        and item.get("stage") == "raw vertical GEMM"
        and item.get("direction") == direction
    ]
    if len(matches) != 1:
        raise ValueError(
            f"provider {provider.get('id')} lacks one raw vertical GEMM/{direction} timing"
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


def analyze(results: list[tuple[Candidate, int, dict]]) -> dict:
    cells: dict[tuple[str, str, str, str], dict[int, dict]] = {}
    maximum_error = 0.0
    all_correct = True
    contract_valid = True
    for candidate, round_number, result in results:
        profile = result["run"]["profile"]
        for provider in result["providers"]:
            if provider["id"] not in PROVIDER_IDS:
                continue
            error = maximum_correctness_error(provider)
            maximum_error = max(maximum_error, error)
            all_correct = all_correct and math.isfinite(error) and error <= 1.0e-12
            scheduling = provider.get("scheduling", {})
            contract_valid = contract_valid and (
                scheduling.get("internalWorkers") == 1
                and scheduling.get("outerWorkers") == candidate.outer_workers
                and provider["executionContract"]["forward"]["nativePlacement"] == "out-of-place"
                and provider["executionContract"]["inverse"]["nativePlacement"] == "out-of-place"
            )
            for direction in DIRECTIONS:
                cells.setdefault(
                    (candidate.id, profile, provider["id"], direction), {}
                )[round_number] = {
                    "seconds": primitive_seconds(provider, direction),
                    "persistentBytes": int(provider["memory"]["persistentBytes"]),
                    "setupSeconds": float(provider["setup"]["totalSeconds"]),
                }

    expected_keys = {
        (candidate.id, profile, provider, direction)
        for candidate in candidate_matrix()
        for profile in PROFILES
        for provider in PROVIDER_IDS
        for direction in DIRECTIONS
    }
    complete = set(cells) == expected_keys
    rows: list[dict] = []
    dynamic_static_ratios: list[float] = []
    split_complex_by_candidate: dict[str, list[float]] = {
        candidate.id: [] for candidate in candidate_matrix()
    }
    dynamic_wins = 0
    comparison_count = 0
    for profile in PROFILES:
        for provider_id in PROVIDER_IDS:
            for direction in DIRECTIONS:
                medians: dict[str, float] = {}
                memory: dict[str, float] = {}
                setup: dict[str, float] = {}
                for candidate in candidate_matrix():
                    entries = cells.get((candidate.id, profile, provider_id, direction), {})
                    if sorted(entries) != list(range(1, REFERENCE_ROUNDS + 1)):
                        complete = False
                        continue
                    medians[candidate.id] = statistics.median(
                        item["seconds"] for item in entries.values()
                    )
                    memory[candidate.id] = statistics.median(
                        item["persistentBytes"] for item in entries.values()
                    )
                    setup[candidate.id] = statistics.median(
                        item["setupSeconds"] for item in entries.values()
                    )
                if len(medians) != len(candidate_matrix()):
                    continue
                ratio = medians["outer-dynamic-16"] / medians["outer-static-12"]
                dynamic_static_ratios.append(ratio)
                comparison_count += 1
                dynamic_wins += int(ratio < 1.0)
                rows.append({
                    "profile": profile,
                    "provider": provider_id,
                    "direction": direction,
                    "dynamic16Seconds": medians["outer-dynamic-16"],
                    "static12Seconds": medians["outer-static-12"],
                    "dynamic16ToStatic12": ratio,
                    "dynamic16PersistentBytes": memory["outer-dynamic-16"],
                    "static12PersistentBytes": memory["outer-static-12"],
                    "dynamic16SetupSeconds": setup["outer-dynamic-16"],
                    "static12SetupSeconds": setup["outer-static-12"],
                })

    for candidate in candidate_matrix():
        for profile in PROFILES:
            for direction in DIRECTIONS:
                complex_entries = cells.get(
                    (candidate.id, profile, "accelerate-zgemm", direction), {}
                )
                split_entries = cells.get(
                    (candidate.id, profile, "accelerate-split-dgemm", direction), {}
                )
                if not complex_entries or not split_entries:
                    continue
                split_complex_by_candidate[candidate.id].append(
                    statistics.median(item["seconds"] for item in split_entries.values())
                    / statistics.median(item["seconds"] for item in complex_entries.values())
                )

    dynamic_ratio = geometric_mean(dynamic_static_ratios) if dynamic_static_ratios else None
    split_ratios = {
        candidate_id: geometric_mean(values) if values else None
        for candidate_id, values in split_complex_by_candidate.items()
    }
    selected = (
        "outer-dynamic-16" if dynamic_ratio is not None and dynamic_ratio <= 1.0
        else "outer-static-12"
    )
    return {
        "schema": "spectral-kernel-vertical-gemm-reference-analysis-v1",
        "experimentId": EXPERIMENT_ID,
        "incrementId": INCREMENT_ID,
        "phase": "reference",
        "completeProductionMatrix": complete,
        "allCorrectWithin1e-12": all_correct,
        "executionContractPassed": contract_valid,
        "maximumCorrectnessError": maximum_error,
        "geometricDynamic16ToStatic12": dynamic_ratio,
        "dynamic16Wins": dynamic_wins,
        "comparisonCount": comparison_count,
        "geometricSplitToComplexByTopology": split_ratios,
        "selectedSchedulingCandidateId": selected,
        "profiles": rows,
        "capabilityDisposition": {
            "variableSizeGroupedGemmBatch": "unsupported-by-public-accelerate-cblas-api",
            "blockedPackAndGemm": "not-a-primitive-gemm-requirement; movement is evaluated by issue #13 and the issue #16 compact-tile pipeline",
        },
        "referenceGate": {
            "threeIndependentProcessesPerCell": complete,
            "correctnessPassed": all_correct,
            "executionContractPassed": contract_valid,
            "zeroSteadyStateAllocationRequired": True,
            "allocationVerification": "skbench-unit macOS allocator interposer",
            "referenceCandidateSetPassed": bool(complete and all_correct and contract_valid),
        },
    }


def command_for(
    executable: Path,
    candidate: Candidate,
    profile: str,
    warmups: int,
    samples: int,
    seed: int,
    output: Path,
) -> list[str]:
    return [
        str(executable), "run",
        "--kernel", "vertical-gemm",
        "--profile", profile,
        "--vertical-gemm-family", "k2-grouped",
        "--vertical-gemm-schedule", candidate.schedule,
        "--vertical-gemm-outer-workers", str(candidate.outer_workers),
        "--warmups", str(warmups),
        "--samples", str(samples),
        "--seed", str(seed),
        "--output", str(output),
    ]


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
        default=repository_root / "results/local" / f"issue8-vertical-reference-{timestamp}",
    )
    parser.add_argument("--profiles", nargs="*")
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

    profiles = arguments.profiles or list(PROFILES)
    unknown = sorted(set(profiles) - set(PROFILES))
    if unknown:
        parser.error(f"unknown profile: {', '.join(unknown)}")
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
    physical_memory = sysctl_uint64("hw.memsize", 0)
    estimated_peaks = {
        profile: estimated_k2_explicit_peak_bytes(profile) for profile in profiles
    }
    if physical_memory and not arguments.allow_memory_risk:
        rejected = [
            profile for profile, estimate in estimated_peaks.items()
            if estimate > arguments.max_memory_fraction * physical_memory
        ]
        if rejected:
            details = ", ".join(
                f"{profile} ({gibibytes(estimated_peaks[profile])})" for profile in rejected
            )
            parser.error(
                f"estimated explicit peak exceeds {arguments.max_memory_fraction:.0%} of "
                f"{gibibytes(physical_memory)} physical memory: {details}"
            )

    candidates = candidate_matrix()
    commands: list[tuple[str, int, str, Candidate, list[str], Path]] = []
    for round_index in range(arguments.rounds):
        round_candidates = rotated(candidates, round_index)
        round_profiles = rotated(profiles, round_index)
        for profile in round_profiles:
            for candidate in round_candidates:
                stem = f"round-{round_index + 1}--{profile}--{candidate.id}"
                result_path = arguments.output / f"{stem}.json"
                commands.append((
                    stem,
                    round_index + 1,
                    profile,
                    candidate,
                    command_for(
                        arguments.executable, candidate, profile, arguments.warmups,
                        arguments.samples, arguments.seed, result_path,
                    ),
                    result_path,
                ))

    if arguments.dry_run:
        for _, _, _, _, command, _ in commands:
            print(f"VECLIB_MAXIMUM_THREADS=1 {' '.join(command)}")
        print(
            f"Planned {len(commands)} isolated run(s): {arguments.rounds} round(s), "
            f"{len(profiles)} profile(s), {len(candidates)} candidate(s)."
        )
        return 0

    arguments.output.mkdir(parents=True, exist_ok=False)
    manifest = {
        "schema": "spectral-kernel-local-sweep-v1",
        "experimentId": EXPERIMENT_ID,
        "incrementId": INCREMENT_ID,
        "phase": "reference",
        "createdAtUtc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "question": "Which frozen persistent topology and complex formulation supplies the reference K-squared-grouped antialiased vertical projection/reconstruction kernel?",
        "baseline": "The 64-run issue #8 portability screen that selected dynamic-16 and retained static-12 as the only finalists.",
        "controlledVariables": [
            "Float64 K-squared-grouped matrices, vertical two-thirds retention, fixtures, and oracle",
            "VECLIB_MAXIMUM_THREADS=1, out-of-place placement, and zero steady-state allocation",
            "pre-arranged operands; horizontal packing and ordering remain excluded",
        ],
        "changedVariables": [
            "outer-dynamic-16 versus weighted outer-static-12 scheduling",
            "complex zgemm versus two real split dgemm calls",
            "production workload and forward/inverse direction",
        ],
        "timedOperations": ["raw forward vertical GEMM", "raw inverse vertical GEMM"],
        "excludedWork": [
            "horizontal transforms, packing, matrix construction, fixture generation, modal work, and nonlinear flux",
        ],
        "capabilityDisposition": {
            "variableSizeGroupedGemmBatch": "unsupported-by-public-accelerate-cblas-api",
        },
        "referenceProtocol": {
            "independentProcesses": arguments.rounds,
            "candidateOrderRotatedByRound": True,
            "profileOrderRotatedByRound": True,
            "warmupsPerProcess": arguments.warmups,
            "samplesPerProcess": arguments.samples,
        },
        "profiles": profiles,
        "candidates": [asdict(candidate) for candidate in candidates],
        "threadEnvironment": {"VECLIB_MAXIMUM_THREADS": "1"},
        "physicalMemoryBytes": physical_memory,
        "estimatedExplicitPeakBytesByProfile": estimated_peaks,
        "sourceTreeGitCommit": source_commit,
        "sourceTreeDirty": source_dirty,
        "runs": [],
    }

    failed = False
    completed_results: list[tuple[Candidate, int, dict]] = []
    for index, (stem, round_number, profile, candidate, command, result_path) in enumerate(commands, start=1):
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
            "profile": profile,
            "candidate": asdict(candidate),
            "primaryProvider": "accelerate-split-dgemm",
            "estimatedExplicitPeakBytes": estimated_peaks[profile],
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
