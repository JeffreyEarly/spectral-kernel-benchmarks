#!/usr/bin/env python3
"""Run the bounded issue #13 horizontal-to-vertical representation crossover."""

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

from run_fftw_native_order_sweep import REFERENCE_PROFILES, rotated
from run_vertical_gemm_sweep import (
    PROFILE_SHAPES,
    gibibytes,
    git_source_state,
    sysctl_uint64,
)


CONTROL_IDS = {
    "wvm-direct--outer-dynamic-16",
    "wvm-packed-split--outer-dynamic-16",
}


@dataclass(frozen=True)
class Candidate:
    id: str
    policy: str
    primary_provider: str
    schedule: str
    outer_workers: int
    control: bool = False
    representation_bridge: bool = False


def candidate_matrix() -> list[Candidate]:
    providers = {
        "wvm-direct": "boundary-wvm-direct",
        "wvm-packed-split": "boundary-wvm-packed-split",
        "pruned-compact-interleaved": "boundary-pruned-compact-interleaved",
        "plane-major-fused-split": "boundary-plane-major-fused-split",
        "plane-major-view": "boundary-plane-major-view",
    }
    candidates: list[Candidate] = []
    for policy, provider in providers.items():
        for schedule, workers in (("outer-dynamic", 16), ("outer-static", 12)):
            candidate_id = f"{policy}--{schedule}-{workers}"
            candidates.append(Candidate(
                candidate_id,
                policy,
                provider,
                schedule,
                workers,
                control=candidate_id in CONTROL_IDS,
                representation_bridge=policy == "plane-major-fused-split",
            ))
    return candidates


def load_screen_analysis(path: Path | None) -> dict | None:
    if path is None:
        return None
    with path.open(encoding="utf-8") as stream:
        analysis = json.load(stream)
    if analysis.get("schema") != "spectral-kernel-boundary-crossover-analysis-v1":
        raise ValueError("--screen-analysis has the wrong schema")
    return analysis


def select_candidates(
    requested: list[str] | None,
    phase: str,
    screen_analysis: dict | None = None,
) -> list[Candidate]:
    available = candidate_matrix()
    by_id = {candidate.id: candidate for candidate in available}
    if requested:
        unknown = sorted(set(requested) - set(by_id))
        if unknown:
            raise ValueError(f"unknown candidate: {', '.join(unknown)}")
        selected = [by_id[candidate_id] for candidate_id in requested]
    elif phase == "screen":
        selected = available
    else:
        if screen_analysis is None:
            raise ValueError("reference phase requires --screen-analysis")
        reference_ids = set(screen_analysis.get("referenceCandidateIds", []))
        selected = [candidate for candidate in available if candidate.id in reference_ids]

    if phase == "reference":
        if screen_analysis is None:
            raise ValueError("reference phase requires --screen-analysis")
        permitted = set(screen_analysis.get("referenceCandidateIds", []))
        rejected = [candidate.id for candidate in selected if candidate.id not in permitted]
        if rejected:
            raise ValueError(
                "reference candidate did not satisfy the preregistered screen rule: "
                + ", ".join(rejected)
            )
        if not CONTROL_IDS.issubset({candidate.id for candidate in selected}):
            raise ValueError("reference phase must retain both dynamic-16 historical controls")
    return selected


def provider_timing(provider: dict, direction: str) -> float:
    for timing in provider["timings"]:
        if (
            timing["scope"] == "uninstrumented-total"
            and timing["stage"] == "composed horizontal-vertical boundary"
            and timing["direction"] == direction
        ):
            return float(timing["medianSeconds"])
    raise ValueError(
        f"provider {provider['id']} lacks composed-boundary {direction} timing"
    )


def analyze(
    results: list[tuple[Candidate, dict]],
    phase: str = "screen",
) -> dict:
    cells: dict[tuple[str, str, str], list[float]] = {}
    for candidate, result in results:
        provider = next(
            item for item in result["providers"]
            if item["id"] == candidate.primary_provider
        )
        profile = result["run"]["profile"]
        for direction in ("forward", "inverse"):
            cells.setdefault((candidate.id, profile, direction), []).append(
                provider_timing(provider, direction)
            )

    profiles = sorted({profile for _, profile, _ in cells})
    summaries: dict[str, dict] = {}
    for candidate in candidate_matrix():
        ratios: list[float] = []
        rows: list[dict] = []
        wins = 0
        complete = True
        maximum_ratio = 0.0
        for profile in profiles:
            for direction in ("forward", "inverse"):
                values = cells.get((candidate.id, profile, direction), [])
                peer_medians = [
                    statistics.median(peer_values)
                    for (peer_id, peer_profile, peer_direction), peer_values in cells.items()
                    if peer_profile == profile and peer_direction == direction
                ]
                if not values or not peer_medians:
                    complete = False
                    continue
                candidate_seconds = statistics.median(values)
                best_seconds = min(peer_medians)
                ratio = candidate_seconds / best_seconds
                ratios.append(ratio)
                maximum_ratio = max(maximum_ratio, ratio)
                if ratio <= 1.0 + 1.0e-12:
                    wins += 1
                rows.append({
                    "profile": profile,
                    "direction": direction,
                    "candidateSeconds": candidate_seconds,
                    "bestCandidateSeconds": best_seconds,
                    "candidateToBest": ratio,
                })
        geometric_ratio = (
            math.exp(sum(math.log(value) for value in ratios) / len(ratios))
            if ratios else None
        )
        competitive = bool(
            complete and geometric_ratio is not None
            and geometric_ratio <= 1.05 and wins > 0
        )
        summaries[candidate.id] = {
            "candidate": candidate.id,
            "policy": candidate.policy,
            "schedule": candidate.schedule,
            "outerWorkers": candidate.outer_workers,
            "control": candidate.control,
            "representationBridge": candidate.representation_bridge,
            "completeProductionMatrix": complete,
            "geometricRatioToCellBest": geometric_ratio,
            "maximumCellRatioToBest": maximum_ratio if ratios else None,
            "workloadDirectionWins": wins,
            "competitiveRulePassed": competitive,
            "cells": rows,
        }

    bridge_candidates = [
        summary for summary in summaries.values()
        if summary["representationBridge"]
        and summary["completeProductionMatrix"]
        and summary["geometricRatioToCellBest"] is not None
    ]
    best_bridge = min(
        bridge_candidates,
        key=lambda item: item["geometricRatioToCellBest"],
        default=None,
    )
    advancing: list[str] = []
    for summary in summaries.values():
        bridge_rule = bool(
            best_bridge is not None
            and summary["candidate"] == best_bridge["candidate"]
            and summary["geometricRatioToCellBest"] <= 1.10
        )
        summary["representationBridgeRulePassed"] = bridge_rule
        summary["qualifiesForReference"] = bool(
            summary["competitiveRulePassed"] or bridge_rule
        )
        if summary["qualifiesForReference"] and not summary["control"]:
            advancing.append(summary["candidate"])

    reference_ids = sorted(CONTROL_IDS | set(advancing))
    paired_summaries: list[dict] = []
    for candidate in candidate_matrix():
        ratios: list[float] = []
        rows: list[dict] = []
        complete = True
        for profile in profiles:
            candidate_forward = cells.get((candidate.id, profile, "forward"), [])
            candidate_inverse = cells.get((candidate.id, profile, "inverse"), [])
            peer_totals = []
            for peer in candidate_matrix():
                peer_forward = cells.get((peer.id, profile, "forward"), [])
                peer_inverse = cells.get((peer.id, profile, "inverse"), [])
                if peer_forward and peer_inverse:
                    peer_totals.append(
                        statistics.median(peer_forward)
                        + statistics.median(peer_inverse)
                    )
            if not candidate_forward or not candidate_inverse or not peer_totals:
                complete = False
                continue
            candidate_seconds = (
                statistics.median(candidate_forward)
                + statistics.median(candidate_inverse)
            )
            best_seconds = min(peer_totals)
            ratio = candidate_seconds / best_seconds
            ratios.append(ratio)
            rows.append({
                "profile": profile,
                "candidatePairedSeconds": candidate_seconds,
                "bestCandidatePairedSeconds": best_seconds,
                "candidateToBestPaired": ratio,
            })
        paired_summaries.append({
            "candidate": candidate.id,
            "completeProductionMatrix": complete,
            "geometricRatioToPairedBest": (
                math.exp(sum(math.log(value) for value in ratios) / len(ratios))
                if ratios else None
            ),
            "maximumProfileRatioToPairedBest": max(ratios) if ratios else None,
            "profileWins": sum(value <= 1.0 + 1.0e-12 for value in ratios),
            "profiles": rows,
        })

    issue9_candidates: list[str] = []
    if phase == "reference":
        eligible = [
            summary for summary in paired_summaries
            if summary["completeProductionMatrix"]
            and summary["geometricRatioToPairedBest"] is not None
            and summary["maximumProfileRatioToPairedBest"] <= 1.10
        ]
        if eligible:
            best_geometric = min(
                item["geometricRatioToPairedBest"] for item in eligible
            )
            issue9_candidates = [
                item["candidate"]
                for item in sorted(
                    eligible,
                    key=lambda value: (
                        value["geometricRatioToPairedBest"], value["candidate"]
                    ),
                )
                if item["geometricRatioToPairedBest"] <= 1.03 * best_geometric
            ][:3]

    return {
        "schema": "spectral-kernel-boundary-crossover-analysis-v1",
        "phase": phase,
        "competitiveThreshold": 1.05,
        "competitiveWinRequirement": 1,
        "representationBridgeThreshold": 1.10,
        "referenceControls": sorted(CONTROL_IDS),
        "advancingCandidateIds": sorted(advancing),
        "referenceCandidateIds": reference_ids,
        "issue9CandidateIds": issue9_candidates,
        "issue9SelectionRule": (
            "At reference depth, pair forward and inverse boundary medians within each "
            "profile, then retain at most three candidates within 3% of the best geometric "
            "paired ratio and no profile more than 10% slower than its best paired candidate."
        ),
        "issue9PairedComparisons": paired_summaries,
        "comparisons": [summaries[candidate.id] for candidate in candidate_matrix()],
    }


def estimated_explicit_peak_bytes(profile: str, policy: str) -> int:
    nx, nz, fields, nkl, groups = PROFILE_SHAPES[profile]
    ny = nx
    nj = 2 * (nz - 1) // 3
    columns = nkl * fields
    family_elements = groups * nz * nj
    source_matrices = 2 * family_elements * 8
    physical = nz * columns * 16
    modal = nj * columns * 16
    full = ny * (nx // 2 + 1) * nz * fields * 16
    full_modal = ny * (nx // 2 + 1) * nj * fields * 16
    real_grid = nx * ny * nz * fields * 8
    if policy in {"wvm-packed-split", "plane-major-fused-split"}:
        provider = 2 * family_elements * 8 + 2 * (physical + modal)
        external = 2 * full + real_grid + physical + modal
    elif policy == "pruned-compact-interleaved":
        provider = 2 * family_elements * 16 + 2 * (physical + modal)
        external = full + real_grid + physical + modal
    else:
        provider = 2 * family_elements * 16
        external = 2 * full + 2 * full_modal + real_grid + physical + modal
    return source_matrices + provider + external + 2 * 1024**2


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
        "--kernel", "spectral-boundary",
        "--boundary-policy", candidate.policy,
        "--profile", profile,
        "--fftw-planning", "measure",
        "--fftw-alignment", "unaligned",
        "--fftw-wisdom", "cold",
        "--fftw-internal-workers", "1",
        "--fftw-outer-workers", "12",
        "--vertical-gemm-family", "k2-grouped",
        "--vertical-gemm-schedule", candidate.schedule,
        "--vertical-gemm-outer-workers", str(candidate.outer_workers),
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
    parser.add_argument("--candidate", action="append", dest="candidates")
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
        candidates = select_candidates(
            arguments.candidates, arguments.phase, screen_analysis,
        )
    except (OSError, ValueError) as error:
        parser.error(str(error))

    if arguments.phase == "screen":
        rounds = arguments.rounds or 1
        warmups = arguments.warmups or 2
        samples = arguments.samples or 9
        increment_id = "spectral-boundary-crossover-screen-v1"
    else:
        rounds = arguments.rounds or 3
        warmups = arguments.warmups or 3
        samples = arguments.samples or 21
        increment_id = "spectral-boundary-crossover-reference-v1"
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
        f"issue13-spectral-boundary-{arguments.phase}-{timestamp}"
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
        "experimentId": "issue-013-ordering-packing-crossover",
        "incrementId": increment_id,
        "phase": arguments.phase,
        "createdAtUtc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "question": (
            "Which complete Float64 representation policy minimizes the composed "
            "antialiased horizontal-transform plus grouped vertical-projection boundary?"
        ),
        "baseline": (
            "The historical WVM direct/no-reorder and WVM radial-pack-to-split policies, "
            "with issue #7 horizontal survivors and issue #8 vertical schedulers."
        ),
        "changedVariables": [
            "full versus pruned horizontal algorithm",
            "WVM, plane-major, compact interleaved, and compact split representations",
            "elided, fused, or explicit retention/packing/embedding",
            "dynamic-16 versus static-12 outer vertical scheduling",
        ],
        "controlledVariables": [
            "Float64 radial two-thirds horizontal retention and vertically retained Nj=floor(2*(Nz-1)/3)",
            "FFTW 3.3.11 MEASURE, unaligned, cold, internal-1/outer-12",
            "K-squared grouped vertical matrices, VECLIB_MAXIMUM_THREADS=1, fixture, seed, and workloads",
            "zero steady-state allocation and no modal work or nonlinear flux",
        ],
        "timedOperations": [
            "raw FFT or exact pruned row/column components",
            "raw vertical MM",
            "retention, conversion, packing, zero fill, and embedding",
            "uninstrumented forward and inverse horizontal-to-vertical boundary totals",
        ],
        "excludedWork": [
            "modal work, nonlinear flux, full issue #9 pipeline, Float32, and GPU work",
            "planning, allocation, fixture generation, and oracle reordering from steady-state totals",
        ],
        "placement": (
            "Every policy runs in an isolated process. FFTs and vertical transforms are out-of-place. "
            "Full-spectrum direct/view inverse totals rebuild zero-padded disposable input every call "
            "because multidimensional FFTW c2r may destroy it."
        ),
        "advancementRule": (
            "Advance a non-control candidate only when its complete-matrix geometric ratio to the "
            "cell best is at most 1.05 with at least one workload-direction win. The best fused-split "
            "representation bridge may advance at 1.10. Dynamic-16 WVM direct and packed controls "
            "remain at reference depth regardless of screen rank."
        ),
        "issue9SelectionRule": (
            "At reference depth, pair forward and inverse boundary medians within each profile, "
            "then retain at most three candidates within 3% of the best geometric paired ratio "
            "and with no profile more than 10% slower than its best paired candidate."
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
    completed_results: list[tuple[Candidate, dict]] = []
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
                completed_results.append((candidate, result))
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
