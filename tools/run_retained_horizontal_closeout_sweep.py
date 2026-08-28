#!/usr/bin/env python3
"""Run the bounded issue #7 representation-boundary close-out campaign."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

from run_fftw_native_order_sweep import REFERENCE_PROFILES, rotated
from run_pruned_horizontal_sweep import estimated_explicit_peak_bytes
from run_vertical_gemm_sweep import gibibytes, git_source_state, sysctl_uint64


CONTROL_IDS = {"fftw-plane-major-control", "fftw-pruned-control"}


@dataclass(frozen=True)
class Candidate:
    id: str
    kind: str
    primary_provider: str
    retained_representation: str
    outer_workers: int = 12
    control: bool = False


def candidate_matrix() -> list[Candidate]:
    return [
        Candidate(
            "fftw-plane-major-control", "full", "fftw", "interleaved",
            control=True,
        ),
        Candidate(
            "fftw-pruned-control", "pruned", "fftw-partial-column-pruned",
            "interleaved", control=True,
        ),
        Candidate(
            "fftw-plane-major-retained-view", "full",
            "fftw-plane-major-retained-view", "view",
        ),
        Candidate(
            "fftw-plane-major-fused-retained-split", "full",
            "fftw-plane-major-fused-retained-split", "split",
        ),
        Candidate(
            "fftw-pruned-fused-retained-split", "pruned",
            "fftw-partial-column-pruned-fused-split", "split",
        ),
    ]


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
        advancing = set(screen_analysis.get("advancingCandidateIds", []))
        selected = [
            candidate for candidate in available
            if candidate.control or candidate.id in advancing
        ]

    if phase == "reference":
        if screen_analysis is None:
            raise ValueError("reference phase requires --screen-analysis")
        advancing = set(screen_analysis.get("advancingCandidateIds", []))
        rejected = [
            candidate.id for candidate in selected
            if not candidate.control and candidate.id not in advancing
        ]
        if rejected:
            raise ValueError(
                "reference candidate did not satisfy the preregistered screen rule: "
                + ", ".join(rejected)
            )
        if not CONTROL_IDS.issubset({candidate.id for candidate in selected}):
            raise ValueError("reference phase must retain both matched controls")
    return selected


def provider_timing(provider: dict, direction: str) -> float:
    for timing in provider["timings"]:
        if (
            timing["scope"] == "uninstrumented-total"
            and timing["direction"] == direction
        ):
            return float(timing["medianSeconds"])
    raise ValueError(
        f"provider {provider['id']} lacks uninstrumented-total {direction} timing"
    )


def analyze(results: list[tuple[Candidate, dict]]) -> dict:
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
    comparisons: list[dict] = []
    advancing: list[str] = []
    for candidate in candidate_matrix():
        if candidate.control:
            continue
        ratios: list[float] = []
        wins = 0
        complete = True
        rows: list[dict] = []
        for profile in profiles:
            for direction in ("forward", "inverse"):
                candidate_values = cells.get((candidate.id, profile, direction), [])
                control_values = [
                    value
                    for control_id in CONTROL_IDS
                    for value in cells.get((control_id, profile, direction), [])
                ]
                if not candidate_values or not control_values:
                    complete = False
                    continue
                candidate_seconds = statistics.median(candidate_values)
                control_seconds = min(
                    statistics.median(
                        cells[(control_id, profile, direction)]
                    )
                    for control_id in CONTROL_IDS
                    if cells.get((control_id, profile, direction))
                )
                ratio = candidate_seconds / control_seconds
                ratios.append(ratio)
                wins += ratio < 1.0
                rows.append({
                    "profile": profile,
                    "direction": direction,
                    "candidateSeconds": candidate_seconds,
                    "bestControlSeconds": control_seconds,
                    "candidateToBestControl": ratio,
                })
        geometric_ratio = (
            math.exp(sum(math.log(value) for value in ratios) / len(ratios))
            if ratios else None
        )
        competitive = bool(
            complete and geometric_ratio is not None
            and geometric_ratio <= 1.05 and wins > 0
        )
        representation_bridge = bool(
            complete and geometric_ratio is not None
            and candidate.retained_representation in {"split", "view"}
            and geometric_ratio <= 1.10
        )
        qualifies = competitive or representation_bridge
        if qualifies:
            advancing.append(candidate.id)
        comparisons.append({
            "candidate": candidate.id,
            "retainedRepresentation": candidate.retained_representation,
            "completeProductionMatrix": complete,
            "geometricRatioToBestControl": geometric_ratio,
            "workloadDirectionWins": wins,
            "competitiveRulePassed": competitive,
            "representationBridgeRulePassed": representation_bridge,
            "qualifiesForReference": qualifies,
            "cells": rows,
        })
    return {
        "schema": "spectral-kernel-retained-horizontal-closeout-analysis-v1",
        "competitiveThreshold": 1.05,
        "representationBridgeThreshold": 1.10,
        "advancingCandidateIds": advancing,
        "comparisons": comparisons,
    }


def command_for(
    executable: Path, candidate: Candidate, profile: str,
    warmups: int, samples: int, seed: int, result_path: Path,
) -> list[str]:
    command = [
        str(executable), "run", "--providers", "fftw",
        "--fftw-layout", "interleaved",
        "--fftw-planning", "measure",
        "--fftw-alignment", "unaligned",
        "--fftw-wisdom", "cold",
        "--fftw-internal-workers", "1",
        "--fftw-outer-workers", str(candidate.outer_workers),
        "--retained-representation", candidate.retained_representation,
        "--profile", profile,
        "--warmups", str(warmups),
        "--samples", str(samples),
        "--seed", str(seed),
        "--output", str(result_path),
    ]
    if candidate.kind == "pruned":
        command[2:2] = ["--kernel", "pruned-horizontal"]
    else:
        command.extend(["--fftw-spectrum-order", "plane-major"])
    return command


def load_screen_analysis(path: Path | None) -> dict | None:
    if path is None:
        return None
    with path.open(encoding="utf-8") as stream:
        analysis = json.load(stream)
    if analysis.get("schema") != "spectral-kernel-retained-horizontal-closeout-analysis-v1":
        raise ValueError("--screen-analysis has the wrong schema")
    return analysis


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
        increment_id = "retained-horizontal-representation-closeout-screen-v1"
    else:
        rounds = arguments.rounds or 3
        warmups = arguments.warmups or 3
        samples = arguments.samples or 21
        increment_id = "retained-horizontal-representation-closeout-reference-v1"
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
        profile: estimated_explicit_peak_bytes(profile) for profile in profiles
    }
    if physical_memory > 0 and not arguments.allow_memory_risk:
        rejected = [
            profile for profile, estimate in estimated_peaks.items()
            if estimate > arguments.max_memory_fraction * physical_memory
        ]
        if rejected:
            details = ", ".join(
                f"{profile} ({gibibytes(estimated_peaks[profile])})"
                for profile in rejected
            )
            parser.error(
                f"estimated explicit peak exceeds {arguments.max_memory_fraction:.0%} of "
                f"{gibibytes(physical_memory)} physical memory: {details}; "
                "use --allow-memory-risk to override"
            )

    output = arguments.output or (
        repository_root / "results/local" /
        f"issue7-retained-horizontal-closeout-{arguments.phase}-{timestamp}"
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
            print(" ".join(command))
        print(
            f"Planned {len(commands)} isolated run(s): {rounds} round(s), "
            f"{len(profiles)} profile(s), {len(candidates)} candidate(s)."
        )
        return 0

    output.mkdir(parents=True, exist_ok=False)
    manifest = {
        "schema": "spectral-kernel-local-sweep-v1",
        "experimentId": "issue-007-retained-horizontal-algorithms",
        "incrementId": increment_id,
        "phase": arguments.phase,
        "createdAtUtc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "question": (
            "Can a zero-copy provider-order retained view or a fused compact split "
            "boundary improve on the established plane-major and pruned Float64 controls?"
        ),
        "baseline": (
            "Matched plane-major full FFTW and partial-column-pruned FFTW with one "
            "internal worker and 12 persistent outer workers."
        ),
        "changedVariables": [
            "persistent full-spectrum index view versus compact split retained storage",
            "full versus partial-column-pruned FFTW algorithm",
            "selection/conversion/normalization fusion policy",
        ],
        "controlledVariables": [
            "Float64 logical radial two-thirds operator, fixture, seed, and no horizontal normalization",
            "FFTW 3.3.11, MEASURE planning, unaligned execution, and outer-12 scheduling",
            "six production workloads and zero steady-state allocation",
        ],
        "timedOperations": [
            "raw FFT or separately exposed row/column primitives",
            "retention, representation conversion, embedding, and zero fill",
            "complete uninstrumented retained forward and inverse operators",
            "diagnostic fused versus separate horizontal normalization for compact split",
        ],
        "excludedWork": [
            "vertical projection, modal work, nonlinear flux, and issue #13 packing policies",
            "construction of a ready zero-padded inverse view, which issue #13 must measure",
            "planning, allocation, fixture restoration, oracle reordering, Float32, and GPU work",
        ],
        "placement": (
            "All full and pruned transforms are out-of-place. The retained view keeps a full "
            "plane-major forward spectrum; its inverse accepts ready disposable zero-padded "
            "provider storage because multidimensional FFTW c2r may destroy its input."
        ),
        "advancementRule": (
            "A new candidate advances when its complete-matrix geometric ratio to the best "
            "matched control is at most 1.05 with at least one workload-direction win, or when "
            "a distinct split/view representation needed by issue #13 is within 1.10."
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
        "physicalMemoryBytes": physical_memory,
        "estimatedExplicitPeakBytesByProfile": estimated_peaks,
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
        with log_path.open("w", encoding="utf-8") as log:
            completed = subprocess.run(
                command, cwd=repository_root, stdout=log,
                stderr=subprocess.STDOUT,
            )
        entry = {
            "id": stem,
            "round": round_number,
            "profile": profile,
            "candidate": asdict(candidate),
            "primaryProvider": candidate.primary_provider,
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
            json.dump(analyze(completed_results), stream, indent=2)
            stream.write("\n")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
