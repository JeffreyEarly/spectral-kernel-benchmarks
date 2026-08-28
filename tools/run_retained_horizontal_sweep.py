#!/usr/bin/env python3
"""Run the bounded issue #7 retained-horizontal finalist campaign."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

from run_fftw_native_order_sweep import REFERENCE_PROFILES, rotated
from run_pruned_horizontal_sweep import estimated_explicit_peak_bytes
from run_vertical_gemm_sweep import gibibytes, git_source_state, sysctl_uint64


GUARD_PROFILES = {
    "vdsp-native-guard-256-w12": "wvm-historical-256-nz65-f3",
    "vdsp-native-guard-512-w16": "wvm-historical-512-nz129-f3",
}


@dataclass(frozen=True)
class Candidate:
    id: str
    kind: str
    primary_provider: str
    outer_workers: int
    applicable_profile: str | None = None


def candidate_matrix() -> list[Candidate]:
    return [
        Candidate("fftw-plane-major-outer12", "plane-major", "fftw", 12),
        Candidate("fftw-pruned-outer4", "pruned", "fftw-partial-column-pruned", 4),
        Candidate("fftw-pruned-outer8", "pruned", "fftw-partial-column-pruned", 8),
        Candidate("fftw-pruned-outer12", "pruned", "fftw-partial-column-pruned", 12),
        Candidate(
            "vdsp-native-guard-256-w12", "vdsp-guard",
            "accelerate-vdsp-native-retained", 12,
            GUARD_PROFILES["vdsp-native-guard-256-w12"],
        ),
        Candidate(
            "vdsp-native-guard-512-w16", "vdsp-guard",
            "accelerate-vdsp-native-retained", 16,
            GUARD_PROFILES["vdsp-native-guard-512-w16"],
        ),
    ]


def select_candidates(
    requested: list[str] | None, phase: str,
) -> list[Candidate]:
    available = candidate_matrix()
    by_id = {candidate.id: candidate for candidate in available}
    if requested:
        unknown = sorted(set(requested) - set(by_id))
        if unknown:
            raise ValueError(f"unknown candidate: {', '.join(unknown)}")
        selected = [by_id[value] for value in requested]
    else:
        selected = (
            [candidate for candidate in available if candidate.kind != "vdsp-guard"]
            if phase == "reference" else available
        )
    if phase == "reference" and any(candidate.kind == "vdsp-guard" for candidate in selected):
        raise ValueError(
            "vDSP guardrails are preliminary-only unless the published 1.25x advancement rule passes"
        )
    return selected


def timing(provider: dict, scope: str, direction: str) -> float:
    for item in provider["timings"]:
        if item["scope"] == scope and item["direction"] == direction:
            return float(item["medianSeconds"])
    raise ValueError(
        f"provider {provider['id']} lacks {scope} {direction} timing"
    )


def analyze(results: list[tuple[Candidate, dict]]) -> dict:
    by_profile: dict[str, list[dict]] = {}
    guards: list[dict] = []
    for candidate, result in results:
        profile = result["run"]["profile"]
        providers = result["providers"]
        fftw_candidates = [
            provider for provider in providers
            if provider["id"] in {
                "fftw", "fftw-full-2d-retained-reference",
                "fftw-partial-column-pruned",
            }
        ]
        by_profile.setdefault(profile, []).extend(fftw_candidates)
        if candidate.kind == "vdsp-guard":
            provider = next(
                item for item in providers
                if item["id"] == "accelerate-vdsp-native-retained"
            )
            guards.append({"candidate": candidate.id, "profile": profile, "provider": provider})

    guard_results: list[dict] = []
    advance = False
    for guard in guards:
        profile = guard["profile"]
        ratios: dict[str, float] = {}
        best: dict[str, float] = {}
        for direction in ("forward", "inverse"):
            best[direction] = min(
                timing(provider, "uninstrumented-total", direction)
                for provider in by_profile[profile]
            )
            ratios[direction] = (
                timing(guard["provider"], "uninstrumented-total", direction)
                / best[direction]
            )
        qualifies = ratios["forward"] <= 1.25 and ratios["inverse"] <= 1.25
        advance = advance or qualifies
        guard_results.append({
            "candidate": guard["candidate"],
            "profile": profile,
            "bestFftwSeconds": best,
            "vdspToBestFftw": ratios,
            "qualifiesForExpansion": qualifies,
        })
    return {
        "schema": "spectral-kernel-retained-horizontal-analysis-v1",
        "vdspExpansionThreshold": 1.25,
        "vdspExpansionTriggered": advance,
        "vdspGuardrails": guard_results,
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
    parser.add_argument("--profiles", nargs="*")
    parser.add_argument("--candidate", action="append", dest="candidates")
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
        candidates = select_candidates(arguments.candidates, arguments.phase)
    except ValueError as error:
        parser.error(str(error))

    if arguments.phase == "screen":
        rounds = arguments.rounds or 1
        warmups = arguments.warmups or 2
        samples = arguments.samples or 9
        increment_id = "retained-horizontal-finalist-screen-v1"
    else:
        rounds = arguments.rounds or 3
        warmups = arguments.warmups or 3
        samples = arguments.samples or 21
        increment_id = "retained-horizontal-finalist-reference-v1"
    if min(rounds, warmups, samples) < 1:
        parser.error("--rounds, --warmups, and --samples must be positive")
    if not 0.0 < arguments.max_memory_fraction <= 1.0:
        parser.error("--max-memory-fraction must be in (0, 1]")

    source_commit, source_dirty = git_source_state(repository_root)
    if source_dirty and not arguments.allow_dirty_tree:
        parser.error(
            "the benchmark source tree is dirty; commit and rebuild for evidence collection or use "
            "--allow-dirty-tree for an explicitly exploratory run"
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
        f"issue7-retained-horizontal-{arguments.phase}-{timestamp}"
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
                if candidate.applicable_profile not in (None, profile):
                    continue
                stem = (
                    f"round-{round_index + 1}--{profile}--{candidate.id}"
                )
                result_path = output / f"{stem}.json"
                common = [
                    "--profile", profile,
                    "--warmups", str(warmups),
                    "--samples", str(samples),
                    "--seed", str(arguments.seed),
                    "--output", str(result_path),
                ]
                if candidate.kind == "plane-major":
                    command = [
                        str(arguments.executable), "run", "--providers", "fftw",
                        "--fftw-layout", "interleaved",
                        "--fftw-spectrum-order", "plane-major",
                        "--fftw-planning", "measure",
                        "--fftw-alignment", "unaligned",
                        "--fftw-wisdom", "cold",
                        "--fftw-internal-workers", "1",
                        "--fftw-outer-workers", str(candidate.outer_workers),
                        *common,
                    ]
                elif candidate.kind == "pruned":
                    command = [
                        str(arguments.executable), "run",
                        "--kernel", "pruned-horizontal",
                        "--providers", "fftw",
                        "--fftw-layout", "interleaved",
                        "--fftw-planning", "measure",
                        "--fftw-alignment", "unaligned",
                        "--fftw-wisdom", "cold",
                        "--fftw-internal-workers", "1",
                        "--fftw-outer-workers", str(candidate.outer_workers),
                        *common,
                    ]
                else:
                    command = [
                        str(arguments.executable), "run",
                        "--providers", "both",
                        "--workers", str(candidate.outer_workers),
                        "--vdsp-strategy", "in-place",
                        "--vdsp-batch-strategy", "direct-persistent",
                        "--fftw-layout", "interleaved",
                        "--fftw-spectrum-order", "wvm",
                        "--fftw-planning", "measure",
                        "--fftw-alignment", "unaligned",
                        "--fftw-wisdom", "cold",
                        "--fftw-internal-workers", "1",
                        "--fftw-outer-workers", "12",
                        *common,
                    ]
                commands.append(
                    (stem, round_index + 1, profile, candidate, command, result_path)
                )

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
            "Which matched Float64 implementation of the radially retained horizontal "
            "forward and inverse operator is fastest when full, pruned, provider-native, "
            "and compact representations are allowed?"
        ),
        "baseline": (
            "WVM-order FFTW MEASURE with one internal worker and 12 persistent outer workers, "
            "measured inside the outer-12 pruned comparison and linked to the issue #4 baseline."
        ),
        "changedVariables": [
            "full WVM-order, plane-major, partially pruned, or vDSP packed-split algorithm",
            "persistent outer worker count for the pruned FFTW candidate",
            "physical retained representation and whether full-spectrum permutation is elided",
        ],
        "controlledVariables": [
            "Float64 logical retained modes, radial two-thirds rule, fixture, seed, and normalization",
            "FFTW 3.3.11 build and planning effort for all FFTW candidates",
            "out-of-place retained-operator boundary and zero steady-state allocation requirement",
        ],
        "timedOperations": [
            "underlying raw provider FFT where separable from the algorithm",
            "retention and inverse embedding",
            "real-grid or representation packing and unpacking",
            "complete uninstrumented retained forward and inverse operations",
        ],
        "excludedWork": [
            "vertical projection, GEMM, modal work, nonlinear flux, and issue #13 ordering crossover",
            "Float32, GPU execution, thread affinity, and cross-Mac generalization",
            "oracle reordering, allocation, planning, and fixture generation from steady-state totals",
        ],
        "placement": (
            "FFTW complete retained calls are out-of-place. vDSP uses an in-place packed split "
            "primitive behind an out-of-place real-grid to compact-retained adapter; caller inputs "
            "are preserved and reusable native buffers are reported."
        ),
        "vdspExpansionRule": (
            "Expand beyond the two guard workloads only if at least one guard is no slower than "
            "1.25x the best matched FFTW complete retained operator in both directions."
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
        commands, start=1
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
