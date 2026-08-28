#!/usr/bin/env python3
"""Run the issue #4 native-order screen or repeated production reference campaign."""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

from run_vertical_gemm_sweep import git_source_state


SCREEN_PROFILES = (
    "wvm-historical-256-nz65-f3",
    "wvm-historical-512-nz129-f3",
)

REFERENCE_PROFILES = (
    "wvm-current-256-nz129-f1",
    "wvm-historical-256-nz65-f3",
    "wvm-current-256-nz129-f4",
    "wvm-current-512-nz257-f1",
    "wvm-historical-512-nz129-f3",
    "wvm-historical-512-nz129-f4",
)


@dataclass(frozen=True)
class Candidate:
    id: str
    spectrum_order: str
    layout: str
    planning: str
    alignment: str
    internal_workers: int
    outer_workers: int

    @property
    def total_workers(self) -> int:
        return self.internal_workers * self.outer_workers


def sysctl_integer(name: str, fallback: int) -> int:
    if sys.platform == "darwin":
        value = ctypes.c_uint32()
        size = ctypes.c_size_t(ctypes.sizeof(value))
        libc = ctypes.CDLL(None)
        if libc.sysctlbyname(
            name.encode(), ctypes.byref(value), ctypes.byref(size), None, 0
        ) == 0:
            return int(value.value)
    return fallback


def candidate_matrix(performance_workers: int, total_workers: int) -> list[Candidate]:
    if performance_workers < 4 or performance_workers % 4 != 0:
        raise ValueError("performance workers must be positive and divisible by four")
    if total_workers < performance_workers:
        raise ValueError("total workers must be at least the performance-worker count")
    topologies = (
        ("estimate-internal-performance", "estimate", "unaligned", performance_workers, 1),
        ("measure-outer-performance", "measure", "unaligned", 1, performance_workers),
        ("measure-outer-total", "measure", "unaligned", 1, total_workers),
        ("measure-hybrid-4", "measure", "unaligned", 4, performance_workers // 4),
    )
    candidates: list[Candidate] = []
    for order in ("wvm", "plane-major"):
        layouts = ("interleaved",) if order == "wvm" else ("interleaved", "split")
        for layout in layouts:
            for topology, planning, alignment, internal, outer in topologies:
                candidates.append(
                    Candidate(
                        f"{order}-{layout}-{topology}",
                        order,
                        layout,
                        planning,
                        alignment,
                        internal,
                        outer,
                    )
                )
    return candidates


def select_candidates(
    requested: list[str] | None, available: list[Candidate], require_explicit: bool
) -> list[Candidate]:
    by_id = {candidate.id: candidate for candidate in available}
    if not requested:
        if require_explicit:
            raise ValueError("reference campaigns require at least one explicit --candidate")
        return available
    unknown = sorted(set(requested) - set(by_id))
    if unknown:
        raise ValueError(f"unknown candidate: {', '.join(unknown)}")
    return [by_id[candidate_id] for candidate_id in requested]


def rotated(values: list[Candidate], offset: int) -> list[Candidate]:
    if not values:
        return []
    shift = offset % len(values)
    return values[shift:] + values[:shift]


def main() -> int:
    repository_root = Path(__file__).resolve().parents[1]
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    default_total = os.cpu_count() or 1
    default_performance = sysctl_integer("hw.perflevel0.physicalcpu", default_total)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("screen", "reference"), default="screen")
    parser.add_argument(
        "--executable", type=Path, default=repository_root / "build/release/skbench"
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--profiles", nargs="*")
    parser.add_argument("--candidate", action="append", dest="candidates")
    parser.add_argument("--performance-workers", type=int, default=default_performance)
    parser.add_argument("--total-workers", type=int, default=default_total)
    parser.add_argument("--rounds", type=int)
    parser.add_argument("--warmups", type=int)
    parser.add_argument("--samples", type=int)
    parser.add_argument("--seed", type=int, default=129)
    parser.add_argument("--allow-dirty-tree", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    arguments = parser.parse_args()

    if arguments.phase == "screen":
        default_profiles = list(SCREEN_PROFILES)
        default_rounds, default_warmups, default_samples = 1, 2, 9
        increment_id = "fftw-native-spectrum-order-screen-v2"
    else:
        default_profiles = list(REFERENCE_PROFILES)
        default_rounds, default_warmups, default_samples = 3, 3, 21
        increment_id = "fftw-production-pareto-reference-v3"
    profiles = arguments.profiles or default_profiles
    rounds = arguments.rounds or default_rounds
    warmups = arguments.warmups or default_warmups
    samples = arguments.samples or default_samples
    if rounds < 1 or warmups < 1 or samples < 1:
        parser.error("--rounds, --warmups, and --samples must be positive")

    known_profiles = set(SCREEN_PROFILES) | set(REFERENCE_PROFILES)
    unknown_profiles = sorted(set(profiles) - known_profiles)
    if unknown_profiles:
        parser.error(f"unknown profile: {', '.join(unknown_profiles)}")
    try:
        available = candidate_matrix(
            arguments.performance_workers, arguments.total_workers
        )
        candidates = select_candidates(
            arguments.candidates,
            available,
            require_explicit=arguments.phase == "reference",
        )
    except ValueError as error:
        parser.error(str(error))

    source_commit, source_dirty = git_source_state(repository_root)
    if source_dirty and not arguments.allow_dirty_tree:
        parser.error(
            "the benchmark source tree is dirty; commit and rebuild for evidence collection or use "
            "--allow-dirty-tree for an explicitly exploratory run"
        )
    output = arguments.output or (
        repository_root / "results/local" / f"issue4-fftw-{arguments.phase}-{timestamp}"
    )

    commands: list[tuple[str, int, str, Candidate, list[str], Path]] = []
    for round_index in range(rounds):
        round_candidates = rotated(candidates, round_index)
        round_profiles = profiles[round_index % len(profiles):] + profiles[:round_index % len(profiles)]
        for profile in round_profiles:
            for candidate in round_candidates:
                stem = f"round-{round_index + 1}--{profile}--{candidate.id}"
                result_path = output / f"{stem}.json"
                command = [
                    str(arguments.executable),
                    "run",
                    "--profile",
                    profile,
                    "--providers",
                    "fftw",
                    "--fftw-layout",
                    candidate.layout,
                    "--fftw-spectrum-order",
                    candidate.spectrum_order,
                    "--fftw-planning",
                    candidate.planning,
                    "--fftw-alignment",
                    candidate.alignment,
                    "--fftw-wisdom",
                    "cold",
                    "--fftw-internal-workers",
                    str(candidate.internal_workers),
                    "--fftw-outer-workers",
                    str(candidate.outer_workers),
                    "--warmups",
                    str(warmups),
                    "--samples",
                    str(samples),
                    "--seed",
                    str(arguments.seed),
                    "--output",
                    str(result_path),
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
        "experimentId": "issue-004-fftw-strategy-sweep",
        "incrementId": increment_id,
        "phase": arguments.phase,
        "createdAtUtc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "question": (
            "Does a provider-native plane-major half-spectrum, with or without split complex, "
            "improve raw FFT or retained-operator performance enough to join the final FFTW Pareto set?"
            if arguments.phase == "screen"
            else "Which screened FFTW strategies form the production-workload Pareto set when independently planned processes and deeper steady-state samples are combined?"
        ),
        "baseline": (
            "The immutable issue #4 WVM-frequency-major planning, scheduling, and paired split/interleaved increments."
        ),
        "changedVariables": [
            "native half-spectrum order",
            "interleaved or split complex layout",
            "planning and worker topology",
        ],
        "controlledVariables": [
            "FFTW 3.3.11 build, Float64 precision, fixture, seed, normalization, and logical operator",
            "radial two-thirds horizontal retention and mode-keyed correctness oracle",
            "out-of-place raw transform contract and zero steady-state allocation requirement",
        ],
        "timedOperations": [
            "raw native forward and inverse FFT",
            "native/WVM order permutation",
            "split/interleaved conversion",
            "WVM-compatible full-spectrum adapter",
            "native-order retained selection and embedding",
            "complete retained forward and inverse operators",
            "empty persistent scheduler dispatch",
        ],
        "excludedWork": [
            "vertical projection, GEMM, modal work, and nonlinear flux calculation",
            "thread affinity, GPU execution, Float32, and generated FFT kernels",
            "allocation and planning from steady-state timings",
        ],
        "placement": (
            "Every measured FFTW path is out-of-place. Split plans use one contiguous "
            "[real][imaginary] allocation and multidimensional split in-place is recorded as unsupported."
        ),
        "referenceProtocol": {
            "independentlyPlannedProcesses": rounds,
            "candidateOrderRotatedByRound": True,
            "profileOrderRotatedByRound": True,
            "warmupsPerProcess": warmups,
            "samplesPerProcess": samples,
            "aggregateRule": "pool samples only within an identical workload/candidate signature; report process medians and pooled intervals",
        },
        "profiles": profiles,
        "performanceWorkers": arguments.performance_workers,
        "totalWorkers": arguments.total_workers,
        "rounds": rounds,
        "warmups": warmups,
        "samples": samples,
        "seed": arguments.seed,
        "sourceTreeGitCommit": source_commit,
        "sourceTreeDirty": source_dirty,
        "candidates": [
            asdict(candidate) | {"total_workers": candidate.total_workers}
            for candidate in candidates
        ],
        "runs": [],
    }

    failed = False
    for index, (stem, round_number, profile, candidate, command, result_path) in enumerate(
        commands, start=1
    ):
        print(f"[{index}/{len(commands)}] {stem}", flush=True)
        log_path = output / f"{stem}.log"
        with log_path.open("w", encoding="utf-8") as log:
            completed = subprocess.run(
                command, cwd=repository_root, stdout=log, stderr=subprocess.STDOUT
            )
        entry = {
            "id": stem,
            "round": round_number,
            "profile": profile,
            "candidate": asdict(candidate) | {"total_workers": candidate.total_workers},
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
            entry.update(
                {
                    "runId": result["run"]["id"],
                    "status": result["status"],
                    "result": result_path.name,
                    "embeddedGitCommit": embedded_commit,
                    "embeddedGitDirty": embedded_dirty,
                    "sourceMetadataMatches": metadata_matches,
                }
            )
            if not metadata_matches:
                entry["exitCode"] = 3
                completed = subprocess.CompletedProcess(command, 3)
                print(
                    f"binary/source metadata mismatch for {stem}: "
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
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
