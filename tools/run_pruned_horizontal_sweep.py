#!/usr/bin/env python3
"""Run the bounded issue #12 partial-column-pruned FFTW comparison."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

from run_vertical_gemm_sweep import (
    PROFILE_SHAPES,
    gibibytes,
    git_source_state,
    sysctl_uint64,
    thread_limits,
)


DEFAULT_PROFILES = (
    "wvm-current-256-nz129-f1",
    "wvm-historical-256-nz65-f3",
    "wvm-current-256-nz129-f4",
    "wvm-current-512-nz257-f1",
    "wvm-historical-512-nz129-f3",
    "wvm-historical-512-nz129-f4",
)


def selected_profiles(values: list[str] | None) -> list[str]:
    profiles = list(DEFAULT_PROFILES) if not values else values
    unknown = sorted(set(profiles) - set(PROFILE_SHAPES))
    if unknown:
        raise ValueError(f"unknown profile: {', '.join(unknown)}")
    return profiles


def estimated_explicit_peak_bytes(profile: str) -> int:
    nx, nz, fields, nkl, _ = PROFILE_SHAPES[profile]
    ny = nx
    planes = nz * fields
    real = nx * ny * planes * 8
    full = (nx // 2 + 1) * ny * planes * 16
    retained = nkl * planes * 16
    return 6 * real + 4 * full + 3 * retained + 2 * 1024**2


def main() -> int:
    repository_root = Path(__file__).resolve().parents[1]
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--executable", type=Path, default=repository_root / "build/release/skbench"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=repository_root / "results/local" / f"issue12-pruned-horizontal-{timestamp}",
    )
    parser.add_argument("--profiles", nargs="*", help="Subset of named WVM profiles")
    parser.add_argument(
        "--workers",
        default="1,performance",
        help="Comma-separated FFTW internal worker counts or performance/total aliases",
    )
    parser.add_argument(
        "--planning", choices=("estimate", "measure", "patient", "exhaustive"),
        default="measure",
    )
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--samples", type=int, default=9)
    parser.add_argument("--seed", type=int, default=129)
    parser.add_argument("--max-memory-fraction", type=float, default=0.5)
    parser.add_argument("--allow-memory-risk", action="store_true")
    parser.add_argument("--allow-dirty-tree", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    arguments = parser.parse_args()

    source_commit, source_dirty = git_source_state(repository_root)
    if source_dirty and not arguments.allow_dirty_tree:
        parser.error(
            "the benchmark source tree is dirty; commit and rebuild for evidence collection or use "
            "--allow-dirty-tree for an explicitly exploratory run"
        )
    if arguments.warmups < 1 or arguments.samples < 1:
        parser.error("--warmups and --samples must be positive")
    if not 0.0 < arguments.max_memory_fraction <= 1.0:
        parser.error("--max-memory-fraction must be in (0, 1]")
    try:
        profiles = selected_profiles(arguments.profiles)
        workers = thread_limits(arguments.workers)
    except ValueError as error:
        parser.error(str(error))

    physical_memory = sysctl_uint64("hw.memsize", 0)
    estimated_peaks = {profile: estimated_explicit_peak_bytes(profile) for profile in profiles}
    if physical_memory > 0 and not arguments.allow_memory_risk:
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
                f"{gibibytes(physical_memory)} physical memory: {details}; "
                "use --allow-memory-risk to override"
            )

    commands: list[tuple[str, str, int, int, list[str], Path]] = []
    for profile in profiles:
        for worker_count in workers:
            stem = f"{profile}--partial-column-pruned--internal-{worker_count}"
            result_path = arguments.output / f"{stem}.json"
            command = [
                str(arguments.executable),
                "run",
                "--kernel",
                "pruned-horizontal",
                "--profile",
                profile,
                "--providers",
                "fftw",
                "--fftw-layout",
                "interleaved",
                "--fftw-planning",
                arguments.planning,
                "--fftw-alignment",
                "unaligned",
                "--fftw-wisdom",
                "cold",
                "--fftw-internal-workers",
                str(worker_count),
                "--fftw-outer-workers",
                "1",
                "--warmups",
                str(arguments.warmups),
                "--samples",
                str(arguments.samples),
                "--seed",
                str(arguments.seed),
                "--output",
                str(result_path),
            ]
            commands.append(
                (stem, profile, worker_count, estimated_peaks[profile], command, result_path)
            )

    if arguments.dry_run:
        for profile in profiles:
            print(
                f"estimated explicit peak {profile}: {gibibytes(estimated_peaks[profile])}"
                + (f" / {gibibytes(physical_memory)} physical" if physical_memory else "")
            )
        for _, _, _, _, command, _ in commands:
            print(" ".join(command))
        print(f"Planned {len(commands)} isolated run(s).")
        return 0

    arguments.output.mkdir(parents=True, exist_ok=False)
    manifest = {
        "schema": "spectral-kernel-local-sweep-v1",
        "experimentId": "issue-012-pruned-horizontal-transforms",
        "incrementId": "fftw-partial-column-pruned-v1",
        "createdAtUtc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "question": (
            "Can a separable FFTW transform that omits high-kx complex column FFTs beat "
            "the matched optimized full 2-D FFTW transform plus radial selection?"
        ),
        "baseline": (
            "Same-run FFTW 3.3.11 full 2-D guru64 transform plus mode-keyed radial "
            "selection or embedding, using the same planning mode and internal worker count."
        ),
        "changedVariables": [
            "integrated full 2-D plan versus row R2C/C2R plus selected-kx column C2C plans",
            "WVM full-spectrum output versus plane-major row-spectrum scratch and compact retained output",
        ],
        "controlledVariables": [
            "FFTW 3.3.11 build and compiler flags",
            "logical retained modes, normalization, fixture, precision, planning effort, and workers",
            "warmups, samples, seed, and out-of-place caller contract within each run",
        ],
        "timedOperations": [
            "full raw 2-D forward and inverse FFT",
            "reference radial selection and embedding",
            "candidate real-row transforms",
            "candidate selected-kx complex-column transforms",
            "candidate direct retained selection and inverse embedding",
            "complete uninstrumented retained forward and inverse operators",
        ],
        "excludedWork": [
            "vertical projection, GEMM, modal work, and nonlinear flux calculation",
            "deeper within-column output pruning and transform-internal transposition",
            "outer batch sharding, split-complex storage, and in-place retained operation",
        ],
        "placement": (
            "Both complete retained operators are out-of-place. The candidate uses in-place "
            "selected column transforms inside private full-sized plane-major scratch."
        ),
        "profiles": profiles,
        "internalWorkers": workers,
        "planning": arguments.planning,
        "physicalMemoryBytes": physical_memory,
        "estimatedExplicitPeakBytesByProfile": estimated_peaks,
        "sourceTreeGitCommit": source_commit,
        "sourceTreeDirty": source_dirty,
        "warmups": arguments.warmups,
        "samples": arguments.samples,
        "seed": arguments.seed,
        "runs": [],
    }

    failed = False
    for index, (stem, profile, worker_count, estimated_peak, command, result_path) in enumerate(
        commands, start=1
    ):
        print(f"[{index}/{len(commands)}] {stem}", flush=True)
        log_path = arguments.output / f"{stem}.log"
        with log_path.open("w", encoding="utf-8") as log:
            completed = subprocess.run(
                command, cwd=repository_root, stdout=log, stderr=subprocess.STDOUT
            )
        entry = {
            "id": stem,
            "profile": profile,
            "internalWorkers": worker_count,
            "estimatedExplicitPeakBytes": estimated_peak,
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
                completed = subprocess.CompletedProcess(command, 1)
                print(
                    f"binary/source metadata mismatch for {stem}: "
                    f"source={source_commit} dirty={source_dirty}, "
                    f"binary={embedded_commit} dirty={embedded_dirty}",
                    file=sys.stderr,
                )
        manifest["runs"].append(entry)
        with (arguments.output / "manifest.json").open("w", encoding="utf-8") as stream:
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
