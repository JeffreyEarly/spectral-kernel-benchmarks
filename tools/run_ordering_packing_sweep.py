#!/usr/bin/env python3
"""Run the bounded issue #13 MATLAB-style ordering/packing baseline."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

from run_vertical_gemm_sweep import (
    PROFILE_SHAPES,
    exact_topologies,
    gibibytes,
    git_source_state,
    sysctl_uint64,
)


DEFAULT_PROFILES = (
    "wvm-historical-256-nz65-f3",
    "wvm-current-256-nz129-f1",
    "wvm-current-256-nz129-f3",
    "wvm-current-256-nz129-f4",
)


def selected_profiles(values: list[str] | None) -> list[str]:
    profiles = list(DEFAULT_PROFILES) if not values else values
    unknown = sorted(set(profiles) - set(PROFILE_SHAPES))
    if unknown:
        raise ValueError(f"unknown profile: {', '.join(unknown)}")
    return profiles


def estimated_explicit_peak_bytes(profile: str) -> int:
    nx, nz, fields, nkl, groups = PROFILE_SHAPES[profile]
    ny = nx
    nj = 2 * (nz - 1) // 3
    columns = nkl * fields
    family_elements = groups * nz * nj
    source_matrices = 2 * family_elements * 8
    provider_matrices = 2 * family_elements * 16 + 2 * family_elements * 8
    physical = nz * columns * 16
    modal = nj * columns * 16
    full = ny * (nx // 2 + 1) * nz * fields * 16
    provider_operands = 4 * (physical + modal)
    bookkeeping_reserve = 2 * 1024**2
    construction = source_matrices + provider_matrices + provider_operands + physical + modal + full
    inspection = provider_matrices + provider_operands + 5 * physical + 3 * modal + 4 * full
    return max(construction, inspection) + bookkeeping_reserve


def main() -> int:
    repository_root = Path(__file__).resolve().parents[1]
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--executable", type=Path, default=repository_root / "build/release/skbench")
    parser.add_argument(
        "--output",
        type=Path,
        default=repository_root / "results/local" / f"issue13-ordering-packing-{timestamp}",
    )
    parser.add_argument("--profiles", nargs="*", help="Subset of named WVM profiles")
    parser.add_argument(
        "--topologies",
        default="outer-static:12,outer-dynamic:16",
        help="Exact comma-separated schedule:worker candidates",
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
    if not 0.0 < arguments.max_memory_fraction <= 1.0:
        parser.error("--max-memory-fraction must be in (0, 1]")
    if arguments.warmups < 1 or arguments.samples < 1:
        parser.error("--warmups and --samples must be positive")

    profiles = selected_profiles(arguments.profiles)
    topologies = exact_topologies(arguments.topologies)
    if any(schedule == "serial" for schedule, _ in topologies):
        parser.error("this bounded issue #13 sweep compares only outer-static and outer-dynamic finalists")
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
        for schedule, workers in topologies:
            stem = f"{profile}--matlab-radial-pack--{schedule}--outer-{workers}"
            result_path = arguments.output / f"{stem}.json"
            command = [
                str(arguments.executable),
                "run",
                "--kernel",
                "ordering-packing",
                "--profile",
                profile,
                "--vertical-gemm-family",
                "k2-grouped",
                "--vertical-gemm-schedule",
                schedule,
                "--vertical-gemm-outer-workers",
                str(workers),
                "--warmups",
                str(arguments.warmups),
                "--samples",
                str(arguments.samples),
                "--seed",
                str(arguments.seed),
                "--output",
                str(result_path),
            ]
            commands.append((stem, schedule, workers, estimated_peaks[profile], command, result_path))

    if arguments.dry_run:
        for profile in profiles:
            print(
                f"estimated explicit peak {profile}: {gibibytes(estimated_peaks[profile])}"
                + (f" / {gibibytes(physical_memory)} physical" if physical_memory else "")
            )
        for _, _, _, _, command, _ in commands:
            print(f"VECLIB_MAXIMUM_THREADS=1 {' '.join(command)}")
        print(f"Planned {len(commands)} isolated run(s).")
        return 0

    arguments.output.mkdir(parents=True, exist_ok=False)
    manifest = {
        "schema": "spectral-kernel-local-sweep-v1",
        "experimentId": "issue-013-ordering-packing-crossover",
        "createdAtUtc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "scope": "bounded Float64 MATLAB-style radial gather/embed plus K2 vertical projection; FFT excluded",
        "profiles": profiles,
        "topologies": [
            {"schedule": schedule, "outerWorkers": workers}
            for schedule, workers in topologies
        ],
        "threadEnvironment": "VECLIB_MAXIMUM_THREADS",
        "threadLimit": 1,
        "reuseCounts": [2, 4, 8],
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
    for index, (stem, schedule, workers, estimated_peak, command, result_path) in enumerate(
        commands, start=1
    ):
        print(f"[{index}/{len(commands)}] {stem}", flush=True)
        log_path = arguments.output / f"{stem}.log"
        environment = os.environ.copy()
        environment["VECLIB_MAXIMUM_THREADS"] = "1"
        with log_path.open("w", encoding="utf-8") as log:
            completed = subprocess.run(
                command,
                cwd=repository_root,
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
        entry = {
            "id": stem,
            "threadLimit": 1,
            "schedule": schedule,
            "outerWorkers": workers,
            "estimatedExplicitPeakBytes": estimated_peak,
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
            entry.update(
                {
                    "runId": result["run"]["id"],
                    "status": result["status"],
                    "result": result_path.name,
                    "reportedEstimatedExplicitPeakBytes": result["workload"]["bytes"].get(
                        "orderingPackingEstimatedExplicitPeak"
                    ),
                    "sourceMetadataMatches": metadata_matches,
                }
            )
            if not metadata_matches and not arguments.allow_dirty_tree:
                entry["exitCode"] = 3
                completed = subprocess.CompletedProcess(command, 3)
                print(
                    f"source metadata mismatch: tree={source_commit[:12]} dirty={source_dirty}, "
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
