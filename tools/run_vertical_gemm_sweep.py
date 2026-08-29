#!/usr/bin/env python3
"""Run bounded issue #8 Float64 vertical GEMM screens in isolated processes."""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path


DEFAULT_PROFILES = (
    "wvm-historical-256-nz65-f3",
    "wvm-historical-512-nz129-f3",
)

PROFILE_SHAPES = {
    "wvm-historical-256-nz65-f3": (256, 65, 3, 11_439, 2_045),
    "wvm-historical-256-nz65-f4": (256, 65, 4, 11_439, 2_045),
    "wvm-historical-512-nz129-f3": (512, 129, 3, 45_765, 7_486),
    "wvm-historical-512-nz129-f4": (512, 129, 4, 45_765, 7_486),
    "wvm-current-256-nz129-f1": (256, 129, 1, 11_439, 2_045),
    "wvm-current-256-nz129-f3": (256, 129, 3, 11_439, 2_045),
    "wvm-current-256-nz129-f4": (256, 129, 4, 11_439, 2_045),
    "wvm-current-512-nz257-f1": (512, 257, 1, 45_765, 7_486),
    "wvm-current-512-nz257-f3": (512, 257, 3, 45_765, 7_486),
    "wvm-current-512-nz257-f4": (512, 257, 4, 45_765, 7_486),
    "wvm-large-512-nz513-f4": (512, 513, 4, 45_765, 7_486),
    "wvm-large-1024-nz129-f4": (1024, 129, 4, 183_037, 27_779),
}

PREFLIGHT_BOOKKEEPING_RESERVE_BYTES = 2 * 1024**2


def sysctl_integer(name: str, fallback: int) -> int:
    if sys.platform == "darwin":
        value = ctypes.c_uint32()
        size = ctypes.c_size_t(ctypes.sizeof(value))
        libc = ctypes.CDLL(None)
        if libc.sysctlbyname(name.encode(), ctypes.byref(value), ctypes.byref(size), None, 0) == 0:
            return int(value.value)
    try:
        result = subprocess.run(
            ["sysctl", "-n", name], check=True, capture_output=True, text=True
        )
        return int(result.stdout.strip())
    except (FileNotFoundError, subprocess.CalledProcessError, ValueError):
        return fallback


def sysctl_uint64(name: str, fallback: int) -> int:
    if sys.platform == "darwin":
        value = ctypes.c_uint64()
        size = ctypes.c_size_t(ctypes.sizeof(value))
        libc = ctypes.CDLL(None)
        if libc.sysctlbyname(name.encode(), ctypes.byref(value), ctypes.byref(size), None, 0) == 0:
            return int(value.value)
    try:
        result = subprocess.run(
            ["sysctl", "-n", name], check=True, capture_output=True, text=True
        )
        return int(result.stdout.strip())
    except (FileNotFoundError, subprocess.CalledProcessError, ValueError):
        return fallback


def thread_limits(specification: str) -> list[int]:
    total = sysctl_integer("hw.physicalcpu", os.cpu_count() or 1)
    performance = sysctl_integer("hw.perflevel0.physicalcpu", total)
    aliases = {"performance": performance, "total": total}
    values: list[int] = []
    for token in specification.split(","):
        token = token.strip()
        value = aliases.get(token)
        if value is None:
            value = int(token)
        if value < 1:
            raise ValueError("thread limits must be positive")
        if value not in values:
            values.append(value)
    return values


def selected_profiles(values: list[str] | None) -> list[str]:
    if not values:
        return list(DEFAULT_PROFILES)
    unknown = sorted(set(values) - set(PROFILE_SHAPES))
    if unknown:
        raise ValueError(f"unknown profile: {', '.join(unknown)}")
    return values


def schedules(specification: str) -> list[str]:
    allowed = {"serial", "outer-static", "outer-dynamic"}
    values = [value.strip() for value in specification.split(",") if value.strip()]
    unknown = sorted(set(values) - allowed)
    if unknown:
        raise ValueError(f"unknown vertical GEMM schedule: {', '.join(unknown)}")
    if not values:
        raise ValueError("at least one vertical GEMM schedule is required")
    return list(dict.fromkeys(values))


def exact_topologies(specification: str) -> list[tuple[str, int]]:
    values: list[tuple[str, int]] = []
    for token in specification.split(","):
        token = token.strip()
        if not token:
            continue
        if token == "serial":
            topology = ("serial", 1)
        else:
            try:
                schedule, worker_specification = token.split(":", maxsplit=1)
            except ValueError as error:
                raise ValueError(
                    "non-serial topologies use schedule:workers, for example outer-static:12"
                ) from error
            if schedule not in {"outer-static", "outer-dynamic"}:
                raise ValueError(f"unknown vertical GEMM topology: {token}")
            workers = thread_limits(worker_specification)
            if len(workers) != 1:
                raise ValueError(f"topology must resolve to one worker count: {token}")
            topology = (schedule, workers[0])
        if topology not in values:
            values.append(topology)
    if not values:
        raise ValueError("at least one vertical GEMM topology is required")
    return values


def estimated_k2_explicit_peak_bytes(profile: str) -> int:
    _, nz, fields, nkl, groups = PROFILE_SHAPES[profile]
    nj = 2 * (nz - 1) // 3
    columns = nkl * fields
    family_elements = groups * nz * nj
    source = 2 * family_elements * 8
    provider_matrices = 2 * family_elements * 16 + 2 * family_elements * 8
    external_operands = (nz + nj) * columns * 16
    provider_operands = 4 * external_operands
    provider_persistent = provider_matrices + provider_operands
    construction_peak = source + provider_persistent + external_operands
    output_inspection_peak = provider_persistent + 3 * external_operands
    return max(construction_peak, output_inspection_peak) + PREFLIGHT_BOOKKEEPING_RESERVE_BYTES


def gibibytes(value: int) -> str:
    return f"{value / (1024 ** 3):.2f} GiB"


def git_source_state(repository_root: Path) -> tuple[str, bool]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=normal"],
        cwd=repository_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return commit, bool(status.strip())


def main() -> int:
    repository_root = Path(__file__).resolve().parents[1]
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--executable", type=Path, default=repository_root / "build/release/skbench")
    parser.add_argument(
        "--output",
        type=Path,
        default=repository_root / "results/local" / f"issue8-vertical-gemm-{timestamp}",
    )
    parser.add_argument("--profiles", nargs="*", help="Subset of the ten named issue #8 WVM profiles")
    parser.add_argument(
        "--family",
        choices=("common", "k2-grouped"),
        default="common",
        help="Vertical matrix family; packing remains excluded for both",
    )
    parser.add_argument(
        "--thread-limits",
        default="1,performance,total",
        help="Comma-separated VECLIB_MAXIMUM_THREADS limits or aliases",
    )
    parser.add_argument(
        "--schedules",
        default="serial",
        help="Comma-separated serial, outer-static, and outer-dynamic schedules",
    )
    parser.add_argument(
        "--outer-workers",
        default="performance,total",
        help="Comma-separated outer worker counts or aliases for non-serial schedules",
    )
    parser.add_argument(
        "--topologies",
        help="Exact comma-separated finalists such as serial,outer-static:12,outer-dynamic:16; overrides the schedule/worker cross-product",
    )
    parser.add_argument(
        "--max-memory-fraction",
        type=float,
        default=0.5,
        help="Reject a K²-grouped profile whose estimated explicit peak exceeds this fraction of physical memory",
    )
    parser.add_argument(
        "--allow-memory-risk",
        action="store_true",
        help="Run even when the explicit peak estimate exceeds --max-memory-fraction",
    )
    parser.add_argument(
        "--allow-dirty-tree",
        action="store_true",
        help="Permit exploratory runs from a dirty tree or stale binary; unsuitable for publication",
    )
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--samples", type=int, default=9)
    parser.add_argument("--seed", type=int, default=129)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    arguments = parser.parse_args()

    source_commit, source_dirty = git_source_state(repository_root)
    if source_dirty and not arguments.allow_dirty_tree:
        parser.error(
            "the benchmark source tree is dirty; commit and rebuild for evidence collection or use "
            "--allow-dirty-tree for an explicitly exploratory run"
        )

    profiles = selected_profiles(arguments.profiles)
    limits = thread_limits(arguments.thread_limits)
    if arguments.topologies:
        topologies = exact_topologies(arguments.topologies)
    else:
        selected_schedules = schedules(arguments.schedules)
        outer_workers = thread_limits(arguments.outer_workers)
        topologies = []
        for schedule in selected_schedules:
            workers_for_schedule = [1] if schedule == "serial" else outer_workers
            topologies.extend((schedule, workers) for workers in workers_for_schedule)
    if any(schedule != "serial" for schedule, _ in topologies):
        if arguments.family != "k2-grouped":
            parser.error("outer schedules require --family k2-grouped")
        if any(limit != 1 for limit in limits):
            parser.error("outer schedules require --thread-limits 1 to prevent nested BLAS threading")
    if not 0.0 < arguments.max_memory_fraction <= 1.0:
        parser.error("--max-memory-fraction must be in (0, 1]")
    if arguments.warmups < 1 or arguments.samples < 1:
        parser.error("--warmups and --samples must be positive")

    physical_memory = sysctl_uint64("hw.memsize", 0)
    estimated_peaks = {
        profile: estimated_k2_explicit_peak_bytes(profile)
        for profile in profiles
    } if arguments.family == "k2-grouped" else {profile: 0 for profile in profiles}
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
                f"{gibibytes(physical_memory)} physical memory: {details}; use --allow-memory-risk to override"
            )

    commands: list[tuple[str, int, str, int, int, list[str], Path]] = []
    for profile in profiles:
        for thread_limit in limits:
            for schedule, workers in topologies:
                stem = (
                    f"{profile}--{arguments.family}--{schedule}--outer-{workers}"
                    f"--veclib-threads-{thread_limit}"
                )
                result_path = arguments.output / f"{stem}.json"
                command = [
                    str(arguments.executable),
                    "run",
                    "--kernel",
                    "vertical-gemm",
                    "--profile",
                    profile,
                    "--vertical-gemm-family",
                    arguments.family,
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
                commands.append(
                    (
                        stem,
                        thread_limit,
                        schedule,
                        workers,
                        estimated_peaks[profile],
                        command,
                        result_path,
                    )
                )

    if arguments.dry_run:
        for profile in profiles:
            print(
                f"estimated explicit peak {profile}: {gibibytes(estimated_peaks[profile])}"
                + (f" / {gibibytes(physical_memory)} physical" if physical_memory else "")
            )
        for _, thread_limit, _, _, _, command, _ in commands:
            print(f"VECLIB_MAXIMUM_THREADS={thread_limit} {' '.join(command)}")
        print(f"Planned {len(commands)} isolated run(s); thread limits={limits}.")
        return 0

    arguments.output.mkdir(parents=True, exist_ok=False)
    manifest = {
        "schema": "spectral-kernel-local-sweep-v1",
        "experimentId": "issue-008-vertical-projection-gemm",
        "createdAtUtc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "scope": f"bounded {arguments.family} Float64 vertical projection; packing excluded",
        "verticalGemmFamily": arguments.family,
        "profiles": profiles,
        "threadEnvironment": "VECLIB_MAXIMUM_THREADS",
        "threadLimits": limits,
        "topologies": [
            {"schedule": schedule, "outerWorkers": workers}
            for schedule, workers in topologies
        ],
        "physicalMemoryBytes": physical_memory,
        "maxMemoryFraction": arguments.max_memory_fraction,
        "estimatedExplicitPeakBytesByProfile": estimated_peaks,
        "sourceTreeGitCommit": source_commit,
        "sourceTreeDirty": source_dirty,
        "warmups": arguments.warmups,
        "samples": arguments.samples,
        "seed": arguments.seed,
        "runs": [],
    }
    failed = False
    for index, (
        stem,
        thread_limit,
        schedule,
        workers,
        estimated_peak,
        command,
        result_path,
    ) in enumerate(commands, start=1):
        print(f"[{index}/{len(commands)}] {stem}", flush=True)
        log_path = arguments.output / f"{stem}.log"
        environment = os.environ.copy()
        environment["VECLIB_MAXIMUM_THREADS"] = str(thread_limit)
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
            "threadLimit": thread_limit,
            "schedule": schedule,
            "outerWorkers": workers,
            "estimatedExplicitPeakBytes": estimated_peak,
            "environment": {"VECLIB_MAXIMUM_THREADS": str(thread_limit)},
            "command": command,
            "exitCode": completed.returncode,
            "log": log_path.name,
            "sourceTreeGitCommit": source_commit,
            "sourceTreeDirty": source_dirty,
        }
        if result_path.is_file():
            with result_path.open(encoding="utf-8") as stream:
                result = json.load(stream)
            entry.update(
                {
                    "runId": result["run"]["id"],
                    "status": result["status"],
                    "result": result_path.name,
                    "reportedEstimatedExplicitPeakBytes": result["workload"]["bytes"].get(
                        "verticalBenchmarkEstimatedExplicitPeak"
                    ),
                }
            )
            embedded_commit = result["environment"].get("gitCommit", "")
            embedded_dirty = result["environment"].get("gitDirty")
            metadata_matches = (
                bool(embedded_commit)
                and embedded_commit != "unknown"
                and source_commit.startswith(embedded_commit)
                and embedded_dirty == source_dirty
            )
            entry["sourceMetadataMatches"] = metadata_matches
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
