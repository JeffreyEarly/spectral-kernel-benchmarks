#!/usr/bin/env python3
"""Run the bounded issue #8 common-matrix Float64 vertical GEMM screen."""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path


PROFILES = (
    "wvm-historical-256-nz65-f3",
    "wvm-historical-512-nz129-f3",
)


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
        return list(PROFILES)
    unknown = sorted(set(values) - set(PROFILES))
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
    parser.add_argument("--profiles", nargs="*", help="Subset of the representative 256 and 512 profiles")
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
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--samples", type=int, default=9)
    parser.add_argument("--seed", type=int, default=129)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    arguments = parser.parse_args()

    profiles = selected_profiles(arguments.profiles)
    limits = thread_limits(arguments.thread_limits)
    selected_schedules = schedules(arguments.schedules)
    outer_workers = thread_limits(arguments.outer_workers)
    if any(schedule != "serial" for schedule in selected_schedules):
        if arguments.family != "k2-grouped":
            parser.error("outer schedules require --family k2-grouped")
        if any(limit != 1 for limit in limits):
            parser.error("outer schedules require --thread-limits 1 to prevent nested BLAS threading")
    if arguments.warmups < 1 or arguments.samples < 1:
        parser.error("--warmups and --samples must be positive")

    commands: list[tuple[str, int, str, int, list[str], Path]] = []
    for profile in profiles:
        for thread_limit in limits:
            for schedule in selected_schedules:
                workers_for_schedule = [1] if schedule == "serial" else outer_workers
                for workers in workers_for_schedule:
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
                    commands.append((stem, thread_limit, schedule, workers, command, result_path))

    if arguments.dry_run:
        for _, thread_limit, _, _, command, _ in commands:
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
        "schedules": selected_schedules,
        "outerWorkers": outer_workers,
        "warmups": arguments.warmups,
        "samples": arguments.samples,
        "seed": arguments.seed,
        "runs": [],
    }
    failed = False
    for index, (stem, thread_limit, schedule, workers, command, result_path) in enumerate(commands, start=1):
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
            "environment": {"VECLIB_MAXIMUM_THREADS": str(thread_limit)},
            "command": command,
            "exitCode": completed.returncode,
            "log": log_path.name,
        }
        if result_path.is_file():
            with result_path.open(encoding="utf-8") as stream:
                result = json.load(stream)
            entry.update(
                {"runId": result["run"]["id"], "status": result["status"], "result": result_path.name}
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
