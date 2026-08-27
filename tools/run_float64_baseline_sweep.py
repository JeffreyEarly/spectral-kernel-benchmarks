#!/usr/bin/env python3
"""Run the issue #3/#5 Float64 workload, placement, scratch, and worker matrix."""

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
    "wvm-historical-256-nz65-f4",
    "wvm-historical-512-nz129-f3",
    "wvm-historical-512-nz129-f4",
    "wvm-current-256-nz129-f1",
    "wvm-current-256-nz129-f3",
    "wvm-current-256-nz129-f4",
    "wvm-current-512-nz257-f1",
    "wvm-current-512-nz257-f3",
    "wvm-current-512-nz257-f4",
)

STRATEGIES = (
    "in-place",
    "in-place-explicit-scratch",
    "out-of-place",
    "out-of-place-explicit-scratch",
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


def worker_counts(specification: str) -> list[int]:
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
            raise ValueError("worker counts must be positive")
        if value not in values:
            values.append(value)
    return values


def selected(values: list[str] | None, available: tuple[str, ...], label: str) -> list[str]:
    if not values:
        return list(available)
    unknown = sorted(set(values) - set(available))
    if unknown:
        raise ValueError(f"unknown {label}: {', '.join(unknown)}")
    return values


def main() -> int:
    repository_root = Path(__file__).resolve().parents[1]
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--executable", type=Path, default=repository_root / "build/release/skbench")
    parser.add_argument("--output", type=Path, default=repository_root / "results/local" / f"float64-baseline-{timestamp}")
    parser.add_argument("--profiles", nargs="*", help="Subset of the ten exact WVM baseline profiles")
    parser.add_argument("--strategies", nargs="*", help="Subset of the four native vDSP strategies")
    parser.add_argument("--workers", default="1,performance,total", help="Comma-separated counts or aliases")
    parser.add_argument("--warmups", type=int, help="Override each profile's warm-up count")
    parser.add_argument("--samples", type=int, help="Override each profile's sample count")
    parser.add_argument("--seed", type=int, default=129)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    arguments = parser.parse_args()

    profiles = selected(arguments.profiles, PROFILES, "profile")
    strategies = selected(arguments.strategies, STRATEGIES, "strategy")
    workers = worker_counts(arguments.workers)
    if arguments.warmups is not None and arguments.warmups < 1:
        parser.error("--warmups must be positive")
    if arguments.samples is not None and arguments.samples < 1:
        parser.error("--samples must be positive")

    commands: list[tuple[str, list[str], Path]] = []
    for profile in profiles:
        for strategy in strategies:
            for worker_count in workers:
                stem = f"{profile}--{strategy}--workers-{worker_count}"
                result_path = arguments.output / f"{stem}.json"
                command = [
                    str(arguments.executable),
                    "run",
                    "--profile",
                    profile,
                    "--vdsp-strategy",
                    strategy,
                    "--workers",
                    str(worker_count),
                    "--seed",
                    str(arguments.seed),
                    "--output",
                    str(result_path),
                ]
                if arguments.warmups is not None:
                    command.extend(("--warmups", str(arguments.warmups)))
                if arguments.samples is not None:
                    command.extend(("--samples", str(arguments.samples)))
                commands.append((stem, command, result_path))

    if arguments.dry_run:
        for _, command, _ in commands:
            print(" ".join(command))
        print(f"Planned {len(commands)} run(s); workers={workers}.")
        return 0

    arguments.output.mkdir(parents=True, exist_ok=False)
    manifest = {
        "schema": "spectral-kernel-local-sweep-v1",
        "createdAtUtc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "profiles": profiles,
        "strategies": strategies,
        "workers": workers,
        "runs": [],
    }
    failed = False
    for index, (stem, command, result_path) in enumerate(commands, start=1):
        print(f"[{index}/{len(commands)}] {stem}", flush=True)
        completed = subprocess.run(command, cwd=repository_root)
        entry = {"id": stem, "command": command, "exitCode": completed.returncode}
        if result_path.is_file():
            with result_path.open(encoding="utf-8") as stream:
                result = json.load(stream)
            entry.update({"runId": result["run"]["id"], "status": result["status"], "result": result_path.name})
        manifest["runs"].append(entry)
        with (arguments.output / "manifest.json").open("w", encoding="utf-8") as stream:
            json.dump(manifest, stream, indent=2)
            stream.write("\n")
        if completed.returncode != 0:
            failed = True
            if not arguments.continue_on_error:
                break
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
