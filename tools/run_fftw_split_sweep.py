#!/usr/bin/env python3
"""Run the bounded issue #4 paired FFTW split/interleaved increment."""

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


PROFILES = (
    "wvm-historical-256-nz65-f3",
    "wvm-historical-512-nz129-f3",
)


@dataclass(frozen=True)
class Candidate:
    id: str
    planning: str
    alignment: str
    internal_workers: int
    outer_workers: int


def sysctl_integer(name: str, fallback: int) -> int:
    if sys.platform == "darwin":
        value = ctypes.c_uint32()
        size = ctypes.c_size_t(ctypes.sizeof(value))
        libc = ctypes.CDLL(None)
        if libc.sysctlbyname(name.encode(), ctypes.byref(value), ctypes.byref(size), None, 0) == 0:
            return int(value.value)
    return fallback


def candidates(performance_workers: int, total_workers: int) -> list[Candidate]:
    if performance_workers < 1 or total_workers < 1:
        raise ValueError("worker counts must be positive")
    if performance_workers % 4 != 0:
        raise ValueError("the bounded hybrid candidate requires a performance-worker count divisible by four")
    return [
        Candidate(f"estimate-unaligned-internal-{performance_workers}", "estimate", "unaligned", performance_workers, 1),
        Candidate(f"measure-unaligned-outer-{performance_workers}", "measure", "unaligned", 1, performance_workers),
        Candidate(f"measure-unaligned-outer-{total_workers}", "measure", "unaligned", 1, total_workers),
        Candidate(
            f"measure-unaligned-hybrid-4x{performance_workers // 4}",
            "measure",
            "unaligned",
            4,
            performance_workers // 4,
        ),
        Candidate(f"measure-aligned-internal-{performance_workers}", "measure", "aligned", performance_workers, 1),
    ]


def main() -> int:
    repository_root = Path(__file__).resolve().parents[1]
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    default_total = os.cpu_count() or 1
    default_performance = sysctl_integer("hw.perflevel0.physicalcpu", default_total)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--executable", type=Path, default=repository_root / "build/release/skbench")
    parser.add_argument(
        "--output",
        type=Path,
        default=repository_root / "results/local" / f"issue4-fftw-split-{timestamp}",
    )
    parser.add_argument("--profiles", nargs="*", choices=PROFILES, default=list(PROFILES))
    parser.add_argument("--performance-workers", type=int, default=default_performance)
    parser.add_argument("--total-workers", type=int, default=default_total)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--samples", type=int, default=9)
    parser.add_argument("--seed", type=int, default=129)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    arguments = parser.parse_args()
    if arguments.warmups < 1 or arguments.samples < 1:
        parser.error("--warmups and --samples must be positive")
    try:
        matrix = candidates(arguments.performance_workers, arguments.total_workers)
    except ValueError as error:
        parser.error(str(error))

    commands: list[tuple[str, Candidate, list[str], Path]] = []
    for profile in arguments.profiles:
        for candidate in matrix:
            stem = f"{profile}--{candidate.id}"
            result_path = arguments.output / f"{stem}.json"
            command = [
                str(arguments.executable),
                "run",
                "--profile",
                profile,
                "--providers",
                "fftw",
                "--fftw-layout",
                "paired",
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
                str(arguments.warmups),
                "--samples",
                str(arguments.samples),
                "--seed",
                str(arguments.seed),
                "--output",
                str(result_path),
            ]
            commands.append((stem, candidate, command, result_path))

    if arguments.dry_run:
        for _, _, command, _ in commands:
            print(" ".join(command))
        print(f"Planned {len(commands)} paired run(s).")
        return 0

    arguments.output.mkdir(parents=True, exist_ok=False)
    manifest = {
        "schema": "spectral-kernel-local-sweep-v1",
        "experimentId": "issue-004-fftw-strategy-sweep",
        "incrementId": "fftw-split-versus-interleaved-v1",
        "createdAtUtc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "question": "Can FFTW's split guru64 API beat the matched interleaved API before or after WVM-compatible conversion and retained-mode selection?",
        "baseline": "The preliminary issue #4 interleaved planning/alignment/scheduling screen at commit c696065.",
        "changedVariables": ["FFTW interleaved versus split API and physical complex layout"],
        "controlledVariables": [
            "FFTW 3.3.11 build and compiler flags",
            "logical WVM transform and frequency-major mode order",
            "planning, alignment, worker topology, fixture, warmups, and samples within each paired run",
        ],
        "timedOperations": [
            "raw forward and inverse FFT",
            "split/interleaved conversion",
            "direct retained-mode selection and embedding",
            "WVM-compatible full-spectrum adapter",
            "persistent split retained operator",
            "empty scheduler dispatch",
        ],
        "excludedWork": [
            "allocation and planning from steady-state timings",
            "vertical projection, GEMM, modal work, and nonlinear flux calculation",
        ],
        "profiles": arguments.profiles,
        "performanceWorkers": arguments.performance_workers,
        "totalWorkers": arguments.total_workers,
        "warmups": arguments.warmups,
        "samples": arguments.samples,
        "seed": arguments.seed,
        "candidates": [asdict(candidate) for candidate in matrix],
        "runs": [],
    }

    failed = False
    for index, (stem, candidate, command, result_path) in enumerate(commands, start=1):
        print(f"[{index}/{len(commands)}] {stem}", flush=True)
        log_path = arguments.output / f"{stem}.log"
        with log_path.open("w", encoding="utf-8") as log:
            completed = subprocess.run(command, cwd=repository_root, stdout=log, stderr=subprocess.STDOUT)
        entry = {
            "id": stem,
            "candidate": asdict(candidate),
            "command": command,
            "exitCode": completed.returncode,
            "log": log_path.name,
        }
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
            print(log_path.read_text(encoding="utf-8")[-4000:], file=sys.stderr)
            if not arguments.continue_on_error:
                break
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
