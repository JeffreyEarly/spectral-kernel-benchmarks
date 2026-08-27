#!/usr/bin/env python3
"""Run the bounded issue #4 FFTW planning, alignment, wisdom, and scheduling screen."""

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
    phase: str
    planning: str
    alignment: str
    wisdom: str
    internal_workers: int
    outer_workers: int
    planning_time_limit_seconds: float = 0.0

    @property
    def total_workers(self) -> int:
        return self.internal_workers * self.outer_workers


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


def scheduling_candidates(workers: list[int]) -> list[Candidate]:
    candidates: list[Candidate] = []
    for count in workers:
        candidates.append(
            Candidate(
                f"internal-{count}", "scheduling", "measure", "unaligned", "cold", count, 1
            )
        )
        if count > 1:
            candidates.append(
                Candidate(
                    f"outer-{count}", "scheduling", "measure", "unaligned", "cold", 1, count
                )
            )
        if count >= 4 and count % 2 == 0:
            candidates.append(
                Candidate(
                    f"hybrid-2x{count // 2}",
                    "scheduling",
                    "measure",
                    "unaligned",
                    "cold",
                    2,
                    count // 2,
                )
            )
        if count >= 8 and count % 4 == 0:
            candidates.append(
                Candidate(
                    f"hybrid-4x{count // 4}",
                    "scheduling",
                    "measure",
                    "unaligned",
                    "cold",
                    4,
                    count // 4,
                )
            )
    return candidates


def planning_candidates(
    workers: int, patient_limit: float, exhaustive_limit: float
) -> list[Candidate]:
    return [
        Candidate("estimate-unaligned", "planning", "estimate", "unaligned", "cold", workers, 1),
        Candidate("estimate-aligned", "planning", "estimate", "aligned", "cold", workers, 1),
        Candidate("measure-unaligned", "planning", "measure", "unaligned", "cold", workers, 1),
        Candidate("measure-aligned", "planning", "measure", "aligned", "cold", workers, 1),
        Candidate(
            "patient-unaligned", "planning", "patient", "unaligned", "cold", workers, 1, patient_limit
        ),
        Candidate("patient-aligned", "planning", "patient", "aligned", "cold", workers, 1, patient_limit),
        Candidate(
            "exhaustive-aligned",
            "planning",
            "exhaustive",
            "aligned",
            "cold",
            workers,
            1,
            exhaustive_limit,
        ),
        Candidate(
            "measure-unaligned-imported",
            "planning",
            "measure",
            "unaligned",
            "generated-import",
            workers,
            1,
        ),
        Candidate(
            "measure-aligned-imported",
            "planning",
            "measure",
            "aligned",
            "generated-import",
            workers,
            1,
        ),
    ]


def deduplicated(candidates: list[Candidate]) -> list[Candidate]:
    result: list[Candidate] = []
    seen: set[tuple[str, str, str, int, int, float]] = set()
    for candidate in candidates:
        key = (
            candidate.planning,
            candidate.alignment,
            candidate.wisdom,
            candidate.internal_workers,
            candidate.outer_workers,
            candidate.planning_time_limit_seconds,
        )
        if key not in seen:
            result.append(candidate)
            seen.add(key)
    return result


def main() -> int:
    repository_root = Path(__file__).resolve().parents[1]
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--executable", type=Path, default=repository_root / "build/release/skbench")
    parser.add_argument(
        "--output",
        type=Path,
        default=repository_root / "results/local" / f"issue4-fftw-strategy-{timestamp}",
    )
    parser.add_argument("--profiles", nargs="*", help="Subset of the representative 256 and 512 profiles")
    parser.add_argument("--matrix", choices=("all", "planning", "scheduling"), default="all")
    parser.add_argument("--workers", default="1,2,4,8,performance,total")
    parser.add_argument("--planning-workers", default="performance")
    parser.add_argument("--patient-time-limit", type=float, default=5.0)
    parser.add_argument("--exhaustive-time-limit", type=float, default=5.0)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--samples", type=int, default=9)
    parser.add_argument("--seed", type=int, default=129)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    arguments = parser.parse_args()

    profiles = selected(arguments.profiles, PROFILES, "profile")
    workers = worker_counts(arguments.workers)
    planning_worker_values = worker_counts(arguments.planning_workers)
    if len(planning_worker_values) != 1:
        parser.error("--planning-workers must resolve to exactly one count")
    if arguments.patient_time_limit <= 0.0 or arguments.exhaustive_time_limit <= 0.0:
        parser.error("planning time limits must be positive")
    if arguments.warmups < 1 or arguments.samples < 1:
        parser.error("--warmups and --samples must be positive")

    candidates: list[Candidate] = []
    if arguments.matrix in ("all", "scheduling"):
        candidates.extend(scheduling_candidates(workers))
    if arguments.matrix in ("all", "planning"):
        candidates.extend(
            planning_candidates(
                planning_worker_values[0],
                arguments.patient_time_limit,
                arguments.exhaustive_time_limit,
            )
        )
    candidates = deduplicated(candidates)

    commands: list[tuple[str, Candidate, list[str], Path]] = []
    for profile in profiles:
        for candidate in candidates:
            stem = f"{profile}--{candidate.id}"
            result_path = arguments.output / f"{stem}.json"
            command = [
                str(arguments.executable),
                "run",
                "--profile",
                profile,
                "--providers",
                "fftw",
                "--fftw-planning",
                candidate.planning,
                "--fftw-alignment",
                candidate.alignment,
                "--fftw-wisdom",
                candidate.wisdom,
                "--fftw-internal-workers",
                str(candidate.internal_workers),
                "--fftw-outer-workers",
                str(candidate.outer_workers),
                "--fftw-planning-time-limit",
                str(candidate.planning_time_limit_seconds),
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
        print(f"Planned {len(commands)} run(s); {len(candidates)} unique candidate(s); workers={workers}.")
        return 0

    arguments.output.mkdir(parents=True, exist_ok=False)
    manifest = {
        "schema": "spectral-kernel-local-sweep-v1",
        "experimentId": "issue-004-fftw-strategy-sweep",
        "createdAtUtc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "profiles": profiles,
        "matrix": arguments.matrix,
        "workers": workers,
        "planningWorkers": planning_worker_values[0],
        "patientTimeLimitSeconds": arguments.patient_time_limit,
        "exhaustiveTimeLimitSeconds": arguments.exhaustive_time_limit,
        "warmups": arguments.warmups,
        "samples": arguments.samples,
        "candidates": [asdict(candidate) | {"total_workers": candidate.total_workers} for candidate in candidates],
        "runs": [],
    }
    failed = False
    for index, (stem, candidate, command, result_path) in enumerate(commands, start=1):
        print(f"[{index}/{len(commands)}] {stem}", flush=True)
        log_path = arguments.output / f"{stem}.log"
        with log_path.open("w", encoding="utf-8") as log:
            completed = subprocess.run(
                command, cwd=repository_root, stdout=log, stderr=subprocess.STDOUT
            )
        entry = {
            "id": stem,
            "candidate": asdict(candidate) | {"total_workers": candidate.total_workers},
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
