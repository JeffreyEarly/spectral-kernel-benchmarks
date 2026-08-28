#!/usr/bin/env python3
"""Run the bounded issue #12 persistent outer-sharding comparison."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

from run_pruned_horizontal_sweep import (
    DEFAULT_PROFILES,
    estimated_explicit_peak_bytes,
    selected_profiles,
)
from run_vertical_gemm_sweep import (
    gibibytes,
    git_source_state,
    sysctl_uint64,
    thread_limits,
)


def outer_worker_counts(value: str) -> list[int]:
    workers = thread_limits(value)
    if not workers:
        raise ValueError("at least one outer worker count is required")
    return workers


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
        default=(
            repository_root / "results/local" /
            f"issue12-pruned-outer-sharding-{timestamp}"
        ),
    )
    parser.add_argument("--profiles", nargs="*", help="Subset of named WVM profiles")
    parser.add_argument(
        "--outer-workers",
        default="1,4,8,performance",
        help="Comma-separated persistent outer worker counts or performance/total aliases",
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
        outer_workers = outer_worker_counts(arguments.outer_workers)
    except ValueError as error:
        parser.error(str(error))

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

    commands: list[tuple[str, str, int, int, list[str], Path]] = []
    for profile in profiles:
        for worker_count in outer_workers:
            stem = f"{profile}--partial-column-pruned--outer-{worker_count}"
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
                "1",
                "--fftw-outer-workers",
                str(worker_count),
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
        "incrementId": "fftw-partial-column-pruned-outer-sharding-v2",
        "createdAtUtc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "question": (
            "Can persistent outer plane/field sharding with single-threaded FFTW plans "
            "turn the partial-column-pruned retained operator into a scalable forward, "
            "inverse, or direction-specific winner?"
        ),
        "baseline": (
            "Within every run, the same persistent outer worker topology executes the full "
            "FFTW 2-D transform plus outer-sharded radial selection/embedding and the "
            "partial-column-pruned separable candidate. The earlier internal-thread cohort "
            "remains immutable prior evidence."
        ),
        "changedVariables": [
            "persistent outer plane/field worker count",
            "internal FFTW pthreads fixed at one",
            "outer-sharded full-spectrum and compact-retained adapters",
        ],
        "controlledVariables": [
            "FFTW 3.3.11 build, compiler flags, planning effort, fixture, and precision",
            "logical retained modes, normalization, workload, warmups, samples, and seed",
            "out-of-place caller contract and aggregate full-half-spectrum scratch capacity",
        ],
        "timedOperations": [
            "full raw 2-D forward and inverse FFT with outer plan sharding",
            "full-reference outer-sharded radial selection and embedding",
            "candidate outer-sharded real-row and selected-column transforms",
            "candidate outer-sharded direct retention and inverse embedding",
            "empty persistent scheduler dispatch",
            "complete uninstrumented retained forward and inverse operators",
        ],
        "excludedWork": [
            "vertical projection, GEMM, modal work, and nonlinear flux calculation",
            "deeper within-column pruning, reduced scratch, split storage, and generated FFTs",
            "thread affinity, efficiency-core-specific scheduling, and caller-visible in-place operation",
        ],
        "placement": (
            "Both complete retained operators are out-of-place. Each candidate worker owns "
            "a disjoint reusable slice of one aggregate full-half-spectrum plane-major scratch "
            "allocation; selected column transforms are in-place only inside that slice."
        ),
        "profiles": profiles,
        "internalWorkers": 1,
        "outerWorkers": outer_workers,
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
            "internalWorkers": 1,
            "outerWorkers": worker_count,
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
