#!/usr/bin/env python3
"""Screen WVM-derived dealiased horizontal-advection topologies for issue #17."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

from run_spectral_pipeline_sweep import geometric_mean, maximum_correctness_error
from run_vertical_gemm_sweep import git_source_state


EXPERIMENT_ID = "issue-017-implicit-hybrid-dealiased-convolution"
INCREMENT_ID = "wvm-derived-horizontal-advection-screen-v1"
COHORT_ID = "issue17-wvm-advection-three-resolution-v1"
PROFILES = (
    "wvm-current-256-nz129-f4",
    "wvm-current-512-nz257-f4",
    "wvm-large-1024-nz129-f4",
)
SERIAL_EXPLICIT_ID = "fftw-explicit-streamed-wvm-advection"
PARALLEL_EXPLICIT_ID = "fftw-explicit-parallel-target-wvm-advection"
SERIAL_IMPLICIT_ID = "fftwpp-streamed-target-wvm-advection"
ALL_TARGET_ID = "fftwpp-all-target-wvm-advection"
PARALLEL_IMPLICIT_ID = "fftwpp-parallel-target-wvm-advection"
PROVIDER_IDS = (
    SERIAL_EXPLICIT_ID,
    PARALLEL_EXPLICIT_ID,
    SERIAL_IMPLICIT_ID,
    ALL_TARGET_ID,
    PARALLEL_IMPLICIT_ID,
)
TOTAL_STAGE = "WVM-like four-target horizontal advection"


def provider(result: dict, provider_id: str) -> dict:
    return next(
        item for item in result["providers"] if item["id"] == provider_id
    )


def total_seconds(record: dict) -> float:
    matches = [
        item for item in record["timings"]
        if item["scope"] == "uninstrumented-total"
        and item["stage"] == TOTAL_STAGE
        and item["direction"] == "forward"
    ]
    if len(matches) != 1:
        raise ValueError(
            f"provider {record['id']} lacks one authoritative advection total"
        )
    return float(matches[0]["medianSeconds"])


def persistent_bytes(record: dict) -> int:
    value = int(record.get("memory", {}).get("persistentBytes", 0))
    if value <= 0:
        raise ValueError(f"provider {record['id']} lacks positive persistent memory")
    return value


def analyze(results: list[dict]) -> dict:
    rows = []
    all_correct = True
    maximum_error = 0.0
    for result in results:
        records = {identifier: provider(result, identifier) for identifier in PROVIDER_IDS}
        explicit = records[PARALLEL_EXPLICIT_ID]
        candidate = records[PARALLEL_IMPLICIT_ID]
        errors = {
            identifier: maximum_correctness_error(record)
            for identifier, record in records.items()
        }
        all_correct = all_correct and all(
            math.isfinite(value) and value <= 1.0e-12
            for value in errors.values()
        )
        maximum_error = max(maximum_error, *errors.values())
        explicit_seconds = total_seconds(explicit)
        candidate_seconds = total_seconds(candidate)
        explicit_memory = persistent_bytes(explicit)
        candidate_memory = persistent_bytes(candidate)
        rows.append({
            "profile": result["run"]["profile"],
            "Nx": int(result["workload"]["Nx"]),
            "providers": {
                identifier: {
                    "seconds": total_seconds(record),
                    "persistentBytes": persistent_bytes(record),
                    "maximumCorrectnessError": errors[identifier],
                }
                for identifier, record in records.items()
            },
            "parallelImplicitToParallelExplicit": candidate_seconds / explicit_seconds,
            "parallelImplicitMemoryToParallelExplicit": candidate_memory / explicit_memory,
        })

    complete = {row["profile"] for row in rows} == set(PROFILES)
    time_ratios = [row["parallelImplicitToParallelExplicit"] for row in rows]
    memory_ratios = [
        row["parallelImplicitMemoryToParallelExplicit"] for row in rows
    ]
    large_ratios = [
        row["parallelImplicitToParallelExplicit"] for row in rows
        if row["Nx"] >= 512
    ]
    geometric_time = geometric_mean(time_ratios) if complete else math.inf
    geometric_memory = geometric_mean(memory_ratios) if complete else math.inf
    large_geometric_time = geometric_mean(large_ratios) if complete else math.inf
    worst_time = max(time_ratios, default=math.inf)
    advance = (
        complete
        and all_correct
        and geometric_time <= 0.95
        and large_geometric_time <= 0.95
        and worst_time <= 1.05
        and geometric_memory <= 0.80
    )
    return {
        "schema": "spectral-kernel-dealiased-convolution-analysis-v1",
        "experimentId": EXPERIMENT_ID,
        "incrementId": INCREMENT_ID,
        "phase": "screen",
        "cohortId": COHORT_ID,
        "complete": complete,
        "allCorrect": all_correct,
        "maximumCorrectnessError": maximum_error,
        "geometricParallelImplicitToParallelExplicit": geometric_time,
        "largeGeometricParallelImplicitToParallelExplicit": large_geometric_time,
        "worstParallelImplicitToParallelExplicit": worst_time,
        "geometricParallelImplicitMemoryToParallelExplicit": geometric_memory,
        "advanceToReference": advance,
        "selectionRule": (
            "Advance the fixed four-target FFTW++ policy when all three profiles "
            "are correct, geometric time is at most 0.95 overall and on the two "
            "larger profiles, no profile exceeds 1.05, and geometric persistent "
            "memory is at most 0.80 of the matched parallel explicit FFTW control."
        ),
        "profiles": sorted(rows, key=lambda row: row["Nx"]),
    }


def main() -> int:
    repository_root = Path(__file__).resolve().parents[1]
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--executable", type=Path,
        default=repository_root / "build" / "issue17" / "skbench",
    )
    parser.add_argument(
        "--output", type=Path,
        default=repository_root / "results" / "local" /
        f"issue17-wvm-advection-screen-{timestamp}",
    )
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--samples", type=int, default=21)
    parser.add_argument("--seed", type=int, default=129)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--allow-dirty-tree", action="store_true")
    arguments = parser.parse_args()

    source_commit, source_dirty = git_source_state(repository_root)
    if source_dirty and not arguments.allow_dirty_tree:
        parser.error(
            "the source tree is dirty; commit the benchmark implementation or "
            "use --allow-dirty-tree for a non-publishable diagnostic"
        )
    if not arguments.executable.is_file():
        parser.error(f"benchmark executable is missing: {arguments.executable}")
    if arguments.warmups < 1 or arguments.samples < 1:
        parser.error("warmups and samples must be positive")

    commands = []
    for profile in PROFILES:
        result_path = arguments.output / f"{profile}.json"
        command = [
            str(arguments.executable), "run",
            "--kernel", "dealiased-convolution",
            "--convolution-map", "wvm-advection",
            "--profile", profile,
            "--warmups", str(arguments.warmups),
            "--samples", str(arguments.samples),
            "--seed", str(arguments.seed),
            "--output", str(result_path),
        ]
        commands.append((profile, result_path, command))

    if arguments.dry_run:
        for _, _, command in commands:
            print(" ".join(command))
        print(
            f"Planned {len(commands)} isolated WVM-derived convolution screens."
        )
        return 0

    arguments.output.mkdir(parents=True, exist_ok=False)
    wvm_repository = repository_root.parent / "wave-vortex-model"
    wvm_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=wvm_repository, check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    manifest = {
        "schema": "spectral-kernel-local-sweep-v1",
        "experimentId": EXPERIMENT_ID,
        "incrementId": INCREMENT_ID,
        "phase": "screen",
        "cohortId": COHORT_ID,
        "createdAtUtc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "question": (
            "Can FFTW++ implicit/hybrid dealiasing beat a fair explicit FFTW "
            "control for the four nonhydrostatic WVM horizontal advection "
            "expressions while reducing persistent memory?"
        ),
        "baseline": (
            "A same-process explicit FFTW control that reconstructs U, V, and W "
            "once and evaluates four derivative targets through a persistent "
            "four-worker schedule."
        ),
        "changedVariables": [
            "explicit full-grid versus centered/Hermitian implicit-hybrid dealiasing",
            "serial streamed, all-target, and persistent four-target scheduling topologies",
        ],
        "controlledVariables": [
            "15 ready compact input spectra: U, V, W and x/y/z derivatives of four targets",
            "four outputs defined by -(U*qx + V*qy + W*qz)",
            "radial two-thirds retention, Float64 fixtures, FFTW MEASURE/unaligned, and one algorithm across sizes",
        ],
        "timedOperation": (
            "Ready compact retained horizontal spectra to four ready compact "
            "retained advective-flux spectra, including embedding, transforms, "
            "multiplication/reduction, retention, and persistent dispatch."
        ),
        "componentLedger": [
            "explicit advecting inverse FFT batch",
            "explicit per-target derivative inverse FFT batch",
            "explicit per-target three-term reduction and forward FFT",
            "fused FFTW++ transform-reduce-transform",
            "authoritative caller-visible total and persistent memory",
        ],
        "excludedWork": [
            "vertical reconstruction/projection and phase evolution",
            "coefficient-space flux accumulation and the complete nonlinear WVM flux",
            "time stepping, state management, I/O, Float32, and general-Mac claims",
        ],
        "allocationPolicy": (
            "All plans, worker pools, buffers, and compact outputs are persistent; "
            "the focused allocator-interposer test must pass separately."
        ),
        "selectionRule": (
            "Advance the fixed four-target FFTW++ policy only if all profiles are "
            "correct, geometric time is at most 0.95 overall and for Nx>=512, "
            "no profile exceeds 1.05, and geometric persistent memory is at most 0.80."
        ),
        "profiles": list(PROFILES),
        "providers": list(PROVIDER_IDS),
        "sourceTreeGitCommit": source_commit,
        "sourceTreeDirty": source_dirty,
        "wvmAuditGitCommit": wvm_commit,
        "wvmAuditSource": (
            "CompiledKernel/src/WVTransformConstantStratificationKernel.cpp "
            "nonlinearFluxImpl"
        ),
        "rounds": 1,
        "warmups": arguments.warmups,
        "samples": arguments.samples,
        "seed": arguments.seed,
        "runs": [],
    }

    failed = False
    results = []
    for index, (profile, result_path, command) in enumerate(commands, start=1):
        print(f"[{index}/{len(commands)}] {profile}", flush=True)
        log_path = arguments.output / f"{profile}.log"
        with log_path.open("w", encoding="utf-8") as log:
            completed = subprocess.run(
                command, cwd=repository_root, stdout=log,
                stderr=subprocess.STDOUT,
            )
        entry = {
            "id": profile,
            "round": 1,
            "profile": profile,
            "primaryProvider": PARALLEL_IMPLICIT_ID,
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
                bool(embedded_commit) and embedded_commit != "unknown"
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
            if completed.returncode == 0 and result["status"] == "passed" and metadata_matches:
                results.append(result)
            else:
                failed = True
        else:
            failed = True
        manifest["runs"].append(entry)
        with (arguments.output / "manifest.json").open("w", encoding="utf-8") as stream:
            json.dump(manifest, stream, indent=2)
            stream.write("\n")
        if failed:
            print(log_path.read_text(encoding="utf-8")[-4000:], file=sys.stderr)
            break

    if results:
        with (arguments.output / "analysis.json").open("w", encoding="utf-8") as stream:
            json.dump(analyze(results), stream, indent=2)
            stream.write("\n")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
