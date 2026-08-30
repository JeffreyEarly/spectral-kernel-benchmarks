#!/usr/bin/env python3
"""Run the issue #20 retained-row constant-stratification vertical screen."""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import subprocess
from datetime import UTC, datetime
from pathlib import Path

from run_spectral_pipeline_sweep import maximum_correctness_error
from run_vertical_gemm_sweep import git_source_state, sysctl_uint64


EXPERIMENT_ID = "issue-020-constant-stratification-type1"
INCREMENT_ID = "production-type1-retained-row-screen-v1"
COHORT_ID = "issue20-m4-production-type1-retained-row-v1"
PROFILES = (
    "wvm-current-256-nz129-f4",
    "wvm-current-512-nz257-f4",
    "wvm-large-1024-nz129-f4",
    "wvm-large-512-nz513-f4",
)
FULL_PROVIDER = "fftw-wvm-type1-full-half"
COMPACT_PROVIDER = "fftw-wvm-type1-retained-compact"
TOTAL_STAGE = (
    "production nonlinear-flux vertical transform schedule "
    "(15 inverse + 4 forward complex channels)"
)


def command_for(
    executable: Path, profile: str, output: Path, workers: int,
    warmups: int, samples: int,
) -> list[str]:
    return [
        str(executable), "run",
        "--kernel", "constant-stratification-vertical",
        "--profile", profile,
        "--fftw-planning", "measure",
        "--fftw-internal-workers", str(workers),
        "--warmups", str(warmups),
        "--samples", str(samples),
        "--output", str(output),
    ]


def provider(result: dict, provider_id: str) -> dict:
    matches = [item for item in result["providers"] if item["id"] == provider_id]
    if len(matches) != 1:
        raise ValueError(f"result lacks one provider {provider_id}")
    return matches[0]


def stage(record: dict, scope: str, name: str, direction: str) -> dict:
    matches = [
        item for item in record["timings"]
        if item["scope"] == scope and item["stage"] == name
        and item["direction"] == direction
    ]
    if len(matches) != 1:
        raise ValueError(
            f"provider {record['id']} lacks one {scope}/{name}/{direction} timing"
        )
    return matches[0]


def profile_row(result: dict) -> dict:
    full = provider(result, FULL_PROVIDER)
    compact = provider(result, COMPACT_PROVIDER)
    full_total = stage(
        full, "uninstrumented-total", TOTAL_STAGE, "complete"
    )["medianSeconds"]
    compact_total = stage(
        compact, "uninstrumented-total", TOTAL_STAGE, "complete"
    )["medianSeconds"]
    full_error = maximum_correctness_error(full)
    compact_error = maximum_correctness_error(compact)
    workload = result["workload"]
    return {
        "runId": result["run"]["id"],
        "profile": result["run"]["profile"],
        "shape": [workload["Nx"], workload["Ny"], workload["Nz"]],
        "fullRows": workload["H"],
        "retainedRows": workload["Nkl"],
        "retainedRowFraction": workload["Nkl"] / workload["H"],
        "fullTotalSeconds": full_total,
        "compactTotalSeconds": compact_total,
        "compactToFullTotal": compact_total / full_total,
        "fullArenaBytes": full["memory"]["algorithmResidentBytes"],
        "compactArenaBytes": compact["memory"]["algorithmResidentBytes"],
        "compactToFullArena": (
            compact["memory"]["algorithmResidentBytes"]
            / full["memory"]["algorithmResidentBytes"]
        ),
        "fullDctSeconds": stage(
            full, "primitive", "raw DCT-I one complex channel", "forward"
        )["medianSeconds"],
        "compactDctSeconds": stage(
            compact, "primitive", "raw DCT-I one complex channel", "forward"
        )["medianSeconds"],
        "fullDstSeconds": stage(
            full, "primitive", "raw DST-I one complex interior channel", "forward"
        )["medianSeconds"],
        "compactDstSeconds": stage(
            compact, "primitive", "raw DST-I one complex interior channel", "forward"
        )["medianSeconds"],
        "maximumCorrectnessError": max(full_error, compact_error),
        "correctWithin1e12": (
            math.isfinite(full_error) and math.isfinite(compact_error)
            and max(full_error, compact_error) <= 1.0e-12
        ),
        "gitDirty": result["environment"]["gitDirty"],
    }


def analyze(results: list[dict], workers: int, benchmark_commit: str) -> dict:
    rows = [profile_row(result) for result in results]
    ratios = [row["compactToFullTotal"] for row in rows]
    memory_ratios = [row["compactToFullArena"] for row in rows]
    all_correct = all(row["correctWithin1e12"] for row in rows)
    complete = {row["profile"] for row in rows} == set(PROFILES)
    return {
        "schema": "spectral-kernel-constant-stratification-vertical-analysis-v1",
        "experimentId": EXPERIMENT_ID,
        "incrementId": INCREMENT_ID,
        "phase": "preliminary-component-screen",
        "cohortId": COHORT_ID,
        "benchmarkCommit": benchmark_commit,
        "fftwInternalWorkers": workers,
        "completeProfileMatrix": complete,
        "allCorrectWithin1e12": all_correct,
        "maximumCorrectnessError": max(
            row["maximumCorrectnessError"] for row in rows
        ),
        "geometricCompactToFullVerticalSchedule": math.exp(
            statistics.fmean(math.log(value) for value in ratios)
        ),
        "worstCompactToFullVerticalSchedule": max(ratios),
        "geometricCompactToFullExplicitArena": math.exp(
            statistics.fmean(math.log(value) for value in memory_ratios)
        ),
        "integrationRecommended": bool(
            complete and all_correct and max(ratios) <= 0.90
        ),
        "interpretation": (
            "This screen isolates the exact production type-I transform schedule. "
            "It can recommend composing retained-row vertical transforms with the "
            "compact horizontal pipeline, but it cannot establish complete nonlinear-"
            "flux adoption."
        ),
        "rows": rows,
    }


def estimated_peak_bytes(profile: str) -> int:
    shapes = {
        "wvm-current-256-nz129-f4": (256, 129),
        "wvm-current-512-nz257-f4": (512, 257),
        "wvm-large-1024-nz129-f4": (1024, 129),
        "wvm-large-512-nz513-f4": (512, 513),
    }
    n, nz = shapes[profile]
    full_rows = (n // 2 + 1) * n
    radius = n // 3
    retained_rows = sum(
        1
        for l in range(-radius, radius + 1)
        for k in range(0, radius + 1)
        if k * k + l * l <= radius * radius
    )
    complex_bytes = 16
    # Four-channel full and compact arenas remain resident. FFTW planning may
    # temporarily add the largest three-channel surrogate.
    return complex_bytes * nz * (7 * full_rows + 4 * retained_rows)


def main() -> int:
    repository_root = Path(__file__).resolve().parents[1]
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--executable", type=Path,
        default=repository_root / "build" / "release" / "skbench",
    )
    parser.add_argument(
        "--output", type=Path,
        default=repository_root / "results" / "local" /
        f"issue20-constant-stratification-vertical-{timestamp}",
    )
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--samples", type=int, default=15)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--allow-dirty-tree", action="store_true")
    parser.add_argument("--allow-memory-risk", action="store_true")
    arguments = parser.parse_args()

    benchmark_commit, source_dirty = git_source_state(repository_root)
    if source_dirty and not arguments.allow_dirty_tree:
        parser.error(
            "the source tree is dirty; commit the benchmark implementation or "
            "use --allow-dirty-tree for a non-publishable diagnostic"
        )
    if not arguments.executable.is_file():
        parser.error(f"benchmark executable is missing: {arguments.executable}")
    if arguments.workers < 1 or arguments.warmups < 1 or arguments.samples < 1:
        parser.error("workers, warmups, and samples must be positive")

    estimates = {profile: estimated_peak_bytes(profile) for profile in PROFILES}
    physical_memory = sysctl_uint64("hw.memsize", 16 * 1024**3)
    if max(estimates.values()) > int(0.75 * physical_memory) and not arguments.allow_memory_risk:
        parser.error(
            "estimated process peak exceeds 75% of physical memory; use "
            "--allow-memory-risk only after reviewing the estimate"
        )

    commands = []
    for profile in PROFILES:
        result_path = arguments.output / f"{profile}.json"
        commands.append((
            profile, result_path,
            command_for(
                arguments.executable.resolve(), profile, result_path,
                arguments.workers, arguments.warmups, arguments.samples,
            ),
        ))
    if arguments.dry_run:
        for _, _, command in commands:
            print(" ".join(str(value) for value in command))
        print(json.dumps({"estimatedProcessPeakBytes": estimates}, indent=2))
        return 0

    arguments.output.mkdir(parents=True, exist_ok=False)
    manifest = {
        "schema": "spectral-kernel-local-sweep-v1",
        "experimentId": EXPERIMENT_ID,
        "incrementId": INCREMENT_ID,
        "phase": "preliminary-component-screen",
        "cohortId": COHORT_ID,
        "createdAtUtc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "question": (
            "Can the exact production FFTW type-I vertical schedule run only "
            "on retained horizontal rows without changing its mathematics?"
        ),
        "baseline": (
            "WVM's full half-spectrum interleaved-complex REDFT00/RODFT00 "
            "schedule at the audited production source commit."
        ),
        "changedVariables": [
            "full half-spectrum horizontal rows versus compact radial retained rows",
        ],
        "controlledVariables": [
            "FFTW 3.3.11 type-I algorithms, planning mode, internal workers, placement, strides, channel families, normalization, and call counts",
            "Float64 four-field antialiased workloads and one tuple across sizes",
        ],
        "timedOperation": (
            "Primitive DCT-I/DST-I, forward normalization, and the production "
            "15-inverse-plus-4-forward complex-channel vertical transform schedule."
        ),
        "excludedWork": [
            "horizontal FFTs and retained-row production",
            "coefficient assembly, phase evaluation, pointwise products, flux projection arithmetic, and coefficient accumulation",
            "complete nonlinear-flux execution",
        ],
        "correctnessOracle": (
            "Independent direct type-I sums plus every retained logical row "
            "compared with the full-half control at tolerance 1e-12."
        ),
        "estimatedProcessPeakBytes": estimates,
        "runs": [],
    }
    results = []
    environment = os.environ.copy()
    environment["VECLIB_MAXIMUM_THREADS"] = "1"
    for profile, result_path, command in commands:
        completed = subprocess.run(
            command, cwd=repository_root, env=environment,
            capture_output=True, text=True, check=False,
        )
        (arguments.output / f"{profile}.stdout.txt").write_text(
            completed.stdout, encoding="utf-8"
        )
        (arguments.output / f"{profile}.stderr.txt").write_text(
            completed.stderr, encoding="utf-8"
        )
        manifest["runs"].append({
            "id": profile,
            "profile": profile,
            "result": result_path.name,
            "exitCode": completed.returncode,
            "primaryProvider": COMPACT_PROVIDER,
        })
        if completed.returncode != 0:
            (arguments.output / "manifest.json").write_text(
                json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
            )
            raise RuntimeError(f"benchmark failed for {profile}")
        with result_path.open(encoding="utf-8") as stream:
            results.append(json.load(stream))

    analysis = analyze(results, arguments.workers, benchmark_commit)
    (arguments.output / "analysis.json").write_text(
        json.dumps(analysis, indent=2) + "\n", encoding="utf-8"
    )
    (arguments.output / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(analysis, indent=2))
    return 0 if analysis["allCorrectWithin1e12"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
