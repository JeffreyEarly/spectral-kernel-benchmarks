#!/usr/bin/env python3
"""Run the issue #20 production-shaped constant-stratification composition screen."""

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
INCREMENT_ID = "production-shaped-type1-composed-screen-v1"
COHORT_ID = "issue20-m4-production-shaped-type1-composed-v1"
PROFILES = (
    "wvm-current-256-nz129-f4",
    "wvm-current-512-nz257-f4",
    "wvm-large-1024-nz129-f4",
    "wvm-large-512-nz513-f4",
)
CONTROL_PROVIDER = "pipeline-constant-stratification-wvm-full-half"
CANDIDATE_PROVIDER = (
    "pipeline-constant-stratification-streaming-pruned-tile16"
)
TOTAL_STAGE = "production-shaped constant-stratification spectral-flux composition"
COMPONENT_STAGES = {
    "coefficientAssembly": (
        "component", "mode-keyed coefficient assembly and retained/full clearing",
        "inverse",
    ),
    "verticalInverse": (
        "component", "15 inverse complex type-I channels", "inverse",
    ),
    "horizontalInverse": (
        "retained-operator-total", "five horizontal inverse transforms", "inverse",
    ),
    "pointwise": (
        "component", "four streamed pointwise advection expressions", "pointwise",
    ),
    "horizontalForward": (
        "retained-operator-total",
        "four horizontal forward transforms and radial retention", "forward",
    ),
    "verticalForward": (
        "component", "four forward complex type-I channels and normalization",
        "forward",
    ),
    "coefficientProjection": (
        "component", "coefficient reset and four target accumulations", "forward",
    ),
}


def command_for(
    executable: Path, profile: str, output: Path, vertical_workers: int,
    horizontal_workers: int, pointwise_workers: int, warmups: int,
    samples: int,
) -> list[str]:
    return [
        str(executable), "run",
        "--kernel", "constant-stratification-flux",
        "--profile", profile,
        "--fftw-planning", "measure",
        "--fftw-internal-workers", str(vertical_workers),
        "--fftw-outer-workers", str(horizontal_workers),
        "--streaming-tile-width", "16",
        "--pointwise-policy", "spatial-static",
        "--pointwise-workers", str(pointwise_workers),
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
    control = provider(result, CONTROL_PROVIDER)
    candidate = provider(result, CANDIDATE_PROVIDER)
    control_total = stage(
        control, "uninstrumented-total", TOTAL_STAGE, "complete"
    )["medianSeconds"]
    candidate_total = stage(
        candidate, "uninstrumented-total", TOTAL_STAGE, "complete"
    )["medianSeconds"]
    control_error = maximum_correctness_error(control)
    candidate_error = maximum_correctness_error(candidate)
    components = {}
    for key, (scope, name, direction) in COMPONENT_STAGES.items():
        control_seconds = stage(control, scope, name, direction)["medianSeconds"]
        candidate_seconds = stage(
            candidate, scope, name, direction
        )["medianSeconds"]
        components[key] = {
            "controlSeconds": control_seconds,
            "candidateSeconds": candidate_seconds,
            "candidateToControl": candidate_seconds / control_seconds,
        }
    workload = result["workload"]
    control_resident = control["memory"]["algorithmResidentBytes"]
    candidate_resident = candidate["memory"]["algorithmResidentBytes"]
    return {
        "runId": result["run"]["id"],
        "profile": result["run"]["profile"],
        "shape": [workload["Nx"], workload["Ny"], workload["Nz"]],
        "fullRows": workload["H"],
        "retainedRows": workload["Nkl"],
        "controlTotalSeconds": control_total,
        "candidateTotalSeconds": candidate_total,
        "candidateToControlTotal": candidate_total / control_total,
        "controlResidentBytes": control_resident,
        "candidateResidentBytes": candidate_resident,
        "candidateToControlResident": candidate_resident / control_resident,
        "maximumCorrectnessError": max(control_error, candidate_error),
        "correctWithin1e12": (
            math.isfinite(control_error) and math.isfinite(candidate_error)
            and max(control_error, candidate_error) <= 1.0e-12
        ),
        "components": components,
        "gitDirty": result["environment"]["gitDirty"],
    }


def geometric(values: list[float]) -> float:
    return math.exp(statistics.fmean(math.log(value) for value in values))


def analyze(
    results: list[dict], vertical_workers: int, horizontal_workers: int,
    pointwise_workers: int, benchmark_commit: str,
) -> dict:
    rows = [profile_row(result) for result in results]
    ratios = [row["candidateToControlTotal"] for row in rows]
    all_correct = all(row["correctWithin1e12"] for row in rows)
    complete = {row["profile"] for row in rows} == set(PROFILES)
    component_geometric = {
        key: geometric([
            row["components"][key]["candidateToControl"] for row in rows
        ])
        for key in COMPONENT_STAGES
    }
    return {
        "schema": "spectral-kernel-constant-stratification-composed-analysis-v1",
        "experimentId": EXPERIMENT_ID,
        "incrementId": INCREMENT_ID,
        "phase": "preliminary-composed-screen",
        "cohortId": COHORT_ID,
        "benchmarkCommit": benchmark_commit,
        "topology": {
            "verticalType1InternalWorkers": vertical_workers,
            "horizontalOuterWorkers": horizontal_workers,
            "pointwiseStaticWorkers": pointwise_workers,
            "streamingTileWidth": 16,
        },
        "completeProfileMatrix": complete,
        "allCorrectWithin1e12": all_correct,
        "maximumCorrectnessError": max(
            row["maximumCorrectnessError"] for row in rows
        ),
        "geometricCandidateToControlTotal": geometric(ratios),
        "worstCandidateToControlTotal": max(ratios),
        "geometricCandidateToControlResident": geometric([
            row["candidateToControlResident"] for row in rows
        ]),
        "componentGeometricCandidateToControl": component_geometric,
        "authoritativeValidationRecommended": bool(
            complete and all_correct and geometric(ratios) <= 0.90
            and max(ratios) <= 1.03
        ),
        "interpretation": (
            "This development screen composes exact type-I family transforms, "
            "the fixed horizontal algorithms, a production-shaped streamed "
            "lifetime, and pointwise work around a declared synthetic coefficient "
            "map. It can recommend exact WVM fixture or in-repository validation; "
            "it cannot establish complete nonlinear-flux adoption."
        ),
        "rows": rows,
    }


def retained_rows(nx: int) -> int:
    radius = nx // 3
    return sum(
        1
        for l in range(-radius, radius + 1)
        for k in range(0, radius + 1)
        if k * k + l * l <= radius * radius
    )


def estimated_peak_bytes(profile: str, horizontal_workers: int) -> int:
    shapes = {
        "wvm-current-256-nz129-f4": (256, 129),
        "wvm-current-512-nz257-f4": (512, 257),
        "wvm-large-1024-nz129-f4": (1024, 129),
        "wvm-large-512-nz513-f4": (512, 513),
    }
    nx, nz = shapes[profile]
    half_rows = (nx // 2 + 1) * nx
    compact_rows = retained_rows(nx)
    nj = 2 * (nz - 1) // 3
    complex_bytes = 16
    full_arena = 4 * half_rows * nz * complex_bytes
    compact_arena = 4 * compact_rows * nz * complex_bytes
    two_graph_real_lifetimes = 14 * nx * nx * nz * 8
    coefficient_fixture_and_outputs = 9 * compact_rows * nj * complex_bytes
    horizontal_scratch = 4 * horizontal_workers * half_rows * complex_bytes
    opaque_allowance = 8 * 1024**3
    return (
        full_arena + compact_arena + two_graph_real_lifetimes
        + coefficient_fixture_and_outputs + horizontal_scratch
        + opaque_allowance
    )


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
        f"issue20-constant-stratification-composed-{timestamp}",
    )
    parser.add_argument("--vertical-workers", type=int, default=16)
    parser.add_argument("--horizontal-workers", type=int, default=12)
    parser.add_argument("--pointwise-workers", type=int, default=8)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--samples", type=int, default=7)
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
    worker_values = (
        arguments.vertical_workers, arguments.horizontal_workers,
        arguments.pointwise_workers, arguments.warmups, arguments.samples,
    )
    if any(value < 1 for value in worker_values):
        parser.error("worker counts, warmups, and samples must be positive")

    estimates = {
        profile: estimated_peak_bytes(profile, arguments.horizontal_workers)
        for profile in PROFILES
    }
    physical_memory = sysctl_uint64("hw.memsize", 16 * 1024**3)
    if max(estimates.values()) > int(0.75 * physical_memory) and not (
        arguments.allow_memory_risk
    ):
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
                arguments.vertical_workers, arguments.horizontal_workers,
                arguments.pointwise_workers, arguments.warmups,
                arguments.samples,
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
        "phase": "preliminary-composed-screen",
        "cohortId": COHORT_ID,
        "createdAtUtc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "question": (
            "Does the retained-row type-I component advantage survive the fixed "
            "horizontal algorithms, streamed pointwise lifetime, coefficient "
            "passes, and output accumulation?"
        ),
        "baseline": (
            "WVM-order full-half type-I arena with full horizontal FFTW, using "
            "the same declared mode-keyed coefficient map and pointwise graph."
        ),
        "changedVariables": [
            "full WVM half-spectrum versus compact split retained type-I rows",
            "full horizontal FFTW versus fixed tile-16 partial-column pruning",
        ],
        "controlledVariables": [
            "Float64, four-field workload, exact type-I family schedule and normalization",
            "mode-keyed coefficient map, four pointwise expressions, output "
            "accumulation, worker topology, and one tuple across sizes",
        ],
        "timedOperation": (
            "Five coefficient assemblies, 15 inverse type-I channels, five "
            "horizontal inverses, four pointwise expressions, four horizontal "
            "forwards, four normalized forward type-I channels, and four "
            "coefficient accumulations; uninstrumented total sampled independently."
        ),
        "excludedWork": [
            "WVM physical coefficient formulas and phase evolution",
            "MATLAB/MEX dispatch, model state, timestep, and I/O",
            "complete authoritative WVM nonlinear-flux validation",
        ],
        "correctnessOracle": (
            "Every compact intermediate and final mode-keyed output is compared "
            "with the full-half control at tolerance 1e-12."
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
            "primaryProvider": CANDIDATE_PROVIDER,
        })
        if completed.returncode != 0:
            (arguments.output / "manifest.json").write_text(
                json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
            )
            raise RuntimeError(f"benchmark failed for {profile}")
        with result_path.open(encoding="utf-8") as stream:
            results.append(json.load(stream))

    analysis = analyze(
        results, arguments.vertical_workers, arguments.horizontal_workers,
        arguments.pointwise_workers, benchmark_commit,
    )
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
