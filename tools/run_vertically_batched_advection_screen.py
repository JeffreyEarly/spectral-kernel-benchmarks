#!/usr/bin/env python3
"""Run the first vertically batched WVM-derived advection screen for issue #18."""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

from run_spectral_pipeline_sweep import maximum_correctness_error
from run_vertical_gemm_sweep import git_source_state, sysctl_uint64


EXPERIMENT_ID = "issue-018-vertically-batched-advection-pipeline"
INCREMENT_ID = "vertically-batched-advection-first-composition-v1"
COHORT_ID = "issue18-m4-first-composition-256-nz129-f4-v1"
PROFILE = "wvm-current-256-nz129-f4"
PROFILE_SHAPE = (256, 129)
TOTAL_STAGE = "vertically batched WVM-derived advection pipeline"
EXPLICIT_PROVIDER = "composed-fftw-explicit-parallel-target-wvm-advection"
FFTWPP_PROVIDER = "composed-fftwpp-parallel-target-wvm-advection"


@dataclass(frozen=True)
class Candidate:
    id: str
    cli_id: str
    provider_id: str
    role: str


def candidates() -> list[Candidate]:
    return [
        Candidate(
            "explicit-parallel",
            "explicit-parallel",
            EXPLICIT_PROVIDER,
            "fixed-wvm-compatible-control",
        ),
        Candidate(
            "fftwpp-parallel",
            "fftwpp-parallel",
            FFTWPP_PROVIDER,
            "fixed-implicit-hybrid-candidate",
        ),
    ]


def stored_disk(n: int) -> list[tuple[int, int]]:
    radius = n // 3
    return [
        (k, l)
        for l in range(radius + 1)
        for k in range(-radius, radius + 1)
        if k * k + l * l <= radius * radius
    ]


def estimated_process_peak_bytes(n: int, nz: int, candidate: Candidate) -> int:
    modes = stored_disk(n)
    nkl = len(modes)
    nj = 2 * (nz - 1) // 3
    groups = len({k * k + l * l for k, l in modes})
    matrix_source = 2 * groups * nz * nj * 8
    directional_matrices = 2 * groups * nz * nj * 8
    directional_split_operands = 2 * 8 * nkl * (
        15 * (nj + nz) + 4 * (nz + nj)
    )
    level_adapter = 15 * nkl * 16
    half_spectrum = n * (n // 2 + 1)
    explicit_horizontal = (
        15 * half_spectrum * 16
        + 15 * n * n * 8
        + 4 * nkl * 16
    )
    implicit_horizontal_conservative = explicit_horizontal
    horizontal = (
        explicit_horizontal
        if candidate.id == "explicit-parallel"
        else implicit_horizontal_conservative
    )
    harness = (
        nkl * nj * 15 * 16
        + nkl * nz * 4 * 16
        + nkl * nj * 4 * 16
        + matrix_source
    )
    return (
        directional_matrices
        + directional_split_operands
        + level_adapter
        + horizontal
        + harness
    )


def command_for(
    executable: Path, output: Path, candidate: Candidate,
    warmups: int, samples: int, seed: int,
) -> list[str]:
    return [
        str(executable), "run",
        "--kernel", "vertically-batched-advection",
        "--profile", PROFILE,
        "--vertical-gemm-family", "k2-grouped",
        "--vertical-gemm-schedule", "outer-dynamic",
        "--vertical-gemm-outer-workers", "12",
        "--convolution-candidate", candidate.cli_id,
        "--warmups", str(warmups),
        "--samples", str(samples),
        "--seed", str(seed),
        "--output", str(output),
    ]


def provider(result: dict, provider_id: str) -> dict:
    matches = [item for item in result["providers"] if item["id"] == provider_id]
    if len(matches) != 1:
        raise ValueError(f"result lacks one provider {provider_id}")
    return matches[0]


def timing(record: dict, scope: str, stage: str) -> float:
    matches = [
        item for item in record["timings"]
        if item["scope"] == scope and item["stage"] == stage
    ]
    if len(matches) != 1:
        raise ValueError(
            f"provider {record['id']} lacks one {scope}/{stage} timing"
        )
    return float(matches[0]["medianSeconds"])


def analyze(results: list[tuple[Candidate, dict]]) -> dict:
    by_id: dict[str, dict] = {}
    all_correct = True
    maximum_error = 0.0
    rows = []
    for candidate, result in results:
        record = provider(result, candidate.provider_id)
        error = maximum_correctness_error(record)
        correct = math.isfinite(error) and error <= 1.0e-12
        all_correct = all_correct and correct
        maximum_error = max(maximum_error, error)
        memory = record["memory"]
        row = {
            "candidateId": candidate.id,
            "providerId": candidate.provider_id,
            "role": candidate.role,
            "totalSeconds": timing(record, "uninstrumented-total", TOTAL_STAGE),
            "inverseVerticalSeconds": timing(
                record, "primitive", "raw inverse vertical GEMM (15 fields)"
            ),
            "horizontalBatchSeconds": timing(
                record, "component",
                "vertically batched horizontal advection including level movement",
            ),
            "movementSeconds": timing(
                record, "adapter-component",
                "all-level split/field-major packing and projected-output scatter",
            ),
            "forwardVerticalSeconds": timing(
                record, "primitive", "raw forward vertical GEMM (4 fields)"
            ),
            "algorithmResidentBytes": int(memory["algorithmResidentBytes"]),
            "scratchBytes": int(memory["scratchBytes"]),
            "benchmarkHarnessBytes": int(memory["benchmarkHarnessBytes"]),
            "estimatedProcessPeakBytes": int(memory["estimatedProcessPeakBytes"]),
            "observedProcessHighWaterBytes": int(
                memory["observedProcessHighWaterBytes"]
            ),
            "maximumCorrectnessError": error,
            "correct": correct,
        }
        rows.append(row)
        by_id[candidate.id] = row

    complete = set(by_id) == {candidate.id for candidate in candidates()}
    time_ratio = None
    memory_ratio = None
    reference_recommended = False
    if complete:
        baseline = by_id["explicit-parallel"]
        candidate = by_id["fftwpp-parallel"]
        time_ratio = candidate["totalSeconds"] / baseline["totalSeconds"]
        memory_ratio = (
            candidate["algorithmResidentBytes"]
            / baseline["algorithmResidentBytes"]
        )
        reference_recommended = bool(
            all_correct
            and (
                time_ratio <= 0.98
                or (memory_ratio <= 0.80 and time_ratio <= 1.05)
            )
        )

    return {
        "schema": "spectral-kernel-vertically-batched-advection-analysis-v1",
        "experimentId": EXPERIMENT_ID,
        "incrementId": INCREMENT_ID,
        "phase": "first-composition",
        "cohortId": COHORT_ID,
        "complete": complete,
        "allCorrectWithin1e-12": all_correct,
        "maximumCorrectnessError": maximum_error,
        "fftwppToExplicitTotal": time_ratio,
        "fftwppToExplicitAlgorithmResident": memory_ratio,
        "referenceCampaignRecommended": reference_recommended,
        "screenRule": (
            "Recommend reference depth when both isolated candidates are correct "
            "and FFTW++ is at most 0.98x in composed time, or when it is at most "
            "0.80x in algorithm-resident memory without exceeding 1.05x in time. "
            "This is a continuation gate, not the 0.9000 adoption threshold."
        ),
        "rows": rows,
    }


def main() -> int:
    repository_root = Path(__file__).resolve().parents[1]
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--executable", type=Path,
        default=repository_root / "build" / "issue18" / "skbench",
    )
    parser.add_argument(
        "--output", type=Path,
        default=repository_root / "results" / "local" /
        f"issue18-first-composition-{timestamp}",
    )
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--seed", type=int, default=129)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--allow-dirty-tree", action="store_true")
    parser.add_argument("--allow-memory-risk", action="store_true")
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

    physical_memory = sysctl_uint64("hw.memsize", 16 * 1024**3)
    safe_limit = int(0.75 * physical_memory)
    estimates = {
        candidate.id: estimated_process_peak_bytes(*PROFILE_SHAPE, candidate)
        for candidate in candidates()
    }
    if max(estimates.values()) > safe_limit and not arguments.allow_memory_risk:
        parser.error(
            "estimated process peak exceeds 75% of physical memory; use "
            "--allow-memory-risk only after reviewing the estimate"
        )

    commands = []
    for candidate in candidates():
        result_path = arguments.output / f"{candidate.id}.json"
        commands.append((
            candidate,
            result_path,
            command_for(
                arguments.executable, result_path, candidate,
                arguments.warmups, arguments.samples, arguments.seed,
            ),
        ))
    if arguments.dry_run:
        for _, _, command in commands:
            print(" ".join(command))
        print(json.dumps({"estimatedProcessPeakBytes": estimates}, indent=2))
        return 0

    arguments.output.mkdir(parents=True, exist_ok=False)
    manifest = {
        "schema": "spectral-kernel-local-sweep-v1",
        "experimentId": EXPERIMENT_ID,
        "incrementId": INCREMENT_ID,
        "phase": "first-composition",
        "cohortId": COHORT_ID,
        "createdAtUtc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "question": (
            "Does the fixed FFTW++ four-target horizontal policy remain useful "
            "after directional vertical reconstruction/projection and required "
            "level movement are composed?"
        ),
        "baseline": (
            "The fixed explicit FFTW four-target control with identical "
            "directional split K2-grouped vertical providers."
        ),
        "changedVariables": [
            "explicit full-grid FFTW versus fixed FFTW++ implicit/hybrid horizontal convolution",
        ],
        "controlledVariables": [
            "256 squared, Nz=129, Nj=floor(2*(Nz-1)/3), and four output fields",
            "15 modal inputs, floor(N/3) radial retained disk, K2-grouped split GEMM, and outer-dynamic-12 vertical scheduling",
            "isolated processes, Float64 fixture, cold FFTW MEASURE planning, and one vertical-level streaming policy",
        ],
        "timedOperation": (
            "Ready retained/truncated 15-input modal coefficients through "
            "directional vertical reconstruction, four horizontal advective "
            "expressions per level, retained compact outputs, and directional "
            "vertical projection to four ready modal outputs."
        ),
        "componentLedger": [
            "raw inverse vertical GEMM for 15 inputs",
            "one-level primitive or fused horizontal operator",
            "all-level packing and output scatter",
            "vertically batched horizontal stage",
            "raw forward vertical GEMM for four outputs",
            "authoritative uninstrumented composed total",
        ],
        "excludedWork": [
            "phase evolution and coefficient-space flux accumulation",
            "remaining nonlinear-flux bookkeeping and complete time stepping",
            "Float32, GPU work, and general-Mac conclusions",
        ],
        "allocationPolicy": (
            "Directional vertical matrices/buffers, horizontal plans, worker "
            "pools, level adapter, and outputs are persistent; focused allocator "
            "interposition verifies zero steady-state application allocations."
        ),
        "selectionRule": (
            "Continue to reference depth when correct and either composed time "
            "is <=0.98x, or algorithm-resident memory is <=0.80x while time is <=1.05x."
        ),
        "profile": PROFILE,
        "candidates": [asdict(candidate) for candidate in candidates()],
        "sourceTreeGitCommit": source_commit,
        "sourceTreeDirty": source_dirty,
        "physicalMemoryBytes": physical_memory,
        "safeMemoryLimitBytes": safe_limit,
        "estimatedProcessPeakBytes": estimates,
        "rounds": 1,
        "warmups": arguments.warmups,
        "samples": arguments.samples,
        "seed": arguments.seed,
        "runs": [],
    }

    environment = os.environ.copy()
    environment["VECLIB_MAXIMUM_THREADS"] = "1"
    failed = False
    results = []
    for index, (candidate, result_path, command) in enumerate(commands, start=1):
        print(f"[{index}/{len(commands)}] {candidate.id}", flush=True)
        log_path = arguments.output / f"{candidate.id}.log"
        with log_path.open("w", encoding="utf-8") as log:
            completed = subprocess.run(
                command, cwd=repository_root, env=environment,
                stdout=log, stderr=subprocess.STDOUT,
            )
        entry = {
            "id": candidate.id,
            "round": 1,
            "profile": PROFILE,
            "candidate": asdict(candidate),
            "primaryProvider": candidate.provider_id,
            "command": command,
            "exitCode": completed.returncode,
            "log": log_path.name,
            "sourceTreeGitCommit": source_commit,
            "sourceTreeDirty": source_dirty,
        }
        if result_path.is_file():
            entry["result"] = result_path.name
            with result_path.open(encoding="utf-8") as stream:
                result = json.load(stream)
            results.append((candidate, result))
        else:
            failed = True
        failed = failed or completed.returncode != 0
        manifest["runs"].append(entry)

    if len(results) == len(commands):
        analysis = analyze(results)
        with (arguments.output / "analysis.json").open(
            "w", encoding="utf-8"
        ) as stream:
            json.dump(analysis, stream, indent=2)
            stream.write("\n")
        manifest["analysis"] = "analysis.json"
        print(json.dumps(analysis, indent=2))
    else:
        failed = True
    with (arguments.output / "manifest.json").open(
        "w", encoding="utf-8"
    ) as stream:
        json.dump(manifest, stream, indent=2)
        stream.write("\n")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
