#!/usr/bin/env python3
"""Preflight and run the non-reference authoritative issue #19 scale-out."""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

from prepare_spectral_flux_fixture import prepare
from run_spectral_pipeline_sweep import maximum_correctness_error
from run_vertical_gemm_sweep import git_source_state


EXPERIMENT_ID = "issue-019-production-lifetime-spectral-flux-composition"
INCREMENT_ID = "production-lifetime-flux-authoritative-scaleout-v1"
CAPACITY_SCHEMA = "spectral-flux-fixture-capacity-v1"
TOLERANCE = 1.0e-12


@dataclass(frozen=True)
class Workload:
    profile: str
    nx: int
    ny: int
    nz: int


@dataclass(frozen=True)
class Candidate:
    id: str
    policy: str
    provider: str


WORKLOADS = (
    Workload("wvm-current-512-nz257-f4", 512, 512, 257),
    Workload("wvm-large-1024-nz129-f4", 1024, 1024, 129),
    Workload("wvm-large-512-nz513-f4", 512, 512, 513),
)

CANDIDATES = (
    Candidate(
        "production-lifetime-wvm-direct-authoritative",
        "wvm-direct",
        "pipeline-production-lifetime-wvm-direct-authoritative",
    ),
    Candidate(
        "production-lifetime-streaming-pruned-tile16-authoritative",
        "streaming-pruned-compact-split",
        "pipeline-production-lifetime-streaming-pruned-tile16-authoritative",
    ),
)


def matlab_quote(value: Path | str) -> str:
    return str(value).replace("'", "''")


def physical_memory_bytes() -> int:
    completed = subprocess.run(
        ["sysctl", "-n", "hw.memsize"], check=True,
        text=True, capture_output=True,
    )
    return int(completed.stdout.strip())


def matlab_capacity(matlab: str, wvm_repository: Path,
                    workload: Workload) -> dict:
    expression = (
        "restoredefaultpath;"
        f"cd('{matlab_quote(wvm_repository)}');"
        "addpath(pwd,fullfile(pwd,'Benchmarks'));"
        f"c=spectralFluxFixtureCapacity(Nxyz=[{workload.nx} "
        f"{workload.ny} {workload.nz}]);"
        "fprintf('SKBENCH_CAPACITY_JSON=%s\\n',jsonencode(c));"
    )
    completed = subprocess.run(
        [matlab, "-batch", expression], check=True,
        text=True, capture_output=True,
    )
    marker = "SKBENCH_CAPACITY_JSON="
    matches = [line[len(marker):] for line in completed.stdout.splitlines()
               if line.startswith(marker)]
    if len(matches) != 1:
        raise ValueError(
            f"MATLAB did not return one capacity record for {workload.profile}"
        )
    result = json.loads(matches[0])
    if result.get("schema") != CAPACITY_SCHEMA:
        raise ValueError(f"unexpected capacity schema for {workload.profile}")
    return result


def graph_capacity(capacity: dict, physical_bytes: int) -> dict:
    workload = capacity["workload"]
    nx = int(workload["Nx"])
    ny = int(workload["Ny"])
    nz = int(workload["Nz"])
    nkl = int(workload["Nkl"])
    nj = int(workload["Nj"])
    segments = int(workload["canonicalGroupSegmentCount"])
    source_operator = int(capacity["payloadBytes"]["verticalOperators"])
    canonical_operator = 32 * nz * nj * segments
    fixture_modal = 16 * nj * nkl * 19
    packed_inputs = 16 * nj * nkl * 15
    real_volume = 8 * nx * ny * nz

    split_operands = 16 * nkl * (nz + nj) * 19
    split_scratch = 16 * nkl * nz * 4 + 7 * real_volume
    split_steady = canonical_operator + split_operands + split_scratch
    split_construction = (
        source_operator // 2 + canonical_operator + split_operands +
        fixture_modal + packed_inputs
    )
    split_post_setup = split_steady + fixture_modal + packed_inputs
    split_peak = max(split_construction, split_post_setup)

    half_rows = (nx // 2 + 1) * ny
    direct_arrays = 16 * half_rows * (nz + nj) * 19
    direct_scratch = 16 * half_rows * nz * 4 + 7 * real_volume
    direct_steady = 2 * canonical_operator + direct_arrays + direct_scratch
    direct_construction = (
        source_operator // 2 + 2 * canonical_operator + direct_arrays +
        fixture_modal + packed_inputs
    )
    direct_post_setup = direct_steady + fixture_modal + packed_inputs
    direct_peak = max(direct_construction, direct_post_setup)

    opaque_reserve = max(4 * 1024**3, math.ceil(0.05 * physical_bytes))

    def record(steady: int, peak: int) -> dict:
        required = peak + opaque_reserve
        return {
            "estimatedSteadyExplicitBytes": steady,
            "estimatedSetupPeakExplicitBytes": peak,
            "opaqueAndSystemReserveBytes": opaque_reserve,
            "requiredPhysicalMemoryBytes": required,
            "physicalMemoryBytes": physical_bytes,
            "feasible": required <= physical_bytes,
            "classification": (
                "preflight-feasible" if required <= physical_bytes
                else "capacity-exclusion"
            ),
        }

    return {
        "canonicalOperatorBytesAfterModeOrdering": canonical_operator,
        CANDIDATES[0].id: record(direct_steady, direct_peak),
        CANDIDATES[1].id: record(split_steady, split_peak),
        "method": (
            "Exact explicit arrays from the frozen graph and WVM source/canonical "
            "group counts, plus max(4 GiB, 5% physical memory) for opaque providers "
            "and the operating system. This is a conservative allocation preflight, "
            "not an observed high-water measurement."
        ),
    }


def preflight(matlab: str, wvm_repository: Path, output: Path,
              physical_memory_override: int | None = None) -> dict:
    physical = physical_memory_override or physical_memory_bytes()
    free_disk = shutil.disk_usage(output).free
    records = []
    disk_required = 0
    for workload in WORKLOADS:
        capacity = matlab_capacity(matlab, wvm_repository, workload)
        disk_required += int(
            capacity["diskBytes"]["recommendedFreeForSourceAndPrepared"]
        )
        records.append({
            "profile": workload.profile,
            "fixture": capacity,
            "graphs": graph_capacity(capacity, physical),
        })
    result = {
        "schema": "spectral-kernel-authoritative-scaleout-preflight-v1",
        "createdAtUtc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "physicalMemoryBytes": physical,
        "freeDiskBytesAtPreflight": free_disk,
        "recommendedAggregateDiskBytes": disk_required,
        "aggregateDiskFeasible": disk_required <= free_disk,
        "workloads": records,
    }
    (output / "preflight.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result


def export_fixture(matlab: str, wvm_repository: Path, root: Path,
                   workload: Workload) -> None:
    fixture = root / workload.profile / "fixture"
    manifest = fixture / "manifest.json"
    if manifest.is_file():
        return
    if fixture.exists() and any(fixture.iterdir()):
        raise ValueError(
            f"partial fixture requires manual recovery before resume: {fixture}"
        )
    fixture.parent.mkdir(parents=True, exist_ok=True)
    expression = (
        "restoredefaultpath;"
        f"cd('{matlab_quote(wvm_repository)}');"
        "addpath(pwd,fullfile(pwd,'Benchmarks'));"
        f"m=exportSpectralFluxFixture('{matlab_quote(fixture)}',"
        f"Nxyz=[{workload.nx} {workload.ny} {workload.nz}],seed=19019);"
        "assert(m.authoritative);"
    )
    log_path = root / workload.profile / "export.log"
    with log_path.open("w", encoding="utf-8") as log:
        completed = subprocess.run(
            [matlab, "-batch", expression], cwd=wvm_repository,
            stdout=log, stderr=subprocess.STDOUT,
        )
    if completed.returncode != 0 or not manifest.is_file():
        raise RuntimeError(
            f"authoritative export failed for {workload.profile}; see {log_path}"
        )


def command_for(executable: Path, prepared: Path, workload: Workload,
                candidate: Candidate, result: Path) -> list[str]:
    return [
        str(executable), "run",
        "--kernel", "production-lifetime-flux",
        "--profile", workload.profile,
        "--boundary-policy", candidate.policy,
        "--spectral-flux-fixture", str(prepared),
        "--fftw-planning", "measure",
        "--fftw-alignment", "unaligned",
        "--fftw-wisdom", "cold",
        "--fftw-internal-workers", "1",
        "--fftw-outer-workers", "12",
        "--streaming-tile-width", "16",
        "--vertical-gemm-family", "k2-grouped",
        "--vertical-gemm-schedule", "outer-dynamic",
        "--vertical-gemm-outer-workers", "16",
        "--warmups", "1",
        "--samples", "1",
        "--output", str(result),
    ]


def provider_result(path: Path, candidate: Candidate) -> dict:
    result = json.loads(path.read_text(encoding="utf-8"))
    providers = result.get("providers", [])
    if len(providers) != 1 or providers[0].get("id") != candidate.provider:
        raise ValueError(f"unexpected provider record in {path}")
    error = maximum_correctness_error(providers[0])
    fixture = result.get("provenance", {}).get("spectralFluxFixture", {})
    passed = bool(
        result.get("status") == "passed" and math.isfinite(error)
        and error <= TOLERANCE and fixture.get("authoritative") is True
        and fixture.get("schema") == "spectral-flux-fixture-v1"
    )
    return {
        "result": path.name,
        "status": result.get("status"),
        "maximumCorrectnessError": error,
        "passed": passed,
        "fixtureHash": fixture.get("fixtureHash"),
        "waveVortexModelCommit": fixture.get("waveVortexModelCommit"),
        "observedProcessHighWaterBytes": int(
            providers[0].get("memory", {}).get(
                "observedProcessHighWaterBytes", 0
            )
        ),
        "algorithmResidentBytes": int(
            providers[0].get("memory", {}).get("algorithmResidentBytes", 0)
        ),
    }


def correctness_trials(executable: Path, root: Path, preflight_record: dict,
                       allow_dirty_tree: bool) -> dict:
    source_commit, source_dirty = git_source_state(
        Path(__file__).resolve().parents[1]
    )
    if source_dirty and not allow_dirty_tree:
        raise ValueError(
            "commit and rebuild the benchmark before authoritative trials"
        )
    preflight_by_profile = {
        item["profile"]: item for item in preflight_record["workloads"]
    }
    trials = []
    for workload in WORKLOADS:
        workload_root = root / workload.profile
        fixture = workload_root / "fixture"
        prepared = workload_root / "prepared-fixture.bin"
        if not prepared.is_file():
            prepare(fixture, prepared)
        graph_capacity_record = preflight_by_profile[workload.profile]["graphs"]
        runs = []
        for candidate in CANDIDATES:
            feasibility = graph_capacity_record[candidate.id]
            if not feasibility["feasible"]:
                runs.append({
                    "candidate": asdict(candidate),
                    "status": "capacity-exclusion",
                    "capacity": feasibility,
                })
                continue
            result_path = workload_root / f"correctness--{candidate.id}.json"
            if not result_path.is_file():
                log_path = result_path.with_suffix(".log")
                environment = os.environ.copy()
                environment["VECLIB_MAXIMUM_THREADS"] = "1"
                with log_path.open("w", encoding="utf-8") as log:
                    completed = subprocess.run(
                        command_for(
                            executable, prepared, workload, candidate,
                            result_path,
                        ),
                        cwd=Path(__file__).resolve().parents[1],
                        env=environment, stdout=log,
                        stderr=subprocess.STDOUT,
                    )
                if completed.returncode != 0:
                    runs.append({
                        "candidate": asdict(candidate),
                        "status": "failed",
                        "log": log_path.name,
                    })
                    continue
            runs.append({
                "candidate": asdict(candidate),
                **provider_result(result_path, candidate),
            })
        trials.append({"profile": workload.profile, "runs": runs})
    result = {
        "schema": "spectral-kernel-authoritative-scaleout-correctness-v1",
        "experimentId": EXPERIMENT_ID,
        "incrementId": INCREMENT_ID,
        "classification": "authoritative-non-reference-correctness-scaleout",
        "eligibleForReference": False,
        "adoptionGateEvaluated": False,
        "timingInterpretation": (
            "Each feasible graph used one warmup and one measured execution only "
            "to exercise its complete boundary. Timing values are diagnostics and "
            "must not enter reference or adoption statistics."
        ),
        "sourceTreeGitCommit": source_commit,
        "sourceTreeDirty": source_dirty,
        "trials": trials,
    }
    (root / "analysis.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result


def main() -> int:
    repository = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wvm-repository", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--executable", type=Path,
                        default=repository / "build/release/skbench")
    parser.add_argument("--matlab", default="matlab")
    parser.add_argument("--physical-memory-bytes", type=int)
    parser.add_argument(
        "--phase", choices=("preflight", "export", "correctness", "all"),
        default="all",
    )
    parser.add_argument("--allow-dirty-tree", action="store_true")
    arguments = parser.parse_args()
    arguments.output.mkdir(parents=True, exist_ok=True)
    if not arguments.wvm_repository.is_dir():
        parser.error("--wvm-repository must be a directory")
    wvm_commit, wvm_dirty = git_source_state(arguments.wvm_repository)
    if wvm_dirty:
        parser.error("the WVM repository must be clean for authoritative export")

    preflight_path = arguments.output / "preflight.json"
    if arguments.phase in {"preflight", "all"} or not preflight_path.is_file():
        preflight_record = preflight(
            arguments.matlab, arguments.wvm_repository, arguments.output,
            arguments.physical_memory_bytes,
        )
    else:
        preflight_record = json.loads(preflight_path.read_text(encoding="utf-8"))
    if not preflight_record["aggregateDiskFeasible"]:
        parser.error("aggregate source-plus-prepared fixture disk preflight failed")
    if arguments.phase == "preflight":
        print(json.dumps(preflight_record, indent=2, sort_keys=True))
        return 0

    if arguments.phase in {"export", "all"}:
        for workload in WORKLOADS:
            print(f"export {workload.profile}", flush=True)
            export_fixture(
                arguments.matlab, arguments.wvm_repository,
                arguments.output, workload,
            )
    if arguments.phase == "export":
        return 0

    if not arguments.executable.is_file():
        parser.error(f"benchmark executable is missing: {arguments.executable}")
    analysis = correctness_trials(
        arguments.executable.resolve(), arguments.output,
        preflight_record, arguments.allow_dirty_tree,
    )
    print(json.dumps(analysis, indent=2, sort_keys=True))
    return 0 if all(
        run["status"] == "capacity-exclusion" or run.get("passed") is True
        for trial in analysis["trials"] for run in trial["runs"]
    ) else 1


if __name__ == "__main__":
    raise SystemExit(main())
