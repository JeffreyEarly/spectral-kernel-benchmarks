#!/usr/bin/env python3
"""Run the frozen issue #3 production FFTW reference campaign."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

from run_float64_baseline_sweep import PROFILES
from run_vertical_gemm_sweep import git_source_state, sysctl_integer


EXPERIMENT_ID = "issue-003-fftw-production-baseline"
INCREMENT_ID = "fftw-production-baseline-reference-v1"
CANDIDATE_ID = "wvm-interleaved-measure-internal-performance"
REFERENCE_ROUNDS = 3


def rotated(values: list[str], offset: int) -> list[str]:
    if not values:
        return []
    shift = offset % len(values)
    return values[shift:] + values[:shift]


def stage(provider: dict, scope: str, name: str, direction: str) -> dict:
    matches = [
        item for item in provider.get("timings", [])
        if item.get("scope") == scope
        and item.get("stage") == name
        and item.get("direction") == direction
    ]
    if len(matches) != 1:
        raise ValueError(
            f"provider {provider.get('id')} lacks one {scope}/{name}/{direction} timing"
        )
    return matches[0]


def maximum_correctness_error(provider: dict) -> float:
    metrics = provider.get("correctness", [])
    if not metrics or not all(metric.get("passed", False) for metric in metrics):
        return math.inf
    return max(float(metric["maximumRelativeError"]) for metric in metrics)


def analyze(results: list[tuple[int, dict]], performance_workers: int) -> dict:
    cells: dict[str, dict[int, dict]] = {}
    maximum_error = 0.0
    all_correct = True
    contracts_valid = True
    for round_number, result in results:
        provider = next(item for item in result["providers"] if item["id"] == "fftw")
        error = maximum_correctness_error(provider)
        maximum_error = max(maximum_error, error)
        all_correct = all_correct and math.isfinite(error) and error <= 1.0e-12
        scheduling = provider.get("scheduling", {})
        planning_configuration = provider.get("planning", {}).get("configuration", "")
        contracts_valid = contracts_valid and (
            planning_configuration.startswith("FFTW_MEASURE|FFTW_UNALIGNED")
            and scheduling.get("internalWorkers") == performance_workers
            and scheduling.get("outerWorkers") == 1
            and provider["executionContract"]["forward"]["nativePlacement"] == "out-of-place"
            and provider["executionContract"]["inverse"]["nativePlacement"] == "out-of-place"
        )
        cells.setdefault(result["run"]["profile"], {})[round_number] = {
            "forward": float(stage(provider, "primitive", "raw FFT", "forward")["medianSeconds"]),
            "inverse": float(stage(provider, "primitive", "raw FFT", "inverse")["medianSeconds"]),
            "setup": float(provider["setup"]["totalSeconds"]),
            "error": error,
        }

    rows: list[dict] = []
    complete = set(cells) == set(PROFILES)
    for profile in PROFILES:
        rounds = cells.get(profile, {})
        if sorted(rounds) != list(range(1, REFERENCE_ROUNDS + 1)):
            complete = False
            continue
        rows.append({
            "profile": profile,
            "forwardSeconds": statistics.median(item["forward"] for item in rounds.values()),
            "inverseSeconds": statistics.median(item["inverse"] for item in rounds.values()),
            "setupSeconds": statistics.median(item["setup"] for item in rounds.values()),
            "forwardRoundSeconds": [rounds[index]["forward"] for index in sorted(rounds)],
            "inverseRoundSeconds": [rounds[index]["inverse"] for index in sorted(rounds)],
        })

    return {
        "schema": "spectral-kernel-fftw-production-reference-analysis-v1",
        "experimentId": EXPERIMENT_ID,
        "incrementId": INCREMENT_ID,
        "candidateId": CANDIDATE_ID,
        "phase": "reference",
        "completeProductionMatrix": complete,
        "allCorrectWithin1e-12": all_correct,
        "productionContractPassed": contracts_valid,
        "maximumCorrectnessError": maximum_error,
        "profiles": rows,
        "referenceGate": {
            "threeIndependentProcessesPerProfile": complete,
            "correctnessPassed": all_correct,
            "productionContractPassed": contracts_valid,
            "zeroSteadyStateAllocationRequired": True,
            "allocationVerification": "skbench-unit macOS allocator interposer",
            "referenceBaselinePassed": bool(complete and all_correct and contracts_valid),
        },
    }


def command_for(
    executable: Path,
    profile: str,
    performance_workers: int,
    warmups: int,
    samples: int,
    seed: int,
    output: Path,
) -> list[str]:
    return [
        str(executable), "run",
        "--profile", profile,
        "--providers", "fftw",
        "--fftw-layout", "interleaved",
        "--fftw-spectrum-order", "wvm",
        "--fftw-planning", "measure",
        "--fftw-alignment", "unaligned",
        "--fftw-wisdom", "cold",
        "--fftw-internal-workers", str(performance_workers),
        "--fftw-outer-workers", "1",
        "--warmups", str(warmups),
        "--samples", str(samples),
        "--seed", str(seed),
        "--output", str(output),
    ]


def main() -> int:
    repository_root = Path(__file__).resolve().parents[1]
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--executable", type=Path,
        default=repository_root / "build/release/skbench",
    )
    parser.add_argument(
        "--output", type=Path,
        default=repository_root / "results/local" / f"issue3-production-reference-{timestamp}",
    )
    parser.add_argument("--profiles", nargs="*")
    parser.add_argument(
        "--performance-workers", type=int,
        default=sysctl_integer("hw.perflevel0.physicalcpu", 1),
    )
    parser.add_argument("--rounds", type=int, default=REFERENCE_ROUNDS)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--samples", type=int, default=21)
    parser.add_argument("--seed", type=int, default=129)
    parser.add_argument("--allow-dirty-tree", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    arguments = parser.parse_args()

    profiles = arguments.profiles or list(PROFILES)
    unknown = sorted(set(profiles) - set(PROFILES))
    if unknown:
        parser.error(f"unknown profile: {', '.join(unknown)}")
    if min(arguments.performance_workers, arguments.rounds, arguments.warmups, arguments.samples) < 1:
        parser.error("workers, rounds, warmups, and samples must be positive")

    source_commit, source_dirty = git_source_state(repository_root)
    if source_dirty and not arguments.allow_dirty_tree:
        parser.error(
            "the benchmark source tree is dirty; commit and rebuild for evidence collection "
            "or use --allow-dirty-tree for an exploratory run"
        )

    commands: list[tuple[str, int, str, list[str], Path]] = []
    for round_index in range(arguments.rounds):
        for profile in rotated(profiles, round_index):
            stem = f"round-{round_index + 1}--{profile}--{CANDIDATE_ID}"
            result_path = arguments.output / f"{stem}.json"
            commands.append((
                stem,
                round_index + 1,
                profile,
                command_for(
                    arguments.executable, profile, arguments.performance_workers,
                    arguments.warmups, arguments.samples, arguments.seed, result_path,
                ),
                result_path,
            ))

    if arguments.dry_run:
        for _, _, _, command, _ in commands:
            print(" ".join(command))
        print(
            f"Planned {len(commands)} isolated run(s): {arguments.rounds} round(s), "
            f"{len(profiles)} profile(s), internal={arguments.performance_workers}, outer=1."
        )
        return 0

    arguments.output.mkdir(parents=True, exist_ok=False)
    manifest = {
        "schema": "spectral-kernel-local-sweep-v1",
        "experimentId": EXPERIMENT_ID,
        "incrementId": INCREMENT_ID,
        "phase": "reference",
        "createdAtUtc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "question": "What are the reference raw forward and inverse costs of WVM's exact production FFTW contract across the complete ten-profile matrix?",
        "baseline": "The preliminary issue #3 matrix and preserved WVM issue #129 vertical slice.",
        "controlledVariables": [
            "pinned FFTW 3.3.11 Float64 build and WVM guru64 frequency-major interleaved strides",
            "FFTW_MEASURE | FFTW_UNALIGNED, cold planning, out-of-place placement",
            "fixture seed, normalization, oracle, warmups, samples, and performance-core topology",
        ],
        "changedVariables": ["Nx, Ny, Nz, field count, and transform direction"],
        "timedOperations": ["raw forward FFT", "raw inverse FFT"],
        "excludedWork": [
            "planning, allocation, retention, representation conversion, vertical projection, modal work, and nonlinear flux",
        ],
        "referenceProtocol": {
            "independentlyPlannedProcesses": arguments.rounds,
            "profileOrderRotatedByRound": True,
            "warmupsPerProcess": arguments.warmups,
            "samplesPerProcess": arguments.samples,
        },
        "profiles": profiles,
        "performanceWorkers": arguments.performance_workers,
        "sourceTreeGitCommit": source_commit,
        "sourceTreeDirty": source_dirty,
        "runs": [],
    }

    failed = False
    completed_results: list[tuple[int, dict]] = []
    for index, (stem, round_number, profile, command, result_path) in enumerate(commands, start=1):
        print(f"[{index}/{len(commands)}] {stem}", flush=True)
        log_path = arguments.output / f"{stem}.log"
        with log_path.open("w", encoding="utf-8") as log:
            completed = subprocess.run(
                command, cwd=repository_root, stdout=log, stderr=subprocess.STDOUT,
            )
        entry = {
            "id": stem,
            "round": round_number,
            "profile": profile,
            "candidate": {"id": CANDIDATE_ID},
            "primaryProvider": "fftw",
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
            entry.update({
                "runId": result["run"]["id"],
                "status": result["status"],
                "result": result_path.name,
                "sourceMetadataMatches": metadata_matches,
            })
            if result["status"] == "passed" and metadata_matches:
                completed_results.append((round_number, result))
            else:
                completed = subprocess.CompletedProcess(command, 1)
        manifest["runs"].append(entry)
        with (arguments.output / "manifest.json").open("w", encoding="utf-8") as stream:
            json.dump(manifest, stream, indent=2)
            stream.write("\n")
        if completed.returncode != 0:
            failed = True
            print(log_path.read_text(encoding="utf-8")[-4000:], file=sys.stderr)
            if not arguments.continue_on_error:
                break

    if completed_results:
        with (arguments.output / "analysis.json").open("w", encoding="utf-8") as stream:
            json.dump(analyze(completed_results, arguments.performance_workers), stream, indent=2)
            stream.write("\n")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
