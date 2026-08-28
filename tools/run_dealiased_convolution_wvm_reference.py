#!/usr/bin/env python3
"""Run the fixed-policy issue #17 WVM-derived convolution reference campaign."""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

from run_fftw_native_order_sweep import rotated
from run_spectral_pipeline_sweep import (
    geometric_mean,
    maximum_correctness_error,
    percentile,
    stratified_geometric_bootstrap,
)
from run_vertical_gemm_sweep import git_source_state


EXPERIMENT_ID = "issue-017-implicit-hybrid-dealiased-convolution"
INCREMENT_ID = "wvm-derived-horizontal-advection-reference-v1"
COHORT_ID = "issue17-wvm-advection-three-profile-f4-reference-v1"
PROFILES = (
    "wvm-current-256-nz129-f4",
    "wvm-current-512-nz257-f4",
    "wvm-large-1024-nz129-f4",
)
REFERENCE_ROUNDS = 3
TOTAL_STAGE = "WVM-like four-target horizontal advection"
EXPLICIT_ID = "fftw-explicit-parallel-target-wvm-advection"
FFTWPP_ID = "fftwpp-parallel-target-wvm-advection"


@dataclass(frozen=True)
class Candidate:
    id: str
    cli_value: str
    primary_provider: str


def reference_candidates() -> list[Candidate]:
    return [
        Candidate("explicit-parallel", "explicit-parallel", EXPLICIT_ID),
        Candidate("fftwpp-parallel", "fftwpp-parallel", FFTWPP_ID),
    ]


def provider_record(candidate: Candidate, result: dict) -> dict:
    providers = [
        provider for provider in result["providers"]
        if provider["id"] == candidate.primary_provider
    ]
    if len(providers) != 1 or len(result["providers"]) != 1:
        raise ValueError(
            f"{candidate.id} result must contain exactly its one finalist provider"
        )
    return providers[0]


def stage_seconds(provider: dict, scope: str, stage: str) -> float | None:
    matches = [
        item for item in provider.get("timings", [])
        if item.get("scope") == scope
        and item.get("stage") == stage
        and item.get("direction") == "forward"
    ]
    if len(matches) > 1:
        raise ValueError(f"provider {provider['id']} repeats {scope}/{stage}")
    return None if not matches else float(matches[0]["medianSeconds"])


def total_seconds(provider: dict) -> float:
    value = stage_seconds(provider, "uninstrumented-total", TOTAL_STAGE)
    if value is None:
        raise ValueError(f"provider {provider['id']} lacks its authoritative total")
    return value


def memory_record(provider: dict) -> dict[str, int]:
    memory = provider.get("memory", {})
    required = (
        "algorithmResidentBytes",
        "benchmarkHarnessBytes",
        "estimatedProcessPeakBytes",
        "observedProcessHighWaterBytes",
    )
    if any(int(memory.get(key, 0)) <= 0 for key in required):
        raise ValueError(f"provider {provider['id']} lacks reference memory fields")
    return {key: int(memory[key]) for key in required}


def placement_valid(provider: dict) -> bool:
    contract = provider.get("executionContract", {}).get("forward", {})
    return bool(
        contract.get("adapterPlacement") == "out-of-place"
        and contract.get("adapterPreservesCallerInput") is True
        and contract.get("requiresPreservationCopyForRepeatedExecution") is True
        and contract.get("preservationIncludedInAdapterTiming") is True
    )


def percentile_bootstrap(
    values: list[float], seed: int = 129, resamples: int = 20000,
) -> tuple[float, float]:
    if not values:
        raise ValueError("percentile bootstrap requires values")
    generator = random.Random(seed)
    draws = [
        statistics.median(generator.choices(values, k=len(values)))
        for _ in range(resamples)
    ]
    return percentile(draws, 0.025), percentile(draws, 0.975)


def analyze(results: list[tuple[Candidate, int, dict]]) -> dict:
    cells: dict[tuple[str, str], list[dict]] = {}
    maximum_error = 0.0
    all_correct = True
    all_placements_valid = True
    for candidate, round_number, result in results:
        provider = provider_record(candidate, result)
        error = maximum_correctness_error(provider)
        maximum_error = max(maximum_error, error)
        all_correct = all_correct and math.isfinite(error) and error <= 1.0e-12
        all_placements_valid = all_placements_valid and placement_valid(provider)
        cells.setdefault((candidate.id, result["run"]["profile"]), []).append({
            "round": round_number,
            "seconds": total_seconds(provider),
            "planningSeconds": float(provider["planning"]["seconds"]),
            "memory": memory_record(provider),
            "maximumCorrectnessError": error,
            "fusedPrimitiveSeconds": stage_seconds(
                provider, "fused-primitive",
                "implicit hybrid transform-reduce-transform",
            ),
            "sharedAdvectorSeconds": stage_seconds(
                provider, "adapter-component",
                "shared advector embedding and inverse FFTs",
            ),
            "parallelTargetSeconds": stage_seconds(
                provider, "adapter-component",
                "parallel target transforms and reduction",
            ),
        })

    profiles: list[dict] = []
    profile_round_ratios: dict[str, list[float]] = {}
    complete = True
    for profile_index, profile in enumerate(PROFILES):
        baseline = {
            item["round"]: item
            for item in cells.get(("explicit-parallel", profile), [])
        }
        candidate = {
            item["round"]: item
            for item in cells.get(("fftwpp-parallel", profile), [])
        }
        rounds = sorted(set(baseline) & set(candidate))
        if (
            rounds != list(range(1, REFERENCE_ROUNDS + 1))
            or len(baseline) != REFERENCE_ROUNDS
            or len(candidate) != REFERENCE_ROUNDS
        ):
            complete = False
            continue
        baseline_times = [baseline[round_number]["seconds"] for round_number in rounds]
        candidate_times = [candidate[round_number]["seconds"] for round_number in rounds]
        ratios = [
            candidate[round_number]["seconds"] /
            baseline[round_number]["seconds"]
            for round_number in rounds
        ]
        profile_round_ratios[profile] = ratios
        lower, upper = percentile_bootstrap(ratios, seed=129 + profile_index)

        memory: dict[str, dict[str, float]] = {}
        for key in baseline[rounds[0]]["memory"]:
            baseline_value = statistics.median(
                baseline[round_number]["memory"][key]
                for round_number in rounds
            )
            candidate_value = statistics.median(
                candidate[round_number]["memory"][key]
                for round_number in rounds
            )
            memory[key] = {
                "baselineBytes": baseline_value,
                "candidateBytes": candidate_value,
                "candidateToBaseline": candidate_value / baseline_value,
            }

        baseline_median = statistics.median(baseline_times)
        candidate_median = statistics.median(candidate_times)
        profiles.append({
            "profile": profile,
            "baselineSeconds": baseline_median,
            "candidateSeconds": candidate_median,
            "candidateToBaseline": candidate_median / baseline_median,
            "roundRatios": ratios,
            "pairedBootstrap95": {"lower": lower, "upper": upper},
            "planningSeconds": {
                "baseline": statistics.median(
                    baseline[round_number]["planningSeconds"]
                    for round_number in rounds
                ),
                "candidate": statistics.median(
                    candidate[round_number]["planningSeconds"]
                    for round_number in rounds
                ),
            },
            "components": {
                "baselineSharedAdvectorSeconds": statistics.median(
                    baseline[round_number]["sharedAdvectorSeconds"]
                    for round_number in rounds
                ),
                "baselineParallelTargetSeconds": statistics.median(
                    baseline[round_number]["parallelTargetSeconds"]
                    for round_number in rounds
                ),
                "candidateFusedPrimitiveSeconds": statistics.median(
                    candidate[round_number]["fusedPrimitiveSeconds"]
                    for round_number in rounds
                ),
            },
            "memory": memory,
        })

    complete = complete and len(profiles) == len(PROFILES)
    time_ratios = [row["candidateToBaseline"] for row in profiles]
    resident_ratios = [
        row["memory"]["algorithmResidentBytes"]["candidateToBaseline"]
        for row in profiles
    ]
    observed_ratios = [
        row["memory"]["observedProcessHighWaterBytes"]["candidateToBaseline"]
        for row in profiles
    ]
    geometric_time = geometric_mean(time_ratios) if time_ratios else None
    maximum_time = max(time_ratios) if time_ratios else None
    geometric_resident = geometric_mean(resident_ratios) if resident_ratios else None
    geometric_observed = geometric_mean(observed_ratios) if observed_ratios else None
    confidence = None
    if complete:
        lower, upper = stratified_geometric_bootstrap(profile_round_ratios)
        confidence = {"lower": lower, "upper": upper}

    improvement_passed = bool(
        complete and geometric_time is not None and geometric_time <= 0.90
    )
    regression_passed = bool(
        complete and maximum_time is not None and maximum_time <= 1.03
    )
    confidence_passed = bool(
        confidence is not None and confidence["upper"] < 1.0
    )
    memory_passed = bool(
        complete and geometric_resident is not None
        and geometric_resident <= 0.80
    )
    passed = bool(
        improvement_passed and regression_passed and confidence_passed
        and memory_passed and all_correct and all_placements_valid
    )
    return {
        "schema": "spectral-kernel-dealiased-convolution-reference-analysis-v1",
        "experimentId": EXPERIMENT_ID,
        "incrementId": INCREMENT_ID,
        "phase": "reference",
        "cohortId": COHORT_ID,
        "completeProductionMatrix": complete,
        "allCorrectWithin1e12": all_correct,
        "allPlacementContractsValid": all_placements_valid,
        "maximumCorrectnessError": maximum_error,
        "geometricCandidateToBaseline": geometric_time,
        "maximumProfileCandidateToBaseline": maximum_time,
        "geometricAlgorithmResidentCandidateToBaseline": geometric_resident,
        "geometricObservedHighWaterCandidateToBaseline": geometric_observed,
        "stratifiedPairedBootstrap95": confidence,
        "profiles": profiles,
        "adoptionGate": {
            "geometricTimeRatioAtMost": 0.90,
            "maximumProfileTimeRatioAtMost": 1.03,
            "geometricAlgorithmResidentRatioAtMost": 0.80,
            "improvementPassed": improvement_passed,
            "regressionPassed": regression_passed,
            "confidenceExcludesTie": confidence_passed,
            "memoryReductionPassed": memory_passed,
            "correctnessPassed": all_correct,
            "placementPassed": all_placements_valid,
            "zeroSteadyStateAllocationRequired": True,
            "allocationVerification": "skbench-unit macOS allocator interposer",
            "adoptionCandidatePassed": passed,
            "sizeDependentDispatchAllowed": False,
            "crossMacReplicationStillRequired": True,
        },
    }


def command_for(
    executable: Path, candidate: Candidate, profile: str,
    warmups: int, samples: int, seed: int, output: Path,
) -> list[str]:
    return [
        str(executable), "run",
        "--kernel", "dealiased-convolution",
        "--convolution-map", "wvm-advection",
        "--convolution-candidate", candidate.cli_value,
        "--profile", profile,
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
        default=repository_root / "build" / "issue17" / "skbench",
    )
    parser.add_argument(
        "--test-executable", type=Path,
        default=repository_root / "build" / "issue17" / "skbench_tests",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--rounds", type=int, default=REFERENCE_ROUNDS)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--samples", type=int, default=21)
    parser.add_argument("--seed", type=int, default=129)
    parser.add_argument("--allow-dirty-tree", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    arguments = parser.parse_args()
    if arguments.rounds != REFERENCE_ROUNDS:
        parser.error(
            f"the preregistered reference campaign requires {REFERENCE_ROUNDS} rounds"
        )
    if arguments.warmups != 3 or arguments.samples != 21:
        parser.error(
            "the preregistered reference campaign requires 3 warmups and 21 samples"
        )
    if not arguments.executable.is_file():
        parser.error(f"benchmark executable is missing: {arguments.executable}")
    if not arguments.test_executable.is_file():
        parser.error(f"test executable is missing: {arguments.test_executable}")

    source_commit, source_dirty = git_source_state(repository_root)
    if source_dirty and not arguments.allow_dirty_tree:
        parser.error(
            "the source tree is dirty; commit and rebuild for reference evidence "
            "or use --allow-dirty-tree for a non-publishable diagnostic"
        )

    output = arguments.output or (
        repository_root / "results" / "local" /
        f"issue17-wvm-advection-reference-{timestamp}"
    )
    candidates = reference_candidates()
    commands = []
    profiles = list(PROFILES)
    for round_index in range(REFERENCE_ROUNDS):
        round_candidates = rotated(candidates, round_index)
        round_profiles = (
            profiles[round_index % len(profiles):]
            + profiles[:round_index % len(profiles)]
        )
        for profile in round_profiles:
            for candidate in round_candidates:
                stem = (
                    f"round-{round_index + 1}--{profile}--{candidate.id}"
                )
                result_path = output / f"{stem}.json"
                commands.append((
                    stem, round_index + 1, profile, candidate, result_path,
                    command_for(
                        arguments.executable, candidate, profile,
                        arguments.warmups, arguments.samples,
                        arguments.seed, result_path,
                    ),
                ))

    if arguments.dry_run:
        print(arguments.test_executable)
        for *_, command in commands:
            print(" ".join(str(value) for value in command))
        print(
            f"Planned {len(commands)} isolated reference run(s): "
            f"{REFERENCE_ROUNDS} rotated rounds, {len(PROFILES)} profiles, "
            f"and {len(candidates)} fixed finalists."
        )
        return 0

    output.mkdir(parents=True, exist_ok=False)
    allocation_log = output / "allocation-verification.log"
    with allocation_log.open("w", encoding="utf-8") as log:
        allocation_check = subprocess.run(
            [str(arguments.test_executable)], cwd=repository_root,
            stdout=log, stderr=subprocess.STDOUT,
        )
    if allocation_check.returncode != 0:
        print(allocation_log.read_text(encoding="utf-8")[-4000:], file=sys.stderr)
        return allocation_check.returncode

    manifest = {
        "schema": "spectral-kernel-local-sweep-v1",
        "experimentId": EXPERIMENT_ID,
        "incrementId": INCREMENT_ID,
        "phase": "reference",
        "cohortId": COHORT_ID,
        "createdAtUtc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "question": (
            "Does one fixed four-target FFTW++ implicit/hybrid policy reliably "
            "beat the matched persistent explicit FFTW control for the WVM-derived "
            "horizontal advection map while reducing resident storage?"
        ),
        "baseline": (
            "A finalist-only explicit FFTW process that reconstructs U, V, and W "
            "once before persistent four-target dispatch."
        ),
        "changedVariables": [
            "explicit full-grid transforms versus FFTW++ centered/Hermitian implicit-hybrid transforms",
        ],
        "controlledVariables": [
            "the four expressions -(U*q_x + V*q_y + W*q_z)",
            "15 ready compact inputs, four ready compact outputs, radial two-thirds retention, and Float64 fixtures",
            "four persistent outer-static target workers and one provider-internal worker",
            "one uniform candidate policy across 256 squared, 512 squared, and 1024 squared",
        ],
        "timedOperation": (
            "Ready compact retained horizontal spectra to four ready compact "
            "retained advective-flux spectra, including embedding, transforms, "
            "reduction, retention, and persistent dispatch."
        ),
        "componentLedger": [
            "candidate setup/planning outside the timed operation",
            "explicit shared-advector and parallel-target components",
            "FFTW++ fused transform-reduce-transform and caller-visible total",
            "algorithm-resident, harness-only, estimated-peak, and observed-high-water memory",
        ],
        "excludedWork": [
            "vertical reconstruction/projection, phase evolution, and coefficient accumulation",
            "the complete nonlinear flux, time stepping, I/O, Float32, and general-Mac claims",
        ],
        "allocationPolicy": (
            "The same-commit skbench-unit allocator interposer must pass before "
            "any campaign process; candidate plans, buffers, outputs, and worker "
            "pools are persistent."
        ),
        "adoptionGate": (
            "Require a complete matched three-round matrix, correctness within 1e-12, "
            "valid placement contracts, at least 10% geometric time improvement, "
            "no profile regression above 3%, a stratified paired-bootstrap 95% "
            "interval excluding a tie, and at least 20% resident-memory reduction."
        ),
        "profiles": list(PROFILES),
        "candidates": [asdict(candidate) for candidate in candidates],
        "candidateOrderRotatedByRound": True,
        "profileOrderRotatedByRound": True,
        "finalistOnlyProcesses": True,
        "allocationVerification": {
            "command": [str(arguments.test_executable)],
            "exitCode": allocation_check.returncode,
            "log": allocation_log.name,
        },
        "sourceTreeGitCommit": source_commit,
        "sourceTreeDirty": source_dirty,
        "rounds": REFERENCE_ROUNDS,
        "warmups": arguments.warmups,
        "samples": arguments.samples,
        "seed": arguments.seed,
        "runs": [],
    }

    failed = False
    completed_results: list[tuple[Candidate, int, dict]] = []
    for index, (
        stem, round_number, profile, candidate, result_path, command,
    ) in enumerate(commands, start=1):
        print(f"[{index}/{len(commands)}] {stem}", flush=True)
        log_path = output / f"{stem}.log"
        with log_path.open("w", encoding="utf-8") as log:
            completed = subprocess.run(
                command, cwd=repository_root,
                stdout=log, stderr=subprocess.STDOUT,
            )
        entry = {
            "id": stem,
            "round": round_number,
            "profile": profile,
            "candidate": asdict(candidate),
            "primaryProvider": candidate.primary_provider,
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
            if (
                completed.returncode == 0 and result["status"] == "passed"
                and metadata_matches
            ):
                completed_results.append((candidate, round_number, result))
            else:
                failed = True
        else:
            failed = True
        manifest["runs"].append(entry)
        with (output / "manifest.json").open("w", encoding="utf-8") as stream:
            json.dump(manifest, stream, indent=2)
            stream.write("\n")
        if failed and not arguments.continue_on_error:
            print(log_path.read_text(encoding="utf-8")[-4000:], file=sys.stderr)
            break

    if completed_results:
        with (output / "analysis.json").open("w", encoding="utf-8") as stream:
            json.dump(analyze(completed_results), stream, indent=2)
            stream.write("\n")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
