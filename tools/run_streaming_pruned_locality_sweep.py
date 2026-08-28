#!/usr/bin/env python3
"""Screen bounded compact tiles for the issue #16 streaming pipeline."""

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

from run_spectral_pipeline_sweep import (
    Candidate,
    estimated_explicit_peak_bytes as issue9_estimated_peak,
    geometric_mean,
    maximum_correctness_error,
    provider_record,
    provider_timing,
)
from run_vertical_gemm_sweep import (
    PROFILE_SHAPES,
    gibibytes,
    git_source_state,
    sysctl_uint64,
)


EXPERIMENT_ID = "issue-016-streaming-pruned-compact-split"
INCREMENT_ID = "streaming-pruned-compact-split-locality-screen-v1"
COHORT_ID = "issue16-locality-three-profile-f4-v1"
BASELINE_ID = "plane-major-fused-split--outer-dynamic-16"
DIRECT_ID = "streaming-pruned-direct-1--outer-dynamic-16"
PROFILES = (
    "wvm-current-256-nz129-f4",
    "wvm-current-512-nz257-f4",
    "wvm-large-1024-nz129-f4",
)
LARGE_PROFILES = (
    "wvm-current-512-nz257-f4",
    "wvm-large-1024-nz129-f4",
)


@dataclass(frozen=True)
class LocalityCandidate:
    id: str
    policy: str
    primary_provider: str
    role: str
    tile_width: int


def candidate_matrix() -> list[LocalityCandidate]:
    candidates = [
        LocalityCandidate(
            BASELINE_ID,
            "plane-major-fused-split",
            "pipeline-plane-major-fused-split",
            "same-commit-issue9-control",
            0,
        ),
        LocalityCandidate(
            DIRECT_ID,
            "streaming-pruned-compact-split",
            "pipeline-streaming-pruned-compact-split",
            "issue16-page-strided-direct-control",
            1,
        ),
    ]
    candidates.extend(
        LocalityCandidate(
            f"streaming-pruned-tiled-{tile_width}--outer-dynamic-16",
            "streaming-pruned-compact-split",
            "pipeline-streaming-pruned-compact-split",
            "issue16-plane-major-compact-tile-candidate",
            tile_width,
        )
        for tile_width in (4, 8, 16)
    )
    return candidates


def command_for(
    executable: Path,
    candidate: LocalityCandidate,
    profile: str,
    warmups: int,
    samples: int,
    seed: int,
    result_path: Path,
) -> list[str]:
    command = [
        str(executable), "run",
        "--kernel", "spectral-pipeline",
        "--boundary-policy", candidate.policy,
        "--profile", profile,
        "--fftw-planning", "measure",
        "--fftw-alignment", "unaligned",
        "--fftw-wisdom", "cold",
        "--fftw-internal-workers", "1",
        "--fftw-outer-workers", "12",
        "--vertical-gemm-family", "k2-grouped",
        "--vertical-gemm-schedule", "outer-dynamic",
        "--vertical-gemm-outer-workers", "16",
        "--warmups", str(warmups),
        "--samples", str(samples),
        "--seed", str(seed),
        "--output", str(result_path),
    ]
    if candidate.tile_width > 0:
        command[4:4] = [
            "--streaming-tile-width", str(candidate.tile_width),
        ]
    return command


def estimated_explicit_peak_bytes(
    profile: str, candidate: LocalityCandidate,
) -> int:
    baseline = issue9_estimated_peak(profile, "plane-major-fused-split")
    if candidate.id == BASELINE_ID:
        return baseline
    nx, nz, fields, retained_modes, _ = PROFILE_SHAPES[profile]
    full_spectrum = nx * (nx // 2 + 1) * nz * fields * 16
    worker_fft_scratch = 12 * nx * (nx // 2 + 1) * 16
    compact_tile = (
        0 if candidate.tile_width == 1
        else 12 * candidate.tile_width * retained_modes * 16
    )
    return baseline - full_spectrum + worker_fft_scratch + compact_tile


def required_memory(provider: dict) -> dict[str, int]:
    keys = (
        "algorithmResidentBytes",
        "benchmarkHarnessBytes",
        "estimatedProcessPeakBytes",
        "observedProcessHighWaterBytes",
    )
    memory = provider.get("memory", {})
    missing = [key for key in keys if int(memory.get(key, 0)) <= 0]
    if missing:
        raise ValueError(
            f"provider {provider['id']} lacks positive memory fields: "
            f"{', '.join(missing)}"
        )
    return {key: int(memory[key]) for key in keys}


def candidate_provider(candidate: LocalityCandidate, result: dict) -> dict:
    compatible = Candidate(
        candidate.id,
        candidate.policy,
        candidate.primary_provider,
        candidate.role,
    )
    return provider_record(compatible, result)


def analyze(results: list[tuple[LocalityCandidate, dict]]) -> dict:
    candidates = candidate_matrix()
    cells: dict[tuple[str, str], dict] = {}
    maximum_error = 0.0
    all_correct = True
    for candidate, result in results:
        provider = candidate_provider(candidate, result)
        error = maximum_correctness_error(provider)
        maximum_error = max(maximum_error, error)
        all_correct = all_correct and math.isfinite(error) and error <= 1.0e-12
        cells[(candidate.id, result["run"]["profile"])] = {
            "seconds": provider_timing(provider),
            "memory": required_memory(provider),
            "maximumCorrectnessError": error,
        }

    expected = {
        (candidate.id, profile)
        for candidate in candidates
        for profile in PROFILES
    }
    complete = set(cells) == expected
    profiles: list[dict] = []
    for profile in PROFILES:
        baseline = cells.get((BASELINE_ID, profile))
        direct = cells.get((DIRECT_ID, profile))
        if baseline is None or direct is None:
            continue
        candidate_rows = []
        for candidate in candidates:
            cell = cells.get((candidate.id, profile))
            if cell is None:
                continue
            candidate_rows.append({
                "candidateId": candidate.id,
                "tileWidth": candidate.tile_width,
                "seconds": cell["seconds"],
                "toBaseline": cell["seconds"] / baseline["seconds"],
                "toDirect": cell["seconds"] / direct["seconds"],
                "memory": {
                    key: {
                        "bytes": value,
                        "toBaseline": value / baseline["memory"][key],
                    }
                    for key, value in cell["memory"].items()
                },
                "maximumCorrectnessError":
                    cell["maximumCorrectnessError"],
            })
        profiles.append({"profile": profile, "candidates": candidate_rows})

    summaries = []
    for candidate in candidates:
        rows_by_profile = {
            profile["profile"]: next(
                (
                    row for row in profile["candidates"]
                    if row["candidateId"] == candidate.id
                ),
                None,
            )
            for profile in profiles
        }
        if any(row is None for row in rows_by_profile.values()) or \
                len(rows_by_profile) != len(PROFILES):
            continue
        rows = [rows_by_profile[profile] for profile in PROFILES]
        large_rows = [rows_by_profile[profile] for profile in LARGE_PROFILES]
        summaries.append({
            "candidateId": candidate.id,
            "tileWidth": candidate.tile_width,
            "geometricToBaseline": geometric_mean(
                [row["toBaseline"] for row in rows]
            ),
            "largeGeometricToBaseline": geometric_mean(
                [row["toBaseline"] for row in large_rows]
            ),
            "geometricToDirect": geometric_mean(
                [row["toDirect"] for row in rows]
            ),
            "largeGeometricToDirect": geometric_mean(
                [row["toDirect"] for row in large_rows]
            ),
            "geometricAlgorithmResidentToBaseline": geometric_mean([
                row["memory"]["algorithmResidentBytes"]["toBaseline"]
                for row in rows
            ]),
        })

    tiled_summaries = [
        summary for summary in summaries
        if summary["tileWidth"] in (4, 8, 16)
        and summary["geometricAlgorithmResidentToBaseline"] <= 0.90
    ]
    selected = min(
        tiled_summaries,
        key=lambda summary: (
            summary["largeGeometricToBaseline"],
            summary["geometricToBaseline"],
            summary["tileWidth"],
        ),
        default=None,
    )
    optimization_successful = bool(
        complete and all_correct and selected is not None
        and selected["largeGeometricToDirect"] <= 0.97
        and selected["geometricToDirect"] <= 1.02
    )

    return {
        "schema": "spectral-kernel-streaming-pruned-locality-analysis-v1",
        "phase": "screen",
        "cohortId": COHORT_ID,
        "baselineCandidateId": BASELINE_ID,
        "directCandidateId": DIRECT_ID,
        "completeProductionMatrix": complete,
        "allCorrectWithin1e-12": all_correct,
        "maximumCorrectnessError": maximum_error,
        "profiles": profiles,
        "candidateSummaries": summaries,
        "selection": {
            "selectedCandidateId": (
                selected["candidateId"] if selected is not None else None
            ),
            "selectedTileWidth": (
                selected["tileWidth"] if selected is not None else None
            ),
            "requiresAlgorithmResidentRatioAtMost": 0.90,
            "requiresLargeCaseImprovementVersusDirect": 0.03,
            "allowsOverallRegressionVersusDirectAtMost": 0.02,
            "optimizationSuccessful": optimization_successful,
            "advanceSelectedTileToReference": optimization_successful,
            "sizeDependentDispatchAllowed": False,
        },
    }


def main() -> int:
    repository_root = Path(__file__).resolve().parents[1]
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--executable", type=Path,
        default=repository_root / "build/release/skbench",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--seed", type=int, default=129)
    parser.add_argument("--max-memory-fraction", type=float, default=0.5)
    parser.add_argument("--allow-memory-risk", action="store_true")
    parser.add_argument("--allow-dirty-tree", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    arguments = parser.parse_args()
    if min(arguments.warmups, arguments.samples) < 1:
        parser.error("--warmups and --samples must be positive")
    if not 0.0 < arguments.max_memory_fraction <= 1.0:
        parser.error("--max-memory-fraction must be in (0, 1]")

    source_commit, source_dirty = git_source_state(repository_root)
    if source_dirty and not arguments.allow_dirty_tree:
        parser.error(
            "the benchmark source tree is dirty; commit and rebuild for evidence "
            "collection or use --allow-dirty-tree for an exploratory run"
        )

    candidates = candidate_matrix()
    estimates = {
        candidate.id: {
            profile: estimated_explicit_peak_bytes(profile, candidate)
            for profile in PROFILES
        }
        for candidate in candidates
    }
    physical_memory = sysctl_uint64("hw.memsize", 0)
    if physical_memory > 0 and not arguments.allow_memory_risk:
        rejected = [
            (candidate_id, profile, estimate)
            for candidate_id, profile_estimates in estimates.items()
            for profile, estimate in profile_estimates.items()
            if estimate > arguments.max_memory_fraction * physical_memory
        ]
        if rejected:
            details = ", ".join(
                f"{candidate}/{profile} ({gibibytes(estimate)})"
                for candidate, profile, estimate in rejected
            )
            parser.error(
                f"estimated explicit peak exceeds {arguments.max_memory_fraction:.0%} "
                f"of {gibibytes(physical_memory)} physical memory: {details}"
            )

    output = arguments.output or (
        repository_root / "results/local" /
        f"issue16-streaming-pruned-locality-{timestamp}"
    )
    commands: list[
        tuple[str, str, LocalityCandidate, list[str], Path]
    ] = []
    for profile_index, profile in enumerate(PROFILES):
        shift = profile_index % len(candidates)
        ordered = candidates[shift:] + candidates[:shift]
        for candidate in ordered:
            stem = f"{profile}--{candidate.id}"
            result_path = output / f"{stem}.json"
            commands.append((
                stem,
                profile,
                candidate,
                command_for(
                    arguments.executable, candidate, profile,
                    arguments.warmups, arguments.samples,
                    arguments.seed, result_path,
                ),
                result_path,
            ))

    if arguments.dry_run:
        for _, _, _, command, _ in commands:
            print(f"VECLIB_MAXIMUM_THREADS=1 {' '.join(command)}")
        print(
            f"Planned {len(commands)} isolated locality run(s) across "
            f"{len(PROFILES)} four-field profiles."
        )
        return 0

    output.mkdir(parents=True, exist_ok=False)
    manifest = {
        "schema": "spectral-kernel-local-sweep-v1",
        "experimentId": EXPERIMENT_ID,
        "incrementId": INCREMENT_ID,
        "phase": "screen",
        "cohortId": COHORT_ID,
        "createdAtUtc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "question": (
            "Can bounded plane-major compact staging recover locality and make "
            "the issue #16 lower-memory pipeline competitive at large sizes?"
        ),
        "baseline": (
            "Same-commit issue #9 fused-split control and issue #16 tile-1 "
            "page-strided streaming control."
        ),
        "changedVariables": [
            "compact staging tile width 1, 4, 8, or 16 planes",
            "page-strided direct split access versus a 32-mode cache-blocked transpose",
        ],
        "controlledVariables": [
            "one uniform selected tile width across all workloads",
            "Float64 retention, FFTW internal-1/outer-12, vertical dynamic-16, modal work, and oracle",
            "one worker-local FFT half-spectrum plane and no batch-sized candidate spectrum",
        ],
        "timedOperation": (
            "One allocation-free ready-real-input to reconstructed-real-output "
            "synthetic antialiased spectral round trip."
        ),
        "componentLedger": [
            "row and selected-column FFT work",
            "compact staging and blocked forward/inverse transpose",
            "retained horizontal forward/inverse totals",
            "raw vertical MM, modal work, and authoritative uninstrumented total",
            "scratch, algorithm-resident, benchmark-only, explicit-peak, and observed-high-water memory",
        ],
        "excludedWork": [
            "workload-size dispatch, full-spectrum tiles, and custom butterflies",
            "reference-depth inference, WVM nonlinear flux, and issue #17",
        ],
        "allocationPolicy": "Zero allocations in every timed steady-state graph.",
        "selectionRule": (
            "Among tiled candidates with geometric algorithm-resident memory at "
            "most 0.90 of fused split, select the lowest two-large-case geometric "
            "time ratio. Advance only if it improves on direct streaming by at "
            "least 3% on the large cases and remains within 2% overall."
        ),
        "profiles": list(PROFILES),
        "largeDecisionProfiles": list(LARGE_PROFILES),
        "candidates": [asdict(candidate) for candidate in candidates],
        "candidateOrderRotatesByProfile": True,
        "threadEnvironment": {"VECLIB_MAXIMUM_THREADS": "1"},
        "physicalMemoryBytes": physical_memory,
        "maximumMemoryFraction": arguments.max_memory_fraction,
        "estimatedExplicitPeakBytes": estimates,
        "sourceTreeGitCommit": source_commit,
        "sourceTreeDirty": source_dirty,
        "rounds": 1,
        "warmups": arguments.warmups,
        "samples": arguments.samples,
        "seed": arguments.seed,
        "runs": [],
    }

    failed = False
    completed_results: list[tuple[LocalityCandidate, dict]] = []
    for index, (stem, profile, candidate, command, result_path) in enumerate(
        commands, start=1,
    ):
        print(f"[{index}/{len(commands)}] {stem}", flush=True)
        log_path = output / f"{stem}.log"
        environment = os.environ.copy()
        environment["VECLIB_MAXIMUM_THREADS"] = "1"
        with log_path.open("w", encoding="utf-8") as log:
            completed = subprocess.run(
                command, cwd=repository_root, env=environment,
                stdout=log, stderr=subprocess.STDOUT,
            )
        entry = {
            "id": stem,
            "round": 1,
            "profile": profile,
            "candidate": asdict(candidate),
            "primaryProvider": candidate.primary_provider,
            "estimatedExplicitPeakBytes": estimates[candidate.id][profile],
            "environment": {"VECLIB_MAXIMUM_THREADS": "1"},
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
            if result["status"] == "passed" and metadata_matches:
                completed_results.append((candidate, result))
            else:
                completed = subprocess.CompletedProcess(command, 1)
        manifest["runs"].append(entry)
        with (output / "manifest.json").open("w", encoding="utf-8") as stream:
            json.dump(manifest, stream, indent=2)
            stream.write("\n")
        if completed.returncode != 0:
            failed = True
            print(log_path.read_text(encoding="utf-8")[-4000:], file=sys.stderr)
            if not arguments.continue_on_error:
                break

    if completed_results:
        with (output / "analysis.json").open("w", encoding="utf-8") as stream:
            json.dump(analyze(completed_results), stream, indent=2)
            stream.write("\n")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
