#!/usr/bin/env python3
"""Run the issue #20 authoritative constant-stratification reference campaign."""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

from prepare_constant_stratification_flux_fixture import (
    fixture_identity,
    prepare,
    validate_and_read,
)
from run_cross_mac_spectral_reference import percentile_bootstrap
from run_spectral_pipeline_sweep import (
    geometric_mean,
    stratified_geometric_bootstrap,
)
from run_vertical_gemm_sweep import git_source_state


EXPERIMENT_ID = "issue-020-constant-stratification-type1"
INCREMENT_ID = "constant-stratification-flux-authoritative-reference-v1"
COHORT_ID = "issue20-m4-authoritative-constant-flux-v1"
CONTROL_PROVIDER = (
    "pipeline-constant-stratification-wvm-full-half-authoritative"
)
CANDIDATE_PROVIDER = (
    "pipeline-constant-stratification-streaming-pruned-tile16-authoritative"
)
TOTAL_STAGE = (
    "authoritative WVM constant-stratification nonlinear-flux composition"
)
REFERENCE_ROUNDS = 3
EXTENDED_REFERENCE_ROUNDS = 5
REFERENCE_WARMUPS = 3
REFERENCE_SAMPLES = 21
TOLERANCE = 1.0e-12
ORACLE_TOLERANCE = 2.0e-12
WVM_COMMIT = "6ad254fb9756ac918bb72e036020d004879df1f2"
PROFILE_SHAPES = {
    "wvm-current-256-nz129-f4": (256, 256, 129),
    "wvm-current-512-nz257-f4": (512, 512, 257),
    "wvm-large-1024-nz129-f4": (1024, 1024, 129),
    "wvm-large-512-nz513-f4": (512, 512, 513),
}
PROFILES = tuple(PROFILE_SHAPES)
COMPONENT_STAGES = {
    "phase": ("component", "phase evaluation and flux reset", "phase"),
    "coefficientAssembly": (
        "component",
        "mode-keyed coefficient assembly and retained/full clearing",
        "inverse",
    ),
    "verticalInverse": (
        "component", "15 inverse complex type-I channels", "inverse",
    ),
    "horizontalInverse": (
        "retained-operator-total",
        "five horizontal inverse transforms",
        "inverse",
    ),
    "pointwise": (
        "component",
        "four streamed pointwise advection expressions",
        "pointwise",
    ),
    "horizontalForward": (
        "retained-operator-total",
        "four horizontal forward transforms and radial retention",
        "forward",
    ),
    "verticalForward": (
        "component",
        "four forward complex type-I channels and normalization",
        "forward",
    ),
    "coefficientProjection": (
        "component", "four authoritative flux-target accumulations", "forward",
    ),
}


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def matlab_quote(value: Path | str) -> str:
    return str(value).replace("'", "''")


def fixture_export_command(
    matlab: str, repository_root: Path, wvm_repository: Path, output: Path,
    profile: str,
) -> list[str]:
    nx, ny, nz = PROFILE_SHAPES[profile]
    expression = (
        f"addpath('{matlab_quote(repository_root / 'tools')}');"
        "exportConstantStratificationFluxFixture("
        f"'{matlab_quote(output)}','{matlab_quote(wvm_repository)}',"
        f"Nxyz=[{nx} {ny} {nz}],"
        f"fixtureId='{profile}-authoritative-v1');"
    )
    return [matlab, "-batch", expression]


def fixture_assignments(values: list[str]) -> dict[str, Path]:
    fixtures: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError("fixture assignments must use PROFILE=EXPORT_DIRECTORY")
        profile, raw_path = value.split("=", 1)
        if profile not in PROFILE_SHAPES:
            raise ValueError(f"unknown fixture profile: {profile}")
        if profile in fixtures:
            raise ValueError(f"duplicate fixture profile: {profile}")
        path = Path(raw_path).expanduser().resolve()
        if not (path / "manifest.json").is_file():
            raise ValueError(f"fixture manifest is missing: {path / 'manifest.json'}")
        fixtures[profile] = path
    if values and set(fixtures) != set(PROFILES):
        missing = sorted(set(PROFILES) - set(fixtures))
        raise ValueError("missing fixture profile(s): " + ", ".join(missing))
    return fixtures


def prepare_fixture(export: Path, prepared: Path, summary_path: Path) -> dict:
    if prepared.is_file() or summary_path.is_file():
        if not prepared.is_file() or not summary_path.is_file():
            raise ValueError("prepared fixture and summary must either both exist or both be absent")
        manifest, manifest_bytes, _, _ = validate_and_read(export)
        summary = load_json(summary_path)
        if summary.get("fixtureHash") != fixture_identity(manifest, manifest_bytes):
            raise ValueError(f"prepared fixture summary is stale: {summary_path}")
        if summary.get("preparedBytes") != prepared.stat().st_size:
            raise ValueError(f"prepared fixture byte count changed: {prepared}")
        return summary
    prepared.parent.mkdir(parents=True, exist_ok=True)
    summary = prepare(export, prepared)
    write_json(summary_path, summary)
    return summary


def command_for(
    executable: Path, profile: str, fixture: Path, comparison_order: str,
    warmups: int, samples: int, output: Path,
) -> list[str]:
    return [
        str(executable), "run",
        "--kernel", "constant-stratification-flux",
        "--profile", profile,
        "--constant-stratification-flux-fixture", str(fixture),
        "--fftw-planning", "measure",
        "--fftw-alignment", "unaligned",
        "--fftw-wisdom", "cold",
        "--fftw-internal-workers", "16",
        "--fftw-outer-workers", "12",
        "--streaming-tile-width", "16",
        "--pointwise-policy", "spatial-static",
        "--pointwise-workers", "8",
        "--comparison-order", comparison_order,
        "--warmups", str(warmups),
        "--samples", str(samples),
        "--output", str(output),
    ]


def provider(result: dict, provider_id: str) -> dict:
    matches = [item for item in result.get("providers", [])
               if item.get("id") == provider_id]
    if len(matches) != 1:
        raise ValueError(f"result lacks one provider {provider_id}")
    return matches[0]


def stage(record: dict, scope: str, name: str, direction: str) -> dict:
    matches = [
        item for item in record.get("timings", [])
        if item.get("scope") == scope and item.get("stage") == name
        and item.get("direction") == direction
    ]
    if len(matches) != 1:
        raise ValueError(
            f"provider {record.get('id')} lacks one {scope}/{name}/{direction} timing"
        )
    return matches[0]


def allocation_ledger_valid(record: dict) -> bool:
    matches = [
        item for item in record.get("componentLedger", [])
        if item.get("stage") == "steady-state application allocation"
    ]
    return bool(len(matches) == 1 and matches[0].get("state") == "elided")


def placement_valid(record: dict) -> bool:
    contract = record.get("executionContract", {}).get("forward", {})
    return bool(
        contract.get("nativePlacement") == "out-of-place"
        and contract.get("adapterPlacement") == "out-of-place"
        and contract.get("destroysNativeInput") is False
        and contract.get("adapterPreservesCallerInput") is True
        and contract.get("requiresPreservationCopyForRepeatedExecution") is False
    )


def provider_record(record: dict) -> dict:
    total = stage(record, "uninstrumented-total", TOTAL_STAGE, "complete")
    components = {
        key: stage(record, scope, name, direction)["medianSeconds"]
        for key, (scope, name, direction) in COMPONENT_STAGES.items()
    }
    memory = record.get("memory", {})
    memory_keys = (
        "persistentBytes", "scratchBytes", "algorithmResidentBytes",
        "benchmarkHarnessBytes", "estimatedProcessPeakBytes",
        "observedProcessHighWaterBytes",
    )
    if any(int(memory.get(key, 0)) <= 0 for key in memory_keys):
        raise ValueError(f"provider {record.get('id')} lacks complete memory evidence")
    correctness = record.get("correctness", [])
    complete_matches = [
        item for item in correctness
        if str(item.get("name", "")).startswith("complete ")
        and "authoritative WVM oracle" in str(item.get("name", ""))
    ]
    if len(complete_matches) != 1:
        raise ValueError(
            f"provider {record.get('id')} lacks one complete WVM oracle metric"
        )
    complete_error = float(complete_matches[0]["maximumRelativeError"])
    complete_l2_error = float(complete_matches[0]["relativeL2Error"])
    equivalence_matches = [
        item for item in correctness
        if item.get("name") ==
        "complete compact composition versus full-half control"
    ]
    if record.get("id") == CANDIDATE_PROVIDER and len(equivalence_matches) != 1:
        raise ValueError(
            "authoritative compact provider lacks one complete control-equivalence metric"
        )
    if record.get("id") == CONTROL_PROVIDER and equivalence_matches:
        raise ValueError(
            "authoritative control unexpectedly contains a control-equivalence metric"
        )
    cross_matches = [
        item for item in correctness
        if item.get("name") ==
        "fixture MATLAB versus compiled WVM nonlinear-flux cross-check"
    ]
    if len(cross_matches) != 1:
        raise ValueError(
            f"provider {record.get('id')} lacks one WVM cross-backend metric"
        )
    return {
        "totalSeconds": float(total["medianSeconds"]),
        "totalSamplesSeconds": [float(value) for value in total["samplesSeconds"]],
        "components": components,
        "memory": {key: int(memory[key]) for key in memory_keys},
        "maximumCorrectnessError": complete_error,
        "relativeL2Error": complete_l2_error,
        "algorithmEquivalenceMaximumScaleNormalizedError": (
            float(equivalence_matches[0]["maximumRelativeError"])
            if equivalence_matches else None
        ),
        "algorithmEquivalenceRelativeL2Error": (
            float(equivalence_matches[0]["relativeL2Error"])
            if equivalence_matches else None
        ),
        "allCorrectnessMetricsPassed": bool(
            correctness and all(item.get("passed") is True for item in correctness)
        ),
        "crossBackendMaximumScaleNormalizedError": float(
            cross_matches[0]["maximumRelativeError"]
        ),
        "crossBackendRelativeL2Error": float(
            cross_matches[0]["relativeL2Error"]
        ),
        "placementValid": placement_valid(record),
        "allocationLedgerValid": allocation_ledger_valid(record),
        "schedulingId": record.get("schedulingId"),
    }


def result_record(
    result: dict, profile: str, round_number: int, fixture: dict,
    source_commit: str, warmups: int, samples: int,
) -> dict:
    control = provider_record(provider(result, CONTROL_PROVIDER))
    candidate = provider_record(provider(result, CANDIDATE_PROVIDER))
    provenance = result.get("provenance", {}).get("spectralFluxFixture", {})
    embedded_commit = result.get("environment", {}).get("gitCommit", "")
    source_matches = bool(
        embedded_commit and embedded_commit != "unknown"
        and source_commit.startswith(embedded_commit)
        and result.get("environment", {}).get("gitDirty") is False
    )
    fixture_matches = bool(
        provenance.get("status") == "authoritative-wvm-export"
        and provenance.get("authoritative") is True
        and provenance.get("schema") == "constant-stratification-flux-fixture-v1"
        and provenance.get("fixtureHash") == fixture["fixtureHash"]
        and provenance.get("waveVortexModelCommit") == WVM_COMMIT
    )
    order = "candidate-first" if round_number % 2 == 0 else "control-first"
    expected_schedule = (
        "vertical-type1-internal-16;horizontal-internal-1-outer-12;"
        f"pointwise-spatial-static-8;comparison-{order}"
    )
    errors = [control["maximumCorrectnessError"],
              candidate["maximumCorrectnessError"]]
    valid = bool(
        result.get("status") == "passed"
        and result.get("run", {}).get("profile") == profile
        and result.get("run", {}).get("warmups") == warmups
        and result.get("run", {}).get("samples") == samples
        and all(len(item["totalSamplesSeconds"]) == samples
                for item in (control, candidate))
        and all(math.isfinite(error) and error <= ORACLE_TOLERANCE
                for error in errors)
        and all(
            item["allCorrectnessMetricsPassed"]
            and math.isfinite(item["relativeL2Error"])
            and item["relativeL2Error"] <= ORACLE_TOLERANCE
            for item in (control, candidate)
        )
        and math.isfinite(
            candidate["algorithmEquivalenceMaximumScaleNormalizedError"]
        )
        and candidate[
            "algorithmEquivalenceMaximumScaleNormalizedError"
        ] <= TOLERANCE
        and math.isfinite(candidate["algorithmEquivalenceRelativeL2Error"])
        and candidate["algorithmEquivalenceRelativeL2Error"] <= TOLERANCE
        and all(item["placementValid"] and item["allocationLedgerValid"]
                for item in (control, candidate))
        and control["schedulingId"] == expected_schedule
        and candidate["schedulingId"] == expected_schedule
        and fixture_matches and source_matches
    )
    return {
        "runId": result.get("run", {}).get("id"),
        "profile": profile,
        "round": round_number,
        "comparisonOrder": order,
        "control": control,
        "candidate": candidate,
        "candidateToControl": (
            candidate["totalSeconds"] / control["totalSeconds"]
        ),
        "maximumCorrectnessError": max(errors),
        "algorithmEquivalenceMaximumScaleNormalizedError": candidate[
            "algorithmEquivalenceMaximumScaleNormalizedError"
        ],
        "algorithmEquivalenceRelativeL2Error": candidate[
            "algorithmEquivalenceRelativeL2Error"
        ],
        "fixtureHash": provenance.get("fixtureHash"),
        "fixtureMatches": fixture_matches,
        "sourceMetadataMatches": source_matches,
        "valid": valid,
    }


def conditional_round_decision(records: list[dict]) -> dict:
    cells = {
        (item["profile"], item["round"]): item["candidateToControl"]
        for item in records
    }
    ratios: dict[str, list[float]] = {}
    triggers: list[dict] = []
    for profile in PROFILES:
        values = [cells[(profile, round_number)]
                  for round_number in range(1, REFERENCE_ROUNDS + 1)
                  if (profile, round_number) in cells]
        if len(values) != REFERENCE_ROUNDS:
            continue
        ratios[profile] = values
        spread = max(values) / min(values)
        if spread > 1.10:
            triggers.append({
                "id": "profile-ratio-spread", "profile": profile,
                "value": spread, "threshold": 1.10,
            })
        if min(values) <= 1.03 <= max(values) and min(values) < max(values):
            triggers.append({
                "id": "profile-straddles-regression-boundary",
                "profile": profile, "lower": min(values),
                "upper": max(values), "boundary": 1.03,
            })
    aggregate: list[float] = []
    if len(ratios) == len(PROFILES):
        aggregate = [
            geometric_mean([ratios[profile][index] for profile in PROFILES])
            for index in range(REFERENCE_ROUNDS)
        ]
        if min(aggregate) <= 0.90 <= max(aggregate) and min(aggregate) < max(aggregate):
            triggers.append({
                "id": "aggregate-straddles-improvement-boundary",
                "lower": min(aggregate), "upper": max(aggregate),
                "boundary": 0.90,
            })
        median = statistics.median(aggregate)
        if 0.85 <= median <= 0.95:
            triggers.append({
                "id": "aggregate-median-near-improvement-boundary",
                "value": median, "lower": 0.85, "upper": 0.95,
            })
    complete = len(ratios) == len(PROFILES)
    return {
        "completeInitialThreeRoundMatrix": complete,
        "profileRoundRatios": ratios,
        "aggregateRoundRatios": aggregate,
        "triggers": triggers,
        "runAdditionalTwoRounds": complete and bool(triggers),
        "finalRoundCount": (
            EXTENDED_REFERENCE_ROUNDS if complete and triggers
            else REFERENCE_ROUNDS
        ),
    }


def analyze(records: list[dict], source_commit: str) -> dict:
    observed_rounds = sorted({item["round"] for item in records})
    expected_rounds = list(range(1, len(observed_rounds) + 1))
    protocol_complete = bool(
        observed_rounds == expected_rounds
        and len(observed_rounds) in {REFERENCE_ROUNDS, EXTENDED_REFERENCE_ROUNDS}
    )
    rows: list[dict] = []
    stratified_ratios: dict[str, list[float]] = {}
    all_valid = True
    for profile_index, profile in enumerate(PROFILES):
        profile_records = sorted(
            [item for item in records if item["profile"] == profile],
            key=lambda item: item["round"],
        )
        if not profile_records:
            all_valid = False
            continue
        if [item["round"] for item in profile_records] != expected_rounds:
            all_valid = False
            continue
        all_valid = all_valid and all(item["valid"] for item in profile_records)
        ratios = [item["candidateToControl"] for item in profile_records]
        stratified_ratios[profile] = ratios
        lower, upper = percentile_bootstrap(ratios, seed=20020 + profile_index)
        components = {}
        for key in COMPONENT_STAGES:
            control_seconds = statistics.median(
                item["control"]["components"][key] for item in profile_records
            )
            candidate_seconds = statistics.median(
                item["candidate"]["components"][key] for item in profile_records
            )
            components[key] = {
                "controlMedianSeconds": control_seconds,
                "candidateMedianSeconds": candidate_seconds,
                "candidateToControl": candidate_seconds / control_seconds,
            }
        control_memory = profile_records[0]["control"]["memory"]
        candidate_memory = profile_records[0]["candidate"]["memory"]
        rows.append({
            "profile": profile,
            "shape": list(PROFILE_SHAPES[profile]),
            "rounds": expected_rounds,
            "comparisonOrders": [item["comparisonOrder"] for item in profile_records],
            "controlMedianSeconds": statistics.median(
                item["control"]["totalSeconds"] for item in profile_records
            ),
            "candidateMedianSeconds": statistics.median(
                item["candidate"]["totalSeconds"] for item in profile_records
            ),
            "candidateToControl": statistics.median(ratios),
            "roundRatios": ratios,
            "empiricalPairedRange": {"lower": lower, "upper": upper},
            "components": components,
            "memory": {
                "control": control_memory,
                "candidate": candidate_memory,
                "candidateToControlAlgorithmResident": (
                    candidate_memory["algorithmResidentBytes"] /
                    control_memory["algorithmResidentBytes"]
                ),
                "observedHighWaterInterpretation": (
                    "common-process measurement containing both provider graphs; "
                    "not an algorithm-specific ratio"
                ),
            },
            "maximumCorrectnessError": max(
                item["maximumCorrectnessError"] for item in profile_records
            ),
        })

    complete = bool(
        protocol_complete and all_valid
        and {item["profile"] for item in rows} == set(PROFILES)
    )
    geometric_ratio = (
        geometric_mean([item["candidateToControl"] for item in rows])
        if complete else None
    )
    maximum_ratio = (
        max(item["candidateToControl"] for item in rows) if complete else None
    )
    interval = None
    if complete:
        lower, upper = stratified_geometric_bootstrap(stratified_ratios)
        interval = {"lower": lower, "upper": upper}
    maximum_error = max(
        (item["maximumCorrectnessError"] for item in records), default=math.inf
    )
    maximum_equivalence_error = max(
        (item["algorithmEquivalenceMaximumScaleNormalizedError"]
         for item in records), default=math.inf
    )
    maximum_equivalence_l2 = max(
        (item["algorithmEquivalenceRelativeL2Error"] for item in records),
        default=math.inf,
    )
    oracle_correctness = bool(complete and maximum_error <= ORACLE_TOLERANCE)
    algorithm_equivalence = bool(
        complete and maximum_equivalence_error <= TOLERANCE
        and maximum_equivalence_l2 <= TOLERANCE
    )
    correctness = oracle_correctness and algorithm_equivalence
    improvement = bool(geometric_ratio is not None and geometric_ratio <= 0.90)
    regression = bool(maximum_ratio is not None and maximum_ratio <= 1.03)
    interval_passed = bool(interval is not None and interval["upper"] < 1.0)
    conditional = conditional_round_decision(records)
    if len(observed_rounds) == EXTENDED_REFERENCE_ROUNDS:
        conditional = {
            **conditional,
            "runAdditionalTwoRounds": False,
            "finalRoundCount": EXTENDED_REFERENCE_ROUNDS,
            "additionalRoundsAlreadyCollected": True,
        }
    return {
        "schema": "spectral-kernel-constant-stratification-authoritative-reference-analysis-v1",
        "experimentId": EXPERIMENT_ID,
        "incrementId": INCREMENT_ID,
        "phase": "authoritative-reference",
        "classification": "reference",
        "benchmarkCommit": source_commit,
        "waveVortexModelCommit": WVM_COMMIT,
        "profilesRequested": list(PROFILES),
        "profilesMatched": [item["profile"] for item in rows],
        "rounds": len(observed_rounds),
        "referenceRoundProtocolComplete": protocol_complete,
        "conditionalRoundDecision": conditional,
        "allRecordsValid": complete,
        "allAlgorithmEquivalenceWithin1e12": algorithm_equivalence,
        "maximumAlgorithmEquivalenceError": maximum_equivalence_error,
        "maximumAlgorithmEquivalenceL2Error": maximum_equivalence_l2,
        "allAuthoritativeOracleWithin2e12": oracle_correctness,
        "maximumAuthoritativeOracleError": maximum_error,
        "geometricCandidateToControl": geometric_ratio,
        "maximumProfileCandidateToControl": maximum_ratio,
        "empiricalStratifiedPairedRange": interval,
        "rows": rows,
        "adoptionGate": {
            "geometricTimeRatioAtMost": 0.90,
            "maximumProfileTimeRatioAtMost": 1.03,
            "improvementPassed": improvement,
            "regressionPassed": regression,
            "empiricalIntervalExcludesTie": interval_passed,
            "correctnessPassed": correctness,
            "algorithmEquivalenceWithin1e12Passed": algorithm_equivalence,
            "authoritativeOracleWithin2e12Passed": oracle_correctness,
            "referenceRoundProtocolPassed": protocol_complete,
            "allocationLedgerPassed": bool(
                complete and all(
                    item[side]["allocationLedgerValid"]
                    for item in records for side in ("control", "candidate")
                )
            ),
            "singleTupleAcrossSizesPassed": complete,
            "advanceConstantStratificationCandidate": bool(
                complete and improvement and regression and interval_passed
                and correctness
            ),
            "completeConstantStratificationNonlinearFluxMeasured": True,
            "generalStratificationClaimAllowed": False,
            "generalMacClaimAllowed": False,
            "sizeDependentDispatchAllowed": False,
        },
        "interpretation": (
            "The authoritative boundary begins with retained Ap/Am/A0 state and "
            "elapsed time and ends with retained Fp/Fm/F0. It includes exact WVM "
            "phase evolution, physical coefficient formulas, analytic type-I "
            "transforms, streamed pointwise products, projection, and phase removal."
        ),
    }


def run_rounds(
    repository_root: Path, executable: Path, output: Path, manifest: dict,
    fixtures: dict[str, dict], rounds: list[int], continue_on_error: bool,
) -> tuple[list[dict], bool]:
    result_dir = output / "runs"
    result_dir.mkdir(parents=True, exist_ok=True)
    existing = {item["id"]: item for item in manifest.get("runs", [])}
    records: list[dict] = []
    failed = False
    plans = []
    for round_number in rounds:
        offset = (round_number - 1) % len(PROFILES)
        profile_order = list(PROFILES[offset:]) + list(PROFILES[:offset])
        order = "candidate-first" if round_number % 2 == 0 else "control-first"
        for profile in profile_order:
            identifier = f"reference-round-{round_number}--{profile}"
            result_path = result_dir / f"{identifier}.json"
            plans.append((identifier, round_number, profile, order, result_path))
    for index, (identifier, round_number, profile, order, result_path) in enumerate(
        plans, start=1
    ):
        command = command_for(
            executable, profile, Path(fixtures[profile]["preparedPath"]), order,
            manifest["warmups"], manifest["samples"], result_path,
        )
        if identifier in existing:
            entry = existing[identifier]
            if entry.get("exitCode") != 0 or not result_path.is_file():
                raise ValueError(f"existing run requires manual recovery: {identifier}")
            result = load_json(result_path)
            record = result_record(
                result, profile, round_number, fixtures[profile],
                manifest["sourceTreeGitCommit"], manifest["warmups"],
                manifest["samples"],
            )
            if not record["valid"]:
                raise ValueError(f"existing run no longer validates: {identifier}")
            records.append(record)
            print(f"[{index}/{len(plans)}] reuse {identifier}", flush=True)
            continue
        print(f"[{index}/{len(plans)}] {identifier}", flush=True)
        log_path = result_path.with_suffix(".log")
        environment = os.environ.copy()
        environment["VECLIB_MAXIMUM_THREADS"] = "1"
        with log_path.open("w", encoding="utf-8") as log:
            completed = subprocess.run(
                command, cwd=repository_root, env=environment,
                stdout=log, stderr=subprocess.STDOUT,
            )
        entry = {
            "id": identifier,
            "round": round_number,
            "profile": profile,
            "comparisonOrder": order,
            "primaryProvider": CANDIDATE_PROVIDER,
            "command": list(map(str, command)),
            "environment": {"VECLIB_MAXIMUM_THREADS": "1"},
            "exitCode": completed.returncode,
            "log": str(log_path.relative_to(output)),
        }
        if result_path.is_file():
            result = load_json(result_path)
            record = result_record(
                result, profile, round_number, fixtures[profile],
                manifest["sourceTreeGitCommit"], manifest["warmups"],
                manifest["samples"],
            )
            entry.update({
                "runId": record["runId"],
                "result": str(result_path.relative_to(output)),
                "authoritativeFixture": record["fixtureMatches"],
                "sourceMetadataMatches": record["sourceMetadataMatches"],
                "valid": record["valid"],
            })
            if completed.returncode == 0 and record["valid"]:
                records.append(record)
            else:
                entry["exitCode"] = 1
        manifest["runs"].append(entry)
        write_json(output / "manifest.json", manifest)
        if entry["exitCode"] != 0:
            failed = True
            print(log_path.read_text(encoding="utf-8")[-4000:], file=sys.stderr)
            if not continue_on_error:
                break
    return records, failed


def main() -> int:
    repository_root = Path(__file__).resolve().parents[1]
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fixture", action="append", default=[],
        help="existing authoritative export as PROFILE=DIRECTORY; repeat four times",
    )
    parser.add_argument(
        "--wvm-repository", type=Path,
        default=repository_root.parent / "wave-vortex-model",
        help="clean audited WVM checkout used when fixtures are generated",
    )
    parser.add_argument("--matlab", default="matlab")
    parser.add_argument(
        "--executable", type=Path,
        default=repository_root / "build/release/skbench",
    )
    parser.add_argument(
        "--output", type=Path,
        default=repository_root / "results/local" /
        f"issue20-constant-flux-authoritative-reference-{timestamp}",
    )
    parser.add_argument("--warmups", type=int, default=REFERENCE_WARMUPS)
    parser.add_argument("--samples", type=int, default=REFERENCE_SAMPLES)
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    arguments = parser.parse_args()
    if min(arguments.warmups, arguments.samples) < 1:
        parser.error("warmups and samples must be positive")
    if not arguments.executable.is_file():
        parser.error(f"benchmark executable is missing: {arguments.executable}")
    source_commit, source_dirty = git_source_state(repository_root)
    if source_dirty:
        parser.error("commit the source tree before authoritative reference collection")
    try:
        supplied = fixture_assignments(arguments.fixture)
    except ValueError as error:
        parser.error(str(error))
    wvm_repository = arguments.wvm_repository.expanduser().resolve()
    if not supplied:
        wvm_commit, wvm_dirty = git_source_state(wvm_repository)
        if wvm_commit != WVM_COMMIT or wvm_dirty:
            parser.error(
                f"WVM must be clean at audited commit {WVM_COMMIT}; found "
                f"{wvm_commit} dirty={wvm_dirty}"
            )

    output = arguments.output.expanduser().resolve()
    if output.exists() and not arguments.resume:
        parser.error(f"output exists; pass --resume to reuse it: {output}")
    output.mkdir(parents=True, exist_ok=True)
    fixtures: dict[str, dict] = {}
    for index, profile in enumerate(PROFILES, start=1):
        profile_dir = output / "fixtures" / profile
        export = supplied.get(profile, profile_dir / "export")
        if profile not in supplied and not (export / "manifest.json").is_file():
            if export.exists() and any(export.iterdir()):
                parser.error(f"partial fixture export requires manual recovery: {export}")
            export.mkdir(parents=True, exist_ok=True)
            command = fixture_export_command(
                arguments.matlab, repository_root, wvm_repository, export,
                profile,
            )
            if arguments.dry_run:
                print(" ".join(command))
                continue
            print(f"[fixture {index}/{len(PROFILES)}] export {profile}", flush=True)
            with (profile_dir / "export.log").open("w", encoding="utf-8") as log:
                completed = subprocess.run(
                    command, cwd=repository_root,
                    stdout=log, stderr=subprocess.STDOUT,
                )
            if completed.returncode != 0:
                parser.error(f"MATLAB fixture export failed: {profile_dir / 'export.log'}")
        if arguments.dry_run:
            continue
        try:
            summary = prepare_fixture(
                export, profile_dir / "prepared.bin",
                profile_dir / "prepared-summary.json",
            )
        except (OSError, ValueError) as error:
            parser.error(str(error))
        summary["preparedPath"] = str((profile_dir / "prepared.bin").resolve())
        fixtures[profile] = summary
    if arguments.dry_run:
        return 0

    manifest_path = output / "manifest.json"
    manifest = {
        "schema": "spectral-kernel-local-sweep-v1",
        "experimentId": EXPERIMENT_ID,
        "incrementId": INCREMENT_ID,
        "phase": "authoritative-reference",
        "cohortId": COHORT_ID,
        "createdAtUtc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "question": (
            "Does the frozen compact tile-16 constant-stratification graph improve "
            "the exact complete WVM nonlinear flux by at least 10%?"
        ),
        "baseline": (
            "The exact current WVM full-half FFTW horizontal and batched type-I "
            "graph, using the same authoritative state, formulas, and oracle."
        ),
        "controlledVariables": [
            "Float64 nonhydrostatic four-field WVM mathematics and fixture",
            "radial horizontal and two-thirds vertical antialiasing",
            "FFTW MEASURE/UNALIGNED/cold and one frozen topology across sizes",
            "streamed three-shared/three-derivative physical lifetime",
        ],
        "changedVariables": [
            "full WVM-order half-spectrum versus compact radial split rows",
            "full horizontal FFTW versus partial-column-pruned tile-16 FFTW",
            "vertical type-I work on full-half versus retained rows only",
        ],
        "timedOperation": (
            "Retained Ap/Am/A0 and elapsed time through exact phase evolution, "
            "coefficient assembly, 15 inverse type-I channels, five horizontal "
            "inverses, four pointwise targets, four retained horizontal forwards, "
            "four normalized forward type-I channels, modal projection, phase "
            "removal, and accumulated Fp/Fm/F0."
        ),
        "excludedWork": [
            "fixture export, validation, preparation, loading, and oracle comparison",
            "MATLAB/MEX dispatch, model-state ownership, time integration, I/O, and diagnostics",
            "general-stratification and cross-Mac inference",
        ],
        "correctnessOracle": (
            "Every retained Fp/Fm/F0 value is compared by logical (k,l,j) to "
            "MATLAB WVM nonlinearFlux cross-checked against its compiled kernel; "
            "compact and control outputs are also compared directly."
        ),
        "allocationPolicy": (
            "All application buffers, plans, phases, mode tables, worker pools, and "
            "outputs persist; timed steady state performs zero application allocation."
        ),
        "profiles": list(PROFILES),
        "fixtures": fixtures,
        "sourceTreeGitCommit": source_commit,
        "sourceTreeDirty": False,
        "topology": {
            "verticalType1InternalWorkers": 16,
            "horizontalOuterWorkers": 12,
            "pointwiseSpatialStaticWorkers": 8,
            "coefficientWorkers": 2,
            "streamingTileWidth": 16,
        },
        "comparisonOrderPolicy": (
            "control-first in odd rounds and candidate-first in even rounds"
        ),
        "threadEnvironment": {"VECLIB_MAXIMUM_THREADS": "1"},
        "warmups": arguments.warmups,
        "samples": arguments.samples,
        "runs": [],
    }
    if manifest_path.is_file():
        existing = load_json(manifest_path)
        for key in (
            "schema", "experimentId", "incrementId", "phase", "profiles",
            "sourceTreeGitCommit", "sourceTreeDirty", "topology", "warmups",
            "samples",
        ):
            if existing.get(key) != manifest.get(key):
                parser.error(f"existing manifest disagrees on {key}")
        manifest = existing
    else:
        write_json(manifest_path, manifest)

    records, failed = run_rounds(
        repository_root, arguments.executable.resolve(), output, manifest,
        fixtures, [1, 2, 3], arguments.continue_on_error,
    )
    if not failed:
        decision = conditional_round_decision(records)
        if decision["runAdditionalTwoRounds"]:
            extended, failed = run_rounds(
                repository_root, arguments.executable.resolve(), output,
                manifest, fixtures, [4, 5], arguments.continue_on_error,
            )
            records.extend(extended)
    analysis = analyze(records, source_commit)
    write_json(output / "analysis.json", analysis)
    print(json.dumps(analysis, indent=2))
    return 0 if not failed and analysis["allRecordsValid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
