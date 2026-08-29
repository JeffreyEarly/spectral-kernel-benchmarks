#!/usr/bin/env python3
"""Load and validate append-only spectral-kernel benchmark publications."""

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path


CATALOG_SCHEMA = "spectral-kernel-publication-catalog-v1"
RESULT_SCHEMA = "spectral-kernel-benchmark-v1"
PUBLICATION_STATUSES = {"preliminary", "reference", "superseded", "withdrawn"}
EXPERIMENT_PHASES = {"planned", "collecting", "complete"}


@dataclass(frozen=True)
class PublishedBundle:
    publication: dict
    result_path: Path
    samples_path: Path
    result: dict


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_keys(value: dict, keys: tuple[str, ...], context: str) -> None:
    missing = [key for key in keys if key not in value]
    if missing:
        raise ValueError(f"{context}: missing required keys: {', '.join(missing)}")


def _require_nonempty_string(value: object, context: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{context}: expected a non-empty string")


def _unique(values: list, context: str) -> None:
    if len(values) != len(set(values)):
        raise ValueError(f"{context}: values must be unique")


def _artifact_path(results_dir: Path, relative_name: str, context: str) -> Path:
    relative = Path(relative_name)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"{context}: artifact path must remain inside results/published")
    path = (results_dir / relative).resolve()
    if results_dir.resolve() not in path.parents:
        raise ValueError(f"{context}: artifact path escapes results/published")
    return path


def _validate_result(path: Path, result: dict, run_id: str, grandfathered: bool) -> None:
    _require_keys(result, ("schema", "status", "run", "workload", "environment", "providers"), str(path))
    if result["schema"] != RESULT_SCHEMA:
        raise ValueError(f"{path}: unsupported result schema {result['schema']!r}")
    if result["run"].get("id") != run_id:
        raise ValueError(f"{path}: run.id does not match catalog run {run_id!r}")
    if result["status"] not in {"passed", "failed"}:
        raise ValueError(f"{path}: invalid numerical status {result['status']!r}")
    if not isinstance(result["providers"], list) or not result["providers"]:
        raise ValueError(f"{path}: provider list is empty")
    if grandfathered:
        return
    _require_keys(result, ("numericType",), str(path))
    numeric_type = result["numericType"]
    _require_keys(numeric_type, ("id", "scalarBits"), f"{path}.numericType")
    expected_bits = {"float32": 32, "float64": 64}
    if numeric_type["id"] not in expected_bits or numeric_type["scalarBits"] != expected_bits[numeric_type["id"]]:
        raise ValueError(f"{path}.numericType: scalar id and bit width are inconsistent")
    direction_keys = (
        "nativePlacement",
        "adapterPlacement",
        "destroysNativeInput",
        "adapterPreservesCallerInput",
        "requiresPreservationCopyForRepeatedExecution",
        "preservationIncludedInPrimitiveTiming",
        "preservationIncludedInAdapterTiming",
        "nativeInputRepresentationId",
        "nativeOutputRepresentationId",
        "adapterInputRepresentationId",
        "adapterOutputRepresentationId",
        "physicalExtents",
        "stridesElements",
        "paddingElements",
        "minimumAlignmentBytes",
        "aliasing",
        "reusableWorkBytes",
        "outputCanFeedOppositeDirection",
    )
    for provider_index, provider in enumerate(result["providers"]):
        context = f"{path}.providers[{provider_index}]"
        _require_keys(provider, ("executionContract",), context)
        execution = provider["executionContract"]
        _require_keys(execution, ("forward", "inverse"), f"{context}.executionContract")
        for direction in ("forward", "inverse"):
            contract = execution[direction]
            _require_keys(contract, direction_keys, f"{context}.executionContract.{direction}")
            for placement_key in ("nativePlacement", "adapterPlacement"):
                allowed_placements = {"in-place", "out-of-place", "unsupported"}
                if placement_key == "adapterPlacement":
                    allowed_placements.add("out-of-place-view")
                if contract[placement_key] not in allowed_placements:
                    raise ValueError(f"{context}.executionContract.{direction}.{placement_key}: invalid placement")
        memory = provider.get("memory", {})
        memory_aware_keys = (
            "algorithmResidentBytes",
            "benchmarkHarnessBytes",
            "estimatedProcessPeakBytes",
            "observedProcessHighWaterBytes",
        )
        if any(key in memory for key in memory_aware_keys):
            _require_keys(memory, memory_aware_keys, f"{context}.memory")
            for key in memory_aware_keys:
                if not isinstance(memory[key], int) or memory[key] <= 0:
                    raise ValueError(f"{context}.memory.{key}: expected a positive integer")
            if (
                memory["algorithmResidentBytes"] + memory["benchmarkHarnessBytes"]
                != memory["estimatedProcessPeakBytes"]
            ):
                raise ValueError(
                    f"{context}.memory: algorithm-resident plus benchmark-harness "
                    "bytes must equal the estimated process peak"
                )
            reported_peak = result["workload"].get("bytes", {}).get(
                "spectralPipelineEstimatedExplicitPeak"
            )
            if reported_peak and reported_peak != memory["estimatedProcessPeakBytes"]:
                raise ValueError(
                    f"{context}.memory: provider and workload spectral-pipeline peaks differ"
                )


def _validate_samples(path: Path, run_id: str) -> None:
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        required = {"run_id", "provider", "scope", "stage", "direction", "state", "bytes_moved", "sample_index", "seconds"}
        if reader.fieldnames is None or set(reader.fieldnames) != required:
            raise ValueError(f"{path}: sample CSV header does not match the v1 contract")
        for row_number, row in enumerate(reader, start=2):
            if row["run_id"] != run_id:
                raise ValueError(f"{path}:{row_number}: run_id does not match catalog run {run_id!r}")


def _validate_experiment(experiment: dict, experiment_ids: set[str]) -> None:
    experiment_id = experiment.get("id", "<missing>")
    context = f"experiment {experiment_id}"
    _require_keys(
        experiment,
        (
            "id",
            "issue",
            "phase",
            "title",
            "question",
            "baseline",
            "priorExperiments",
            "downstreamIssues",
            "definition",
            "correctness",
            "requiredTables",
        ),
        context,
    )
    if experiment["phase"] not in EXPERIMENT_PHASES:
        raise ValueError(f"{context}: invalid phase {experiment['phase']!r}")
    for key in ("id", "title", "question", "baseline"):
        _require_nonempty_string(experiment[key], f"{context}.{key}")
    if not isinstance(experiment["issue"], int) or experiment["issue"] < 1:
        raise ValueError(f"{context}.issue: expected a positive issue number")
    if not isinstance(experiment["priorExperiments"], list):
        raise ValueError(f"{context}.priorExperiments: expected an array")
    _unique(experiment["priorExperiments"], f"{context}.priorExperiments")
    for prior in experiment["priorExperiments"]:
        if prior not in experiment_ids:
            raise ValueError(f"{context}: unknown prior experiment {prior!r}")
        if prior == experiment_id:
            raise ValueError(f"{context}: cannot list itself as prior evidence")
    if not isinstance(experiment["downstreamIssues"], list):
        raise ValueError(f"{context}.downstreamIssues: expected an array")
    _unique(experiment["downstreamIssues"], f"{context}.downstreamIssues")
    if any(not isinstance(issue, int) or issue < 1 for issue in experiment["downstreamIssues"]):
        raise ValueError(f"{context}.downstreamIssues: expected positive issue numbers")

    definition_keys = (
        "unitOfWork",
        "candidates",
        "workloads",
        "controlledVariables",
        "changedVariables",
        "timedBoundary",
        "excludedWork",
        "allocationPolicy",
    )
    correctness_keys = ("oracle", "tolerance", "capabilityHandling", "canSupport", "cannotSupport")
    if not isinstance(experiment["definition"], dict):
        raise ValueError(f"{context}.definition: expected an object")
    if not isinstance(experiment["correctness"], dict):
        raise ValueError(f"{context}.correctness: expected an object")
    _require_keys(experiment["definition"], definition_keys, f"{context}.definition")
    _require_keys(experiment["correctness"], correctness_keys, f"{context}.correctness")
    for key in definition_keys:
        _require_nonempty_string(experiment["definition"][key], f"{context}.definition.{key}")
    for key in correctness_keys:
        _require_nonempty_string(experiment["correctness"][key], f"{context}.correctness.{key}")
    if not isinstance(experiment["requiredTables"], list) or not experiment["requiredTables"]:
        raise ValueError(f"{context}.requiredTables: expected a non-empty array")
    _unique(experiment["requiredTables"], f"{context}.requiredTables")
    for index, table in enumerate(experiment["requiredTables"]):
        _require_nonempty_string(table, f"{context}.requiredTables[{index}]")


def load_and_validate(results_dir: Path, baseline_ref: str | None = None) -> tuple[dict, list[PublishedBundle]]:
    results_dir = results_dir.resolve()
    catalog_path = results_dir / "catalog.json"
    if not catalog_path.is_file():
        raise ValueError(f"Publication catalog is missing: {catalog_path}")
    with catalog_path.open(encoding="utf-8") as stream:
        catalog = json.load(stream)
    _require_keys(catalog, ("schema", "runs", "experiments"), str(catalog_path))
    if catalog["schema"] != CATALOG_SCHEMA:
        raise ValueError(f"{catalog_path}: unsupported catalog schema {catalog['schema']!r}")
    if not isinstance(catalog["runs"], list) or not isinstance(catalog["experiments"], list):
        raise ValueError(f"{catalog_path}: runs and experiments must be arrays")

    experiment_ids = [experiment.get("id") for experiment in catalog["experiments"]]
    if any(not isinstance(experiment_id, str) for experiment_id in experiment_ids):
        raise ValueError(f"{catalog_path}: every experiment requires a string id")
    _unique(experiment_ids, f"{catalog_path}: experiment ids")
    issue_numbers = [experiment.get("issue") for experiment in catalog["experiments"]]
    _unique(issue_numbers, f"{catalog_path}: experiment issue numbers")
    experiment_id_set = set(experiment_ids)
    experiments_by_id = {experiment["id"]: experiment for experiment in catalog["experiments"]}
    for experiment in catalog["experiments"]:
        _validate_experiment(experiment, experiment_id_set)

    run_ids = [run.get("id") for run in catalog["runs"]]
    if any(not isinstance(run_id, str) for run_id in run_ids):
        raise ValueError(f"{catalog_path}: every run requires a string id")
    _unique(run_ids, f"{catalog_path}: run ids")
    run_id_set = set(run_ids)
    catalog_artifacts: set[str] = set()
    legacy_urls: set[str] = set()
    bundles: list[PublishedBundle] = []
    seen_result_run_ids: set[str] = set()

    for run in catalog["runs"]:
        run_id = run["id"]
        context = f"run {run_id}"
        _require_keys(
            run,
            (
                "id",
                "status",
                "statusReason",
                "summary",
                "issues",
                "experiments",
                "priorExperiments",
                "supersedes",
                "grandfathered",
                "artifacts",
                "legacyUrls",
            ),
            context,
        )
        if run["status"] not in PUBLICATION_STATUSES:
            raise ValueError(f"{context}: invalid publication status {run['status']!r}")
        for key in ("id", "statusReason", "summary"):
            _require_nonempty_string(run[key], f"{context}.{key}")
        if "incrementId" in run:
            _require_nonempty_string(run["incrementId"], f"{context}.incrementId")
        if not isinstance(run["issues"], list) or not run["issues"]:
            raise ValueError(f"{context}.issues: expected a non-empty array")
        if not isinstance(run["experiments"], list) or not run["experiments"]:
            raise ValueError(f"{context}.experiments: expected a non-empty array")
        _unique(run["issues"], f"{context}.issues")
        _unique(run["experiments"], f"{context}.experiments")
        for experiment_id in run["experiments"]:
            if experiment_id not in experiment_id_set:
                raise ValueError(f"{context}: unknown experiment {experiment_id!r}")
        expected_issues = sorted(experiments_by_id[experiment_id]["issue"] for experiment_id in run["experiments"])
        if sorted(run["issues"]) != expected_issues:
            raise ValueError(f"{context}: issue numbers do not match associated experiments")
        for relation_key in ("priorExperiments", "supersedes", "legacyUrls"):
            if not isinstance(run[relation_key], list):
                raise ValueError(f"{context}.{relation_key}: expected an array")
            _unique(run[relation_key], f"{context}.{relation_key}")
        for prior in run["priorExperiments"]:
            if prior not in experiment_id_set:
                raise ValueError(f"{context}: unknown prior experiment {prior!r}")
        for superseded in run["supersedes"]:
            if superseded not in run_id_set:
                raise ValueError(f"{context}: unknown superseded run {superseded!r}")
            if superseded == run_id:
                raise ValueError(f"{context}: cannot supersede itself")
        for legacy_url in run["legacyUrls"]:
            if not isinstance(legacy_url, str) or not legacy_url.startswith("/") or ".." in Path(legacy_url).parts:
                raise ValueError(f"{context}: invalid legacy URL {legacy_url!r}")
            if legacy_url in legacy_urls:
                raise ValueError(f"{context}: legacy URL is already assigned to another run: {legacy_url}")
            legacy_urls.add(legacy_url)
        if not isinstance(run["grandfathered"], bool):
            raise ValueError(f"{context}.grandfathered: expected a boolean")

        artifacts = run["artifacts"]
        _require_keys(artifacts, ("result", "samples", "resultSha256", "samplesSha256"), f"{context}.artifacts")
        result_path = _artifact_path(results_dir, artifacts["result"], f"{context}.artifacts.result")
        samples_path = _artifact_path(results_dir, artifacts["samples"], f"{context}.artifacts.samples")
        if not run["grandfathered"]:
            expected_result = results_dir / "runs" / run_id / "result.json"
            expected_samples = results_dir / "runs" / run_id / "samples.csv"
            if result_path != expected_result.resolve() or samples_path != expected_samples.resolve():
                raise ValueError(f"{context}: new bundles must use results/published/runs/<run-id>/result.json and samples.csv")
        for path, digest_key in ((result_path, "resultSha256"), (samples_path, "samplesSha256")):
            if not path.is_file():
                raise ValueError(f"{context}: published artifact is missing: {path}")
            expected_digest = artifacts[digest_key]
            if not isinstance(expected_digest, str) or len(expected_digest) != 64:
                raise ValueError(f"{context}.{digest_key}: expected a lowercase SHA-256 digest")
            actual_digest = sha256_file(path)
            if actual_digest != expected_digest:
                raise ValueError(f"{context}: immutable artifact changed: {path.name}")
            relative = path.relative_to(results_dir).as_posix()
            if relative in catalog_artifacts:
                raise ValueError(f"{context}: artifact is already assigned to another run: {relative}")
            catalog_artifacts.add(relative)

        with result_path.open(encoding="utf-8") as stream:
            result = json.load(stream)
        _validate_result(result_path, result, run_id, run["grandfathered"])
        _validate_samples(samples_path, run_id)
        if run_id in seen_result_run_ids:
            raise ValueError(f"{context}: duplicate run.id in published results")
        seen_result_run_ids.add(run_id)
        if run["status"] == "reference":
            if result["status"] != "passed":
                raise ValueError(f"{context}: a failed numerical run cannot be reference evidence")
            if result["environment"].get("gitDirty") is not False:
                raise ValueError(f"{context}: reference evidence must come from a clean benchmark tree")
        bundles.append(PublishedBundle(run, result_path, samples_path, result))

    runs_by_id = {run["id"]: run for run in catalog["runs"]}
    superseded_targets = {run_id for run in catalog["runs"] for run_id in run["supersedes"]}
    for run in catalog["runs"]:
        for superseded in run["supersedes"]:
            if runs_by_id[superseded]["status"] != "superseded":
                raise ValueError(f"run {run['id']}: superseded target {superseded} must have status 'superseded'")
        if run["status"] == "superseded" and run["id"] not in superseded_targets:
            raise ValueError(f"run {run['id']}: superseded status requires a superseding run relation")

    discovered_artifacts = {
        path.relative_to(results_dir).as_posix()
        for path in results_dir.rglob("*")
        if path.is_file()
        and path != catalog_path
        and not (
            path.suffix == ".json"
            and path.relative_to(results_dir).parts[0] == "decisions"
        )
        and (path.suffix == ".csv" or (path.suffix == ".json" and "schema" not in path.parts))
    }
    unlisted = sorted(discovered_artifacts - catalog_artifacts)
    if unlisted:
        raise ValueError(f"Published result artifacts are missing from the catalog: {', '.join(unlisted)}")

    if baseline_ref is not None:
        _validate_against_git_ref(catalog_path, catalog, baseline_ref)

    bundles.sort(key=lambda bundle: bundle.result["environment"]["timestampUtc"], reverse=True)
    return catalog, bundles


def _git_show(repository_root: Path, ref: str, repository_path: Path) -> bytes | None:
    relative = repository_path.resolve().relative_to(repository_root.resolve()).as_posix()
    process = subprocess.run(
        ["git", "show", f"{ref}:{relative}"],
        cwd=repository_root,
        check=False,
        capture_output=True,
    )
    if process.returncode == 0:
        return process.stdout
    return None


def _validate_against_git_ref(catalog_path: Path, catalog: dict, baseline_ref: str) -> None:
    repository_root = catalog_path.resolve().parents[2]
    commit_check = subprocess.run(
        ["git", "cat-file", "-e", f"{baseline_ref}^{{commit}}"],
        cwd=repository_root,
        check=False,
        capture_output=True,
    )
    if commit_check.returncode != 0:
        raise ValueError(f"Baseline ref is not a commit: {baseline_ref}")
    baseline_catalog_bytes = _git_show(repository_root, baseline_ref, catalog_path)
    if baseline_catalog_bytes is None:
        return
    baseline_catalog = json.loads(baseline_catalog_bytes)
    current_runs = {run["id"]: run for run in catalog["runs"]}
    for baseline_run in baseline_catalog.get("runs", []):
        run_id = baseline_run["id"]
        if run_id not in current_runs:
            raise ValueError(f"Append-only violation: published run {run_id} was removed from the catalog")
        current_run = current_runs[run_id]
        for key in ("grandfathered", "artifacts"):
            if current_run.get(key) != baseline_run.get(key):
                raise ValueError(f"Append-only violation: immutable metadata changed for run {run_id}: {key}")
        for artifact_key in ("result", "samples"):
            artifact_path = catalog_path.parent / baseline_run["artifacts"][artifact_key]
            baseline_bytes = _git_show(repository_root, baseline_ref, artifact_path)
            if baseline_bytes is None:
                raise ValueError(f"Append-only violation: baseline artifact is missing for run {run_id}: {artifact_key}")
            if not artifact_path.is_file() or artifact_path.read_bytes() != baseline_bytes:
                raise ValueError(f"Append-only violation: published artifact changed for run {run_id}: {artifact_key}")
