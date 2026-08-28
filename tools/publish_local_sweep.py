#!/usr/bin/env python3
"""Stage reviewed local benchmark bundles in the append-only publication archive."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from publication import PUBLICATION_STATUSES, sha256_file


def load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def main() -> int:
    repository_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Local sweep directory containing manifest.json")
    parser.add_argument("--experiment", required=True, help="Existing stable experiment ID")
    parser.add_argument("--status", choices=sorted(PUBLICATION_STATUSES), default="preliminary")
    parser.add_argument("--status-reason", required=True)
    parser.add_argument("--prior-experiment", action="append", default=[])
    parser.add_argument(
        "--primary-provider",
        help="Provider used in catalog summaries; defaults to vDSP for issue #6 and otherwise the first provider",
    )
    parser.add_argument("--allow-dirty", action="store_true", help="Permit bundles whose embedded benchmark tree was dirty")
    parser.add_argument("--apply", action="store_true", help="Copy artifacts and update catalog.json")
    arguments = parser.parse_args()

    input_dir = arguments.input.resolve()
    manifest_path = input_dir / "manifest.json"
    if not manifest_path.is_file():
        parser.error(f"manifest is missing: {manifest_path}")
    manifest = load_json(manifest_path)
    if not isinstance(manifest.get("runs"), list) or not manifest["runs"]:
        parser.error("manifest contains no runs")
    increment_id = manifest.get("incrementId")
    if increment_id is not None and (
        not isinstance(increment_id, str) or not increment_id.strip()
    ):
        parser.error("manifest incrementId must be a non-empty string when present")

    published_dir = repository_root / "results" / "published"
    catalog_path = published_dir / "catalog.json"
    catalog = load_json(catalog_path)
    experiments = {item["id"]: item for item in catalog["experiments"]}
    if arguments.experiment not in experiments:
        parser.error(f"unknown experiment: {arguments.experiment}")
    for prior in arguments.prior_experiment:
        if prior not in experiments:
            parser.error(f"unknown prior experiment: {prior}")

    known_ids = {item["id"] for item in catalog["runs"]}
    planned: list[tuple[dict, Path, Path, Path, Path]] = []
    seen_ids: set[str] = set()
    for manifest_entry in manifest["runs"]:
        if manifest_entry.get("exitCode") != 0:
            parser.error(f"manifest run failed: {manifest_entry.get('id', '<unknown>')}")
        result_name = manifest_entry.get("result")
        if not isinstance(result_name, str):
            parser.error(f"manifest run lacks a result path: {manifest_entry.get('id', '<unknown>')}")
        source_result = input_dir / result_name
        source_samples = source_result.with_suffix(".csv")
        if not source_result.is_file() or not source_samples.is_file():
            parser.error(f"missing JSON/CSV pair for {result_name}")
        result = load_json(source_result)
        run_id = result["run"]["id"]
        if run_id in known_ids or run_id in seen_ids:
            parser.error(f"duplicate published run ID: {run_id}")
        seen_ids.add(run_id)
        if result["environment"].get("gitDirty") is not False and not arguments.allow_dirty:
            parser.error(f"run {run_id} records a dirty benchmark tree")
        destination_dir = published_dir / "runs" / run_id
        destination_result = destination_dir / "result.json"
        destination_samples = destination_dir / "samples.csv"
        if destination_dir.exists() or destination_result.exists() or destination_samples.exists():
            parser.error(f"publication destination already exists: {destination_dir}")

        workload = result["workload"]
        primary_provider = arguments.primary_provider
        if primary_provider is None and arguments.experiment == "issue-006-vdsp-batching-scheduling":
            primary_provider = "accelerate-vdsp"
        if primary_provider is None:
            provider = result["providers"][0]
        else:
            provider = next(
                (item for item in result["providers"] if item["id"] == primary_provider), None
            )
            if provider is None:
                parser.error(f"run {run_id} lacks primary provider {primary_provider}")
        scheduling = provider.get("scheduling", {})
        worker_description = (
            f"internal={scheduling['internalWorkers']}, outer={scheduling['outerWorkers']}"
            if scheduling
            else f"logical workers={provider['workers']}"
        )
        summary = (
            f"{result['environment']['cpuBrand']} {result['numericType']['id']} "
            f"{workload['Nx']} by {workload['Ny']}, Nz={workload['Nz']}, fields={workload['fields']} "
            f"using {provider['algorithmId']} ({worker_description})."
        )
        publication = {
            "id": run_id,
            **({"incrementId": increment_id} if increment_id is not None else {}),
            "status": arguments.status,
            "statusReason": arguments.status_reason,
            "summary": summary,
            "issues": [experiments[arguments.experiment]["issue"]],
            "experiments": [arguments.experiment],
            "priorExperiments": arguments.prior_experiment,
            "supersedes": [],
            "grandfathered": False,
            "artifacts": {
                "result": f"runs/{run_id}/result.json",
                "samples": f"runs/{run_id}/samples.csv",
                "resultSha256": sha256_file(source_result),
                "samplesSha256": sha256_file(source_samples),
            },
            "legacyUrls": [],
        }
        planned.append((publication, source_result, source_samples, destination_result, destination_samples))

    print(f"Prepared {len(planned)} immutable bundle(s) for {arguments.experiment} as {arguments.status}.")
    if not arguments.apply:
        print("Dry run only; pass --apply after reviewing the sweep and publication metadata.")
        return 0

    for publication, source_result, source_samples, destination_result, destination_samples in planned:
        destination_result.parent.mkdir(parents=True, exist_ok=False)
        shutil.copyfile(source_result, destination_result)
        shutil.copyfile(source_samples, destination_samples)
        catalog["runs"].append(publication)
    experiments[arguments.experiment]["phase"] = "collecting"
    with catalog_path.open("w", encoding="utf-8") as stream:
        json.dump(catalog, stream, indent=2)
        stream.write("\n")
    print(f"Updated {catalog_path}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
