#!/usr/bin/env python3
"""Complete experiment pages and associate existing immutable evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def finalize_catalog(
    catalog: dict,
    complete_experiments: list[str],
    associations: list[tuple[str, str]],
) -> dict[str, int]:
    experiments = {item["id"]: item for item in catalog["experiments"]}
    unknown = sorted(set(complete_experiments) - experiments.keys())
    unknown.extend(
        experiment_id
        for experiment_id, _ in associations
        if experiment_id not in experiments
    )
    if unknown:
        raise ValueError(f"unknown experiment: {', '.join(sorted(set(unknown)))}")

    for experiment_id in complete_experiments:
        experiments[experiment_id]["phase"] = "complete"

    associated = 0
    for experiment_id, increment_id in associations:
        issue = experiments[experiment_id]["issue"]
        matched = 0
        for run in catalog["runs"]:
            if run.get("incrementId") != increment_id:
                continue
            matched += 1
            if experiment_id not in run["experiments"]:
                run["experiments"].append(experiment_id)
                run["experiments"].sort()
                run["issues"].append(issue)
                run["issues"].sort()
                associated += 1
        if matched == 0:
            raise ValueError(
                f"increment {increment_id!r} has no published runs to associate"
            )
    return {
        "completedExperiments": len(set(complete_experiments)),
        "newAssociations": associated,
    }


def parse_association(value: str) -> tuple[str, str]:
    experiment_id, separator, increment_id = value.partition("=")
    if not separator or not experiment_id or not increment_id:
        raise argparse.ArgumentTypeError(
            "association must be EXPERIMENT_ID=INCREMENT_ID"
        )
    return experiment_id, increment_id


def main() -> int:
    repository_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--catalog",
        type=Path,
        default=repository_root / "results" / "published" / "catalog.json",
    )
    parser.add_argument("--complete-experiment", action="append", default=[])
    parser.add_argument(
        "--associate-increment",
        action="append",
        default=[],
        type=parse_association,
        metavar="EXPERIMENT_ID=INCREMENT_ID",
    )
    parser.add_argument("--apply", action="store_true")
    arguments = parser.parse_args()
    if not arguments.complete_experiment and not arguments.associate_increment:
        parser.error("at least one completion or association is required")

    with arguments.catalog.open(encoding="utf-8") as stream:
        catalog = json.load(stream)
    try:
        summary = finalize_catalog(
            catalog,
            arguments.complete_experiment,
            arguments.associate_increment,
        )
    except ValueError as error:
        parser.error(str(error))
    print(
        f"Prepared {summary['completedExperiments']} completion(s) and "
        f"{summary['newAssociations']} new run association(s)."
    )
    if not arguments.apply:
        print("Dry run only; pass --apply after reviewing the requested changes.")
        return 0
    with arguments.catalog.open("w", encoding="utf-8") as stream:
        json.dump(catalog, stream, indent=2)
        stream.write("\n")
    print(f"Updated {arguments.catalog}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
