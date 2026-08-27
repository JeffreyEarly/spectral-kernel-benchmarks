#!/usr/bin/env python3
"""Validate the append-only benchmark publication catalog and artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path

from publication import load_and_validate


def main() -> int:
    repository_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=repository_root / "results" / "published")
    parser.add_argument("--baseline-ref", help="Git commit whose published runs must remain byte-identical")
    arguments = parser.parse_args()
    catalog, bundles = load_and_validate(arguments.results, arguments.baseline_ref)
    print(f"Validated {len(bundles)} published run(s) and {len(catalog['experiments'])} experiment(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
