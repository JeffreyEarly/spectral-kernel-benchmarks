import copy
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "tools"))

import run_portable_machine_tuning as portable  # noqa: E402


def machine(performance: int = 8, efficiency: int = 2) -> dict:
    return {
        "hostname": "matilda",
        "cpuBrand": "Apple M1 Max",
        "hardwareModel": "MacBookPro18,4",
        "performanceCores": performance,
        "efficiencyCores": efficiency,
        "totalPhysicalCores": performance + efficiency,
        "physicalMemoryBytes": 64 * 1024 ** 3,
    }


def candidate_row(identifier: str, order: int, workers: dict,
                  seconds: float, memory: int = 1000) -> dict:
    return {
        "candidateId": identifier,
        "candidateOrder": order,
        "workers": workers,
        "complete": True,
        "valid": True,
        "geometricSeconds": seconds,
        "maximumAlgorithmResidentBytes": memory,
        "profiles": [],
    }


class PortableMachineTuningTests(unittest.TestCase):
    def test_registry_freezes_three_distinct_boundaries(self) -> None:
        registry = portable.implementations()
        self.assertEqual(3, len(registry))
        self.assertEqual({"general", "constant"}, {
            item.fixture_kind for item in registry
        })
        self.assertEqual("wvm-native-optimized-v1", registry[0].id)
        self.assertNotEqual(registry[1].mathematical_boundary,
                            registry[2].mathematical_boundary)

    def test_candidate_counts_cover_powers_performance_and_total(self) -> None:
        self.assertEqual((1, 2, 4, 8, 10), portable.candidate_counts(8, 10))
        self.assertEqual((1, 2, 4, 8, 12, 16), portable.candidate_counts(12, 16))
        with self.assertRaises(ValueError):
            portable.candidate_counts(12, 8)

    def test_portable_seed_matches_stage_semantics(self) -> None:
        self.assertEqual(
            portable.WorkerTuple(12, 8, 16, 16),
            portable.portable_seed(12, 16),
        )

    def test_one_factor_matrix_never_changes_two_knobs_at_once(self) -> None:
        implementation = portable.implementations()[0]
        seed = portable.portable_seed(8, 10).applicable(implementation)
        candidates = portable.one_factor_candidates(implementation, 8, 10)
        for candidate in candidates:
            changed = sum(
                value != seed[name]
                for name, value in candidate.applicable(implementation).items()
            )
            self.assertLessEqual(changed, 1)
        self.assertEqual(len(candidates), len({
            portable.tuple_identifier(implementation, item) for item in candidates
        }))

    def test_general_commands_freeze_algorithm_and_vary_only_workers(self) -> None:
        wvm, compact, _ = portable.implementations()
        workers = portable.WorkerTuple(8, 4, 10, 1)
        for implementation in (wvm, compact):
            command = portable.command_for(
                Path("skbench"), implementation, Path("fixture.bin"),
                portable.PROFILES[0], workers, 2, 7, Path("result.json"),
            )
            joined = " ".join(map(str, command))
            self.assertIn(f"--boundary-policy {implementation.boundary}", joined)
            self.assertIn("--fftw-internal-workers 1", joined)
            self.assertIn("--fftw-outer-workers 8", joined)
            self.assertIn("--vertical-gemm-outer-workers 10", joined)
            self.assertIn("--pointwise-workers 4", joined)
            self.assertIn("--streaming-tile-width 16", joined)

    def test_constant_command_maps_type1_workers_without_general_gemm(self) -> None:
        implementation = portable.implementations()[2]
        workers = portable.WorkerTuple(8, 4, 1, 10)
        command = portable.command_for(
            Path("skbench"), implementation, Path("constant.bin"),
            portable.PROFILES[0], workers, 2, 7, Path("result.json"),
        )
        joined = " ".join(map(str, command))
        self.assertIn("--fftw-internal-workers 10", joined)
        self.assertIn("--fftw-outer-workers 8", joined)
        self.assertNotIn("--vertical-gemm-outer-workers", joined)
        self.assertIn("--comparison-order candidate-first", joined)

    def test_selection_uses_one_percent_then_workers_memory_order(self) -> None:
        implementation = portable.implementations()[0]
        rows = [
            candidate_row("fast", 0, {
                "horizontal": 12, "general_vertical": 16, "pointwise": 12,
            }, 1.0, 900),
            candidate_row("fewer", 1, {
                "horizontal": 8, "general_vertical": 10, "pointwise": 4,
            }, 1.009, 1000),
            candidate_row("outside", 2, {
                "horizontal": 1, "general_vertical": 1, "pointwise": 1,
            }, 1.011, 100),
        ]
        self.assertEqual("fewer", portable.select_candidate(
            implementation, rows,
        )["candidateId"])

    def test_manifest_validation_rejects_duplicate_run_ids(self) -> None:
        manifest = {
            "schema": portable.SCHEMA,
            "experimentId": portable.EXPERIMENT_ID,
            "intendedUse": portable.INTENDED_USE,
            "productionValidated": False,
            "machine": machine(),
            "identity": {"executableSha256": "sha256:" + "a" * 64},
            "runs": [{"id": "same"}, {"id": "same"}],
        }
        errors = portable.validate_manifest(manifest)
        self.assertTrue(any("duplicate" in error for error in errors))

    def test_machine_compatibility_includes_topology_and_memory(self) -> None:
        recorded = machine()
        self.assertTrue(portable.machine_matches(recorded, dict(recorded)))
        changed = dict(recorded)
        changed["totalPhysicalCores"] = 12
        self.assertFalse(portable.machine_matches(recorded, changed))

    def test_atomic_write_round_trips(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "machine-tuning.json"
            portable.write_json_atomic(path, {"value": 3})
            self.assertEqual({"value": 3}, portable.load_json(path))
            self.assertFalse(path.with_suffix(".json.tmp").exists())

    def test_combined_neighborhood_is_added_only_after_complete_screen(self) -> None:
        implementation = portable.implementations()[0]
        topology = machine()
        matrix = portable.candidate_matrix((implementation,), topology)
        manifest = {
            "machine": topology,
            "implementations": [portable.contract_record(implementation)],
            "candidateMatrix": matrix,
            "runs": [],
        }
        for candidate in matrix[implementation.id]:
            for profile in portable.PROFILES:
                workers = candidate["workers"]
                seconds = 1.0 - 0.001 * sum(workers.values())
                manifest["runs"].append({
                    "id": f"{candidate['id']}--{profile}",
                    "implementationId": implementation.id,
                    "candidateId": candidate["id"],
                    "profile": profile,
                    "valid": True,
                    "record": {
                        "seconds": seconds,
                        "maximumCorrectnessError": 1.0e-15,
                        "valid": True,
                        "memory": {"algorithmResidentBytes": 1000},
                    },
                })
        before = copy.deepcopy(manifest["candidateMatrix"])
        self.assertTrue(portable.add_combined_neighborhood(manifest))
        self.assertGreater(
            len(manifest["candidateMatrix"][implementation.id]),
            len(before[implementation.id]),
        )
        self.assertTrue(any(
            item["phase"] == "combined-winner-neighborhood"
            for item in manifest["candidateMatrix"][implementation.id]
        ))

    def test_publication_dry_run_accepts_machine_tuning_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runs = root / "runs"
            runs.mkdir()
            result = {
                "run": {"id": "portable-test-run"},
                "environment": {"gitDirty": False, "cpuBrand": "Apple Test"},
                "numericType": {"id": "float64"},
                "workload": {"Nx": 8, "Ny": 8, "Nz": 5, "fields": 4},
                "providers": [{
                    "id": "selected-provider",
                    "algorithmId": "frozen-test-algorithm",
                    "workers": 2,
                }],
            }
            (runs / "cell.json").write_text(
                json.dumps(result), encoding="utf-8",
            )
            (runs / "cell.csv").write_text("sample\n1\n", encoding="utf-8")
            manifest = {
                "schema": portable.SCHEMA,
                "runs": [{
                    "id": "cell", "exitCode": 0,
                    "result": "runs/cell.json", "samples": "runs/cell.csv",
                    "primaryProvider": "selected-provider",
                    "candidateId": "h2--gv2--p2",
                }],
            }
            (root / "machine-tuning.json").write_text(
                json.dumps(manifest), encoding="utf-8",
            )
            completed = subprocess.run(
                [
                    sys.executable,
                    str(REPOSITORY_ROOT / "tools/publish_local_sweep.py"),
                    "--input", str(root),
                    "--experiment", portable.EXPERIMENT_ID,
                    "--status-reason", "portable manifest test",
                ],
                cwd=REPOSITORY_ROOT, capture_output=True, text=True,
            )
            self.assertEqual(0, completed.returncode, completed.stderr)
            self.assertIn("Prepared 1 immutable bundle", completed.stdout)

    def test_publication_summary_preserves_trace_and_run_links(self) -> None:
        contracts = [portable.contract_record(item)
                     for item in portable.implementations()]
        candidate_matrix = {}
        selections = {}
        runs = []
        for contract in contracts:
            implementation = portable.implementation_named(contract["id"])
            seed = portable.portable_seed(8, 10)
            workers = seed.applicable(implementation)
            candidate_id = portable.tuple_identifier(implementation, seed)
            candidate_matrix[implementation.id] = [{
                "id": candidate_id,
                "workers": workers,
                "phase": "one-factor-screen",
            }]
            profiles = []
            for profile in portable.PROFILES:
                run_id = f"run-{implementation.id}-{profile}"
                profiles.append({
                    "profile": profile,
                    "seconds": 1.0,
                    "maximumCorrectnessError": 1.0e-15,
                })
                runs.append({
                    "id": run_id,
                    "implementationId": implementation.id,
                    "candidateId": candidate_id,
                    "profile": profile,
                    "runId": run_id,
                    "valid": True,
                })
            candidate = candidate_row(candidate_id, 0, workers, 1.0)
            candidate["profiles"] = profiles
            selections[implementation.id] = {
                "candidates": [candidate],
                "selected": candidate,
                "intendedUse": portable.INTENDED_USE,
                "productionValidated": False,
            }
        identity = {
            "sourceTreeGitCommit": "a" * 40,
            "sourceTreeDirty": False,
            "executableSha256": "sha256:" + "b" * 64,
            "implementationContractHashes": {
                contract["id"]: contract["contractHash"]
                for contract in contracts
            },
            "fixtureHashes": {},
        }
        manifest = {
            "schema": portable.SCHEMA,
            "experimentId": portable.EXPERIMENT_ID,
            "incrementId": portable.INCREMENT_ID,
            "intendedUse": portable.INTENDED_USE,
            "productionValidated": False,
            "createdAtUtc": "2026-08-31T00:00:00Z",
            "machine": machine(),
            "identity": identity,
            "compatibilityHash": portable.canonical_hash(identity),
            "implementations": contracts,
            "profiles": list(portable.PROFILES),
            "candidateMatrix": candidate_matrix,
            "candidatePolicy": {},
            "threadEnvironment": portable.THREAD_ENVIRONMENT,
            "warmups": 2,
            "samples": 7,
            "runs": runs,
            "analysis": {
                "selectionRule": {"score": "test"},
                "selections": selections,
            },
        }
        summary = portable.publication_summary(manifest)
        self.assertEqual(portable.PUBLICATION_SCHEMA, summary["schema"])
        self.assertEqual(6, summary["campaign"]["validRunCount"])
        self.assertEqual(
            runs[0]["runId"],
            summary["implementations"][0]["selected"]["profiles"][0]["runId"],
        )
        self.assertFalse(summary["productionValidated"])


if __name__ == "__main__":
    unittest.main()
