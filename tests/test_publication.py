import copy
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "tools"))

from publication import load_and_validate  # noqa: E402


PUBLISHED = REPOSITORY_ROOT / "results" / "published"


class PublicationValidationTests(unittest.TestCase):
    def copy_publication(self, root: Path) -> Path:
        destination = root / "published"
        shutil.copytree(PUBLISHED, destination)
        return destination

    def load_catalog(self, published: Path) -> dict:
        return json.loads((published / "catalog.json").read_text(encoding="utf-8"))

    def write_catalog(self, published: Path, catalog: dict) -> None:
        (published / "catalog.json").write_text(json.dumps(catalog, indent=2) + "\n", encoding="utf-8")

    def test_current_catalog_and_grandfathered_hashes_validate(self) -> None:
        catalog, bundles = load_and_validate(PUBLISHED)
        self.assertEqual("spectral-kernel-publication-catalog-v1", catalog["schema"])
        self.assertEqual(len(catalog["runs"]), len(bundles))
        grandfathered = next(bundle for bundle in bundles if bundle.publication["grandfathered"])
        self.assertEqual("20260827T185428Z-lyra", grandfathered.publication["id"])
        self.assertEqual("preliminary", grandfathered.publication["status"])
        self.assertEqual(14, len(catalog["experiments"]))
        experiment_phases = {
            experiment["id"]: experiment["phase"]
            for experiment in catalog["experiments"]
        }
        for experiment_id in (
            "issue-003-fftw-production-baseline",
            "issue-008-vertical-projection-gemm",
            "issue-013-ordering-packing-crossover",
        ):
            self.assertEqual("complete", experiment_phases[experiment_id])
        self.assertEqual(
            "collecting",
            experiment_phases["issue-009-combined-spectral-pipeline"],
        )
        fftw_production_reference = [
            bundle for bundle in bundles
            if bundle.publication.get("incrementId") ==
            "fftw-production-baseline-reference-v1"
        ]
        self.assertEqual(30, len(fftw_production_reference))
        self.assertEqual(10, len({
            bundle.result["run"]["profile"]
            for bundle in fftw_production_reference
        }))
        self.assertEqual({1, 2, 3}, {
            bundle.publication["campaignRound"]
            for bundle in fftw_production_reference
        })
        self.assertEqual({"9c28eab8d148"}, {
            bundle.result["environment"]["gitCommit"]
            for bundle in fftw_production_reference
        })
        vertical_finalist_reference = [
            bundle for bundle in bundles
            if bundle.publication.get("incrementId") ==
            "vertical-k2-grouped-finalists-reference-v1"
        ]
        self.assertEqual(60, len(vertical_finalist_reference))
        self.assertEqual(
            {"outer-dynamic-16", "outer-static-12"},
            {
                bundle.publication["campaignCandidateId"]
                for bundle in vertical_finalist_reference
            },
        )
        self.assertEqual({"28d8cd90710a"}, {
            bundle.result["environment"]["gitCommit"]
            for bundle in vertical_finalist_reference
        })
        deep_vertical_reference = [
            bundle for bundle in bundles
            if bundle.publication.get("incrementId") ==
            "synthetic-spectral-pipeline-deep-vertical-reference-v1"
        ]
        self.assertEqual(6, len(deep_vertical_reference))
        self.assertEqual({"wvm-large-512-nz513-f4"}, {
            bundle.result["run"]["profile"]
            for bundle in deep_vertical_reference
        })
        self.assertEqual({"a7989508e60a"}, {
            bundle.result["environment"]["gitCommit"]
            for bundle in deep_vertical_reference
        })
        for bundle in deep_vertical_reference:
            provider = bundle.result["providers"][0]
            self.assertTrue(any(
                timing["scope"] == "setup-component"
                and timing["stage"] ==
                "release correctness-only benchmark storage"
                and timing["state"] == "setup-only"
                for timing in provider["timings"]
            ))
        pipeline_screen = [
            bundle for bundle in bundles
            if bundle.publication.get("incrementId") ==
            "synthetic-spectral-pipeline-screen-v1"
        ]
        pipeline_reference = [
            bundle for bundle in bundles
            if bundle.publication.get("incrementId") ==
            "synthetic-spectral-pipeline-reference-v1"
        ]
        self.assertEqual(12, len(pipeline_screen))
        self.assertEqual(36, len(pipeline_reference))
        self.assertTrue(all(bundle.publication["status"] == "preliminary" for bundle in pipeline_screen))
        self.assertTrue(all(bundle.publication["status"] == "reference" for bundle in pipeline_reference))
        self.assertEqual({1, 2, 3}, {
            bundle.publication["campaignRound"] for bundle in pipeline_reference
        })
        self.assertEqual(
            {
                "wvm-direct--outer-dynamic-16",
                "plane-major-fused-split--outer-dynamic-16",
            },
            {
                bundle.publication["campaignCandidateId"]
                for bundle in pipeline_reference
            },
        )
        pipeline_large_f4_screen = [
            bundle for bundle in bundles
            if bundle.publication.get("incrementId") ==
            "synthetic-spectral-pipeline-large-f4-screen-v1"
        ]
        pipeline_large_f4_reference = [
            bundle for bundle in bundles
            if bundle.publication.get("incrementId") ==
            "synthetic-spectral-pipeline-large-f4-reference-v1"
        ]
        self.assertEqual(8, len(pipeline_large_f4_screen))
        self.assertEqual(24, len(pipeline_large_f4_reference))
        self.assertTrue(all(
            bundle.publication["status"] == "preliminary"
            for bundle in pipeline_large_f4_screen
        ))
        self.assertTrue(all(
            bundle.publication["status"] == "reference"
            and not bundle.result["environment"]["gitDirty"]
            for bundle in pipeline_large_f4_reference
        ))
        self.assertEqual({1, 2, 3}, {
            bundle.publication["campaignRound"]
            for bundle in pipeline_large_f4_reference
        })
        self.assertEqual(
            {
                "wvm-direct--outer-dynamic-16",
                "plane-major-fused-split--outer-dynamic-16",
            },
            {
                bundle.publication["campaignCandidateId"]
                for bundle in pipeline_large_f4_reference
            },
        )
        self.assertEqual(
            {
                "wvm-current-256-nz129-f4",
                "wvm-historical-512-nz129-f4",
                "wvm-current-512-nz257-f4",
                "wvm-large-1024-nz129-f4",
            },
            {
                bundle.result["run"]["profile"]
                for bundle in pipeline_large_f4_reference
            },
        )
        for bundle in pipeline_large_f4_reference:
            provider = bundle.result["providers"][0]
            memory = provider["memory"]
            self.assertGreater(memory["algorithmResidentBytes"], 0)
            self.assertGreater(memory["benchmarkHarnessBytes"], 0)
            self.assertGreater(memory["observedProcessHighWaterBytes"], 0)
            self.assertEqual(
                memory["estimatedProcessPeakBytes"],
                memory["algorithmResidentBytes"] +
                memory["benchmarkHarnessBytes"],
            )
            self.assertEqual(
                memory["estimatedProcessPeakBytes"],
                bundle.result["workload"]["bytes"]
                ["spectralPipelineEstimatedExplicitPeak"],
            )
        streaming_pruned_screen = [
            bundle for bundle in bundles
            if bundle.publication.get("incrementId") ==
            "streaming-pruned-compact-split-screen-v1"
        ]
        self.assertEqual(6, len(streaming_pruned_screen))
        self.assertTrue(all(
            bundle.publication["status"] == "preliminary"
            and not bundle.result["environment"]["gitDirty"]
            for bundle in streaming_pruned_screen
        ))
        self.assertEqual(
            {
                "plane-major-fused-split--outer-dynamic-16",
                "streaming-pruned-compact-split--outer-dynamic-16",
            },
            {
                bundle.publication["campaignCandidateId"]
                for bundle in streaming_pruned_screen
            },
        )
        self.assertEqual(
            {
                "wvm-current-256-nz129-f4",
                "wvm-current-512-nz257-f4",
                "wvm-large-1024-nz129-f4",
            },
            {
                bundle.result["run"]["profile"]
                for bundle in streaming_pruned_screen
            },
        )
        streaming_candidates = [
            bundle for bundle in streaming_pruned_screen
            if bundle.result["providers"][0]["id"] ==
            "pipeline-streaming-pruned-compact-split"
        ]
        self.assertEqual(3, len(streaming_candidates))
        for bundle in streaming_candidates:
            provider = bundle.result["providers"][0]
            memory = provider["memory"]
            self.assertGreater(memory["algorithmResidentBytes"], 0)
            self.assertGreater(memory["benchmarkHarnessBytes"], 0)
            self.assertGreater(memory["scratchBytes"], 0)
            self.assertLess(
                memory["scratchBytes"],
                bundle.result["workload"]["bytes"]["fullSpectrum"],
            )
            self.assertGreater(memory["observedProcessHighWaterBytes"], 0)
            self.assertEqual(
                memory["estimatedProcessPeakBytes"],
                memory["algorithmResidentBytes"] +
                memory["benchmarkHarnessBytes"],
            )
            self.assertEqual(
                memory["estimatedProcessPeakBytes"],
                bundle.result["workload"]["bytes"]
                ["spectralPipelineEstimatedExplicitPeak"],
            )
        streaming_locality_screen = [
            bundle for bundle in bundles
            if bundle.publication.get("incrementId") ==
            "streaming-pruned-compact-split-locality-screen-v1"
        ]
        self.assertEqual(15, len(streaming_locality_screen))
        self.assertTrue(all(
            bundle.publication["status"] == "preliminary"
            and not bundle.result["environment"]["gitDirty"]
            for bundle in streaming_locality_screen
        ))
        self.assertEqual(
            {
                "plane-major-fused-split--outer-dynamic-16",
                "streaming-pruned-direct-1--outer-dynamic-16",
                "streaming-pruned-tiled-4--outer-dynamic-16",
                "streaming-pruned-tiled-8--outer-dynamic-16",
                "streaming-pruned-tiled-16--outer-dynamic-16",
            },
            {
                bundle.publication["campaignCandidateId"]
                for bundle in streaming_locality_screen
            },
        )
        self.assertEqual(
            {
                "wvm-current-256-nz129-f4",
                "wvm-current-512-nz257-f4",
                "wvm-large-1024-nz129-f4",
            },
            {
                bundle.result["run"]["profile"]
                for bundle in streaming_locality_screen
            },
        )
        locality_candidates = [
            bundle for bundle in streaming_locality_screen
            if bundle.publication["campaignCandidateId"].startswith(
                "streaming-pruned-tiled-"
            )
        ]
        self.assertEqual(9, len(locality_candidates))
        for bundle in locality_candidates:
            provider = bundle.result["providers"][0]
            memory = provider["memory"]
            self.assertEqual(
                "pipeline-streaming-pruned-compact-split",
                provider["id"],
            )
            self.assertIn("compact-tile", provider["algorithmId"])
            self.assertIn(
                "plane-major compact tile with 32-mode blocked transpose",
                provider["planning"]["configuration"],
            )
            self.assertGreater(memory["scratchBytes"], 0)
            self.assertLess(
                memory["scratchBytes"],
                bundle.result["workload"]["bytes"]["fullSpectrum"],
            )
            self.assertEqual(
                memory["estimatedProcessPeakBytes"],
                memory["algorithmResidentBytes"] +
                memory["benchmarkHarnessBytes"],
            )
        streaming_reference = [
            bundle for bundle in bundles
            if bundle.publication.get("incrementId") ==
            "streaming-pruned-compact-split-reference-v1"
        ]
        self.assertEqual(18, len(streaming_reference))
        self.assertTrue(all(
            bundle.publication["status"] == "reference"
            and not bundle.result["environment"]["gitDirty"]
            for bundle in streaming_reference
        ))
        self.assertEqual({1, 2, 3}, {
            bundle.publication["campaignRound"]
            for bundle in streaming_reference
        })
        self.assertEqual(
            {
                "plane-major-fused-split--outer-dynamic-16",
                "streaming-pruned-tiled-16--outer-dynamic-16",
            },
            {
                bundle.publication["campaignCandidateId"]
                for bundle in streaming_reference
            },
        )
        self.assertEqual(
            {
                "wvm-current-256-nz129-f4",
                "wvm-current-512-nz257-f4",
                "wvm-large-1024-nz129-f4",
            },
            {
                bundle.result["run"]["profile"]
                for bundle in streaming_reference
            },
        )
        self.assertEqual(
            {"7c212a904353"},
            {
                bundle.result["environment"]["gitCommit"]
                for bundle in streaming_reference
            },
        )
        self.assertTrue(all(
            bundle.publication["issues"] == [13, 16]
            and bundle.publication["experiments"] == [
                "issue-013-ordering-packing-crossover",
                "issue-016-streaming-pruned-compact-split",
            ]
            for bundle in streaming_reference
        ))
        for bundle in streaming_reference:
            provider = bundle.result["providers"][0]
            memory = provider["memory"]
            self.assertEqual(
                memory["estimatedProcessPeakBytes"],
                memory["algorithmResidentBytes"] +
                memory["benchmarkHarnessBytes"],
            )
            self.assertGreater(memory["observedProcessHighWaterBytes"], 0)
        native_bridge_reference = [
            bundle for bundle in bundles
            if bundle.publication.get("incrementId") ==
            "synthetic-spectral-pipeline-three-way-native-control-reference-v1"
        ]
        self.assertEqual(27, len(native_bridge_reference))
        self.assertTrue(all(
            bundle.publication["status"] == "reference"
            and not bundle.result["environment"]["gitDirty"]
            for bundle in native_bridge_reference
        ))
        self.assertEqual({1, 2, 3}, {
            bundle.publication["campaignRound"]
            for bundle in native_bridge_reference
        })
        self.assertEqual(
            {
                "wvm-direct--outer-dynamic-16",
                "plane-major-fused-split--outer-dynamic-16",
                "streaming-pruned-tiled-16--outer-dynamic-16",
            },
            {
                bundle.publication["campaignCandidateId"]
                for bundle in native_bridge_reference
            },
        )
        self.assertEqual(
            {
                "wvm-current-256-nz129-f4",
                "wvm-current-512-nz257-f4",
                "wvm-large-1024-nz129-f4",
            },
            {
                bundle.result["run"]["profile"]
                for bundle in native_bridge_reference
            },
        )
        self.assertEqual(
            {"3f0c535887ab"},
            {
                bundle.result["environment"]["gitCommit"]
                for bundle in native_bridge_reference
            },
        )
        self.assertEqual(
            {"issue-009-combined-spectral-pipeline"},
            {
                experiment
                for bundle in native_bridge_reference
                for experiment in bundle.publication["experiments"]
            },
        )
        for bundle in native_bridge_reference:
            provider = bundle.result["providers"][0]
            memory = provider["memory"]
            self.assertEqual(
                memory["estimatedProcessPeakBytes"],
                memory["algorithmResidentBytes"] +
                memory["benchmarkHarnessBytes"],
            )
            self.assertGreater(memory["observedProcessHighWaterBytes"], 0)
            self.assertEqual("out-of-place", provider["executionContract"]
                             ["forward"]["nativePlacement"])
            self.assertEqual("out-of-place", provider["executionContract"]
                             ["inverse"]["nativePlacement"])
        outer_increment = [
            bundle for bundle in bundles
            if bundle.publication.get("incrementId") ==
            "fftw-partial-column-pruned-outer-sharding-v2"
        ]
        self.assertEqual(24, len(outer_increment))
        self.assertTrue(all(not bundle.result["environment"]["gitDirty"] for bundle in outer_increment))
        retained_views = [
            provider
            for bundle in bundles
            for provider in bundle.result["providers"]
            if provider["id"] == "fftw-plane-major-retained-view"
        ]
        self.assertEqual(24, len(retained_views))
        self.assertTrue(all(
            provider["executionContract"][direction]["adapterPlacement"] ==
                "out-of-place-view"
            for provider in retained_views
            for direction in ("forward", "inverse")
        ))

    def test_duplicate_run_id_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            published = self.copy_publication(Path(temporary_directory))
            catalog = self.load_catalog(published)
            catalog["runs"].append(copy.deepcopy(catalog["runs"][0]))
            self.write_catalog(published, catalog)
            with self.assertRaisesRegex(ValueError, "run ids: values must be unique"):
                load_and_validate(published)

    def test_modified_artifact_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            published = self.copy_publication(Path(temporary_directory))
            result = published / "m4-max-quick-20260827.json"
            result.write_bytes(result.read_bytes() + b"\n")
            with self.assertRaisesRegex(ValueError, "immutable artifact changed"):
                load_and_validate(published)

    def test_missing_pair_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            published = self.copy_publication(Path(temporary_directory))
            (published / "m4-max-quick-20260827.csv").unlink()
            with self.assertRaisesRegex(ValueError, "published artifact is missing"):
                load_and_validate(published)

    def test_unlisted_artifact_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            published = self.copy_publication(Path(temporary_directory))
            (published / "orphan.csv").write_text("run_id\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "missing from the catalog"):
                load_and_validate(published)

    def test_reviewed_decision_analysis_is_not_a_run_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            published = self.copy_publication(Path(temporary_directory))
            decisions = published / "decisions"
            decisions.mkdir(exist_ok=True)
            (decisions / "issue-011-example.json").write_text(
                json.dumps({"schema": "spectral-kernel-cross-mac-reference-analysis-v1"}) + "\n",
                encoding="utf-8",
            )
            catalog, bundles = load_and_validate(published)
            self.assertEqual(len(catalog["runs"]), len(bundles))

    def test_new_run_must_use_run_id_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            published = self.copy_publication(Path(temporary_directory))
            catalog = self.load_catalog(published)
            catalog["runs"][0]["grandfathered"] = False
            self.write_catalog(published, catalog)
            with self.assertRaisesRegex(ValueError, "new bundles must use"):
                load_and_validate(published)

    def test_dirty_run_cannot_be_reference_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            published = self.copy_publication(Path(temporary_directory))
            catalog = self.load_catalog(published)
            catalog["runs"][0]["status"] = "reference"
            catalog["runs"][0]["statusReason"] = "Attempted promotion."
            self.write_catalog(published, catalog)
            with self.assertRaisesRegex(ValueError, "reference evidence must come from a clean"):
                load_and_validate(published)


if __name__ == "__main__":
    unittest.main()
