import sys
import unittest
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "tools"))

import run_spectral_pipeline_native_bridge as bridge  # noqa: E402


def fake_result(
    candidate: bridge.LocalityCandidate,
    profile: str,
    seconds: float,
    resident: int,
    observed: int,
) -> dict:
    return {
        "status": "passed",
        "run": {"profile": profile},
        "providers": [{
            "id": candidate.primary_provider,
            "timings": [{
                "scope": "uninstrumented-total",
                "stage": "synthetic antialiased spectral pipeline",
                "direction": "round-trip",
                "medianSeconds": seconds,
            }],
            "correctness": [{
                "passed": True,
                "maximumRelativeError": 1.0e-15,
            }],
            "memory": {
                "algorithmResidentBytes": resident,
                "benchmarkHarnessBytes": 500,
                "estimatedProcessPeakBytes": resident + 500,
                "observedProcessHighWaterBytes": observed,
            },
            "setup": {
                "totalSeconds": 0.1,
                "allocationSeconds": 0.01,
            },
            "planning": {
                "seconds": 0.02,
                "temporaryBytes": 100,
            },
            "executionContract": {
                "forward": {"nativePlacement": "out-of-place"},
                "inverse": {"nativePlacement": "out-of-place"},
            },
        }],
    }


class SpectralPipelineNativeBridgeTests(unittest.TestCase):
    def results(self) -> list:
        candidates = bridge.candidate_matrix()
        ratios = {
            bridge.WVM_ID: 1.0,
            bridge.FUSED_ID: 0.80,
            bridge.TILED_ID: 0.60,
        }
        residents = {
            bridge.WVM_ID: 1200,
            bridge.FUSED_ID: 1000,
            bridge.TILED_ID: 850,
        }
        results = []
        for round_number in range(1, bridge.REFERENCE_ROUNDS + 1):
            for profile in bridge.PROFILES:
                for candidate in candidates:
                    results.append((candidate, round_number, fake_result(
                        candidate, profile, ratios[candidate.id],
                        residents[candidate.id], residents[candidate.id] + 700,
                    )))
        return results

    def test_candidate_set_is_exact_and_frozen(self) -> None:
        candidates = bridge.candidate_matrix()
        self.assertEqual(
            [bridge.WVM_ID, bridge.FUSED_ID, bridge.TILED_ID],
            [candidate.id for candidate in candidates],
        )
        self.assertEqual([0, 0, 16], [candidate.tile_width for candidate in candidates])

    def test_analysis_selects_distinct_deployment_winners(self) -> None:
        analysis = bridge.analyze(self.results())
        self.assertTrue(analysis["completeProductionMatrix"])
        self.assertTrue(analysis["referenceGate"]["singlePolicyPersistentEnginePassed"])
        decisions = analysis["deploymentDecisions"]
        self.assertEqual(
            bridge.TILED_ID,
            decisions["persistentCompiledEngine"]["selectedCandidateId"],
        )
        self.assertEqual(
            bridge.WVM_ID,
            decisions["matlabOwnedWvmNativeSpectralBoundary"]["selectedCandidateId"],
        )
        summaries = {item["id"]: item for item in analysis["pairwiseSummaries"]}
        self.assertAlmostEqual(
            0.6,
            summaries[bridge.pair_key(bridge.TILED_ID, bridge.WVM_ID)]
            ["geometricTimeRatio"],
        )
        self.assertAlmostEqual(
            0.75,
            summaries[bridge.pair_key(bridge.TILED_ID, bridge.FUSED_ID)]
            ["geometricTimeRatio"],
        )

    def test_command_distinguishes_native_and_fixed_tile(self) -> None:
        wvm, _, tiled = bridge.candidate_matrix()
        wvm_command = bridge.command_for(
            Path("skbench"), wvm, bridge.PROFILES[0],
            3, 21, 129, Path("wvm.json"),
        )
        tiled_command = bridge.command_for(
            Path("skbench"), tiled, bridge.PROFILES[0],
            3, 21, 129, Path("tiled.json"),
        )
        self.assertIn("--boundary-policy wvm-direct", " ".join(wvm_command))
        self.assertNotIn("--streaming-tile-width", " ".join(wvm_command))
        self.assertIn("--streaming-tile-width 16", " ".join(tiled_command))

    def test_memory_estimates_cover_all_candidates_under_machine_limit(self) -> None:
        limit = 64 * 1024**3
        for candidate in bridge.candidate_matrix():
            for profile in bridge.PROFILES:
                self.assertLess(
                    bridge.estimated_explicit_peak_bytes(profile, candidate),
                    limit,
                )


if __name__ == "__main__":
    unittest.main()
