import sys
import unittest
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "tools"))

import run_streaming_pruned_reference as reference  # noqa: E402
import run_streaming_pruned_locality_sweep as locality  # noqa: E402


def fake_result(
    candidate: locality.LocalityCandidate,
    profile: str,
    seconds: float,
    resident: int,
    observed: int,
) -> dict:
    horizontal_stage = (
        "full FFT fused compact split horizontal operator"
        if candidate.id == locality.BASELINE_ID
        else "streaming pruned compact split horizontal operator"
    )
    timings = [
        ("retained-operator-total", horizontal_stage, "forward", 0.2 * seconds),
        ("retained-operator-total", horizontal_stage, "inverse", 0.2 * seconds),
        ("primitive", "raw vertical MM", "forward", 0.15 * seconds),
        ("component", "mode-keyed modal work", "modal", 0.1 * seconds),
        ("primitive", "raw vertical MM", "inverse", 0.15 * seconds),
        (
            "uninstrumented-total", "synthetic antialiased spectral pipeline",
            "round-trip", seconds,
        ),
    ]
    return {
        "status": "passed",
        "run": {"profile": profile},
        "providers": [{
            "id": candidate.primary_provider,
            "timings": [
                {
                    "scope": scope,
                    "stage": stage,
                    "direction": direction,
                    "medianSeconds": value,
                }
                for scope, stage, direction, value in timings
            ],
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
        }],
    }


class StreamingPrunedReferenceTests(unittest.TestCase):
    def results(self, candidate_ratio: float = 0.75) -> list:
        baseline, candidate = reference.reference_candidates()
        results = []
        for round_number in range(1, reference.REFERENCE_ROUNDS + 1):
            for profile in locality.PROFILES:
                results.append((baseline, round_number, fake_result(
                    baseline, profile, 1.0, 1000, 1500,
                )))
                results.append((candidate, round_number, fake_result(
                    candidate, profile, candidate_ratio, 850, 1300,
                )))
        return results

    def test_candidate_set_freezes_one_uniform_tile(self) -> None:
        candidates = reference.reference_candidates()
        self.assertEqual(
            [locality.BASELINE_ID, reference.TILED_ID],
            [candidate.id for candidate in candidates],
        )
        self.assertEqual([0, 16], [candidate.tile_width for candidate in candidates])

    def test_reference_gate_uses_time_confidence_memory_and_correctness(self) -> None:
        analysis = reference.analyze(self.results())
        gate = analysis["referenceGate"]
        self.assertTrue(analysis["completeProductionMatrix"])
        self.assertTrue(gate["improvementPassed"])
        self.assertTrue(gate["regressionPassed"])
        self.assertTrue(gate["confidenceExcludesTie"])
        self.assertTrue(gate["memoryReductionPassed"])
        self.assertTrue(gate["singleUniformPolicyPassed"])
        self.assertFalse(gate["sizeDependentDispatchAllowed"])
        self.assertEqual(3, len(analysis["profiles"]))
        for profile in analysis["profiles"]:
            self.assertEqual(3, len(profile["roundRatios"]))
            self.assertAlmostEqual(
                0.75, profile["components"]["uninstrumentedTotal"]["candidateToBaseline"],
            )

    def test_memory_gate_is_independent_of_timing_win(self) -> None:
        baseline, candidate = reference.reference_candidates()
        results = []
        for round_number in range(1, reference.REFERENCE_ROUNDS + 1):
            for profile in locality.PROFILES:
                results.append((baseline, round_number, fake_result(
                    baseline, profile, 1.0, 1000, 1500,
                )))
                results.append((candidate, round_number, fake_result(
                    candidate, profile, 0.75, 920, 1400,
                )))
        gate = reference.analyze(results)["referenceGate"]
        self.assertTrue(gate["improvementPassed"])
        self.assertFalse(gate["memoryReductionPassed"])
        self.assertFalse(gate["singleUniformPolicyPassed"])

    def test_command_fixes_reference_depth_and_tile_width(self) -> None:
        candidate = reference.reference_candidates()[1]
        command = locality.command_for(
            Path("skbench"), candidate, locality.PROFILES[-1],
            3, 21, 129, Path("result.json"),
        )
        joined = " ".join(str(value) for value in command)
        self.assertIn("--streaming-tile-width 16", joined)
        self.assertIn("--warmups 3", joined)
        self.assertIn("--samples 21", joined)
        self.assertIn("--fftw-outer-workers 12", joined)
        self.assertIn("--vertical-gemm-outer-workers 16", joined)


if __name__ == "__main__":
    unittest.main()
