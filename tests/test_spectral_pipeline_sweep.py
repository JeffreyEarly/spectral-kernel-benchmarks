import sys
import unittest
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "tools"))

import run_spectral_pipeline_sweep as sweep  # noqa: E402


def fake_result(candidate: sweep.Candidate, profile: str, seconds: float) -> dict:
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
        }],
    }


class SpectralPipelineSweepTests(unittest.TestCase):
    def test_candidate_matrix_is_bounded_to_control_and_selected_candidate(self) -> None:
        candidates = sweep.candidate_matrix()
        self.assertEqual(len(candidates), 2)
        self.assertEqual(
            {candidate.id for candidate in candidates},
            {sweep.BASELINE_ID, sweep.CANDIDATE_ID},
        )

    def test_screen_gate_advances_material_improvement(self) -> None:
        baseline, candidate = sweep.candidate_matrix()
        results = []
        for profile in sweep.REFERENCE_PROFILES:
            results.append((baseline, 1, fake_result(baseline, profile, 1.0)))
            results.append((candidate, 1, fake_result(candidate, profile, 0.85)))
        analysis = sweep.analyze(results, "screen")
        self.assertTrue(analysis["completeProductionMatrix"])
        self.assertTrue(analysis["advanceToReference"])
        self.assertAlmostEqual(analysis["geometricCandidateToBaseline"], 0.85)

    def test_screen_gate_rejects_one_large_regression(self) -> None:
        baseline, candidate = sweep.candidate_matrix()
        results = []
        for index, profile in enumerate(sweep.REFERENCE_PROFILES):
            results.append((baseline, 1, fake_result(baseline, profile, 1.0)))
            results.append((candidate, 1, fake_result(
                candidate, profile, 1.11 if index == 0 else 0.75,
            )))
        analysis = sweep.analyze(results, "screen")
        self.assertFalse(analysis["advanceToReference"])
        self.assertFalse(analysis["screenGate"]["regressionPassed"])

    def test_reference_gate_uses_rotated_round_ratios_and_confidence(self) -> None:
        baseline, candidate = sweep.candidate_matrix()
        results = []
        for round_number in (1, 2, 3):
            for profile in sweep.REFERENCE_PROFILES:
                results.append((baseline, round_number, fake_result(
                    baseline, profile, 1.0 + 0.01 * round_number,
                )))
                results.append((candidate, round_number, fake_result(
                    candidate, profile, 0.82 + 0.005 * round_number,
                )))
        analysis = sweep.analyze(results, "reference")
        gate = analysis["referenceGate"]
        self.assertTrue(gate["improvementPassed"])
        self.assertTrue(gate["regressionPassed"])
        self.assertTrue(gate["confidenceExcludesTie"])
        self.assertTrue(gate["m4AdoptionStatisticsPassed"])

    def test_command_fixes_issue13_survivor_controls(self) -> None:
        candidate = sweep.candidate_matrix()[1]
        command = sweep.command_for(
            Path("skbench"), candidate, sweep.REFERENCE_PROFILES[0],
            2, 9, 129, Path("result.json"),
        )
        joined = " ".join(str(value) for value in command)
        self.assertIn("--kernel spectral-pipeline", joined)
        self.assertIn("--fftw-internal-workers 1", joined)
        self.assertIn("--fftw-outer-workers 12", joined)
        self.assertIn("--vertical-gemm-schedule outer-dynamic", joined)
        self.assertIn("--vertical-gemm-outer-workers 16", joined)


if __name__ == "__main__":
    unittest.main()
