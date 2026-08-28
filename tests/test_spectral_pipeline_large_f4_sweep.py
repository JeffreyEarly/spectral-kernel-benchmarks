import sys
import unittest
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "tools"))

import run_spectral_pipeline_large_f4_sweep as sweep  # noqa: E402


def fake_result(candidate: sweep.Candidate, profile: str, seconds: float,
                resident: int, observed: int) -> dict:
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
                "benchmarkHarnessBytes": 400,
                "estimatedProcessPeakBytes": resident + 400,
                "observedProcessHighWaterBytes": observed,
            },
        }],
    }


class SpectralPipelineLargeF4SweepTests(unittest.TestCase):
    def test_cohort_is_four_field_and_includes_large_horizontal_case(self) -> None:
        self.assertEqual(len(sweep.PROFILES), 4)
        self.assertIn("wvm-current-512-nz257-f4", sweep.PROFILES)
        self.assertIn("wvm-large-1024-nz129-f4", sweep.PROFILES)
        self.assertTrue(all(profile.endswith("f4") for profile in sweep.PROFILES))

    def test_screen_performance_does_not_block_reference_depth(self) -> None:
        baseline, candidate = sweep.candidate_matrix()
        results = []
        for profile in sweep.PROFILES:
            results.append((baseline, 1, fake_result(
                baseline, profile, 1.0, 1000, 1400,
            )))
            results.append((candidate, 1, fake_result(
                candidate, profile, 1.20, 800, 1200,
            )))
        analysis = sweep.analyze(results, "screen")
        self.assertTrue(analysis["completeProductionMatrix"])
        self.assertTrue(analysis["advanceToReference"])
        self.assertTrue(
            analysis["screenGate"]["performanceDoesNotGateReferenceCollection"]
        )

    def test_reference_classifies_tie_with_memory_advantage(self) -> None:
        baseline, candidate = sweep.candidate_matrix()
        results = []
        for round_number in (1, 2, 3):
            for profile in sweep.PROFILES:
                results.append((baseline, round_number, fake_result(
                    baseline, profile, 1.0, 1000, 1500,
                )))
                results.append((candidate, round_number, fake_result(
                    candidate, profile, 1.01, 800, 1250,
                )))
        analysis = sweep.analyze(results, "reference")
        self.assertEqual(
            "tie-with-memory-advantage",
            analysis["referenceGate"]["classification"],
        )
        self.assertAlmostEqual(
            0.8,
            analysis["geometricAlgorithmResidentCandidateToBaseline"],
        )

    def test_reference_selects_one_overall_winner_despite_small_case_regression(self) -> None:
        baseline, candidate = sweep.candidate_matrix()
        ratios = (1.04, 0.98, 0.76, 0.83)
        results = []
        for round_number in (1, 2, 3):
            for profile, ratio in zip(sweep.PROFILES, ratios):
                results.append((baseline, round_number, fake_result(
                    baseline, profile, 1.0, 1000, 1500,
                )))
                results.append((candidate, round_number, fake_result(
                    candidate, profile, ratio, 780, 1450,
                )))
        analysis = sweep.analyze(results, "reference")
        gate = analysis["referenceGate"]
        self.assertEqual(
            "fused-split-overall-winner-with-smallest-case-regression",
            gate["classification"],
        )
        self.assertFalse(gate["regressionPassed"])
        self.assertTrue(gate["boundedSinglePolicyRegressionPassed"])
        self.assertTrue(gate["m4NonhydrostaticAdoptionStatisticsPassed"])
        self.assertFalse(gate["sizeDependentDispatchAllowed"])

    def test_safe_profile_estimates_remain_below_half_of_128_gib(self) -> None:
        limit = 64 * 1024**3
        for candidate in sweep.candidate_matrix():
            for profile in sweep.PROFILES:
                self.assertLess(
                    sweep.estimated_explicit_peak_bytes(profile, candidate.policy),
                    limit,
                )

    def test_command_keeps_the_issue9_algorithm_tuple_fixed(self) -> None:
        candidate = sweep.candidate_matrix()[1]
        command = sweep.command_for(
            Path("skbench"), candidate, sweep.PROFILES[-1],
            1, 5, 129, Path("result.json"),
        )
        joined = " ".join(str(value) for value in command)
        self.assertIn("--kernel spectral-pipeline", joined)
        self.assertIn("--profile wvm-large-1024-nz129-f4", joined)
        self.assertIn("--fftw-outer-workers 12", joined)
        self.assertIn("--vertical-gemm-outer-workers 16", joined)


if __name__ == "__main__":
    unittest.main()
