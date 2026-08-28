import sys
import unittest
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "tools"))

import run_spectral_boundary_sweep as sweep  # noqa: E402


def fake_result(candidate: sweep.Candidate, forward: float, inverse: float) -> dict:
    return {
        "run": {"profile": "wvm-current-256-nz129-f1"},
        "providers": [{
            "id": candidate.primary_provider,
            "timings": [
                {
                    "scope": "uninstrumented-total",
                    "stage": "composed horizontal-vertical boundary",
                    "direction": "forward",
                    "medianSeconds": forward,
                },
                {
                    "scope": "uninstrumented-total",
                    "stage": "composed horizontal-vertical boundary",
                    "direction": "inverse",
                    "medianSeconds": inverse,
                },
            ],
        }],
    }


class SpectralBoundarySweepTests(unittest.TestCase):
    def test_candidate_matrix_crosses_five_policies_and_two_schedulers(self) -> None:
        candidates = sweep.candidate_matrix()
        self.assertEqual(len(candidates), 10)
        self.assertEqual(
            {candidate.policy for candidate in candidates},
            {
                "wvm-direct",
                "wvm-packed-split",
                "pruned-compact-interleaved",
                "plane-major-fused-split",
                "plane-major-view",
            },
        )
        self.assertEqual(
            {candidate.id for candidate in candidates if candidate.control},
            sweep.CONTROL_IDS,
        )

    def test_screen_analysis_applies_competitive_and_bridge_rules(self) -> None:
        results = []
        for candidate in sweep.candidate_matrix():
            seconds = 1.20
            if candidate.id == "pruned-compact-interleaved--outer-dynamic-16":
                seconds = 0.90
            elif candidate.id == "plane-major-fused-split--outer-static-12":
                seconds = 0.95
            elif candidate.control:
                seconds = 1.00
            results.append((candidate, fake_result(candidate, seconds, seconds)))

        analysis = sweep.analyze(results)
        self.assertIn(
            "pruned-compact-interleaved--outer-dynamic-16",
            analysis["advancingCandidateIds"],
        )
        self.assertIn(
            "plane-major-fused-split--outer-static-12",
            analysis["advancingCandidateIds"],
        )
        self.assertTrue(
            sweep.CONTROL_IDS.issubset(set(analysis["referenceCandidateIds"]))
        )
        selected = sweep.select_candidates(None, "reference", analysis)
        self.assertEqual(
            {candidate.id for candidate in selected},
            set(analysis["referenceCandidateIds"]),
        )
        reference_analysis = sweep.analyze(results, phase="reference")
        self.assertEqual(
            reference_analysis["issue9CandidateIds"],
            ["pruned-compact-interleaved--outer-dynamic-16"],
        )
        paired = next(
            item for item in reference_analysis["issue9PairedComparisons"]
            if item["candidate"] == "pruned-compact-interleaved--outer-dynamic-16"
        )
        self.assertEqual(paired["profileWins"], 1)
        self.assertEqual(paired["maximumProfileRatioToPairedBest"], 1.0)

    def test_memory_estimate_is_policy_specific_and_bounded(self) -> None:
        profile = "wvm-current-256-nz129-f1"
        packed = sweep.estimated_explicit_peak_bytes(profile, "wvm-packed-split")
        direct = sweep.estimated_explicit_peak_bytes(profile, "wvm-direct")
        self.assertNotEqual(packed, direct)
        self.assertGreater(packed, 0)
        self.assertLess(max(packed, direct), 16 * 1024**3)

    def test_command_fixes_horizontal_and_vertical_controls(self) -> None:
        candidate = next(
            item for item in sweep.candidate_matrix()
            if item.id == "plane-major-view--outer-static-12"
        )
        command = sweep.command_for(
            Path("skbench"), candidate, "wvm-current-256-nz129-f1",
            2, 9, 129, Path("result.json"),
        )
        joined = " ".join(str(value) for value in command)
        self.assertIn("--kernel spectral-boundary", joined)
        self.assertIn("--fftw-internal-workers 1", joined)
        self.assertIn("--fftw-outer-workers 12", joined)
        self.assertIn("--vertical-gemm-schedule outer-static", joined)
        self.assertIn("--vertical-gemm-outer-workers 12", joined)


if __name__ == "__main__":
    unittest.main()
