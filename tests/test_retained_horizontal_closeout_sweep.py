import sys
import unittest
from pathlib import Path


TOOLS = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(TOOLS))

import run_retained_horizontal_closeout_sweep as sweep  # noqa: E402


def result(candidate: sweep.Candidate, profile: str, forward: float, inverse: float) -> tuple:
    return candidate, {
        "run": {"profile": profile},
        "providers": [{
            "id": candidate.primary_provider,
            "timings": [
                {
                    "scope": "uninstrumented-total",
                    "direction": "forward",
                    "medianSeconds": forward,
                },
                {
                    "scope": "uninstrumented-total",
                    "direction": "inverse",
                    "medianSeconds": inverse,
                },
            ],
        }],
    }


class RetainedHorizontalCloseoutSweepTests(unittest.TestCase):
    def test_screen_contains_controls_and_three_new_candidates(self) -> None:
        selected = sweep.select_candidates(None, "screen")
        self.assertEqual(len(selected), 5)
        self.assertEqual(
            {candidate.id for candidate in selected if candidate.control},
            sweep.CONTROL_IDS,
        )

    def test_reference_uses_only_qualifying_candidates_and_controls(self) -> None:
        analysis = {
            "advancingCandidateIds": ["fftw-plane-major-retained-view"],
        }
        selected = sweep.select_candidates(None, "reference", analysis)
        self.assertEqual(
            {candidate.id for candidate in selected},
            sweep.CONTROL_IDS | {"fftw-plane-major-retained-view"},
        )
        with self.assertRaises(ValueError):
            sweep.select_candidates(
                [
                    "fftw-plane-major-control",
                    "fftw-pruned-control",
                    "fftw-plane-major-fused-retained-split",
                ],
                "reference",
                analysis,
            )

    def test_advancement_rules_are_complete_matrix_and_preregistered(self) -> None:
        candidates = {candidate.id: candidate for candidate in sweep.candidate_matrix()}
        profiles = ["a", "b"]
        results = []
        for profile in profiles:
            results.extend([
                result(candidates["fftw-plane-major-control"], profile, 1.0, 1.0),
                result(candidates["fftw-pruned-control"], profile, 0.9, 0.9),
                result(candidates["fftw-plane-major-retained-view"], profile, 0.88, 0.95),
                result(candidates["fftw-plane-major-fused-retained-split"], profile, 0.98, 0.98),
                result(candidates["fftw-pruned-fused-retained-split"], profile, 1.2, 1.2),
            ])
        analysis = sweep.analyze(results)
        self.assertEqual(
            set(analysis["advancingCandidateIds"]),
            {
                "fftw-plane-major-retained-view",
                "fftw-plane-major-fused-retained-split",
            },
        )
        pruned_split = next(
            item for item in analysis["comparisons"]
            if item["candidate"] == "fftw-pruned-fused-retained-split"
        )
        self.assertFalse(pruned_split["qualifiesForReference"])

    def test_commands_preserve_algorithm_and_representation_identity(self) -> None:
        candidates = {candidate.id: candidate for candidate in sweep.candidate_matrix()}
        pruned = sweep.command_for(
            Path("skbench"), candidates["fftw-pruned-fused-retained-split"],
            "wvm-historical-256-nz65-f3", 2, 9, 129, Path("result.json"),
        )
        self.assertIn("pruned-horizontal", pruned)
        self.assertIn("split", pruned)
        view = sweep.command_for(
            Path("skbench"), candidates["fftw-plane-major-retained-view"],
            "wvm-historical-256-nz65-f3", 2, 9, 129, Path("result.json"),
        )
        self.assertIn("plane-major", view)
        self.assertIn("view", view)


if __name__ == "__main__":
    unittest.main()
