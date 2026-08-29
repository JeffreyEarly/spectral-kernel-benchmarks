import sys
import unittest
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "tools"))

import run_spectral_pipeline_deep_vertical_reference as reference  # noqa: E402


class SpectralPipelineDeepVerticalReferenceTests(unittest.TestCase):
    def test_elided_baseline_ratio_is_explicit(self) -> None:
        self.assertIsNone(reference.ratio_or_none(1.0, 0.0))
        self.assertEqual(reference.ratio_or_none(0.5, 2.0), 0.25)

    def test_frozen_candidates_and_profile(self) -> None:
        self.assertEqual(reference.PROFILE, "wvm-large-512-nz513-f4")
        self.assertEqual(
            [candidate.id for candidate in reference.candidate_matrix()],
            [reference.BASELINE_ID, reference.CANDIDATE_ID],
        )
        self.assertEqual(reference.candidate_matrix()[1].tile_width, 16)

    def test_commands_do_not_introduce_size_dispatch(self) -> None:
        for candidate in reference.candidate_matrix():
            command = reference.command_for(
                Path("skbench"), candidate, reference.PROFILE, 3, 21, 129,
                Path("result.json"),
            )
            joined = " ".join(str(item) for item in command)
            self.assertIn("--kernel spectral-pipeline", joined)
            self.assertIn("--vertical-gemm-schedule outer-dynamic", joined)
            if candidate.id == reference.CANDIDATE_ID:
                self.assertIn("--streaming-tile-width 16", joined)


if __name__ == "__main__":
    unittest.main()
