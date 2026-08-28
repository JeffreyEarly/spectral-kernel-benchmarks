import sys
import unittest
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "tools"))

import run_fftw_native_order_sweep as sweep  # noqa: E402


class FftwNativeOrderSweepTests(unittest.TestCase):
    def test_matrix_covers_order_layout_and_topology(self) -> None:
        candidates = sweep.candidate_matrix(12, 16)
        self.assertEqual(len(candidates), 12)
        identifiers = {candidate.id for candidate in candidates}
        self.assertIn("wvm-interleaved-estimate-internal-performance", identifiers)
        self.assertIn("plane-major-interleaved-measure-outer-total", identifiers)
        self.assertIn("plane-major-split-measure-hybrid-4", identifiers)
        self.assertFalse(any(candidate.layout == "split" and candidate.spectrum_order == "wvm"
                             for candidate in candidates))

    def test_reference_selection_is_explicit(self) -> None:
        candidates = sweep.candidate_matrix(12, 16)
        with self.assertRaises(ValueError):
            sweep.select_candidates(None, candidates, require_explicit=True)
        selected = sweep.select_candidates(
            ["plane-major-interleaved-measure-outer-performance"],
            candidates,
            require_explicit=True,
        )
        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0].outer_workers, 12)

    def test_round_rotation_preserves_every_candidate(self) -> None:
        candidates = sweep.candidate_matrix(12, 16)
        rotated = sweep.rotated(candidates, 5)
        self.assertEqual({candidate.id for candidate in rotated},
                         {candidate.id for candidate in candidates})
        self.assertNotEqual(rotated[0], candidates[0])


if __name__ == "__main__":
    unittest.main()
