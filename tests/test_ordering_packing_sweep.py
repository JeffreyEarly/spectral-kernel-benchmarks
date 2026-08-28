import sys
import unittest
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "tools"))

import run_ordering_packing_sweep as sweep  # noqa: E402


class OrderingPackingSweepTests(unittest.TestCase):
    def test_default_profiles_are_bounded_256_cases(self) -> None:
        self.assertEqual(
            sweep.selected_profiles(None),
            [
                "wvm-historical-256-nz65-f3",
                "wvm-current-256-nz129-f1",
                "wvm-current-256-nz129-f3",
                "wvm-current-256-nz129-f4",
            ],
        )

    def test_explicit_peak_estimate_increases_with_fields_and_depth(self) -> None:
        historical = sweep.estimated_explicit_peak_bytes("wvm-historical-256-nz65-f3")
        current_one = sweep.estimated_explicit_peak_bytes("wvm-current-256-nz129-f1")
        current_four = sweep.estimated_explicit_peak_bytes("wvm-current-256-nz129-f4")
        self.assertLess(historical, current_four)
        self.assertLess(current_one, current_four)
        self.assertLess(current_four, 8 * 1024**3)


if __name__ == "__main__":
    unittest.main()
