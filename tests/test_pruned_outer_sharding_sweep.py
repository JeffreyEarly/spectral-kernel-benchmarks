import sys
import unittest
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "tools"))

import run_pruned_outer_sharding_sweep as sweep  # noqa: E402


class PrunedOuterShardingSweepTests(unittest.TestCase):
    def test_default_profiles_preserve_the_initial_cohort(self) -> None:
        self.assertEqual(
            sweep.selected_profiles(None),
            list(sweep.DEFAULT_PROFILES),
        )

    def test_exact_outer_worker_counts_are_preserved(self) -> None:
        self.assertEqual(sweep.outer_worker_counts("1,4,8"), [1, 4, 8])
        with self.assertRaises(ValueError):
            sweep.outer_worker_counts("0")

    def test_peak_estimate_preserves_bounded_memory_order(self) -> None:
        medium = sweep.estimated_explicit_peak_bytes("wvm-historical-256-nz65-f3")
        large = sweep.estimated_explicit_peak_bytes("wvm-historical-512-nz129-f4")
        self.assertGreater(large, medium)
        self.assertLess(large, 32 * 1024**3)


if __name__ == "__main__":
    unittest.main()
