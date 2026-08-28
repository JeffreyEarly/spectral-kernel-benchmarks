import sys
import unittest
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "tools"))

import run_vertical_gemm_sweep as sweep  # noqa: E402


class VerticalGemmSweepTests(unittest.TestCase):
    def test_named_profiles_and_exact_topologies(self) -> None:
        profiles = sweep.selected_profiles(
            ["wvm-current-256-nz129-f1", "wvm-current-512-nz257-f4"]
        )
        self.assertEqual(
            profiles,
            ["wvm-current-256-nz129-f1", "wvm-current-512-nz257-f4"],
        )
        self.assertEqual(
            sweep.exact_topologies("serial,outer-static:12,outer-dynamic:16"),
            [("serial", 1), ("outer-static", 12), ("outer-dynamic", 16)],
        )
        with self.assertRaises(ValueError):
            sweep.exact_topologies("outer-static")

    def test_peak_estimate_orders_bounded_workloads(self) -> None:
        smallest = sweep.estimated_k2_explicit_peak_bytes("wvm-historical-256-nz65-f3")
        largest = sweep.estimated_k2_explicit_peak_bytes("wvm-current-512-nz257-f4")
        self.assertGreater(largest, smallest)
        self.assertLess(largest, 32 * 1024**3)
        self.assertEqual(sweep.gibibytes(largest), "25.32 GiB")

    def test_git_source_state_is_explicit(self) -> None:
        commit, dirty = sweep.git_source_state(REPOSITORY_ROOT)
        self.assertEqual(len(commit), 40)
        self.assertIsInstance(dirty, bool)


if __name__ == "__main__":
    unittest.main()
