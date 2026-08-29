import sys
import unittest
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "tools"))

import run_vertical_gemm_reference as reference  # noqa: E402


class VerticalGemmReferenceTests(unittest.TestCase):
    def test_frozen_finalists_and_matrix(self) -> None:
        self.assertEqual(len(reference.PROFILES), 10)
        self.assertEqual(
            [candidate.id for candidate in reference.candidate_matrix()],
            ["outer-dynamic-16", "outer-static-12"],
        )

    def test_command_holds_primitive_boundary(self) -> None:
        candidate = reference.candidate_matrix()[0]
        command = reference.command_for(
            Path("skbench"), candidate, reference.PROFILES[0], 3, 21, 129,
            Path("result.json"),
        )
        joined = " ".join(str(item) for item in command)
        self.assertIn("--kernel vertical-gemm", joined)
        self.assertIn("--vertical-gemm-family k2-grouped", joined)
        self.assertIn("--vertical-gemm-schedule outer-dynamic", joined)
        self.assertIn("--vertical-gemm-outer-workers 16", joined)
        self.assertNotIn("spectral-pipeline", joined)


if __name__ == "__main__":
    unittest.main()
