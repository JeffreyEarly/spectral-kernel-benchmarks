import sys
import unittest
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "tools"))

import run_fftw_production_reference as reference  # noqa: E402


class FftwProductionReferenceTests(unittest.TestCase):
    def test_complete_matrix_and_rotation(self) -> None:
        self.assertEqual(len(reference.PROFILES), 10)
        rotated = reference.rotated(list(reference.PROFILES), 3)
        self.assertEqual(set(rotated), set(reference.PROFILES))
        self.assertNotEqual(rotated[0], reference.PROFILES[0])

    def test_command_freezes_production_contract(self) -> None:
        command = reference.command_for(
            Path("skbench"), reference.PROFILES[0], 12, 3, 21, 129,
            Path("result.json"),
        )
        joined = " ".join(str(item) for item in command)
        self.assertIn("--fftw-spectrum-order wvm", joined)
        self.assertIn("--fftw-layout interleaved", joined)
        self.assertIn("--fftw-planning measure", joined)
        self.assertIn("--fftw-alignment unaligned", joined)
        self.assertIn("--fftw-internal-workers 12", joined)
        self.assertIn("--fftw-outer-workers 1", joined)


if __name__ == "__main__":
    unittest.main()
