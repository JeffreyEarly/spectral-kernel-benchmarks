import sys
import unittest
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "tools"))

from run_authoritative_spectral_flux_scaleout import (  # noqa: E402
    CANDIDATES,
    graph_capacity,
)


class AuthoritativeSpectralFluxScaleoutTests(unittest.TestCase):
    def capacity(self, *, nz: int, nj: int, segments: int,
                 source_operator: int) -> dict:
        return {
            "workload": {
                "Nx": 512,
                "Ny": 512,
                "Nz": nz,
                "Nkl": 45765,
                "Nj": nj,
                "canonicalGroupSegmentCount": segments,
            },
            "payloadBytes": {"verticalOperators": source_operator},
        }

    def test_deep_vertical_preflight_excludes_only_wvm_direct_on_128_gib(self) -> None:
        result = graph_capacity(
            self.capacity(
                nz=513, nj=341, segments=14431,
                source_operator=50397497568,
            ),
            128 * 1024**3,
        )
        self.assertFalse(result[CANDIDATES[0].id]["feasible"])
        self.assertTrue(result[CANDIDATES[1].id]["feasible"])
        self.assertGreater(
            result[CANDIDATES[0].id]["estimatedSteadyExplicitBytes"],
            result[CANDIDATES[1].id]["estimatedSteadyExplicitBytes"],
        )

    def test_medium_case_fits_both_graphs(self) -> None:
        result = graph_capacity(
            self.capacity(
                nz=257, nj=170, segments=14431,
                source_operator=12586914240,
            ),
            128 * 1024**3,
        )
        self.assertTrue(result[CANDIDATES[0].id]["feasible"])
        self.assertTrue(result[CANDIDATES[1].id]["feasible"])


if __name__ == "__main__":
    unittest.main()
