import sys
import unittest
from pathlib import Path


TOOLS = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(TOOLS))

import run_retained_horizontal_sweep as sweep  # noqa: E402


def provider(provider_id: str, forward: float, inverse: float) -> dict:
    return {
        "id": provider_id,
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
    }


class RetainedHorizontalSweepTests(unittest.TestCase):
    def test_reference_defaults_exclude_vdsp_guardrails(self) -> None:
        selected = sweep.select_candidates(None, "reference")
        self.assertEqual(len(selected), 4)
        self.assertTrue(all(candidate.kind != "vdsp-guard" for candidate in selected))

    def test_guardrail_requires_both_directions(self) -> None:
        profile = "wvm-historical-256-nz65-f3"
        fftw_candidate = sweep.candidate_matrix()[0]
        vdsp_candidate = next(
            candidate for candidate in sweep.candidate_matrix()
            if candidate.id == "vdsp-native-guard-256-w12"
        )
        fftw_result = {
            "run": {"profile": profile},
            "providers": [provider("fftw", 1.0, 1.0)],
        }
        vdsp_result = {
            "run": {"profile": profile},
            "providers": [
                provider("fftw", 1.0, 1.0),
                provider("accelerate-vdsp-native-retained", 1.2, 1.3),
            ],
        }
        analysis = sweep.analyze([
            (fftw_candidate, fftw_result),
            (vdsp_candidate, vdsp_result),
        ])
        self.assertFalse(analysis["vdspExpansionTriggered"])
        self.assertFalse(
            analysis["vdspGuardrails"][0]["qualifiesForExpansion"]
        )
        self.assertEqual(
            analysis["vdspGuardrails"][0]["vdspToBestFftw"],
            {"forward": 1.2, "inverse": 1.3},
        )


if __name__ == "__main__":
    unittest.main()
