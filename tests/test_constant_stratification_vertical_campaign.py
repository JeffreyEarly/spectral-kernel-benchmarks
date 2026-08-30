import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import run_constant_stratification_vertical_campaign as campaign  # noqa: E402


def timing(scope: str, stage: str, direction: str, seconds: float) -> dict:
    return {
        "scope": scope,
        "stage": stage,
        "direction": direction,
        "medianSeconds": seconds,
    }


def provider(provider_id: str, total: float, resident: int) -> dict:
    return {
        "id": provider_id,
        "memory": {"algorithmResidentBytes": resident},
        "correctness": [{
            "passed": True,
            "maximumRelativeError": 1.0e-15,
        }],
        "timings": [
            timing("uninstrumented-total", campaign.TOTAL_STAGE, "complete", total),
            timing("primitive", "raw DCT-I one complex channel", "forward", total / 20),
            timing("primitive", "raw DST-I one complex interior channel", "forward", total / 19),
        ],
    }


def result(profile: str, compact_ratio: float = 0.4) -> dict:
    return {
        "run": {"id": f"run-{profile}", "profile": profile},
        "workload": {"Nx": 256, "Ny": 256, "Nz": 129, "H": 100, "Nkl": 35},
        "environment": {"gitDirty": False},
        "providers": [
            provider(campaign.FULL_PROVIDER, 1.0, 1000),
            provider(campaign.COMPACT_PROVIDER, compact_ratio, 350),
        ],
    }


class ConstantStratificationVerticalCampaignTests(unittest.TestCase):
    def test_command_pins_production_type1_policy(self) -> None:
        command = campaign.command_for(
            Path("skbench"), campaign.PROFILES[0], Path("result.json"),
            16, 3, 15,
        )
        self.assertEqual(
            "constant-stratification-vertical",
            command[command.index("--kernel") + 1],
        )
        self.assertEqual("measure", command[command.index("--fftw-planning") + 1])
        self.assertEqual("16", command[command.index("--fftw-internal-workers") + 1])

    def test_analysis_recommends_integrating_large_uniform_win(self) -> None:
        analysis = campaign.analyze(
            [result(profile) for profile in campaign.PROFILES], 16, "a" * 40
        )
        self.assertTrue(analysis["completeProfileMatrix"])
        self.assertTrue(analysis["allCorrectWithin1e12"])
        self.assertAlmostEqual(0.4, analysis["geometricCompactToFullVerticalSchedule"])
        self.assertAlmostEqual(0.35, analysis["geometricCompactToFullExplicitArena"])
        self.assertTrue(analysis["integrationRecommended"])

    def test_analysis_does_not_confuse_component_screen_with_weak_candidate(self) -> None:
        analysis = campaign.analyze(
            [result(profile, 0.95) for profile in campaign.PROFILES],
            16, "b" * 40,
        )
        self.assertFalse(analysis["integrationRecommended"])
        self.assertIn("cannot establish complete", analysis["interpretation"])


if __name__ == "__main__":
    unittest.main()
