import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import run_constant_stratification_flux_campaign as campaign  # noqa: E402


def timing(scope: str, stage: str, direction: str, seconds: float) -> dict:
    return {
        "scope": scope,
        "stage": stage,
        "direction": direction,
        "medianSeconds": seconds,
    }


def provider(provider_id: str, total: float, resident: int) -> dict:
    timings = [
        timing("uninstrumented-total", campaign.TOTAL_STAGE, "complete", total)
    ]
    for scope, stage, direction in campaign.COMPONENT_STAGES.values():
        timings.append(timing(scope, stage, direction, total / 8.0))
    return {
        "id": provider_id,
        "memory": {"algorithmResidentBytes": resident},
        "correctness": [{
            "passed": True,
            "maximumRelativeError": 1.0e-15,
        }],
        "timings": timings,
    }


def result(profile: str, candidate_ratio: float = 0.7) -> dict:
    return {
        "run": {"id": f"run-{profile}", "profile": profile},
        "workload": {"Nx": 256, "Ny": 256, "Nz": 129, "H": 100, "Nkl": 35},
        "environment": {"gitDirty": False},
        "providers": [
            provider(campaign.CONTROL_PROVIDER, 1.0, 1000),
            provider(campaign.CANDIDATE_PROVIDER, candidate_ratio, 800),
        ],
    }


class ConstantStratificationFluxCampaignTests(unittest.TestCase):
    def test_command_freezes_composed_m4_policy(self) -> None:
        command = campaign.command_for(
            Path("skbench"), campaign.PROFILES[0], Path("result.json"),
            16, 12, 8, 2, 7,
        )
        self.assertEqual(
            "constant-stratification-flux",
            command[command.index("--kernel") + 1],
        )
        self.assertEqual("16", command[command.index("--fftw-internal-workers") + 1])
        self.assertEqual("12", command[command.index("--fftw-outer-workers") + 1])
        self.assertEqual("8", command[command.index("--pointwise-workers") + 1])
        self.assertEqual("16", command[command.index("--streaming-tile-width") + 1])

    def test_analysis_recommends_authoritative_validation(self) -> None:
        analysis = campaign.analyze(
            [result(profile) for profile in campaign.PROFILES],
            16, 12, 8, "a" * 40,
        )
        self.assertTrue(analysis["completeProfileMatrix"])
        self.assertTrue(analysis["allCorrectWithin1e12"])
        self.assertAlmostEqual(0.7, analysis["geometricCandidateToControlTotal"])
        self.assertAlmostEqual(0.8, analysis["geometricCandidateToControlResident"])
        self.assertTrue(analysis["authoritativeValidationRecommended"])
        self.assertIn("cannot establish", analysis["interpretation"])

    def test_analysis_rejects_weak_composed_candidate(self) -> None:
        analysis = campaign.analyze(
            [result(profile, 0.95) for profile in campaign.PROFILES],
            16, 12, 8, "b" * 40,
        )
        self.assertFalse(analysis["authoritativeValidationRecommended"])


if __name__ == "__main__":
    unittest.main()
