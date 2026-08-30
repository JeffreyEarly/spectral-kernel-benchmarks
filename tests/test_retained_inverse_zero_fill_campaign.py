import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import run_retained_inverse_zero_fill_campaign as campaign  # noqa: E402


def record(profile: str, candidate_id: str, total: float,
           inverse: float, resident: int = 1000) -> dict:
    return {
        "runId": f"run-{profile}-{candidate_id}",
        "profile": profile,
        "candidateId": candidate_id,
        "totalSeconds": total,
        "inverseBoundarySeconds": inverse,
        "inversePreparationSeconds": inverse / 4.0,
        "tileLoadSeconds": inverse / 8.0,
        "clearSeconds": 0.0,
        "clearState": "elided",
        "scatterSeconds": inverse / 8.0,
        "bytes": {},
        "memory": {
            "algorithmResidentBytes": resident,
            "scratchBytes": resident // 2,
            "estimatedProcessPeakBytes": resident,
            "observedProcessHighWaterBytes": resident,
        },
        "maximumCorrectnessError": 1.0e-15,
        "allocationLedgerValid": True,
        "sourceMetadataMatches": True,
        "valid": True,
    }


class RetainedInverseZeroFillCampaignTests(unittest.TestCase):
    def test_command_freezes_pipeline_and_changes_inverse_policy(self) -> None:
        candidate = campaign.candidate_matrix()[2]
        command = campaign.command_for(
            Path("skbench"), Path("fixture.bin"), campaign.PROFILES[0],
            candidate, 2, 7, Path("result.json"),
        )
        expected = {
            "--boundary-policy": campaign.BOUNDARY_POLICY,
            "--fftw-outer-workers": "12",
            "--streaming-tile-width": "16",
            "--streaming-inverse-policy": "compact-preserved",
            "--vertical-gemm-outer-workers": "16",
            "--pointwise-policy": "spatial-static",
            "--pointwise-workers": "8",
        }
        for option, value in expected.items():
            self.assertEqual(value, command[command.index(option) + 1])

    def test_analysis_requires_inverse_and_total_screen_wins(self) -> None:
        candidates = campaign.candidate_matrix()
        records = []
        for profile in campaign.PROFILES:
            for candidate in candidates:
                if candidate.id == "full-zero-control":
                    records.append(record(profile, candidate.id, 1.0, 0.6))
                elif candidate.id == "compact-preserved-input":
                    records.append(record(profile, candidate.id, 0.94, 0.50))
                else:
                    records.append(record(profile, candidate.id, 0.96, 0.61))
        analysis = campaign.analyze(records, candidates, "a" * 40)
        self.assertTrue(analysis["completeMatchedMatrix"])
        self.assertEqual(
            ["compact-preserved-input"], analysis["referenceCandidateIds"]
        )
        selected = next(
            item for item in analysis["candidateSummaries"]
            if item["candidate"]["id"] == "compact-preserved-input"
        )
        self.assertAlmostEqual(0.94, selected["geometricTotalToControl"])
        self.assertAlmostEqual(
            0.50 / 0.60, selected["geometricInverseBoundaryToControl"]
        )

    def test_analysis_rejects_total_only_noise_without_inverse_win(self) -> None:
        candidates = campaign.candidate_matrix()
        records = []
        for profile in campaign.PROFILES:
            for candidate in candidates:
                if candidate.id == "full-zero-control":
                    records.append(record(profile, candidate.id, 1.0, 0.6))
                else:
                    records.append(record(profile, candidate.id, 0.94, 0.61))
        analysis = campaign.analyze(records, candidates, "b" * 40)
        self.assertEqual([], analysis["referenceCandidateIds"])
        self.assertIn("retain the frozen full-zero policy", analysis["disposition"])


if __name__ == "__main__":
    unittest.main()
