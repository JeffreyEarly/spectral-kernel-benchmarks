import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import run_pointwise_advection_screen as screen  # noqa: E402


def record(profile: str, candidate_id: str, pointwise: float,
           total: float) -> dict:
    return {
        "runId": f"run-{profile}-{candidate_id}",
        "profile": profile,
        "candidateId": candidate_id,
        "pointwiseSeconds": pointwise,
        "totalSeconds": total,
        "pointwiseFractionOfTotal": pointwise / total,
        "effectiveBytes": 100,
        "effectiveGigabytesPerSecond": 1.0,
        "schedulerSeconds": 0.0,
        "algorithmResidentBytes": 1000,
        "scratchBytes": 500,
        "observedProcessHighWaterBytes": 2000,
        "maximumCorrectnessError": 1.0e-15,
        "valid": True,
    }


class PointwiseAdvectionScreenTests(unittest.TestCase):
    def test_command_fixes_issue21_control_and_pointwise_policy(self) -> None:
        candidate = screen.candidate_matrix([12])[-1]
        command = screen.command_for(
            Path("skbench"), Path("fixture.bin"), screen.PROFILES[0],
            candidate, 2, 7, Path("result.json"),
        )
        expected = {
            "--boundary-policy": screen.BOUNDARY_POLICY,
            "--fftw-outer-workers": "12",
            "--vertical-gemm-outer-workers": "16",
            "--pointwise-policy": "spatial-static",
            "--pointwise-workers": "12",
        }
        for option, value in expected.items():
            self.assertEqual(value, command[command.index(option) + 1])

    def test_analysis_selects_uniform_complete_pipeline_winner(self) -> None:
        candidates = screen.candidate_matrix([4, 12])
        timings = {
            "serial-1": (0.20, 1.00),
            "vector-serial-1": (0.20, 1.01),
            "spatial-static-4": (0.08, 0.90),
            "spatial-static-12": (0.07, 0.84),
        }
        records = [
            record(profile, candidate.id, *timings[candidate.id])
            for profile in screen.PROFILES for candidate in candidates
        ]
        analysis = screen.analyze(records, candidates, "a" * 40, True)
        self.assertTrue(analysis["completeMatchedMatrix"])
        self.assertTrue(analysis["allCorrectWithin1e12"])
        self.assertEqual(
            "spatial-static-12",
            analysis["selectedCandidate"]["candidate"]["id"],
        )
        self.assertAlmostEqual(
            0.84, analysis["selectedCandidate"]["geometricTotalToSerial"]
        )
        self.assertFalse(
            analysis["screenGate"]["activatePointwiseFftFusion"]
        )

    def test_result_record_uses_schema_dimension_names(self) -> None:
        candidate = screen.candidate_matrix([12])[-1]
        commit = "a" * 40
        result = {
            "status": "passed",
            "run": {"id": "run", "profile": screen.PROFILES[0]},
            "workload": {"Nx": 256, "Ny": 256, "Nz": 129},
            "environment": {"gitCommit": commit[:12], "gitDirty": True},
            "provenance": {"spectralFluxFixture": {"authoritative": True}},
            "providers": [{
                "id": candidate.provider,
                "timings": [
                    {"stage": screen.POINTWISE_STAGE, "medianSeconds": 0.01},
                    {"stage": screen.TOTAL_STAGE, "medianSeconds": 0.1},
                ],
                "correctness": [{
                    "passed": True, "maximumRelativeError": 1.0e-15,
                }],
                "memory": {},
            }],
        }
        parsed = screen.result_record(
            result, candidate, screen.PROFILES[0], commit, True,
        )
        self.assertTrue(parsed["valid"])
        self.assertEqual(28 * 256 * 256 * 129 * 8, parsed["effectiveBytes"])


if __name__ == "__main__":
    unittest.main()
