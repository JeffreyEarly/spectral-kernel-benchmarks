import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import run_vertically_batched_advection_screen as screen


def result(candidate: screen.Candidate, total: float, resident: int) -> dict:
    def timing(scope: str, stage: str, seconds: float) -> dict:
        return {
            "scope": scope,
            "stage": stage,
            "medianSeconds": seconds,
        }

    return {
        "providers": [{
            "id": candidate.provider_id,
            "correctness": [{
                "passed": True,
                "maximumRelativeError": 1.0e-15,
            }],
            "memory": {
                "algorithmResidentBytes": resident,
                "scratchBytes": 10,
                "benchmarkHarnessBytes": 20,
                "estimatedProcessPeakBytes": resident + 20,
                "observedProcessHighWaterBytes": resident + 100,
            },
            "timings": [
                timing("uninstrumented-total", screen.TOTAL_STAGE, total),
                timing("primitive", "raw inverse vertical GEMM (15 fields)", 1.0),
                timing(
                    "component",
                    "vertically batched horizontal advection including level movement",
                    2.0,
                ),
                timing(
                    "adapter-component",
                    "all-level split/field-major packing and projected-output scatter",
                    0.25,
                ),
                timing("primitive", "raw forward vertical GEMM (4 fields)", 0.5),
            ],
        }],
    }


class VerticallyBatchedAdvectionScreenTests(unittest.TestCase):
    def test_command_fixes_first_composition_policy(self) -> None:
        candidate = screen.candidates()[1]
        command = screen.command_for(
            Path("skbench"), Path("result.json"), candidate, 1, 3, 129,
        )
        self.assertIn("vertically-batched-advection", command)
        self.assertIn("wvm-current-256-nz129-f4", command)
        self.assertEqual("12", command[command.index("--vertical-gemm-outer-workers") + 1])
        self.assertEqual(
            "fftwpp-parallel",
            command[command.index("--convolution-candidate") + 1],
        )

    def test_analysis_separates_continuation_from_adoption(self) -> None:
        explicit, fftwpp = screen.candidates()
        analysis = screen.analyze([
            (explicit, result(explicit, 10.0, 1000)),
            (fftwpp, result(fftwpp, 9.5, 780)),
        ])
        self.assertTrue(analysis["complete"])
        self.assertAlmostEqual(0.95, analysis["fftwppToExplicitTotal"])
        self.assertAlmostEqual(
            0.78, analysis["fftwppToExplicitAlgorithmResident"]
        )
        self.assertTrue(analysis["referenceCampaignRecommended"])
        self.assertIn("continuation gate", analysis["screenRule"])

    def test_memory_estimate_is_positive_and_candidate_independent_conservative(self) -> None:
        estimates = [
            screen.estimated_process_peak_bytes(256, 129, candidate)
            for candidate in screen.candidates()
        ]
        self.assertTrue(all(value > 1_000_000_000 for value in estimates))
        self.assertEqual(estimates[0], estimates[1])


if __name__ == "__main__":
    unittest.main()
