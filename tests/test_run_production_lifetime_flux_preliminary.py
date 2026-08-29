import sys
import unittest
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "tools"))

from run_production_lifetime_flux_preliminary import (  # noqa: E402
    PROFILE,
    TOTAL_STAGE,
    analyze,
    candidate_matrix,
    command_for,
)


def result_for(candidate, seconds: float, resident_bytes: int) -> dict:
    provider = {
        "id": candidate.primary_provider,
        "timings": [
            {
                "scope": "primitive",
                "stage": "raw inverse vertical MM (15 modal inputs)",
                "direction": "inverse",
                "medianSeconds": 0.25 * seconds,
            },
            {
                "scope": "uninstrumented-total",
                "stage": TOTAL_STAGE,
                "direction": "forward",
                "medianSeconds": seconds,
            },
        ],
        "correctness": [
            {
                "passed": True,
                "maximumRelativeError": 2.0e-15,
                "relativeL2Error": 1.0e-15,
            }
        ],
        "memory": {
            "algorithmResidentBytes": resident_bytes,
            "scratchBytes": resident_bytes // 4,
            "estimatedProcessPeakBytes": resident_bytes * 2,
            "observedProcessHighWaterBytes": resident_bytes * 3,
        },
    }
    return {
        "run": {"profile": PROFILE},
        "provenance": {
            "spectralFluxFixture": {
                "status": "provider-independent-synthetic-development",
                "authoritative": False,
                "waveVortexModelCommit": "",
            }
        },
        "providers": [provider],
    }


class ProductionLifetimeFluxPreliminaryTests(unittest.TestCase):
    def test_commands_freeze_the_issue_19_pair(self) -> None:
        candidates = candidate_matrix()
        command = command_for(
            Path("skbench"), candidates[1], 2, 9, 129, Path("result.json")
        )
        self.assertIn("production-lifetime-flux", command)
        self.assertIn("streaming-pruned-compact-split", command)
        self.assertEqual("16", command[command.index("--streaming-tile-width") + 1])
        self.assertEqual("12", command[command.index("--fftw-outer-workers") + 1])
        self.assertEqual(
            "16", command[command.index("--vertical-gemm-outer-workers") + 1]
        )
        self.assertEqual(PROFILE, command[command.index("--profile") + 1])

    def test_analysis_is_descriptive_and_never_evaluates_the_gate(self) -> None:
        control, candidate = candidate_matrix()
        analysis = analyze(
            [
                (control, result_for(control, 2.0, 1000)),
                (candidate, result_for(candidate, 1.0, 600)),
            ]
        )
        self.assertTrue(analysis["completePair"])
        self.assertTrue(analysis["allCorrectWithin1e-12"])
        self.assertTrue(analysis["allFixturesSyntheticDevelopment"])
        self.assertEqual(0.5, analysis["candidateToControl"]["time"])
        self.assertEqual(
            0.6, analysis["candidateToControl"]["algorithmResidentBytes"]
        )
        self.assertFalse(analysis["interpretation"]["eligibleForReference"])
        self.assertFalse(analysis["interpretation"]["adoptionGateEvaluated"])
        self.assertIn(
            "Authoritative versioned WVM fixtures",
            analysis["interpretation"]["referenceBlocker"],
        )


if __name__ == "__main__":
    unittest.main()
