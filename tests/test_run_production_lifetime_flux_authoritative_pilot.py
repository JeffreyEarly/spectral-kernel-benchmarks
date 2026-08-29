import sys
import unittest
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "tools"))

from run_production_lifetime_flux_authoritative_pilot import (  # noqa: E402
    PROFILE,
    TOTAL_STAGE,
    analyze,
    candidate_matrix,
    command_for,
)


FIXTURE_HASH = "sha256:" + "a" * 64
WVM_COMMIT = "b" * 40


def result_for(candidate, seconds: float, resident_bytes: int) -> dict:
    provider = {
        "id": candidate.primary_provider,
        "timings": [
            {
                "scope": "primitive",
                "stage": "raw inverse vertical MM (15 modal inputs; exact WVM F/G families)",
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
        "provenance": {
            "spectralFluxFixture": {
                "schema": "spectral-flux-fixture-v1",
                "status": "authoritative-wvm-export",
                "authoritative": True,
                "fixtureHash": FIXTURE_HASH,
                "waveVortexModelCommit": WVM_COMMIT,
            }
        },
        "providers": [provider],
    }


class ProductionLifetimeFluxAuthoritativePilotTests(unittest.TestCase):
    def test_commands_require_the_prepared_fixture_for_both_frozen_graphs(self) -> None:
        fixture = Path("prepared-fixture.bin")
        for candidate in candidate_matrix():
            command = command_for(
                Path("skbench"), fixture, candidate, 2, 9,
                Path(f"{candidate.id}.json"),
            )
            self.assertEqual(
                str(fixture), command[command.index("--spectral-flux-fixture") + 1]
            )
            self.assertEqual(candidate.policy,
                             command[command.index("--boundary-policy") + 1])
            self.assertEqual(PROFILE, command[command.index("--profile") + 1])
            self.assertEqual("12", command[command.index("--fftw-outer-workers") + 1])
            self.assertEqual(
                "16", command[command.index("--vertical-gemm-outer-workers") + 1]
            )

    def test_analysis_requires_one_authoritative_fixture_but_does_not_run_gate(self) -> None:
        control, candidate = candidate_matrix()
        analysis = analyze(
            [
                (control, result_for(control, 2.0, 1000)),
                (candidate, result_for(candidate, 1.0, 600)),
            ],
            {
                "fixtureHash": FIXTURE_HASH,
                "waveVortexModelCommit": WVM_COMMIT,
            },
        )
        self.assertTrue(analysis["completePair"])
        self.assertTrue(analysis["allCorrectWithin1e-12"])
        self.assertTrue(analysis["allFixturesAuthoritative"])
        self.assertTrue(analysis["singleFixtureHash"])
        self.assertTrue(analysis["singleWaveVortexModelCommit"])
        self.assertEqual(0.5, analysis["candidateToControl"]["time"])
        self.assertEqual(
            0.6, analysis["candidateToControl"]["algorithmResidentBytes"]
        )
        self.assertFalse(analysis["interpretation"]["eligibleForReference"])
        self.assertFalse(analysis["interpretation"]["adoptionGateEvaluated"])
        self.assertIn("multi-workload", analysis["interpretation"]["referenceBlocker"])


if __name__ == "__main__":
    unittest.main()
