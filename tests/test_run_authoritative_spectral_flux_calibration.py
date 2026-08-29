import sys
import unittest
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "tools"))

from run_authoritative_spectral_flux_calibration import (  # noqa: E402
    PROFILES,
    SAMPLES,
    WARMUPS,
    analyze,
    command_for,
)
from run_cross_mac_spectral_reference import topology_matrix  # noqa: E402
from run_production_lifetime_flux_authoritative_pilot import (  # noqa: E402
    candidate_matrix,
)


WVM_COMMIT = "a" * 40


def result_for(candidate, topology, profile: str, seconds: float) -> dict:
    return {
        "status": "passed",
        "run": {
            "id": f"run-{candidate.id}-{topology.id}-{profile}",
            "profile": profile,
        },
        "provenance": {
            "spectralFluxFixture": {
                "schema": "spectral-flux-fixture-v1",
                "status": "authoritative-wvm-export",
                "authoritative": True,
                "fixtureHash": "sha256:" + ("b" if profile == PROFILES[0] else "c") * 64,
                "waveVortexModelCommit": WVM_COMMIT,
            }
        },
        "providers": [
            {
                "id": candidate.primary_provider,
                "schedulingId": (
                    f"horizontal-outer-{topology.horizontal_workers};"
                    f"vertical-{topology.vertical_schedule}-"
                    f"{topology.vertical_workers}-per-operator-family"
                ),
                "timings": [
                    {
                        "scope": "uninstrumented-total",
                        "stage": (
                            "authoritative production-lifetime streamed four-target "
                            "spectral-flux composition"
                        ),
                        "direction": "forward",
                        "medianSeconds": seconds,
                        "samplesSeconds": [seconds] * SAMPLES,
                    }
                ],
                "correctness": [
                    {
                        "passed": True,
                        "maximumRelativeError": 2.0e-15,
                        "relativeL2Error": 1.0e-15,
                    }
                ],
            }
        ],
    }


class AuthoritativeSpectralFluxCalibrationTests(unittest.TestCase):
    def test_command_applies_the_topology_to_both_schedulers(self) -> None:
        candidate = candidate_matrix()[1]
        topology = topology_matrix(12, 16)[2]
        command = command_for(
            Path("skbench"), Path("fixture.bin"), PROFILES[0],
            candidate, topology, Path("result.json"),
        )
        self.assertEqual(
            str(topology.horizontal_workers),
            command[command.index("--fftw-outer-workers") + 1],
        )
        self.assertEqual(
            topology.vertical_schedule,
            command[command.index("--vertical-gemm-schedule") + 1],
        )
        self.assertEqual(
            str(topology.vertical_workers),
            command[command.index("--vertical-gemm-outer-workers") + 1],
        )
        self.assertEqual(str(WARMUPS), command[command.index("--warmups") + 1])
        self.assertEqual(str(SAMPLES), command[command.index("--samples") + 1])

    def test_analysis_prefers_standard_topology_only_within_two_percent(self) -> None:
        topologies = topology_matrix(12, 16)
        preferred = next(
            topology for topology in topologies
            if topology.horizontal_worker_class == "performance"
            and topology.vertical_schedule == "outer-dynamic"
        )
        alternative = next(topology for topology in topologies if topology != preferred)
        results = []
        control, compact = candidate_matrix()
        for candidate in (control, compact):
            for topology in topologies:
                for profile in PROFILES:
                    seconds = 1.2
                    if topology == alternative:
                        seconds = 1.0
                    if topology == preferred:
                        seconds = 1.01 if candidate == control else 1.03
                    results.append((
                        candidate,
                        topology,
                        result_for(candidate, topology, profile, seconds),
                    ))
        analysis = analyze(
            results,
            topologies,
            {"hostname": "test", "performanceCores": 12, "totalPhysicalCores": 16},
            "d" * 40,
        )
        self.assertTrue(analysis["topologiesFrozenForReference"])
        self.assertTrue(analysis["singleWvmCommitAcrossCalibrationFixtures"])
        self.assertEqual(
            preferred.id,
            analysis["selections"][control.id]["selectedTopology"]["id"],
        )
        self.assertEqual(
            alternative.id,
            analysis["selections"][compact.id]["selectedTopology"]["id"],
        )
        self.assertFalse(analysis["calibrationContributesToReferenceInference"])
        self.assertFalse(analysis["adoptionGateEvaluated"])
        self.assertFalse(analysis["sizeDependentDispatchAllowed"])


if __name__ == "__main__":
    unittest.main()
