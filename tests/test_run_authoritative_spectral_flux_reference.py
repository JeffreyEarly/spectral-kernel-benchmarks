import sys
import unittest
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "tools"))

from run_authoritative_spectral_flux_reference import (  # noqa: E402
    CANDIDATE_ID,
    CONTROL_ID,
    REFERENCE_SAMPLES,
    REFERENCE_WARMUPS,
    TIMING_PROFILES,
    analyze,
    capacity_and_fixture_evidence,
    command_for,
    conditional_round_decision,
    result_record,
)
from run_cross_mac_spectral_reference import ScheduleTopology  # noqa: E402
from run_production_lifetime_flux_authoritative_pilot import (  # noqa: E402
    candidate_matrix,
)


BENCHMARK_COMMIT = "a" * 40
WVM_COMMIT = "b" * 40
TOPOLOGY = ScheduleTopology(
    id="horizontal-performance-12--vertical-dynamic-total-16",
    horizontal_workers=12,
    vertical_schedule="outer-dynamic",
    vertical_workers=16,
    horizontal_worker_class="performance",
    vertical_worker_class="total",
)


def fake_result(candidate, profile: str, seconds: float, samples: int = 21) -> dict:
    return {
        "status": "passed",
        "run": {
            "id": f"run-{candidate.id}-{profile}",
            "profile": profile,
            "warmups": 3,
            "samples": samples,
        },
        "environment": {
            "gitCommit": BENCHMARK_COMMIT[:12],
            "gitDirty": False,
        },
        "provenance": {
            "spectralFluxFixture": {
                "schema": "spectral-flux-fixture-v1",
                "status": "authoritative-wvm-export",
                "authoritative": True,
                "fixtureHash": "sha256:" + "c" * 64,
                "waveVortexModelCommit": WVM_COMMIT,
            }
        },
        "providers": [{
            "id": candidate.primary_provider,
            "schedulingId": (
                "horizontal-outer-12;vertical-outer-dynamic-16-per-operator-family"
            ),
            "executionContract": {
                "forward": {
                    "nativePlacement": "out-of-place",
                    "adapterPlacement": "out-of-place",
                    "destroysNativeInput": False,
                    "adapterPreservesCallerInput": True,
                    "requiresPreservationCopyForRepeatedExecution": False,
                }
            },
            "componentLedger": [{
                "stage": "steady-state allocation",
                "state": "elided",
                "detail": "all providers and reusable buffers are persistent",
            }],
            "setup": {"totalSeconds": 0.2},
            "planning": {"seconds": 0.1},
            "memory": {
                "algorithmResidentBytes": 100,
                "scratchBytes": 20,
                "estimatedProcessPeakBytes": 140,
                "observedProcessHighWaterBytes": 160,
            },
            "timings": [
                {
                    "scope": "primitive",
                    "stage": "raw inverse vertical MM",
                    "direction": "inverse",
                    "medianSeconds": 0.1,
                    "samplesSeconds": [0.1] * samples,
                },
                {
                    "scope": "uninstrumented-total",
                    "stage": (
                        "authoritative production-lifetime streamed four-target "
                        "spectral-flux composition"
                    ),
                    "direction": "forward",
                    "medianSeconds": seconds,
                    "samplesSeconds": [seconds] * samples,
                },
            ],
            "correctness": [{
                "passed": True,
                "maximumRelativeError": 2.0e-15,
                "relativeL2Error": 1.0e-15,
            }],
        }],
    }


def analysis_record(seconds: float, resident: int) -> dict:
    return {
        "runId": "run",
        "seconds": seconds,
        "components": {"raw inverse vertical MM": seconds / 10.0},
        "setup": {"totalSeconds": 0.2, "planningSeconds": 0.1},
        "memory": {
            "algorithmResidentBytes": resident,
            "scratchBytes": max(1, resident // 10),
            "estimatedProcessPeakBytes": resident + 20,
            "observedProcessHighWaterBytes": resident + 30,
        },
        "maximumCorrectnessError": 2.0e-15,
        "authoritativeFixture": True,
        "fixtureHash": "sha256:" + "c" * 64,
        "waveVortexModelCommit": WVM_COMMIT,
        "schedulingId": (
            "horizontal-outer-12;vertical-outer-dynamic-16-per-operator-family"
        ),
        "placementContractValid": True,
        "allocationLedgerValid": True,
        "sourceMetadataMatches": True,
        "valid": True,
    }


def calibration() -> dict:
    return {
        "sourceTreeGitCommit": BENCHMARK_COMMIT,
        "waveVortexModelCommit": WVM_COMMIT,
        "selections": {
            candidate.id: {"selectedTopology": TOPOLOGY.__dict__}
            for candidate in candidate_matrix()
        },
    }


class AuthoritativeSpectralFluxReferenceTests(unittest.TestCase):
    def test_command_freezes_reference_depth_and_topology(self) -> None:
        candidate = candidate_matrix()[1]
        command = command_for(
            Path("skbench"), Path("fixture.bin"), TIMING_PROFILES[0],
            candidate, TOPOLOGY, REFERENCE_WARMUPS, REFERENCE_SAMPLES,
            Path("result.json"),
        )
        self.assertEqual("12", command[command.index("--fftw-outer-workers") + 1])
        self.assertEqual(
            "outer-dynamic",
            command[command.index("--vertical-gemm-schedule") + 1],
        )
        self.assertEqual(
            "16", command[command.index("--vertical-gemm-outer-workers") + 1]
        )
        self.assertEqual("3", command[command.index("--warmups") + 1])
        self.assertEqual("21", command[command.index("--samples") + 1])

    def test_result_requires_authoritative_fixture_and_frozen_contract(self) -> None:
        candidate = candidate_matrix()[0]
        result = fake_result(candidate, TIMING_PROFILES[0], 1.0)
        record = result_record(
            candidate, TOPOLOGY, result,
            {
                "fixtureHash": "sha256:" + "c" * 64,
                "waveVortexModelCommit": WVM_COMMIT,
            },
            BENCHMARK_COMMIT, 3, 21,
        )
        self.assertTrue(record["valid"])
        self.assertTrue(record["placementContractValid"])
        self.assertTrue(record["allocationLedgerValid"])
        result["provenance"]["spectralFluxFixture"]["fixtureHash"] = "wrong"
        invalid = result_record(
            candidate, TOPOLOGY, result,
            {
                "fixtureHash": "sha256:" + "c" * 64,
                "waveVortexModelCommit": WVM_COMMIT,
            },
            BENCHMARK_COMMIT, 3, 21,
        )
        self.assertFalse(invalid["valid"])
        result["provenance"]["spectralFluxFixture"]["fixtureHash"] = (
            "sha256:" + "c" * 64
        )
        result["environment"]["gitCommit"] = ""
        missing_source = result_record(
            candidate, TOPOLOGY, result,
            {
                "fixtureHash": "sha256:" + "c" * 64,
                "waveVortexModelCommit": WVM_COMMIT,
            },
            BENCHMARK_COMMIT, 3, 21,
        )
        self.assertFalse(missing_source["sourceMetadataMatches"])
        self.assertFalse(missing_source["valid"])

    def test_conditional_rule_stops_clear_win_after_three_rounds(self) -> None:
        records = []
        for profile in TIMING_PROFILES:
            for round_number in (1, 2, 3):
                records.extend([
                    {
                        "candidateId": CONTROL_ID, "profile": profile,
                        "round": round_number,
                        "record": analysis_record(1.0, 100),
                    },
                    {
                        "candidateId": CANDIDATE_ID, "profile": profile,
                        "round": round_number,
                        "record": analysis_record(0.6, 60),
                    },
                ])
        decision = conditional_round_decision(records)
        self.assertTrue(decision["completeInitialThreeRoundMatrix"])
        self.assertFalse(decision["runAdditionalTwoRounds"])
        self.assertEqual(3, decision["finalRoundCount"])

    def test_conditional_rule_adds_two_rounds_near_gate(self) -> None:
        records = []
        ratios = (0.88, 0.91, 0.90)
        for profile in TIMING_PROFILES:
            for round_number, ratio in enumerate(ratios, start=1):
                records.extend([
                    {
                        "candidateId": CONTROL_ID, "profile": profile,
                        "round": round_number,
                        "record": analysis_record(1.0, 100),
                    },
                    {
                        "candidateId": CANDIDATE_ID, "profile": profile,
                        "round": round_number,
                        "record": analysis_record(ratio, 60),
                    },
                ])
        decision = conditional_round_decision(records)
        self.assertTrue(decision["runAdditionalTwoRounds"])
        self.assertEqual(5, decision["finalRoundCount"])
        self.assertIn(
            "aggregate-median-near-improvement-boundary",
            {item["id"] for item in decision["triggers"]},
        )

    def test_analysis_passes_clear_reference_result_with_capacity_disposition(self) -> None:
        timing = []
        memory = []
        for profile in TIMING_PROFILES:
            for round_number in (1, 2, 3):
                timing.extend([
                    {
                        "candidateId": CONTROL_ID, "profile": profile,
                        "round": round_number,
                        "record": analysis_record(1.0, 100),
                    },
                    {
                        "candidateId": CANDIDATE_ID, "profile": profile,
                        "round": round_number,
                        "record": analysis_record(0.6, 60),
                    },
                ])
            memory.extend([
                {
                    "candidateId": CONTROL_ID, "profile": profile,
                    "round": 1, "record": analysis_record(1.0, 100),
                },
                {
                    "candidateId": CANDIDATE_ID, "profile": profile,
                    "round": 1, "record": analysis_record(0.6, 60),
                },
            ])
        exclusions = [
            {
                "profile": "wvm-large-512-nz513-f4",
                "candidateId": candidate_id,
            }
            for candidate_id in (CONTROL_ID, CANDIDATE_ID)
        ]
        result = analyze(
            timing, memory, exclusions, calibration(),
            {"exitCode": 0}, "d" * 40,
        )
        self.assertTrue(result["completeMatchedFeasibleMatrix"])
        self.assertTrue(result["completeFullWorkloadDisposition"])
        self.assertTrue(result["memoryEvidenceComplete"])
        self.assertAlmostEqual(0.6, result["geometricCandidateToBaseline"])
        self.assertTrue(
            result["adoptionGate"]["advanceToWvmIntegrationExperiment"]
        )
        self.assertFalse(result["adoptionGate"]["completeNonlinearFluxMeasured"])

    def test_capacity_evidence_requires_both_deep_exclusions(self) -> None:
        calibration_evidence = {
            "waveVortexModelCommit": WVM_COMMIT,
            "fixtureHashes": {
                TIMING_PROFILES[0]: "sha256:" + "1" * 64,
                TIMING_PROFILES[1]: "sha256:" + "2" * 64,
            },
        }
        capacity = {
            "schema": (
                "spectral-kernel-authoritative-scaleout-capacity-publication-v1"
            ),
            "waveVortexModelCommit": WVM_COMMIT,
            "machine": {"physicalMemoryBytes": 128},
            "workloads": [
                {
                    "profile": TIMING_PROFILES[1],
                    "fixtureHash": "sha256:" + "2" * 64,
                    "graphs": [],
                },
                {
                    "profile": TIMING_PROFILES[2],
                    "fixtureHash": "sha256:" + "3" * 64,
                    "graphs": [],
                },
                {
                    "profile": "wvm-large-512-nz513-f4",
                    "fixtureHash": None,
                    "graphs": [
                        {
                            "candidateId": candidate_id,
                            "status": "capacity-exclusion",
                            "requiredPhysicalMemoryBytes": 256,
                        }
                        for candidate_id in (CONTROL_ID, CANDIDATE_ID)
                    ],
                },
            ],
        }
        fixtures, exclusions = capacity_and_fixture_evidence(
            calibration_evidence, capacity
        )
        self.assertEqual(set(TIMING_PROFILES), set(fixtures))
        self.assertEqual(2, len(exclusions))


if __name__ == "__main__":
    unittest.main()
