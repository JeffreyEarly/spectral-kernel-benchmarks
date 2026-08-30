import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import run_constant_stratification_flux_authoritative_reference as campaign  # noqa: E402


def timing(scope: str, stage: str, direction: str, seconds: float) -> dict:
    return {
        "scope": scope,
        "stage": stage,
        "direction": direction,
        "medianSeconds": seconds,
        "samplesSeconds": [seconds] * campaign.REFERENCE_SAMPLES,
    }


def provider(provider_id: str, seconds: float, order: str) -> dict:
    timings = [
        timing("uninstrumented-total", campaign.TOTAL_STAGE, "complete", seconds)
    ]
    for scope, stage, direction in campaign.COMPONENT_STAGES.values():
        timings.append(timing(scope, stage, direction, seconds / 9.0))
    correctness = [{
        "name": (
            "complete compact composition versus authoritative WVM oracle"
            if provider_id == campaign.CANDIDATE_PROVIDER
            else "complete full-half composition versus authoritative WVM oracle"
        ),
        "passed": True,
        "maximumRelativeError": 1.0e-14,
        "relativeL2Error": 2.0e-14,
    }]
    if provider_id == campaign.CANDIDATE_PROVIDER:
        correctness.append({
            "name": "complete compact composition versus full-half control",
            "passed": True,
            "maximumRelativeError": 3.0e-15,
            "relativeL2Error": 4.0e-15,
        })
    correctness.append({
        "name": "fixture MATLAB versus compiled WVM nonlinear-flux cross-check",
        "passed": True,
        "maximumRelativeError": 5.0e-12,
        "relativeL2Error": 2.0e-12,
    })
    return {
        "id": provider_id,
        "timings": timings,
        "schedulingId": (
            "vertical-type1-internal-16;horizontal-internal-1-outer-12;"
            f"pointwise-spatial-static-8;comparison-{order}"
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
        "memory": {
            "persistentBytes": 100,
            "scratchBytes": 200,
            "algorithmResidentBytes": 300,
            "benchmarkHarnessBytes": 50,
            "estimatedProcessPeakBytes": 350,
            "observedProcessHighWaterBytes": 1000,
        },
        "componentLedger": [{
            "stage": "steady-state application allocation",
            "state": "elided",
        }],
        "correctness": correctness,
    }


def result(profile: str, round_number: int, ratio: float = 0.7) -> dict:
    order = "candidate-first" if round_number % 2 == 0 else "control-first"
    return {
        "status": "passed",
        "run": {
            "id": f"run-{profile}-{round_number}",
            "profile": profile,
            "warmups": campaign.REFERENCE_WARMUPS,
            "samples": campaign.REFERENCE_SAMPLES,
        },
        "environment": {"gitCommit": "a" * 12, "gitDirty": False},
        "provenance": {"spectralFluxFixture": {
            "status": "authoritative-wvm-export",
            "authoritative": True,
            "schema": "constant-stratification-flux-fixture-v1",
            "fixtureHash": f"sha256:{profile}",
            "waveVortexModelCommit": campaign.WVM_COMMIT,
        }},
        "providers": [
            provider(campaign.CONTROL_PROVIDER, 1.0, order),
            provider(campaign.CANDIDATE_PROVIDER, ratio, order),
        ],
    }


class AuthoritativeConstantFluxCampaignTests(unittest.TestCase):
    def test_command_freezes_topology_and_measurement_order(self) -> None:
        command = campaign.command_for(
            Path("skbench"), campaign.PROFILES[0], Path("fixture.bin"),
            "candidate-first", 3, 21, Path("result.json"),
        )
        self.assertEqual(
            "candidate-first", command[command.index("--comparison-order") + 1]
        )
        self.assertEqual("16", command[command.index("--fftw-internal-workers") + 1])
        self.assertEqual("12", command[command.index("--fftw-outer-workers") + 1])
        self.assertEqual("8", command[command.index("--pointwise-workers") + 1])

    def test_result_requires_authoritative_fixture_and_complete_contract(self) -> None:
        profile = campaign.PROFILES[0]
        record = campaign.result_record(
            result(profile, 1), profile, 1,
            {"fixtureHash": f"sha256:{profile}"}, "a" * 40,
            campaign.REFERENCE_WARMUPS, campaign.REFERENCE_SAMPLES,
        )
        self.assertTrue(record["valid"])
        self.assertAlmostEqual(0.7, record["candidateToControl"])

    def test_analysis_passes_reference_gate_for_stable_improvement(self) -> None:
        records = []
        for round_number in range(1, 4):
            for profile in campaign.PROFILES:
                records.append(campaign.result_record(
                    result(profile, round_number), profile, round_number,
                    {"fixtureHash": f"sha256:{profile}"}, "a" * 40,
                    campaign.REFERENCE_WARMUPS, campaign.REFERENCE_SAMPLES,
                ))
        analysis = campaign.analyze(records, "a" * 40)
        self.assertTrue(analysis["allRecordsValid"])
        self.assertTrue(analysis["algorithmEquivalenceDualNormPassed"])
        self.assertTrue(analysis["authoritativeOracleDualNormPassed"])
        self.assertAlmostEqual(0.7, analysis["geometricCandidateToControl"])
        self.assertTrue(
            analysis["adoptionGate"]["advanceConstantStratificationCandidate"]
        )
        self.assertFalse(
            analysis["conditionalRoundDecision"]["runAdditionalTwoRounds"]
        )

    def test_conditional_rounds_trigger_near_adoption_boundary(self) -> None:
        records = []
        for round_number in range(1, 4):
            for profile in campaign.PROFILES:
                records.append(campaign.result_record(
                    result(profile, round_number, 0.9), profile, round_number,
                    {"fixtureHash": f"sha256:{profile}"}, "a" * 40,
                    campaign.REFERENCE_WARMUPS, campaign.REFERENCE_SAMPLES,
                ))
        decision = campaign.conditional_round_decision(records)
        self.assertTrue(decision["runAdditionalTwoRounds"])
        self.assertEqual(5, decision["finalRoundCount"])


if __name__ == "__main__":
    unittest.main()
