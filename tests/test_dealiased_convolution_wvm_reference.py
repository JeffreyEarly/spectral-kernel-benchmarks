import sys
import unittest
from pathlib import Path
from types import SimpleNamespace


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "tools"))

import run_dealiased_convolution_wvm_reference as reference  # noqa: E402
import build_site  # noqa: E402


def fake_result(
    candidate: reference.Candidate,
    profile: str,
    seconds: float,
    resident: int,
    observed: int,
) -> dict:
    if candidate.primary_provider == reference.EXPLICIT_ID:
        timings = [
            ("adapter-component", "shared advector embedding and inverse FFTs", 0.2),
            ("adapter-component", "parallel target transforms and reduction", 0.7),
            ("uninstrumented-total", reference.TOTAL_STAGE, seconds),
        ]
    else:
        timings = [
            ("fused-primitive", "implicit hybrid transform-reduce-transform", 0.7),
            ("uninstrumented-total", reference.TOTAL_STAGE, seconds),
        ]
    return {
        "status": "passed",
        "run": {"profile": profile},
        "providers": [{
            "id": candidate.primary_provider,
            "planning": {"seconds": 0.5},
            "timings": [
                {
                    "scope": scope,
                    "stage": stage,
                    "direction": "forward",
                    "medianSeconds": value,
                }
                for scope, stage, value in timings
            ],
            "correctness": [{
                "passed": True,
                "maximumRelativeError": 1.0e-15,
            }],
            "memory": {
                "algorithmResidentBytes": resident,
                "benchmarkHarnessBytes": 500,
                "estimatedProcessPeakBytes": resident + 500,
                "observedProcessHighWaterBytes": observed,
            },
            "executionContract": {"forward": {
                "adapterPlacement": "out-of-place",
                "adapterPreservesCallerInput": True,
                "requiresPreservationCopyForRepeatedExecution": True,
                "preservationIncludedInAdapterTiming": True,
            }},
        }],
    }


class DealiasedConvolutionWvmReferenceTests(unittest.TestCase):
    def results(
        self, time_ratio: float = 0.85, memory_ratio: float = 0.65,
    ) -> list:
        baseline, candidate = reference.reference_candidates()
        results = []
        for round_number in range(1, reference.REFERENCE_ROUNDS + 1):
            for profile in reference.PROFILES:
                results.append((baseline, round_number, fake_result(
                    baseline, profile, 1.0, 1000, 1500,
                )))
                results.append((candidate, round_number, fake_result(
                    candidate, profile, time_ratio,
                    round(1000 * memory_ratio), 1300,
                )))
        return results

    def test_reference_matrix_freezes_two_finalists(self) -> None:
        candidates = reference.reference_candidates()
        self.assertEqual(
            ["explicit-parallel", "fftwpp-parallel"],
            [candidate.id for candidate in candidates],
        )
        self.assertEqual(3, len(reference.PROFILES))
        self.assertTrue(all(profile.endswith("f4") for profile in reference.PROFILES))

    def test_reference_gate_uses_paired_confidence_and_memory(self) -> None:
        analysis = reference.analyze(self.results())
        gate = analysis["referenceGate"]
        self.assertTrue(analysis["completeProductionMatrix"])
        self.assertTrue(analysis["allPlacementContractsValid"])
        self.assertTrue(gate["improvementPassed"])
        self.assertTrue(gate["regressionPassed"])
        self.assertTrue(gate["confidenceExcludesTie"])
        self.assertTrue(gate["memoryReductionPassed"])
        self.assertTrue(gate["singleUniformPolicyPassed"])
        self.assertFalse(gate["sizeDependentDispatchAllowed"])
        self.assertEqual(3, len(analysis["profiles"]))

    def test_memory_gate_is_independent_of_speed(self) -> None:
        gate = reference.analyze(
            self.results(time_ratio=0.85, memory_ratio=0.85)
        )["referenceGate"]
        self.assertTrue(gate["improvementPassed"])
        self.assertFalse(gate["memoryReductionPassed"])
        self.assertFalse(gate["singleUniformPolicyPassed"])

    def test_improvement_gate_rejects_small_win(self) -> None:
        gate = reference.analyze(
            self.results(time_ratio=0.92, memory_ratio=0.65)
        )["referenceGate"]
        self.assertFalse(gate["improvementPassed"])
        self.assertFalse(gate["singleUniformPolicyPassed"])

    def test_command_is_finalist_only_and_reference_depth(self) -> None:
        candidate = reference.reference_candidates()[1]
        command = reference.command_for(
            Path("skbench"), candidate, reference.PROFILES[-1],
            3, 21, 129, Path("result.json"),
        )
        joined = " ".join(str(value) for value in command)
        self.assertIn("--convolution-candidate fftwpp-parallel", joined)
        self.assertIn("--warmups 3", joined)
        self.assertIn("--samples 21", joined)

    def test_site_synthesis_pairs_separate_finalist_runs(self) -> None:
        bundles = []
        for candidate, round_number, result in self.results():
            bundles.append(SimpleNamespace(
                publication={
                    "incrementId": reference.INCREMENT_ID,
                    "status": "reference",
                    "campaignCandidateId": candidate.id,
                    "campaignRound": round_number,
                },
                result=result,
            ))
        synthesis = build_site.dealiased_convolution_reference_synthesis(bundles)
        self.assertIn("Rotated finalist-only M4 reference-depth campaign", synthesis)
        self.assertIn("FFTW++ is 0.850× explicit FFTW", synthesis)
        self.assertIn("passes the M4 horizontal-kernel reference gate", synthesis)
        self.assertIn("complete nonlinear flux", synthesis)


if __name__ == "__main__":
    unittest.main()
