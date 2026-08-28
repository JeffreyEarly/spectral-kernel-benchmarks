import sys
import unittest
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import run_vertically_batched_advection_reference as reference  # noqa: E402
import build_site  # noqa: E402


def fake_result(
    candidate: reference.Candidate,
    profile: str,
    seconds: float,
    resident: int,
    observed: int,
) -> dict:
    horizontal_scope = (
        "operator-component"
        if candidate.id == reference.EXPLICIT_ID
        else "fused-primitive"
    )

    def timing(
        scope: str, stage: str, direction: str, value: float,
    ) -> dict:
        return {
            "scope": scope,
            "stage": stage,
            "direction": direction,
            "medianSeconds": value,
        }

    return {
        "status": "passed",
        "run": {"profile": profile},
        "providers": [{
            "id": candidate.provider_id,
            "timings": [
                timing(
                    "setup-shared-component",
                    "K2-grouped vertical fixture generation", "shared", 0.1,
                ),
                timing(
                    "setup-component", "directional vertical matrix preparation",
                    "shared", 0.2,
                ),
                timing(
                    "setup-component",
                    "horizontal planning and persistent scheduler setup",
                    "shared", 0.3,
                ),
                timing(
                    "primitive", "raw inverse vertical GEMM (15 fields)",
                    "inverse", 1.0,
                ),
                timing(
                    horizontal_scope,
                    "one physical-level four-target horizontal advection",
                    "horizontal", 0.01,
                ),
                timing(
                    "adapter-component",
                    "all-level split/field-major packing and projected-output scatter",
                    "horizontal", 2.0,
                ),
                timing(
                    "component",
                    "vertically batched horizontal advection including level movement",
                    "horizontal", 7.0,
                ),
                timing(
                    "primitive", "raw forward vertical GEMM (4 fields)",
                    "forward", 0.5,
                ),
                timing(
                    "uninstrumented-total", reference.TOTAL_STAGE,
                    "forward", seconds,
                ),
            ],
            "correctness": [{
                "passed": True,
                "maximumRelativeError": 1.0e-15,
            }],
            "memory": {
                "algorithmResidentBytes": resident,
                "scratchBytes": 100,
                "benchmarkHarnessBytes": 500,
                "estimatedProcessPeakBytes": resident + 500,
                "observedProcessHighWaterBytes": observed,
            },
            "executionContract": {"forward": {
                "nativePlacement": "out-of-place",
                "adapterPlacement": "out-of-place",
                "adapterPreservesCallerInput": True,
                "requiresPreservationCopyForRepeatedExecution": True,
                "preservationIncludedInAdapterTiming": True,
            }},
        }],
    }


class VerticallyBatchedAdvectionReferenceTests(unittest.TestCase):
    def results(
        self, time_ratio: float = 0.85, memory_ratio: float = 0.75,
    ) -> list:
        baseline, candidate = reference.reference_candidates()
        results = []
        for round_number in range(1, reference.REFERENCE_ROUNDS + 1):
            for profile in reference.PROFILES:
                results.append((baseline, round_number, fake_result(
                    baseline, profile, 10.0, 1000, 1500,
                )))
                results.append((candidate, round_number, fake_result(
                    candidate, profile, 10.0 * time_ratio,
                    round(1000 * memory_ratio), 1300,
                )))
        return results

    def test_reference_matrix_has_four_fixed_four_field_workloads(self) -> None:
        self.assertEqual(
            [reference.EXPLICIT_ID, reference.FFTWPP_ID],
            [candidate.id for candidate in reference.reference_candidates()],
        )
        self.assertEqual(4, len(reference.PROFILES))
        self.assertIn("wvm-large-512-nz513-f4", reference.PROFILES)
        self.assertTrue(all(profile.endswith("f4") for profile in reference.PROFILES))

    def test_reference_gate_uses_time_confidence_memory_and_placement(self) -> None:
        analysis = reference.analyze(self.results())
        gate = analysis["adoptionGate"]
        self.assertTrue(analysis["completeProductionMatrix"])
        self.assertTrue(analysis["allPlacementContractsValid"])
        self.assertTrue(gate["improvementPassed"])
        self.assertTrue(gate["regressionPassed"])
        self.assertTrue(gate["confidenceExcludesTie"])
        self.assertTrue(gate["memoryReductionPassed"])
        self.assertTrue(gate["adoptionCandidatePassed"])
        self.assertFalse(gate["sizeDependentDispatchAllowed"])
        self.assertEqual(4, len(analysis["profiles"]))

    def test_memory_gate_remains_independent_of_speed(self) -> None:
        gate = reference.analyze(
            self.results(time_ratio=0.85, memory_ratio=0.95)
        )["adoptionGate"]
        self.assertTrue(gate["improvementPassed"])
        self.assertFalse(gate["memoryReductionPassed"])
        self.assertFalse(gate["adoptionCandidatePassed"])

    def test_small_composed_win_does_not_pass_adoption(self) -> None:
        gate = reference.analyze(
            self.results(time_ratio=0.95, memory_ratio=0.75)
        )["adoptionGate"]
        self.assertFalse(gate["improvementPassed"])
        self.assertFalse(gate["adoptionCandidatePassed"])

    def test_command_fixes_vertical_and_horizontal_policies(self) -> None:
        candidate = reference.reference_candidates()[1]
        command = reference.command_for(
            Path("skbench"), candidate, reference.PROFILES[-1],
            3, 21, 129, Path("result.json"),
        )
        joined = " ".join(str(value) for value in command)
        self.assertIn("--kernel vertically-batched-advection", joined)
        self.assertIn("--vertical-gemm-schedule outer-dynamic", joined)
        self.assertIn("--vertical-gemm-outer-workers 12", joined)
        self.assertIn("--convolution-candidate fftwpp-parallel", joined)
        self.assertIn("--warmups 3", joined)
        self.assertIn("--samples 21", joined)

    def test_memory_preflight_covers_large_vertical_and_horizontal_cases(self) -> None:
        candidate = reference.reference_candidates()[0]
        large_vertical = reference.estimated_process_peak_bytes(
            *reference.PROFILE_SHAPES["wvm-large-512-nz513-f4"], candidate,
        )
        large_horizontal = reference.estimated_process_peak_bytes(
            *reference.PROFILE_SHAPES["wvm-large-1024-nz129-f4"], candidate,
        )
        self.assertGreater(large_vertical, 50 * 1024**3)
        self.assertGreater(large_horizontal, 20 * 1024**3)

    def test_site_synthesis_keeps_components_and_adoption_separate(self) -> None:
        bundles = [
            SimpleNamespace(
                publication={
                    "incrementId": reference.INCREMENT_ID,
                    "status": "reference",
                    "campaignCandidateId": candidate.id,
                    "campaignRound": round_number,
                },
                result=result,
            )
            for candidate, round_number, result in self.results()
        ]
        synthesis = build_site.vertically_batched_advection_reference_synthesis(
            bundles
        )
        self.assertIn("Four-workload composed M4 reference campaign", synthesis)
        self.assertIn("FFTW++ is 0.850× explicit geometrically", synthesis)
        self.assertIn("passes the M4 adoption-statistics gate", synthesis)
        self.assertIn("Inverse vertical", synthesis)
        self.assertIn("Movement alone", synthesis)
        self.assertIn("complete nonlinear flux", synthesis)


if __name__ == "__main__":
    unittest.main()
