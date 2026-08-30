import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import run_pointwise_advection_reference as reference  # noqa: E402


def measured(total: float, pointwise: float, resident: int = 1_000_000) -> dict:
    return {
        "runId": "run",
        "seconds": total,
        "pointwiseSeconds": pointwise,
        "pointwiseFractionOfTotal": pointwise / total,
        "pointwiseBytes": 1000,
        "effectivePointwiseGigabytesPerSecond": 200.0,
        "components": {},
        "memory": {
            "persistentBytes": resident // 2,
            "scratchBytes": resident // 2,
            "algorithmResidentBytes": resident,
            "benchmarkHarnessBytes": 0,
            "estimatedProcessPeakBytes": resident,
            "observedProcessHighWaterBytes": resident + 100,
        },
        "maximumCorrectnessError": 1.0e-15,
        "authoritativeFixture": True,
        "sourceMetadataMatches": True,
        "schedulingId": "schedule",
        "totalLogicalWorkers": 1,
        "placementContractValid": True,
        "allocationLedgerValid": True,
        "valid": True,
    }


def issue19() -> dict:
    return {
        "waveVortexModelCommit": "w" * 40,
        "machine": {"physicalMemoryBytes": 137_438_953_472},
        "capacityExclusions": [{
            "profile": reference.DEEP_PROFILE,
            "candidateId": (
                "production-lifetime-streaming-pruned-tile16-authoritative"
            ),
            "requiredPhysicalMemoryBytes": 140_096_220_868,
        }],
    }


class PointwiseAdvectionReferenceTests(unittest.TestCase):
    def test_command_preserves_issue21_graph_and_changes_pointwise_workers(self) -> None:
        candidate = reference.candidate_matrix()[3]
        command = reference.command_for(
            Path("skbench"), Path("fixture.bin"), reference.PROFILES[0],
            candidate, 3, 21, Path("result.json"),
        )
        expected = {
            "--boundary-policy": reference.BOUNDARY_POLICY,
            "--fftw-outer-workers": "12",
            "--vertical-gemm-outer-workers": "16",
            "--pointwise-policy": "spatial-static",
            "--pointwise-workers": "12",
            "--warmups": "3",
            "--samples": "21",
        }
        for option, value in expected.items():
            self.assertEqual(value, command[command.index(option) + 1])

    def test_plans_cover_three_rotated_complete_rounds(self) -> None:
        fixtures = {profile: Path(f"{profile}.bin") for profile in reference.PROFILES}
        plans = reference.planned_runs(
            Path("skbench"), fixtures, Path("output"), [1, 2, 3],
            3, 21, "reference",
        )
        self.assertEqual(45, len(plans))
        self.assertEqual(45, len({plan["id"] for plan in plans}))
        first_by_round = [
            next(plan for plan in plans if plan["round"] == round_number)
            for round_number in (1, 2, 3)
        ]
        self.assertEqual(3, len({plan["profile"] for plan in first_by_round}))
        self.assertEqual(3, len({plan["candidate"].id for plan in first_by_round}))

    def test_selection_uses_smallest_worker_within_one_percent_of_fastest(self) -> None:
        totals = {
            "serial-1": (1.0, 0.20),
            "spatial-static-4": (0.86, 0.09),
            "spatial-static-8": (0.845, 0.08),
            "spatial-static-12": (0.84, 0.078),
            "spatial-static-16": (0.842, 0.078),
        }
        timing = []
        memory = []
        for profile in reference.PROFILES:
            for candidate in reference.candidate_matrix():
                total, pointwise = totals[candidate.id]
                for round_number in range(1, 4):
                    timing.append({
                        "profile": profile, "candidateId": candidate.id,
                        "round": round_number,
                        "record": measured(total, pointwise),
                    })
                memory.append({
                    "profile": profile, "candidateId": candidate.id,
                    "round": 1,
                    "record": measured(
                        total, pointwise,
                        1_000_000 + (0 if candidate.id == "serial-1" else 100),
                    ),
                })
        capacity = reference.deep_capacity_exclusions(issue19())
        analysis = reference.analyze(
            timing, memory, {"exitCode": 0}, {"passed": True}, capacity,
            "b" * 40, "r" * 40, issue19(),
        )
        self.assertTrue(analysis["completeMatchedMatrix"])
        self.assertEqual(
            "spatial-static-12",
            analysis["fastestEligibleCandidate"]["candidate"]["id"],
        )
        self.assertEqual(
            "spatial-static-8",
            analysis["selectedCandidate"]["candidate"]["id"],
        )
        self.assertTrue(
            analysis["adoptionGate"]["freezeSelectedM4PointwisePolicy"]
        )
        self.assertFalse(
            analysis["fusionContinuation"]["activateBoundedPointwiseFftFusion"]
        )

    def test_direct_views_do_not_make_deep_case_safe(self) -> None:
        exclusions = reference.deep_capacity_exclusions(issue19())
        self.assertEqual(5, len(exclusions))
        self.assertEqual(
            140_096_220_868,
            next(item for item in exclusions
                 if item["candidateId"] == "serial-1")
            ["requiredPhysicalMemoryBytes"],
        )
        self.assertTrue(all(
            item["requiredPhysicalMemoryBytes"] > item["physicalMemoryBytes"]
            for item in exclusions
        ))
        self.assertTrue(all(
            item["directFamilyViewSteadyStateSavingsBytes"] ==
                64 * reference.DEEP_NKL * reference.DEEP_NZ
            for item in exclusions
        ))


if __name__ == "__main__":
    unittest.main()
