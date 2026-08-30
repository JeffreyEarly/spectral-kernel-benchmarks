import sys
import unittest
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "tools"))

from run_fused_vertical_views_campaign import (  # noqa: E402
    CANDIDATE_ID,
    CONTROL_ID,
    TIMING_PROFILES,
    analyze,
    candidate_matrix,
    command_for,
)


def record(seconds: float, resident: int) -> dict:
    return {
        "seconds": seconds,
        "components": {},
        "timingDetails": {
            "raw inverse vertical MM": seconds / 10.0,
            "native split triple extraction": seconds / 2.0,
        },
        "setup": {"totalSeconds": 0.1, "planningSeconds": 0.01},
        "memory": {
            "algorithmResidentBytes": resident,
            "scratchBytes": resident // 2,
            "estimatedProcessPeakBytes": resident + 20,
            "observedProcessHighWaterBytes": resident + 30,
        },
        "maximumCorrectnessError": 1.0e-15,
        "valid": True,
    }


def issue19() -> dict:
    return {"waveVortexModelCommit": "a" * 40}


class FusedVerticalViewsCampaignTests(unittest.TestCase):
    def test_command_changes_only_the_family_bridge_policy(self) -> None:
        control, candidate = candidate_matrix()
        control_command = command_for(
            Path("skbench"), Path("fixture.bin"), TIMING_PROFILES[0],
            control, 2, 7, Path("control.json"),
        )
        candidate_command = command_for(
            Path("skbench"), Path("fixture.bin"), TIMING_PROFILES[0],
            candidate, 2, 7, Path("candidate.json"),
        )
        self.assertIn("streaming-pruned-compact-split", control_command)
        self.assertIn(
            "streaming-pruned-compact-split-fused-vertical-views",
            candidate_command,
        )
        for option, value in (
            ("--fftw-outer-workers", "12"),
            ("--vertical-gemm-schedule", "outer-dynamic"),
            ("--vertical-gemm-outer-workers", "16"),
            ("--warmups", "2"),
            ("--samples", "7"),
        ):
            self.assertEqual(
                value, candidate_command[candidate_command.index(option) + 1]
            )

    def test_screen_advances_clear_complete_boundary_win(self) -> None:
        timing = []
        memory = []
        for profile in TIMING_PROFILES:
            timing.extend([
                {
                    "candidateId": CONTROL_ID, "profile": profile,
                    "round": 1, "record": record(1.0, 100),
                },
                {
                    "candidateId": CANDIDATE_ID, "profile": profile,
                    "round": 1, "record": record(0.4, 90),
                },
            ])
            memory.extend([
                {
                    "candidateId": CONTROL_ID, "profile": profile,
                    "round": 1, "record": record(1.0, 100),
                },
                {
                    "candidateId": CANDIDATE_ID, "profile": profile,
                    "round": 1, "record": record(0.4, 90),
                },
            ])
        result = analyze(
            "screen", timing, memory, {"exitCode": 0}, "b" * 40,
            issue19(),
        )
        self.assertTrue(result["completeMatchedMatrix"])
        self.assertTrue(result["screenAdvanceToReference"])
        self.assertAlmostEqual(0.4, result["geometricCandidateToControl"])

    def test_reference_gate_requires_three_rounds_and_memory(self) -> None:
        timing = []
        memory = []
        for profile in TIMING_PROFILES:
            for round_number, ratio in enumerate((0.39, 0.40, 0.41), start=1):
                timing.extend([
                    {
                        "candidateId": CONTROL_ID, "profile": profile,
                        "round": round_number, "record": record(1.0, 100),
                    },
                    {
                        "candidateId": CANDIDATE_ID, "profile": profile,
                        "round": round_number, "record": record(ratio, 90),
                    },
                ])
            memory.extend([
                {
                    "candidateId": CONTROL_ID, "profile": profile,
                    "round": 1, "record": record(1.0, 100),
                },
                {
                    "candidateId": CANDIDATE_ID, "profile": profile,
                    "round": 1, "record": record(0.4, 90),
                },
            ])
        result = analyze(
            "reference", timing, memory, {"exitCode": 0}, "b" * 40,
            issue19(),
        )
        self.assertTrue(
            result["adoptionGate"]["advanceFusedViewsToWvmIntegration"]
        )
        self.assertTrue(result["adoptionGate"]["memoryDoesNotRegress"])
        self.assertLess(
            result["empiricalStratifiedPairedRange"]["upper"], 1.0
        )


if __name__ == "__main__":
    unittest.main()
