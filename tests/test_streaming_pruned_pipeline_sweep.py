import sys
import unittest
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "tools"))

import run_streaming_pruned_pipeline_sweep as sweep  # noqa: E402


def fake_result(
    candidate: sweep.ScreenCandidate,
    profile: str,
    seconds: float,
    resident: int,
) -> dict:
    return {
        "status": "passed",
        "run": {"profile": profile},
        "providers": [{
            "id": candidate.primary_provider,
            "timings": [{
                "scope": "uninstrumented-total",
                "stage": "synthetic antialiased spectral pipeline",
                "direction": "round-trip",
                "medianSeconds": seconds,
            }],
            "correctness": [{
                "passed": True,
                "maximumRelativeError": 1.0e-15,
            }],
            "memory": {
                "algorithmResidentBytes": resident,
                "benchmarkHarnessBytes": 500,
                "estimatedProcessPeakBytes": resident + 500,
                "observedProcessHighWaterBytes": resident + 700,
            },
        }],
    }


class StreamingPrunedPipelineSweepTests(unittest.TestCase):
    def results(self, ratios: tuple[float, float, float], resident_ratio: float) -> list:
        baseline, candidate = sweep.candidate_matrix()
        results = []
        for profile, ratio in zip(sweep.PROFILES, ratios):
            results.append((baseline, fake_result(baseline, profile, 1.0, 1000)))
            results.append((candidate, fake_result(
                candidate, profile, ratio, round(1000 * resident_ratio),
            )))
        return results

    def test_cohort_and_candidates_are_exact(self) -> None:
        self.assertEqual(3, len(sweep.PROFILES))
        self.assertEqual(2, len(sweep.LARGE_PROFILES))
        self.assertTrue(all(profile.endswith("f4") for profile in sweep.PROFILES))
        self.assertEqual(
            [sweep.BASELINE_ID, sweep.CANDIDATE_ID],
            [candidate.id for candidate in sweep.candidate_matrix()],
        )

    def test_large_case_time_path_advances(self) -> None:
        analysis = sweep.analyze(self.results((1.08, 0.94, 0.95), 0.95))
        gate = analysis["screenGate"]
        self.assertTrue(gate["largeCaseTimePassed"])
        self.assertTrue(gate["advanceToReference"])
        self.assertEqual("advances-on-large-case-time", gate["classification"])
        self.assertFalse(gate["sizeDependentDispatchAllowed"])

    def test_memory_path_requires_the_overall_time_bound(self) -> None:
        passing = sweep.analyze(self.results((1.01, 1.01, 1.01), 0.85))
        self.assertTrue(passing["screenGate"]["memoryWithinTimePassed"])
        failing = sweep.analyze(self.results((1.04, 1.04, 1.04), 0.85))
        self.assertFalse(failing["screenGate"]["advanceToReference"])
        self.assertEqual(
            "negative-preliminary-screen",
            failing["screenGate"]["classification"],
        )

    def test_streaming_estimate_removes_full_batch_spectrum(self) -> None:
        for profile in sweep.PROFILES:
            baseline = sweep.estimated_explicit_peak_bytes(
                profile, "plane-major-fused-split",
            )
            streaming = sweep.estimated_explicit_peak_bytes(
                profile, "streaming-pruned-compact-split",
            )
            self.assertLess(streaming, baseline)
            self.assertLess(streaming, 64 * 1024**3)

    def test_command_keeps_the_issue16_tuple_fixed(self) -> None:
        candidate = sweep.candidate_matrix()[1]
        command = sweep.command_for(
            Path("skbench"), candidate, sweep.PROFILES[-1],
            1, 5, 129, Path("result.json"),
        )
        joined = " ".join(str(value) for value in command)
        self.assertIn("--boundary-policy streaming-pruned-compact-split", joined)
        self.assertIn("--fftw-internal-workers 1", joined)
        self.assertIn("--fftw-outer-workers 12", joined)
        self.assertIn("--vertical-gemm-outer-workers 16", joined)


if __name__ == "__main__":
    unittest.main()
