import sys
import unittest
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "tools"))

import run_streaming_pruned_locality_sweep as sweep  # noqa: E402


def fake_result(
    candidate: sweep.LocalityCandidate,
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


class StreamingPrunedLocalitySweepTests(unittest.TestCase):
    def results(self) -> list:
        candidates = sweep.candidate_matrix()
        ratios = {
            sweep.BASELINE_ID: (1.0, 1.0, 1.0),
            sweep.DIRECT_ID: (0.75, 1.05, 1.10),
            "streaming-pruned-tiled-4--outer-dynamic-16":
                (0.72, 0.93, 0.96),
            "streaming-pruned-tiled-8--outer-dynamic-16":
                (0.70, 0.82, 0.84),
            "streaming-pruned-tiled-16--outer-dynamic-16":
                (0.71, 0.84, 0.86),
        }
        residents = {candidate.id: 850 for candidate in candidates}
        residents[sweep.BASELINE_ID] = 1000
        results = []
        for candidate in candidates:
            for profile, ratio in zip(
                sweep.PROFILES, ratios[candidate.id], strict=True,
            ):
                results.append((candidate, fake_result(
                    candidate, profile, ratio, residents[candidate.id],
                )))
        return results

    def test_cohort_and_fixed_tile_candidates_are_exact(self) -> None:
        candidates = sweep.candidate_matrix()
        self.assertEqual(5, len(candidates))
        self.assertEqual([0, 1, 4, 8, 16], [
            candidate.tile_width for candidate in candidates
        ])
        self.assertEqual(3, len(sweep.PROFILES))
        self.assertEqual(2, len(sweep.LARGE_PROFILES))
        self.assertTrue(all(profile.endswith("f4") for profile in sweep.PROFILES))

    def test_selects_one_uniform_memory_qualified_tile(self) -> None:
        analysis = sweep.analyze(self.results())
        selection = analysis["selection"]
        self.assertTrue(analysis["completeProductionMatrix"])
        self.assertTrue(analysis["allCorrectWithin1e-12"])
        self.assertEqual(
            "streaming-pruned-tiled-8--outer-dynamic-16",
            selection["selectedCandidateId"],
        )
        self.assertEqual(8, selection["selectedTileWidth"])
        self.assertTrue(selection["optimizationSuccessful"])
        self.assertTrue(selection["advanceSelectedTileToReference"])
        self.assertFalse(selection["sizeDependentDispatchAllowed"])

    def test_memory_estimates_include_bounded_compact_tile(self) -> None:
        candidates = sweep.candidate_matrix()
        for profile in sweep.PROFILES:
            estimates = [
                sweep.estimated_explicit_peak_bytes(profile, candidate)
                for candidate in candidates
            ]
            self.assertLess(estimates[1], estimates[0])
            self.assertLess(estimates[2], estimates[3])
            self.assertLess(estimates[3], estimates[4])
            self.assertLess(estimates[4], estimates[0])

    def test_command_records_tile_width_and_fixed_worker_tuple(self) -> None:
        candidate = sweep.candidate_matrix()[3]
        command = sweep.command_for(
            Path("skbench"), candidate, sweep.PROFILES[-1],
            1, 5, 129, Path("result.json"),
        )
        joined = " ".join(str(value) for value in command)
        self.assertIn("--streaming-tile-width 8", joined)
        self.assertIn("--boundary-policy streaming-pruned-compact-split", joined)
        self.assertIn("--fftw-internal-workers 1", joined)
        self.assertIn("--fftw-outer-workers 12", joined)
        self.assertIn("--vertical-gemm-outer-workers 16", joined)


if __name__ == "__main__":
    unittest.main()
