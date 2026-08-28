import sys
import unittest
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "tools"))

import run_dealiased_convolution_wvm_sweep as sweep  # noqa: E402


def fake_provider(identifier: str, seconds: float, memory: int) -> dict:
    return {
        "id": identifier,
        "timings": [{
            "scope": "uninstrumented-total",
            "stage": sweep.TOTAL_STAGE,
            "direction": "forward",
            "medianSeconds": seconds,
        }],
        "correctness": [{
            "passed": True,
            "maximumRelativeError": 1.0e-15,
        }],
        "memory": {"persistentBytes": memory},
    }


def fake_result(profile: str, nx: int, time_ratio: float, memory_ratio: float) -> dict:
    explicit_seconds = 1.0
    explicit_memory = 1000
    providers = []
    for identifier in sweep.PROVIDER_IDS:
        seconds = explicit_seconds
        memory = explicit_memory
        if identifier == sweep.PARALLEL_IMPLICIT_ID:
            seconds *= time_ratio
            memory = round(memory * memory_ratio)
        providers.append(fake_provider(identifier, seconds, memory))
    return {
        "status": "passed",
        "run": {"profile": profile},
        "workload": {"Nx": nx},
        "providers": providers,
    }


class DealiasedConvolutionWvmSweepTests(unittest.TestCase):
    def results(self) -> list[dict]:
        return [
            fake_result(sweep.PROFILES[0], 256, 1.04, 0.71),
            fake_result(sweep.PROFILES[1], 512, 0.94, 0.76),
            fake_result(sweep.PROFILES[2], 1024, 0.78, 0.64),
        ]

    def test_exact_three_resolution_four_field_cohort(self) -> None:
        self.assertEqual(3, len(sweep.PROFILES))
        self.assertTrue(all(profile.endswith("f4") for profile in sweep.PROFILES))
        self.assertEqual(5, len(sweep.PROVIDER_IDS))

    def test_reference_gate_rewards_large_case_speed_and_memory(self) -> None:
        analysis = sweep.analyze(self.results())
        self.assertTrue(analysis["complete"])
        self.assertTrue(analysis["allCorrect"])
        self.assertLess(
            analysis["geometricParallelImplicitToParallelExplicit"], 0.95,
        )
        self.assertLess(
            analysis["largeGeometricParallelImplicitToParallelExplicit"], 0.95,
        )
        self.assertLess(
            analysis["geometricParallelImplicitMemoryToParallelExplicit"], 0.80,
        )
        self.assertTrue(analysis["advanceToReference"])

    def test_worst_profile_bound_blocks_reference(self) -> None:
        results = self.results()
        results[0] = fake_result(sweep.PROFILES[0], 256, 1.06, 0.71)
        self.assertFalse(sweep.analyze(results)["advanceToReference"])


if __name__ == "__main__":
    unittest.main()
