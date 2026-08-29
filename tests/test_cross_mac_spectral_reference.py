import sys
import unittest
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "tools"))

import run_cross_mac_spectral_reference as cross_mac  # noqa: E402


def fake_result(
    algorithm: cross_mac.AlgorithmGraph,
    profile: str,
    seconds: float,
    resident: int = 1000,
) -> dict:
    return {
        "status": "passed",
        "run": {"profile": profile},
        "environment": {
            "cpuBrand": "Fake Apple",
            "gitCommit": "abc123",
            "gitDirty": False,
        },
        "providers": [{
            "id": algorithm.primary_provider,
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
            "executionContract": {
                "forward": {"nativePlacement": "out-of-place"},
                "inverse": {"nativePlacement": "out-of-place"},
            },
        }],
    }


class CrossMacSpectralReferenceTests(unittest.TestCase):
    def test_sysctl_string_returns_fallback_for_missing_key(self) -> None:
        self.assertEqual(
            "fallback",
            cross_mac.sysctl_string(
                "skbench.this-key-does-not-exist", "fallback",
            ),
        )

    def test_calibration_must_match_reference_machine(self) -> None:
        machine = {
            "hostname": "matilda",
            "cpuBrand": "Apple M1",
            "hardwareModel": "MacBookPro17,1",
            "performanceCores": 4,
            "efficiencyCores": 4,
            "totalPhysicalCores": 8,
        }
        self.assertTrue(cross_mac.calibration_matches_machine(
            {"machine": machine}, dict(machine),
        ))
        other = dict(machine)
        other["hostname"] = "lyra"
        self.assertFalse(cross_mac.calibration_matches_machine(
            {"machine": machine}, other,
        ))

    def test_topology_matrix_uses_machine_core_classes(self) -> None:
        topologies = cross_mac.topology_matrix(4, 8)
        self.assertEqual(4, len(topologies))
        self.assertEqual({4, 8}, {
            topology.horizontal_workers for topology in topologies
        })
        self.assertEqual(
            {("outer-dynamic", 8), ("outer-static", 4)},
            {(topology.vertical_schedule, topology.vertical_workers)
             for topology in topologies},
        )

    def test_command_freezes_algorithm_but_varies_topology(self) -> None:
        wvm, tiled = cross_mac.algorithm_graphs()
        topology = cross_mac.topology_matrix(4, 8)[0]
        wvm_command = cross_mac.command_for(
            Path("skbench"), wvm, topology, cross_mac.PROFILES[0],
            3, 21, 129, Path("wvm.json"),
        )
        tiled_command = cross_mac.command_for(
            Path("skbench"), tiled, topology, cross_mac.PROFILES[0],
            3, 21, 129, Path("tile.json"),
        )
        self.assertIn("--fftw-internal-workers 1", " ".join(wvm_command))
        self.assertIn("--fftw-outer-workers 4", " ".join(wvm_command))
        self.assertIn("--vertical-gemm-outer-workers 8", " ".join(wvm_command))
        self.assertNotIn("--streaming-tile-width", " ".join(wvm_command))
        self.assertIn("--streaming-tile-width 16", " ".join(tiled_command))

    def test_calibration_selects_preferred_semantic_topology_within_two_percent(self) -> None:
        algorithms = cross_mac.algorithm_graphs()
        topologies = cross_mac.topology_matrix(4, 8)
        preferred = next(
            topology for topology in topologies
            if topology.horizontal_worker_class == "performance"
            and topology.vertical_schedule == "outer-dynamic"
        )
        fastest = next(
            topology for topology in topologies
            if topology.horizontal_worker_class == "total"
            and topology.vertical_schedule == "outer-dynamic"
        )
        results = []
        for algorithm in algorithms:
            for topology in topologies:
                for profile in cross_mac.CALIBRATION_PROFILES:
                    seconds = 1.0
                    if topology == fastest:
                        seconds = 0.99
                    elif topology == preferred:
                        seconds = 1.0
                    results.append((algorithm, topology, fake_result(
                        algorithm, profile, seconds,
                    )))
        analysis = cross_mac.calibration_analysis(
            results, list(cross_mac.CALIBRATION_PROFILES), topologies,
        )
        for algorithm in algorithms:
            selected = analysis["selections"][algorithm.id]["selectedTopology"]
            self.assertEqual(preferred.id, selected["id"])

    def reference_results(self, ratios: list[float]) -> list:
        baseline, candidate = cross_mac.algorithm_graphs()
        results = []
        for round_number, ratio in enumerate(ratios, start=1):
            for profile in cross_mac.PROFILES[:2]:
                results.append((baseline, round_number, fake_result(
                    baseline, profile, 1.0,
                )))
                results.append((candidate, round_number, fake_result(
                    candidate, profile, ratio, 700,
                )))
        return results

    def test_conditional_rule_stops_clear_win_after_three_rounds(self) -> None:
        decision = cross_mac.conditional_round_decision(
            self.reference_results([0.60, 0.62, 0.61]),
            list(cross_mac.PROFILES[:2]),
        )
        self.assertTrue(decision["completeInitialThreeRoundMatrix"])
        self.assertFalse(decision["runAdditionalTwoRounds"])
        self.assertEqual(3, decision["finalRoundCount"])

    def test_conditional_rule_adds_two_rounds_near_gate(self) -> None:
        decision = cross_mac.conditional_round_decision(
            self.reference_results([0.88, 0.91, 0.90]),
            list(cross_mac.PROFILES[:2]),
        )
        self.assertTrue(decision["runAdditionalTwoRounds"])
        self.assertEqual(5, decision["finalRoundCount"])
        self.assertTrue(any(
            trigger["id"] == "aggregate-median-near-improvement-boundary"
            for trigger in decision["triggers"]
        ))

    def test_reference_analysis_keeps_memory_only_separate(self) -> None:
        timing = self.reference_results([0.60, 0.62, 0.61])
        baseline, candidate = cross_mac.algorithm_graphs()
        memory = []
        for profile in cross_mac.PROFILES[:2]:
            memory.append((baseline, fake_result(baseline, profile, 9.0, 1000)))
            memory.append((candidate, fake_result(candidate, profile, 9.0, 700)))
        topology = cross_mac.topology_matrix(4, 8)[0]
        calibration = {
            "selections": {
                algorithm.id: {"selectedTopology": cross_mac.asdict(topology)}
                for algorithm in (baseline, candidate)
            },
        }
        analysis = cross_mac.reference_analysis(
            timing, memory, list(cross_mac.PROFILES[:2]), [], calibration,
        )
        self.assertTrue(analysis["completeMatchedFeasibleMatrix"])
        self.assertTrue(analysis["completeFullWorkloadMatrix"])
        self.assertAlmostEqual(0.61, analysis["geometricCandidateToBaseline"])
        self.assertAlmostEqual(
            0.7,
            analysis["memoryOnlyGeometricRatios"]["algorithmResidentBytes"],
        )
        self.assertTrue(
            analysis["decisionGate"]["portabilityCandidatePassedOnThisMachine"]
        )
        self.assertFalse(analysis["decisionGate"]["generalMacClaimAllowed"])

    def test_explicit_capacity_exclusion_allows_scoped_matched_result(self) -> None:
        profiles = list(cross_mac.PROFILES[:3])
        timing = self.reference_results([0.60, 0.62, 0.61])
        baseline, candidate = cross_mac.algorithm_graphs()
        memory = []
        for profile in profiles[:2]:
            memory.append((baseline, fake_result(baseline, profile, 9.0, 1000)))
            memory.append((candidate, fake_result(candidate, profile, 9.0, 700)))
        topology = cross_mac.topology_matrix(4, 8)[0]
        calibration = {
            "selections": {
                algorithm.id: {"selectedTopology": cross_mac.asdict(topology)}
                for algorithm in (baseline, candidate)
            },
        }
        exclusions = [{
            "profile": profiles[2],
            "algorithmId": baseline.id,
        }]
        analysis = cross_mac.reference_analysis(
            timing, memory, profiles, exclusions, calibration,
        )
        self.assertTrue(analysis["completeMatchedFeasibleMatrix"])
        self.assertFalse(analysis["completeFullWorkloadMatrix"])
        self.assertTrue(
            analysis["decisionGate"]["portabilityCandidatePassedOnThisMachine"]
        )

    def test_cross_machine_synthesis_requires_distinct_machines_and_same_commit(self) -> None:
        def analysis(model: str, cpu: str) -> dict:
            return {
                "schema": "spectral-kernel-cross-mac-reference-analysis-v1",
                "sourceTreeGitCommit": "abc123",
                "sourceTreeDirty": False,
                "machine": {
                    "hardwareModel": model,
                    "cpuBrand": cpu,
                },
                "profilesMatched": list(cross_mac.PROFILES[:2]),
                "capacityExclusions": [],
                "calibrationSelections": {},
                "geometricCandidateToBaseline": 0.7,
                "maximumProfileCandidateToBaseline": 0.8,
                "empiricalStratifiedPairedRange": {
                    "lower": 0.6, "upper": 0.9,
                },
                "memoryOnlyGeometricRatios": {},
                "decisionGate": {
                    "portabilityCandidatePassedOnThisMachine": True,
                },
            }

        synthesis = cross_mac.combine_machine_analyses([
            analysis("Mac16,9", "Apple M4 Max"),
            analysis("MacBookPro17,1", "Apple M1"),
        ])
        self.assertTrue(
            synthesis["crossMacPortabilityGate"]
            ["portabilityQualifiedAcrossTestedMachines"]
        )
        self.assertFalse(
            synthesis["crossMacPortabilityGate"]["generalMacClaimAllowed"]
        )
        with self.assertRaises(ValueError):
            cross_mac.combine_machine_analyses([
                analysis("Mac16,9", "Apple M4 Max"),
                analysis("Mac16,9", "Apple M4 Max"),
            ])


if __name__ == "__main__":
    unittest.main()
