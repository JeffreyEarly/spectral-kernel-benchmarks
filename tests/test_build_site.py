import hashlib
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from urllib.parse import unquote, urlsplit


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
GENERATOR = REPOSITORY_ROOT / "tools" / "build_site.py"
VALIDATOR = REPOSITORY_ROOT / "tools" / "validate_publication.py"
PUBLISHED = REPOSITORY_ROOT / "results" / "published"


def content_hashes(root: Path) -> dict[str, str]:
    return {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def assert_internal_links_resolve(test: unittest.TestCase, root: Path) -> None:
    for page in root.rglob("*.html"):
        for href in re.findall(r'href="([^"]+)"', page.read_text(encoding="utf-8")):
            parsed = urlsplit(href)
            if parsed.scheme or parsed.netloc or not parsed.path:
                continue
            target = (page.parent / unquote(parsed.path)).resolve()
            if target.is_dir():
                target /= "index.html"
            test.assertTrue(target.is_file(), f"{page.relative_to(root)} has broken link {href}")


class BuildSiteTests(unittest.TestCase):
    def test_builds_deterministic_append_only_archive(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            output = Path(temporary_directory) / "site"
            subprocess.run([sys.executable, str(VALIDATOR)], cwd=REPOSITORY_ROOT, check=True, capture_output=True, text=True)
            command = [sys.executable, str(GENERATOR), "--output", str(output)]
            subprocess.run(command, cwd=REPOSITORY_ROOT, check=True, capture_output=True, text=True)
            first_hashes = content_hashes(output)

            index = (output / "index.html").read_text(encoding="utf-8")
            legacy_page = output / "runs" / "m4-max-quick-20260827.html"
            canonical_page = output / "runs" / "20260827T185428Z-lyra" / "index.html"
            split_run_page = output / "runs" / "20260828T012502664620Z-lyra" / "index.html"
            issue_3_page = output / "experiments" / "issue-003-fftw-production-baseline" / "index.html"
            issue_4_page = output / "experiments" / "issue-004-fftw-strategy-sweep" / "index.html"
            issue_5_page = output / "experiments" / "issue-005-vdsp-native-baseline" / "index.html"
            issue_6_page = output / "experiments" / "issue-006-vdsp-batching-scheduling" / "index.html"
            issue_7_page = output / "experiments" / "issue-007-retained-horizontal-algorithms" / "index.html"
            issue_8_page = output / "experiments" / "issue-008-vertical-projection-gemm" / "index.html"
            issue_9_page = output / "experiments" / "issue-009-combined-spectral-pipeline" / "index.html"
            issue_12_page = output / "experiments" / "issue-012-pruned-horizontal-transforms" / "index.html"
            issue_13_page = output / "experiments" / "issue-013-ordering-packing-crossover" / "index.html"
            float32_page = output / "experiments" / "issue-015-float32-spectral-kernel-screen" / "index.html"
            issue_16_page = output / "experiments" / "issue-016-streaming-pruned-compact-split" / "index.html"
            issue_17_page = output / "experiments" / "issue-017-implicit-hybrid-dealiased-convolution" / "index.html"
            issue_18_page = output / "experiments" / "issue-018-vertically-batched-advection-pipeline" / "index.html"
            methods_page = output / "methods" / "operators-and-representations" / "index.html"
            decision_page = output / "decisions" / "v1" / "index.html"
            result_artifact = output / "artifacts" / "20260827T185428Z-lyra" / "result.json"
            samples_artifact = output / "artifacts" / "20260827T185428Z-lyra" / "samples.csv"

            self.assertIn("Which spectral kernels are actually fastest?", index)
            self.assertIn("preliminary", index)
            self.assertIn("Raw FFT", index)
            self.assertIn("dealiased four-field convolution", index)
            self.assertTrue(legacy_page.is_file())
            self.assertTrue(canonical_page.is_file())
            self.assertTrue(split_run_page.is_file())
            self.assertTrue(issue_3_page.is_file())
            self.assertTrue(issue_4_page.is_file())
            self.assertTrue(issue_5_page.is_file())
            self.assertTrue(issue_6_page.is_file())
            self.assertTrue(issue_7_page.is_file())
            self.assertTrue(issue_8_page.is_file())
            self.assertTrue(issue_9_page.is_file())
            self.assertTrue(issue_12_page.is_file())
            self.assertTrue(issue_13_page.is_file())
            self.assertTrue(float32_page.is_file())
            self.assertTrue(issue_16_page.is_file())
            self.assertTrue(issue_17_page.is_file())
            self.assertTrue(issue_18_page.is_file())
            self.assertTrue(methods_page.is_file())
            self.assertTrue(decision_page.is_file())
            self.assertIn("What this experiment measures", issue_3_page.read_text(encoding="utf-8"))
            issue_4_html = issue_4_page.read_text(encoding="utf-8")
            self.assertIn("Reproducible Pareto screen", issue_4_html)
            self.assertIn("deterministic percentile-bootstrap 95% intervals", issue_4_html)
            self.assertIn("infeasible within its planning budget", issue_4_html)
            self.assertIn("wvm-guru64-measure-unaligned-cold", issue_4_html)
            self.assertIn("20260827T223146704900Z-lyra", issue_4_html)
            self.assertIn("15 workload-candidates are Pareto", issue_4_html)
            self.assertIn("15 are dominated", issue_4_html)
            self.assertIn("0 are marked infeasible", issue_4_html)
            self.assertIn("Provider-native order result", issue_4_html)
            self.assertIn("0.650× (6/6 wins", issue_4_html)
            self.assertIn("0.825× (6/6 wins", issue_4_html)
            self.assertIn("4.831× (0/6 wins", issue_4_html)
            self.assertIn("9.429× (0/6 wins", issue_4_html)
            self.assertIn("final issue #4 M4 Max reference Pareto set", issue_4_html)
            self.assertIn("f187eac53ff8", issue_4_html)
            self.assertIn("Paired split-versus-interleaved increment", issue_4_html)
            self.assertIn("Split-layout diagnostic", issue_4_html)
            self.assertIn("Raw CV interleaved / split", issue_4_html)
            self.assertIn("Planning memory", issue_4_html)
            self.assertIn("1.217× forward and 1.166× inverse", issue_4_html)
            self.assertIn("Append-only FFTW strategy archive", issue_4_html)
            self.assertIn("wvm-guru64-split-measure-aligned-cold", issue_4_html)
            self.assertIn("20260828T012502664620Z-lyra", issue_4_html)
            self.assertIn("one contiguous allocation with a fixed [real][imaginary] component separation", issue_4_html)
            split_run_html = split_run_page.read_text(encoding="utf-8")
            self.assertIn("FFTW split", split_run_html)
            self.assertIn("wvm-frequency-major-split-half-spectrum", split_run_html)
            self.assertIn("exact WVM-order split in-place", split_run_html)
            issue_5_html = issue_5_page.read_text(encoding="utf-8")
            self.assertIn("No reference run has been published", issue_5_html)
            issue_12_html = issue_12_page.read_text(encoding="utf-8")
            self.assertIn("Partial-column-pruned feasibility increment", issue_12_html)
            self.assertIn("immutable initial internally threaded cohort contains 24", issue_12_html)
            self.assertIn("internal=1, outer=1: forward 0.996× (2/6 wins), inverse 0.930× (6/6 wins)", issue_12_html)
            self.assertIn("internal=12, outer=1: forward 2.094× (0/6 wins), inverse 1.505× (0/6 wins)", issue_12_html)
            self.assertIn("internally threaded performance-core tuple remains a negative result", issue_12_html)
            self.assertIn("Across 36 outer-sharded workload/topology/direction cells", issue_12_html)
            self.assertIn("internal=1, outer=4: forward 0.630× (6/6 wins), inverse 0.669× (6/6 wins)", issue_12_html)
            self.assertIn("fully pruned candidate therefore advances to issue #7", issue_12_html)
            self.assertIn("Empty dispatch", issue_12_html)
            self.assertIn("Scratch aggregate / max shard", issue_12_html)
            issue_13_html = issue_13_page.read_text(encoding="utf-8")
            self.assertIn("Composed representation-crossover increment", issue_13_html)
            self.assertIn("Composed horizontal-to-vertical boundary evidence", issue_13_html)
            self.assertIn("Full-spectrum inverse views include their required per-execution zero-padding rebuild", issue_13_html)
            self.assertIn(
                "modal work and the nonlinear flux calculation remain explicitly excluded",
                issue_13_html.lower(),
            )
            self.assertIn("Primitive FFT", issue_5_html)
            self.assertIn("Retained total", issue_5_html)
            self.assertIn("out-of-place-explicit-scratch", issue_5_html)
            self.assertIn("20260827T205846767331Z-lyra", issue_5_html)
            issue_6_html = issue_6_page.read_text(encoding="utf-8")
            self.assertIn("deterministic percentile-bootstrap 95% intervals", issue_6_html)
            self.assertIn("Raw CV", issue_6_html)
            self.assertIn("Empty dispatch", issue_6_html)
            self.assertIn("vdsp-radix2-native-2d-in-place-direct-gcd", issue_6_html)
            self.assertIn("vdsp-radix2-separable-packed-real-in-place-separable-persistent", issue_6_html)
            self.assertIn("20260827T220056957967Z-lyra", issue_6_html)
            self.assertIn("No new GCD or separable candidate clears the 10% advancement screen", issue_6_html)
            self.assertIn("Direct-persistent remains the only issue #6 candidate carried forward", issue_6_html)
            self.assertIn("Qualified negative feasibility conclusion", issue_6_html)
            self.assertIn("deliberately stops this Float64 scheduling sweep", issue_6_html)
            self.assertIn("only as an issue #7 guardrail", issue_6_html)
            self.assertIn("comes within 1.25×", issue_6_html)
            self.assertIn("not reference adoption statistics", issue_6_html)
            issue_7_html = issue_7_page.read_text(encoding="utf-8")
            self.assertIn("Representation-boundary close-out", issue_7_html)
            self.assertIn("0.815× geometric versus the faster matched control, 10/12 workload-direction wins", issue_7_html)
            self.assertIn("1.086× geometric versus the faster matched control, 0/12 workload-direction wins", issue_7_html)
            self.assertIn("fused/separate normalized-total geometric ratio is 0.861× across 18 processes", issue_7_html)
            self.assertIn("ready disposable zero-padded provider-order spectrum", issue_7_html)
            self.assertIn("Placement and inverse lifetime are algorithm contracts", issue_7_html)
            self.assertIn("Issue #7 handoff", issue_7_html)
            self.assertIn("pruned outer-12 interleaved compact storage as the self-contained materialized control", issue_7_html)
            self.assertIn("wvm-current-256-nz129-f4 forward 1.232×", issue_7_html)
            self.assertIn("Fused split has no reference cell win", issue_7_html)
            self.assertIn("Matched reference finalist campaign", issue_7_html)
            self.assertIn("FFTW plane-major full outer-12: 0.736× geometric across 12 direction-workload cells, 12/12 wins", issue_7_html)
            self.assertIn("FFTW pruned outer-12: 0.694× geometric across 12 direction-workload cells, 12/12 wins", issue_7_html)
            self.assertIn("FFTW pruned outer-4: 1.377× geometric across 12 direction-workload cells, 0/12 wins", issue_7_html)
            self.assertIn("Bounded vDSP native-layout guardrail", issue_7_html)
            self.assertIn("9.118×", issue_7_html)
            self.assertIn("9.309×", issue_7_html)
            self.assertIn("Neither guard workload meets the rule", issue_7_html)
            self.assertIn("vDSP guard runs remain preliminary", issue_7_html)
            self.assertIn("fused or not measured", issue_7_html)
            vdsp_retained_run_html = (output / "runs" / "20260828T061536765767Z-lyra" / "index.html").read_text(encoding="utf-8")
            self.assertIn("Accelerate/vDSP native retained split", vdsp_retained_run_html)
            self.assertIn("persistent native split retained horizontal operator", vdsp_retained_run_html)
            self.assertIn("plane-major-radial-retained-split-complex", vdsp_retained_run_html)
            issue_8_html = issue_8_page.read_text(encoding="utf-8")
            self.assertIn("Bounded common-matrix screen", issue_8_html)
            self.assertIn("deterministic percentile-bootstrap 95% intervals", issue_8_html)
            self.assertIn("Split / complex", issue_8_html)
            self.assertIn("No packing or representation conversion is timed", issue_8_html)
            self.assertIn("20260828T015555382629Z-lyra", issue_8_html)
            self.assertIn("0.540× / 0.496×", issue_8_html)
            self.assertIn("K²-grouped matrix-family penalty", issue_8_html)
            self.assertIn("explicit peak is the harness allocation high-water estimate", issue_8_html)
            self.assertIn("2045", issue_8_html)
            self.assertIn("7486", issue_8_html)
            self.assertIn("2.18× / 5.48×", issue_8_html)
            self.assertIn("20260828T021134464144Z-lyra", issue_8_html)
            self.assertIn("source setup-only 1.22 GiB", issue_8_html)
            self.assertIn("Persistent outer group scheduling", issue_8_html)
            self.assertIn("speedup 4.35× / 4.92×", issue_8_html)
            self.assertIn("speedup 4.44× / 4.76×", issue_8_html)
            self.assertIn("20260828T022558741431Z-lyra", issue_8_html)
            self.assertIn("0.8192 · static-12 · 4.35×", issue_8_html)
            self.assertIn("Finalist portability across fields and vertical depth", issue_8_html)
            self.assertIn("dynamic-16 is fastest in 30 cells and static-12 in 2", issue_8_html)
            self.assertIn("4.03× geometrically, spanning 2.95×–6.00×", issue_8_html)
            self.assertIn("Both finalists beat serial in 64 of 64", issue_8_html)
            self.assertIn("778.64 MiB–25.32 GiB", issue_8_html)
            self.assertIn("20260828T025900272351Z-lyra", issue_8_html)
            self.assertIn("20260828T030055673684Z-lyra", issue_8_html)
            self.assertIn("Every published scheduling run reports zero exactly equivalent adjacent matrix pairs.", issue_8_html)
            issue_9_html = issue_9_page.read_text(encoding="utf-8")
            self.assertIn("Synthetic antialiased spectral-pipeline campaign", issue_9_html)
            self.assertIn("0.699× the WVM direct/no-reorder control", issue_9_html)
            self.assertIn("0.694×–0.702×", issue_9_html)
            self.assertIn("The M4 adoption-statistics gate passes", issue_9_html)
            self.assertIn("explicitly excludes the WVM nonlinear flux calculation", issue_9_html)
            self.assertIn("Raw FFT", issue_9_html)
            self.assertIn("Raw vertical MM", issue_9_html)
            self.assertIn("Movement / rebuild", issue_9_html)
            self.assertIn("Modal work", issue_9_html)
            self.assertIn("Uninstrumented total", issue_9_html)
            pipeline_run_html = (
                output / "runs" / "20260828T141610711264Z-lyra" / "index.html"
            ).read_text(encoding="utf-8")
            self.assertIn("Plane-major fused-split pipeline", pipeline_run_html)
            self.assertIn("Synthetic antialiased spectral pipeline", pipeline_run_html)
            self.assertIn("Mode-keyed modal work", pipeline_run_html)
            self.assertIn("Large four-field nonhydrostatic cohort", issue_9_html)
            self.assertIn("0.896× the WVM control geometrically", issue_9_html)
            self.assertIn("1.040× worst workload", issue_9_html)
            self.assertIn("algorithm-resident ratio is 0.778×", issue_9_html)
            self.assertIn("observed high-water ratio is 0.994×", issue_9_html)
            self.assertIn("0.885×–0.909×", issue_9_html)
            self.assertIn(
                "fused-split overall winner with a documented smallest-case regression",
                issue_9_html,
            )
            self.assertIn("size-dependent algorithm dispatch is not recommended", issue_9_html)
            self.assertNotIn("size-specific dispatch", issue_9_html)
            self.assertIn("wvm-current-256-nz129-f4", issue_9_html)
            self.assertIn("wvm-current-512-nz257-f4", issue_9_html)
            self.assertIn("wvm-large-1024-nz129-f4", issue_9_html)
            self.assertIn("512²/Nz=513/fields=4", issue_9_html)
            self.assertIn("Three-way native-control M4 reference", issue_9_html)
            self.assertIn("0.620× WVM direct geometrically", issue_9_html)
            self.assertIn("0.606×–0.640×", issue_9_html)
            self.assertIn("Against fused split it is 0.691×", issue_9_html)
            self.assertIn("0.684×–0.708×", issue_9_html)
            self.assertIn("algorithm-resident memory is 0.656× WVM direct", issue_9_html)
            self.assertIn("Persistent compiled-engine selection", issue_9_html)
            self.assertIn("MATLAB-owned WVM-native boundary selection", issue_9_html)
            self.assertIn("Component ledger", issue_9_html)
            self.assertIn("Setup, placement, and memory", issue_9_html)
            self.assertIn("No steady-state timed-loop allocation is permitted", issue_9_html)
            issue_17_html = issue_17_page.read_text(encoding="utf-8")
            self.assertIn("Bounded feasibility synthesis", issue_17_html)
            self.assertIn("synthetic quadratic convolution, not WVM’s nonlinear flux", issue_17_html)
            self.assertIn("Analytical 1024² memory projection", issue_17_html)
            self.assertIn("three independently planned four-output applications", issue_17_html)
            self.assertIn("WVM-derived four-target horizontal-advection screen", issue_17_html)
            self.assertIn("FFTW++ is 0.889× the explicit FFTW time", issue_17_html)
            self.assertIn("0.661× its counted algorithm-resident storage", issue_17_html)
            self.assertIn("1024 × 1024", issue_17_html)
            self.assertIn("not a reference result or full-flux claim", issue_17_html)
            self.assertIn("Rotated finalist-only M4 reference-depth campaign", issue_17_html)
            self.assertIn("FFTW++ is 0.902× explicit FFTW geometrically", issue_17_html)
            self.assertIn("0.900×–0.917×", issue_17_html)
            self.assertIn("campaign is reference-quality", issue_17_html)
            self.assertIn("does not meet the M4 adoption threshold", issue_17_html)
            self.assertIn("reference-depth conclusion", issue_17_html)
            issue_18_html = issue_18_page.read_text(encoding="utf-8")
            self.assertIn("First vertically batched composition", issue_18_html)
            self.assertIn("15 ready retained and vertically truncated modal inputs", issue_18_html)
            self.assertIn("complete nonlinear flux remain excluded", issue_18_html)
            self.assertIn("FFTW++ is 0.971× explicit", issue_18_html)
            self.assertIn("algorithm-resident storage is 0.995×", issue_18_html)
            self.assertIn("preliminary continuation gate passes", issue_18_html)
            self.assertIn("0.9000 multi-workload adoption threshold", issue_18_html)
            self.assertIn("169–171 ms is movement alone", issue_18_html)
            self.assertIn("20260828T222553Z-issue18-n256-nz129-explicit-parallel", issue_18_html)
            self.assertIn("20260828T222558Z-issue18-n256-nz129-fftwpp-parallel", issue_18_html)
            self.assertIn("24 reference run(s) currently contribute", issue_18_html)
            self.assertIn("Four-workload composed M4 reference campaign", issue_18_html)
            self.assertIn("FFTW++ is 0.952× explicit geometrically", issue_18_html)
            self.assertIn("0.885×–1.063×", issue_18_html)
            self.assertIn("worst workload of 0.980×", issue_18_html)
            self.assertIn("Algorithm-resident storage is 0.997×", issue_18_html)
            self.assertIn("maximum mode-keyed error is 1.305e-15", issue_18_html)
            self.assertIn("does not meet the M4 adoption threshold", issue_18_html)
            self.assertIn("10% geometric time improvement", issue_18_html)
            self.assertIn("confidence excluding a tie", issue_18_html)
            self.assertIn("20% algorithm-resident-memory reduction", issue_18_html)
            self.assertIn(
                "wvm-current-512-nz257-f4: 0.778×–1.136×",
                issue_18_html,
            )
            self.assertIn(
                "wvm-large-512-nz513-f4: 0.860×–1.217×",
                issue_18_html,
            )
            self.assertIn("No size-dependent dispatch is permitted", issue_18_html)
            self.assertIn("20260828T231215Z-issue18-n512-nz513-explicit-parallel", issue_18_html)
            self.assertIn("20260829T015353Z-issue18-n1024-nz129-fftwpp-parallel", issue_18_html)
            large_f4_run_page = (
                output / "runs" / "20260828T150329017392Z-lyra" / "index.html"
            )
            self.assertTrue(large_f4_run_page.is_file())
            large_f4_run_html = large_f4_run_page.read_text(encoding="utf-8")
            self.assertIn("Plane-major fused-split pipeline", large_f4_run_html)
            self.assertIn("spectralPipelineEstimatedExplicitPeak", (
                output / "artifacts" / "20260828T150329017392Z-lyra" /
                "result.json"
            ).read_text(encoding="utf-8"))
            vertical_run_html = (output / "runs" / "20260828T015555382629Z-lyra" / "index.html").read_text(encoding="utf-8")
            self.assertIn("Raw vertical GEMM", vertical_run_html)
            self.assertIn("Accelerate split dgemm", vertical_run_html)
            self.assertIn("L2", vertical_run_html)
            grouped_run_html = (output / "runs" / "20260828T021134464144Z-lyra" / "index.html").read_text(encoding="utf-8")
            self.assertIn("accelerate-two-dgemm-k2-group-loop-split", grouped_run_html)
            self.assertIn("synthetic-k2-grouped-orthonormal-dct2-pair-rotation-v1", grouped_run_html)
            scheduled_run_html = (output / "runs" / "20260828T022558741431Z-lyra" / "index.html").read_text(encoding="utf-8")
            self.assertIn("accelerate-two-dgemm-k2-group-outer-dynamic-split", scheduled_run_html)
            self.assertIn("public variable-size grouped BLAS batch API=unavailable", scheduled_run_html)
            self.assertIn("exactly equivalent adjacent matrix pairs=0", scheduled_run_html)
            self.assertIn("empty group dispatch", scheduled_run_html)
            issue_13_html = issue_13_page.read_text(encoding="utf-8")
            self.assertIn("First bounded MATLAB-style baseline", issue_13_html)
            self.assertIn("The latest same-commit cohort contains 8 runs across 4 profiles", issue_13_html)
            self.assertIn("Dynamic scheduling wins 7 cells and static 1", issue_13_html)
            self.assertIn("split storage wins 8 and interleaved 0", issue_13_html)
            self.assertIn("Movement costs span 1.095×–6.264×", issue_13_html)
            self.assertIn("32 of 32 representation/schedule/direction cells", issue_13_html)
            self.assertIn("32 cross at R=2", issue_13_html)
            self.assertIn("1.47 GiB–4.47 GiB", issue_13_html)
            self.assertIn("Direct WVM-order no-reorder increment", issue_13_html)
            self.assertIn("direct no-reorder wins 12", issue_13_html)
            self.assertIn("2.712× at the primitive boundary and 0.594× for the one-shot total", issue_13_html)
            self.assertIn("fields=1: 0/4 wins, 1.237× geometric", issue_13_html)
            self.assertIn("fields=3: 8/8 wins, 0.474× geometric", issue_13_html)
            self.assertIn("fields=4: 4/4 wins, 0.448× geometric", issue_13_html)
            self.assertIn("avoided movement offsets the efficiency loss", issue_13_html)
            self.assertIn("11439–11439 GEMM calls", issue_13_html)
            self.assertIn("20260828T032621211425Z-lyra", issue_13_html)
            self.assertIn("20260828T032729237033Z-lyra", issue_13_html)
            ordering_run_html = (output / "runs" / "20260828T032729237033Z-lyra" / "index.html").read_text(encoding="utf-8")
            self.assertIn("WVM retained gather and radial pack", ordering_run_html)
            self.assertIn("persistent-compact-boundary-once", ordering_run_html)
            direct_ordering_run_html = (output / "runs" / "20260828T034224811058Z-lyra" / "index.html").read_text(encoding="utf-8")
            self.assertIn("Direct WVM-order Accelerate zgemm", direct_ordering_run_html)
            self.assertIn("one-shot no-reorder vertical projection", direct_ordering_run_html)
            self.assertIn("persistent-provider-order-no-movement", direct_ordering_run_html)
            self.assertIn("GEMM calls per execution", direct_ordering_run_html)
            self.assertIn("orderingPackingEstimatedExplicitPeak", (output / "artifacts" / "20260828T032729237033Z-lyra" / "result.json").read_text(encoding="utf-8"))
            self.assertIn("in-place", float32_page.read_text(encoding="utf-8").lower())
            issue_16_html = issue_16_page.read_text(encoding="utf-8")
            self.assertIn("Streaming pruned-to-compact-split pipeline", issue_16_html)
            self.assertIn("worker-local-plane", issue_16_html)
            self.assertIn("18 reference run(s) currently contribute", issue_16_html)
            self.assertNotIn("No reference run has been published", issue_16_html)
            self.assertIn("Streaming pruned-to-compact-split preliminary screen", issue_16_html)
            self.assertIn("0.952× geometrically", issue_16_html)
            self.assertIn("two large decision cases are 1.080×", issue_16_html)
            self.assertIn("Algorithm-resident memory is 0.832×", issue_16_html)
            self.assertIn("Maximum correctness error is 1.026e-15", issue_16_html)
            self.assertIn("advances on memory within the time bound", issue_16_html)
            self.assertIn("Cache-local compact-tile optimization screen", issue_16_html)
            self.assertIn("Tile 16 is the preregistered uniform winner", issue_16_html)
            self.assertIn("0.704× fused split geometrically", issue_16_html)
            self.assertIn("0.809× across the two large cases", issue_16_html)
            self.assertIn("0.758× overall", issue_16_html)
            self.assertIn("Algorithm-resident memory remains 0.851×", issue_16_html)
            self.assertIn("1.026e-15", issue_16_html)
            self.assertIn("32-mode cache-blocked transpose", issue_16_html)
            self.assertIn("Fixed tile-16 reference campaign", issue_16_html)
            self.assertIn("0.752× fused split geometrically", issue_16_html)
            self.assertIn("0.734×–0.771×", issue_16_html)
            self.assertIn("worst profile remains a win at 0.865×", issue_16_html)
            self.assertIn("Algorithm-resident memory is 0.851×", issue_16_html)
            self.assertIn("observed process high water is 0.894×", issue_16_html)
            self.assertIn("supersedes fused split as the synthetic-pipeline winner", issue_16_html)
            self.assertIn("The speedup resides in the retained horizontal operator", issue_16_html)
            self.assertIn("no size-dependent selection or dispatch was permitted", issue_16_html)
            streaming_run_page = (
                output / "runs" / "20260828T153842645834Z-lyra" / "index.html"
            )
            self.assertTrue(streaming_run_page.is_file())
            streaming_run_html = streaming_run_page.read_text(encoding="utf-8")
            self.assertIn("Streaming pruned compact-split pipeline", streaming_run_html)
            self.assertIn("selected-kx complex column FFTs", streaming_run_html)
            self.assertIn("streamed radial direct split write", streaming_run_html)
            self.assertIn("one worker-local half-spectrum plane per outer worker", streaming_run_html)
            tiled_streaming_run_page = (
                output / "runs" / "20260828T161727772904Z-lyra" /
                "index.html"
            )
            self.assertTrue(tiled_streaming_run_page.is_file())
            tiled_streaming_run_html = tiled_streaming_run_page.read_text(
                encoding="utf-8"
            )
            self.assertIn("Streaming pruned compact-split pipeline", tiled_streaming_run_html)
            self.assertIn(
                "plane-major compact staging and blocked split transpose",
                tiled_streaming_run_html,
            )
            self.assertIn("streaming compact tile width=16", tiled_streaming_run_html)
            reference_streaming_run_page = (
                output / "runs" / "20260828T165627728052Z-lyra" /
                "index.html"
            )
            self.assertTrue(reference_streaming_run_page.is_file())
            reference_streaming_run_html = reference_streaming_run_page.read_text(
                encoding="utf-8"
            )
            self.assertIn("reference", reference_streaming_run_html)
            self.assertIn("streaming compact tile width=16", reference_streaming_run_html)
            native_bridge_run_page = (
                output / "runs" / "20260828T175442514946Z-lyra" /
                "index.html"
            )
            self.assertTrue(native_bridge_run_page.is_file())
            native_bridge_run_html = native_bridge_run_page.read_text(
                encoding="utf-8"
            )
            self.assertIn("WVM direct/no-reorder pipeline", native_bridge_run_html)
            self.assertIn("wvm-frequency-major-interleaved", native_bridge_run_html)
            decision_html = decision_page.read_text(encoding="utf-8")
            self.assertIn("M4 deployment decision recorded", decision_html)
            self.assertIn("Persistent compiled spectral engine", decision_html)
            self.assertIn("MATLAB + compiled core with WVM-native arrays", decision_html)
            self.assertIn("0.620× WVM direct", decision_html)
            self.assertIn("0.691× fused split", decision_html)
            self.assertIn("general-Mac default recommendation", decision_html)
            self.assertNotIn("Adoption decision not yet ready", decision_html)
            self.assertIn("predates the explicit placement contract", legacy_page.read_text(encoding="utf-8"))
            self.assertEqual((PUBLISHED / "m4-max-quick-20260827.json").read_bytes(), result_artifact.read_bytes())
            self.assertEqual((PUBLISHED / "m4-max-quick-20260827.csv").read_bytes(), samples_artifact.read_bytes())
            self.assertTrue((output / "results" / "m4-max-quick-20260827.json").is_file())
            self.assertTrue((output / "results" / "m4-max-quick-20260827.csv").is_file())
            self.assertTrue((output / "schema" / "spectral-kernel-benchmark-v1.schema.json").is_file())
            self.assertTrue((output / "schema" / "spectral-kernel-publication-catalog-v1.schema.json").is_file())
            self.assertTrue((output / "catalog.json").is_file())
            self.assertTrue((output / ".nojekyll").is_file())
            assert_internal_links_resolve(self, output)

            subprocess.run(command, cwd=REPOSITORY_ROOT, check=True, capture_output=True, text=True)
            self.assertEqual(first_hashes, content_hashes(output))


if __name__ == "__main__":
    unittest.main()
