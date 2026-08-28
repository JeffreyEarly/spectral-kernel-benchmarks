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
            issue_8_page = output / "experiments" / "issue-008-vertical-projection-gemm" / "index.html"
            issue_12_page = output / "experiments" / "issue-012-pruned-horizontal-transforms" / "index.html"
            issue_13_page = output / "experiments" / "issue-013-ordering-packing-crossover" / "index.html"
            float32_page = output / "experiments" / "issue-015-float32-spectral-kernel-screen" / "index.html"
            methods_page = output / "methods" / "operators-and-representations" / "index.html"
            decision_page = output / "decisions" / "v1" / "index.html"
            result_artifact = output / "artifacts" / "20260827T185428Z-lyra" / "result.json"
            samples_artifact = output / "artifacts" / "20260827T185428Z-lyra" / "samples.csv"

            self.assertIn("Which spectral kernels are actually fastest?", index)
            self.assertIn("preliminary", index)
            self.assertTrue(legacy_page.is_file())
            self.assertTrue(canonical_page.is_file())
            self.assertTrue(split_run_page.is_file())
            self.assertTrue(issue_3_page.is_file())
            self.assertTrue(issue_4_page.is_file())
            self.assertTrue(issue_5_page.is_file())
            self.assertTrue(issue_6_page.is_file())
            self.assertTrue(issue_8_page.is_file())
            self.assertTrue(issue_12_page.is_file())
            self.assertTrue(issue_13_page.is_file())
            self.assertTrue(float32_page.is_file())
            self.assertTrue(methods_page.is_file())
            self.assertTrue(decision_page.is_file())
            self.assertIn("What this experiment measures", issue_3_page.read_text(encoding="utf-8"))
            issue_4_html = issue_4_page.read_text(encoding="utf-8")
            self.assertIn("Reproducible Pareto screen", issue_4_html)
            self.assertIn("deterministic percentile-bootstrap 95% intervals", issue_4_html)
            self.assertIn("infeasible within its planning budget", issue_4_html)
            self.assertIn("wvm-guru64-measure-unaligned-cold", issue_4_html)
            self.assertIn("20260827T223146704900Z-lyra", issue_4_html)
            self.assertIn("7 workload-candidates are Pareto", issue_4_html)
            self.assertIn("3 are dominated", issue_4_html)
            self.assertIn("0 are marked infeasible", issue_4_html)
            self.assertIn("Paired split-versus-interleaved increment", issue_4_html)
            self.assertIn("Split-layout diagnostic", issue_4_html)
            self.assertIn("Raw CV interleaved / split", issue_4_html)
            self.assertIn("Planning memory", issue_4_html)
            self.assertIn("1.217× forward and 1.166× inverse", issue_4_html)
            self.assertIn("Append-only interleaved strategy archive", issue_4_html)
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
            self.assertIn("Empty dispatch", issue_12_html)
            self.assertIn("Scratch aggregate / max shard", issue_12_html)
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
            self.assertIn("Direct-persistent therefore remains the only issue #6 candidate carried forward", issue_6_html)
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
            self.assertIn("Adoption decision not yet ready", decision_page.read_text(encoding="utf-8"))
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
