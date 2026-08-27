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
            issue_3_page = output / "experiments" / "issue-003-fftw-production-baseline" / "index.html"
            issue_5_page = output / "experiments" / "issue-005-vdsp-native-baseline" / "index.html"
            issue_6_page = output / "experiments" / "issue-006-vdsp-batching-scheduling" / "index.html"
            float32_page = output / "experiments" / "issue-015-float32-spectral-kernel-screen" / "index.html"
            methods_page = output / "methods" / "operators-and-representations" / "index.html"
            decision_page = output / "decisions" / "v1" / "index.html"
            result_artifact = output / "artifacts" / "20260827T185428Z-lyra" / "result.json"
            samples_artifact = output / "artifacts" / "20260827T185428Z-lyra" / "samples.csv"

            self.assertIn("Which spectral kernels are actually fastest?", index)
            self.assertIn("preliminary", index)
            self.assertTrue(legacy_page.is_file())
            self.assertTrue(canonical_page.is_file())
            self.assertTrue(issue_3_page.is_file())
            self.assertTrue(issue_5_page.is_file())
            self.assertTrue(issue_6_page.is_file())
            self.assertTrue(float32_page.is_file())
            self.assertTrue(methods_page.is_file())
            self.assertTrue(decision_page.is_file())
            self.assertIn("What this experiment measures", issue_3_page.read_text(encoding="utf-8"))
            issue_5_html = issue_5_page.read_text(encoding="utf-8")
            self.assertIn("No reference run has been published", issue_5_html)
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
