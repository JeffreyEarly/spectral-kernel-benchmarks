import hashlib
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
GENERATOR = REPOSITORY_ROOT / "tools" / "build_site.py"


def content_hashes(root: Path) -> dict[str, str]:
    return {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


class BuildSiteTests(unittest.TestCase):
    def test_builds_deterministic_dashboard_and_downloads(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            output = Path(temporary_directory) / "site"
            command = [sys.executable, str(GENERATOR), "--output", str(output)]
            subprocess.run(command, cwd=REPOSITORY_ROOT, check=True, capture_output=True, text=True)
            first_hashes = content_hashes(output)

            index = (output / "index.html").read_text(encoding="utf-8")
            run_page = output / "runs" / "m4-max-quick-20260827.html"
            self.assertIn("Which spectral kernels are actually fastest?", index)
            self.assertIn("Apple M4 Max", index)
            self.assertIn("Raw FFT", index)
            self.assertTrue(run_page.is_file())
            self.assertTrue((output / "results" / "m4-max-quick-20260827.json").is_file())
            self.assertTrue((output / "results" / "m4-max-quick-20260827.csv").is_file())
            self.assertTrue((output / "schema" / "spectral-kernel-benchmark-v1.schema.json").is_file())
            self.assertTrue((output / "assets" / "favicon.svg").is_file())
            self.assertTrue((output / ".nojekyll").is_file())

            subprocess.run(command, cwd=REPOSITORY_ROOT, check=True, capture_output=True, text=True)
            self.assertEqual(first_hashes, content_hashes(output))


if __name__ == "__main__":
    unittest.main()
