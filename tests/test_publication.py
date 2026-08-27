import copy
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "tools"))

from publication import load_and_validate  # noqa: E402


PUBLISHED = REPOSITORY_ROOT / "results" / "published"


class PublicationValidationTests(unittest.TestCase):
    def copy_publication(self, root: Path) -> Path:
        destination = root / "published"
        shutil.copytree(PUBLISHED, destination)
        return destination

    def load_catalog(self, published: Path) -> dict:
        return json.loads((published / "catalog.json").read_text(encoding="utf-8"))

    def write_catalog(self, published: Path, catalog: dict) -> None:
        (published / "catalog.json").write_text(json.dumps(catalog, indent=2) + "\n", encoding="utf-8")

    def test_current_catalog_and_grandfathered_hashes_validate(self) -> None:
        catalog, bundles = load_and_validate(PUBLISHED)
        self.assertEqual("spectral-kernel-publication-catalog-v1", catalog["schema"])
        self.assertEqual(13, len(bundles))
        grandfathered = next(bundle for bundle in bundles if bundle.publication["grandfathered"])
        self.assertEqual("20260827T185428Z-lyra", grandfathered.publication["id"])
        self.assertEqual("preliminary", grandfathered.publication["status"])
        self.assertEqual(11, len(catalog["experiments"]))

    def test_duplicate_run_id_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            published = self.copy_publication(Path(temporary_directory))
            catalog = self.load_catalog(published)
            catalog["runs"].append(copy.deepcopy(catalog["runs"][0]))
            self.write_catalog(published, catalog)
            with self.assertRaisesRegex(ValueError, "run ids: values must be unique"):
                load_and_validate(published)

    def test_modified_artifact_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            published = self.copy_publication(Path(temporary_directory))
            result = published / "m4-max-quick-20260827.json"
            result.write_bytes(result.read_bytes() + b"\n")
            with self.assertRaisesRegex(ValueError, "immutable artifact changed"):
                load_and_validate(published)

    def test_missing_pair_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            published = self.copy_publication(Path(temporary_directory))
            (published / "m4-max-quick-20260827.csv").unlink()
            with self.assertRaisesRegex(ValueError, "published artifact is missing"):
                load_and_validate(published)

    def test_unlisted_artifact_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            published = self.copy_publication(Path(temporary_directory))
            (published / "orphan.csv").write_text("run_id\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "missing from the catalog"):
                load_and_validate(published)

    def test_new_run_must_use_run_id_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            published = self.copy_publication(Path(temporary_directory))
            catalog = self.load_catalog(published)
            catalog["runs"][0]["grandfathered"] = False
            self.write_catalog(published, catalog)
            with self.assertRaisesRegex(ValueError, "new bundles must use"):
                load_and_validate(published)

    def test_dirty_run_cannot_be_reference_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            published = self.copy_publication(Path(temporary_directory))
            catalog = self.load_catalog(published)
            catalog["runs"][0]["status"] = "reference"
            catalog["runs"][0]["statusReason"] = "Attempted promotion."
            self.write_catalog(published, catalog)
            with self.assertRaisesRegex(ValueError, "reference evidence must come from a clean"):
                load_and_validate(published)


if __name__ == "__main__":
    unittest.main()
