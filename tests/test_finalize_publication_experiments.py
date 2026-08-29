import copy
import sys
import unittest
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "tools"))

import finalize_publication_experiments as finalize  # noqa: E402


class FinalizePublicationExperimentsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.catalog = {
            "experiments": [
                {"id": "issue-a", "issue": 3, "phase": "collecting"},
                {"id": "issue-b", "issue": 13, "phase": "collecting"},
            ],
            "runs": [
                {
                    "id": "run-1",
                    "incrementId": "shared-reference-v1",
                    "issues": [3],
                    "experiments": ["issue-a"],
                }
            ],
        }

    def test_completes_and_associates_without_touching_artifacts(self) -> None:
        catalog = copy.deepcopy(self.catalog)
        summary = finalize.finalize_catalog(
            catalog,
            ["issue-a", "issue-b"],
            [("issue-b", "shared-reference-v1")],
        )
        self.assertEqual(
            {"completedExperiments": 2, "newAssociations": 1}, summary
        )
        self.assertEqual(
            ["complete", "complete"],
            [experiment["phase"] for experiment in catalog["experiments"]],
        )
        self.assertEqual([3, 13], catalog["runs"][0]["issues"])
        self.assertEqual(
            ["issue-a", "issue-b"], catalog["runs"][0]["experiments"]
        )

    def test_repeated_association_is_idempotent(self) -> None:
        catalog = copy.deepcopy(self.catalog)
        finalize.finalize_catalog(
            catalog, ["issue-b"], [("issue-b", "shared-reference-v1")]
        )
        summary = finalize.finalize_catalog(
            catalog, ["issue-b"], [("issue-b", "shared-reference-v1")]
        )
        self.assertEqual(0, summary["newAssociations"])

    def test_missing_increment_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "has no published runs"):
            finalize.finalize_catalog(
                copy.deepcopy(self.catalog), [], [("issue-b", "missing")]
            )


if __name__ == "__main__":
    unittest.main()
