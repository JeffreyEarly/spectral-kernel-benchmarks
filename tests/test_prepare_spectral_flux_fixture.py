import hashlib
import json
import struct
import sys
import tempfile
import unittest
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "tools"))

from prepare_spectral_flux_fixture import (  # noqa: E402
    DERIVATIVE_CONVENTION,
    FixtureError,
    GROUP_RULE,
    INPUT_FAMILIES,
    INPUT_NAMES,
    MAGIC,
    TARGET_FAMILIES,
    TARGET_NAMES,
    expected_payloads,
    prepare,
    retained_modes,
)


class SpectralFluxFixturePreparationTests(unittest.TestCase):
    def make_fixture(self, root: Path, *, split_repeated_key: bool = False) -> Path:
        fixture = root / "fixture"
        fixture.mkdir()
        nx = ny = 8
        nz = 7
        nj = 4
        modes = retained_modes(nx, ny)
        nkl = len(modes)
        keys = [k * k + l * l for k, l in modes]
        distinct: list[int] = []
        indices: list[int] = []
        for key in keys:
            if (not distinct or distinct[-1] != key or
                    (split_repeated_key and key == 1)):
                distinct.append(key)
            indices.append(len(distinct) - 1)
        group_count = len(distinct)
        specifications = expected_payloads(
            nx, ny, nz, nkl, nj, group_count
        )
        values = {
            "horizontal-mode-keys.i32le": struct.pack(
                f"<{2 * nkl}i", *(value for mode in modes for value in mode)
            ),
            "vertical-mode-keys.i32le": struct.pack("<4i", 0, 1, 2, 3),
            "mode-group-indices.u32le": struct.pack(
                f"<{nkl}I", *indices
            ),
            "group-keys.u64le": struct.pack(
                f"<{group_count}Q", *distinct
            ),
        }
        for specification in specifications[4:]:
            values[specification["path"]] = bytes(
                specification["elementBytes"]
                * __import__("math").prod(specification["shape"])
            )
        payloads = []
        for specification in specifications:
            path = specification["path"]
            content = values[path]
            (fixture / path).write_bytes(content)
            payloads.append(
                {
                    "path": path,
                    "byteCount": len(content),
                    "elementType": specification["type"],
                    "logicalAxes": (
                        specification["axes"][0]
                        if len(specification["axes"]) == 1
                        else specification["axes"]
                    ),
                    "shape": (
                        specification["shape"][0]
                        if len(specification["shape"]) == 1
                        else specification["shape"]
                    ),
                    "stridesElements": (
                        specification["strides"][0]
                        if len(specification["strides"]) == 1
                        else specification["strides"]
                    ),
                    "isComplex": specification["complex"],
                    "sha256": hashlib.sha256(content).hexdigest(),
                }
            )
        manifest = {
            "schema": "spectral-flux-fixture-v1",
            "fixtureId": "format-test-only",
            "status": "authoritative-wvm-export",
            "authoritative": True,
            "createdAtUtc": "2026-08-29T00:00:00Z",
            "numericType": "float64",
            "byteOrder": "little-endian",
            "provenance": {
                "repository": "JeffreyEarly/wave-vortex-model",
                "commit": "1" * 40,
                "tree": "2" * 40,
                "dirtyTree": False,
                "matlabVersion": "test",
                "matlabRelease": "test",
                "architecture": "test",
            },
            "generator": {
                "path": "Benchmarks/exportSpectralFluxFixture.m",
                "command": (
                    "exportSpectralFluxFixture(outputDirectory,"
                    "Nxyz=[8 8 7],Lxyz=[15000 15000 1300],latitude=45,seed=19)"
                ),
            },
            "workload": {
                "Nx": nx,
                "Ny": ny,
                "Nz": nz,
                "H": (nx // 2 + 1) * ny,
                "Nkl": nkl,
                "Nj": nj,
                "fields": 4,
                "Lxyz": [15000, 15000, 1300],
                "latitude": 45,
            },
            "retention": {
                "horizontalPolicy": "radial-two-thirds",
                "horizontalCutoffFraction": 2.0 / 3.0,
                "verticalPolicy": "floor(2*(Nz-1)/3)",
                "verticalRetainedFraction": 2.0 / 3.0,
            },
            "modeOrder": {
                "logicalAxes": ["k", "l", "j", "field"],
                "horizontal": "WVM radial magnitude then k then l",
                "vertical": "j ascending from zero",
            },
            "normalization": {
                "horizontalForward": "raw FFT coefficients",
                "horizontalInverse": (
                    "raw inverse FFT followed by division by Nx*Ny before each physical factor"
                ),
                "pointwiseScale": 1.0 / float((nx * ny) ** 2),
            },
            "operatorContract": {
                "familyIds": ["wave-f", "wave-g"],
                "inputFieldNames": INPUT_NAMES,
                "inputFieldFamilies": INPUT_FAMILIES,
                "targetNames": TARGET_NAMES,
                "targetFieldFamilies": TARGET_FAMILIES,
                "groupCount": group_count,
                "groupRule": GROUP_RULE,
            },
            "derivativeConvention": DERIVATIVE_CONVENTION,
            "oracle": {
                "identity": (
                    "WVM MATLAB Fourier transforms plus WVM Fw/Gw projection matrices "
                    "and direct -(U*qx+V*qy+W*qz) evaluation"
                ),
                "maximumScaleNormalizedErrorTolerance": 1.0e-12,
                "relativeL2ErrorTolerance": 1.0e-12,
            },
            "payloads": payloads,
        }
        (fixture / "manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
        return fixture

    def test_valid_fixture_prepares_once_with_sha256_identity(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = self.make_fixture(root)
            manifest_path = fixture / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            for payload in manifest["payloads"]:
                payload["byteCount"] = float(payload["byteCount"])
            manifest_path.write_text(
                json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
            )
            output = root / "prepared.bin"
            summary = prepare(fixture, output)
            self.assertTrue(output.read_bytes().startswith(MAGIC))
            self.assertRegex(summary["fixtureHash"], r"^sha256:[0-9a-f]{64}$")
            self.assertEqual("1" * 40, summary["waveVortexModelCommit"])
            with self.assertRaisesRegex(FixtureError, "refusing to overwrite"):
                prepare(fixture, output)

    def test_payload_hash_mismatch_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = self.make_fixture(root)
            payload = fixture / "modal-inputs.c128le"
            content = bytearray(payload.read_bytes())
            content[-1] = 1
            payload.write_bytes(content)
            with self.assertRaisesRegex(FixtureError, "SHA-256"):
                prepare(fixture, root / "prepared.bin")

    def test_repeated_integer_keys_preserve_distinct_wvm_groups(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = self.make_fixture(root, split_repeated_key=True)
            summary = prepare(fixture, root / "prepared.bin")
            self.assertEqual("format-test-only", summary["fixtureId"])

    def test_dirty_tree_cannot_claim_authoritative_status(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = self.make_fixture(root)
            manifest_path = fixture / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["provenance"]["dirtyTree"] = True
            manifest_path.write_text(
                json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
            )
            with self.assertRaisesRegex(FixtureError, "clean WVM tree"):
                prepare(fixture, root / "prepared.bin")


if __name__ == "__main__":
    unittest.main()
