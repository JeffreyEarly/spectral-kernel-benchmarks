import hashlib
import json
import math
import struct
import sys
import tempfile
import unittest
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "tools"))

from prepare_constant_stratification_flux_fixture import (  # noqa: E402
    AUDITED_SOURCES,
    AUDITED_WVM_COMMIT,
    FixtureError,
    MAGIC,
    expected_payloads,
    prepare,
    retained_modes,
)


class ConstantStratificationFixturePreparationTests(unittest.TestCase):
    def make_fixture(self, root: Path, *, reverse_modes: bool = False) -> Path:
        fixture = root / "fixture"
        fixture.mkdir()
        nx = ny = 8
        nz = 7
        nj = 4
        canonical_modes = retained_modes(nx, ny)
        source_modes = list(reversed(canonical_modes)) if reverse_modes else canonical_modes
        nkl = len(source_modes)
        mode_keys = struct.pack(
            f"<{2 * nkl}i",
            *(coordinate for mode in source_modes for coordinate in mode),
        )

        state_scalars: list[float] = []
        flux_scalars: list[float] = []
        for k, l in source_modes:
            mode_scalar_offset = len(state_scalars)
            for field in range(3):
                for j in range(nj):
                    real = 0.01 * (1 + j + 7 * field + 13 * (k + 4) + 17 * (l + 4))
                    imaginary = 0.02 * (1 + j + field)
                    if k == 0 and l == 0:
                        if field == 1:
                            ap_offset = mode_scalar_offset + 2 * j
                            real = state_scalars[ap_offset]
                            imaginary = -state_scalars[ap_offset + 1]
                        elif field == 2:
                            imaginary = 0.0
                    state_scalars.extend((real, imaginary))
                    flux_scalars.extend((real * 0.25, imaginary * 0.25))
        payload_values = {
            "horizontal-mode-keys.i32le": mode_keys,
            "modal-state.c128le": struct.pack(
                f"<{len(state_scalars)}d", *state_scalars
            ),
            "expected-modal-flux.c128le": struct.pack(
                f"<{len(flux_scalars)}d", *flux_scalars
            ),
        }
        payload_records = []
        for specification in expected_payloads(nkl, nj):
            content = payload_values[specification["path"]]
            (fixture / specification["path"]).write_bytes(content)
            payload_records.append({
                "path": specification["path"],
                "byteCount": len(content),
                "elementType": specification["type"],
                "logicalAxes": specification["axes"],
                "shape": specification["shape"],
                "stridesElements": specification["strides"],
                "isComplex": specification["complex"],
                "sha256": hashlib.sha256(content).hexdigest(),
            })

        manifest = {
            "schema": "constant-stratification-flux-fixture-v1",
            "fixtureId": "constant-format-test-only",
            "status": "authoritative-wvm-export",
            "authoritative": True,
            "createdAtUtc": "2026-08-30T00:00:00Z",
            "numericType": "float64",
            "byteOrder": "little-endian",
            "provenance": {
                "repository": "JeffreyEarly/wave-vortex-model",
                "commit": AUDITED_WVM_COMMIT,
                "tree": "1" * 40,
                "dirtyTree": False,
            },
            "generator": {
                "path": "tools/exportConstantStratificationFluxFixture.m",
                "repository": "JeffreyEarly/spectral-kernel-benchmarks",
                "commit": "2" * 40,
                "tree": "3" * 40,
                "dirtyTree": False,
                "command": "exportConstantStratificationFluxFixture(...)"
            },
            "auditedSources": [
                {"path": path, "sha256": digest}
                for path, digest in AUDITED_SOURCES.items()
            ],
            "compiledBackend": {
                "module": {
                    "sha256": "4" * 64,
                    "identityValidated": True,
                }
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
            },
            "physicalConfiguration": {
                "N0": 5.2e-3,
                "rotationRate": 7.2921e-5,
                "latitude": 45,
                "g": 9.81,
                "isHydrostatic": False,
                "shouldAntialias": True,
                "elapsedTime": 123.5,
                "matlabModalNormalizationGravity": 9.81,
            },
            "retention": {
                "horizontalPolicy": "radial-two-thirds",
                "horizontalCutoffFraction": 2.0 / 3.0,
                "verticalPolicy": "floor(2*(Nz-1)/3)",
                "verticalRetainedFraction": nj / float(nz - 1),
            },
            "modeOrder": {
                "logicalAxes": ["k", "l", "j", "coefficient"],
                "horizontal": "WVM radial magnitude then k then l",
                "vertical": "j ascending from zero",
                "coefficientNames": ["Ap", "Am", "A0"],
                "fluxNames": ["Fp", "Fm", "F0"],
            },
            "normalization": {
                "pointwiseScale": 1.0,
            },
            "coefficientContract": {
                "identity": (
                    "WVM constant-stratification natural-dimensional-prescaled "
                    "nonlinear flux"
                ),
            },
            "oracle": {
                "identity": (
                    "WVM MATLAB nonlinearFlux cross-checked against the compiled "
                    "WVTransformConstantStratificationKernel nonlinearFlux"
                ),
                "maximumScaleNormalizedError": 2.0e-14,
                "relativeL2Error": 3.0e-14,
                "maximumScaleNormalizedErrorTolerance": 1.0e-11,
                "relativeL2ErrorTolerance": 1.0e-11,
                "benchmarkMaximumScaleNormalizedErrorTolerance": 1.0e-12,
                "benchmarkRelativeL2ErrorTolerance": 1.0e-12,
            },
            "payloads": payload_records,
        }
        (fixture / "manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
        return fixture

    def test_valid_fixture_prepares_once_with_sha256_identity(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = self.make_fixture(root)
            output = root / "prepared.bin"
            summary = prepare(fixture, output)
            self.assertTrue(output.read_bytes().startswith(MAGIC))
            self.assertRegex(summary["fixtureHash"], r"^sha256:[0-9a-f]{64}$")
            self.assertEqual(AUDITED_WVM_COMMIT,
                             summary["waveVortexModelCommit"])
            with self.assertRaisesRegex(FixtureError, "refusing to overwrite"):
                prepare(fixture, output)

    def test_reversed_source_mode_order_is_prepared_canonically(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = self.make_fixture(root, reverse_modes=True)
            output = root / "prepared.bin"
            prepare(fixture, output)
            self.assertGreater(output.stat().st_size, 2 * 1024)

    def test_payload_hash_mismatch_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = self.make_fixture(root)
            payload = fixture / "expected-modal-flux.c128le"
            content = bytearray(payload.read_bytes())
            content[-1] ^= 1
            payload.write_bytes(content)
            with self.assertRaisesRegex(FixtureError, "SHA-256"):
                prepare(fixture, root / "prepared.bin")

    def test_dirty_wvm_or_generator_tree_is_rejected(self) -> None:
        for key in ("provenance", "generator"):
            with self.subTest(key=key), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                fixture = self.make_fixture(root)
                manifest_path = fixture / "manifest.json"
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                manifest[key]["dirtyTree"] = True
                manifest_path.write_text(
                    json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
                )
                with self.assertRaisesRegex(FixtureError, "clean"):
                    prepare(fixture, root / "prepared.bin")

    def test_wrong_audited_source_hash_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = self.make_fixture(root)
            manifest_path = fixture / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["auditedSources"][0]["sha256"] = "f" * 64
            manifest_path.write_text(
                json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
            )
            with self.assertRaisesRegex(FixtureError, "audited source hashes"):
                prepare(fixture, root / "prepared.bin")


if __name__ == "__main__":
    unittest.main()
