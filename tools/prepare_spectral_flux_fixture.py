#!/usr/bin/env python3
"""Validate a WVM spectral-flux export and prepare one strict skbench input."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import pathlib
import re
import struct
import sys
from typing import Any


SCHEMA = "spectral-flux-fixture-v1"
MAGIC = b"SKFXP001"
VERSION = 1
ENDIAN_MARKER = 0x01020304
WVM_REPOSITORY = "JeffreyEarly/wave-vortex-model"
COMMIT_PATTERN = re.compile(r"[0-9a-f]{40}")
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
INPUT_FAMILIES = [0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 1, 1, 0]
TARGET_FAMILIES = [0, 0, 1, 1]
INPUT_NAMES = [
    "U", "V", "W",
    "q0_x", "q0_y", "q0_z",
    "q1_x", "q1_y", "q1_z",
    "q2_x", "q2_y", "q2_z",
    "q3_x", "q3_y", "q3_z",
]
TARGET_NAMES = ["Fu", "Fv", "Fw", "Feta"]
DERIVATIVE_CONVENTION = (
    "Input slots 0..2 are U,V,W; slots 3+3*t..5+3*t are "
    "q[t].x,q[t].y,q[t].z. q[3].z denotes the complete eta "
    "vertical-advection factor, including eta*dLnN2 when assembled by WVM."
)
NORMALIZATION_ID = (
    "raw FFT coefficients; raw inverse divided by Nx*Ny for each physical "
    "factor; pointwise scale=1/(Nx*Ny)^2; raw forward output"
)
MODE_MAPPING_ID = (
    "logical (k,l,j,field); WVM radial magnitude then k then l; "
    "j-fastest canonical payloads; wave-f/wave-g field mapping"
)


class FixtureError(ValueError):
    """Raised when the exported fixture violates its evidence contract."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise FixtureError(message)


def mapping(value: Any, label: str) -> dict[str, Any]:
    require(isinstance(value, dict), f"{label} must be an object")
    return value


def sequence(value: Any, label: str) -> list[Any]:
    require(isinstance(value, list), f"{label} must be an array")
    return value


def integer(value: Any, label: str, *, positive: bool = False) -> int:
    require(isinstance(value, (int, float)) and not isinstance(value, bool),
            f"{label} must be an integer")
    require(math.isfinite(float(value)) and float(value).is_integer(),
            f"{label} must be an integer")
    value = int(value)
    if positive:
        require(value > 0, f"{label} must be positive")
    return value


def number(value: Any, label: str, *, positive: bool = False) -> float:
    require(isinstance(value, (int, float)) and not isinstance(value, bool),
            f"{label} must be numeric")
    result = float(value)
    require(math.isfinite(result), f"{label} must be finite")
    if positive:
        require(result > 0.0, f"{label} must be positive")
    return result


def text(value: Any, label: str) -> str:
    require(isinstance(value, str) and value, f"{label} must be a nonempty string")
    require("\x00" not in value, f"{label} cannot contain NUL")
    return value


def scalar_or_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else [value]


def dft_modes(count: int) -> list[int]:
    return list(range((count + 1) // 2)) + list(range(-(count // 2), 0))


def self_conjugate(mode: int, count: int) -> bool:
    return mode == 0 or (count % 2 == 0 and mode == -(count // 2))


def primary_mode(k: int, l: int, nx: int, ny: int) -> bool:
    l_self = self_conjugate(l, ny)
    return l > 0 or (l_self and (k > 0 or self_conjugate(k, nx)))


def retained_modes(nx: int, ny: int) -> list[tuple[int, int]]:
    require(nx == ny and nx % 2 == 0,
            "the pilot requires an even square horizontal grid")
    result: list[tuple[int, int]] = []
    for l in dft_modes(ny):
        for k in dft_modes(nx):
            if not primary_mode(k, l, nx, ny):
                continue
            if k == -(nx // 2) or l == -(ny // 2):
                continue
            if 9 * (k * k + l * l) > nx * nx:
                continue
            result.append((k, l))
    result.sort(key=lambda mode: (mode[0] * mode[0] + mode[1] * mode[1],
                                  mode[0], mode[1]))
    return result


def expected_payloads(nx: int, ny: int, nz: int, nkl: int, nj: int,
                      group_count: int) -> list[dict[str, Any]]:
    del nx, ny
    return [
        {"path": "horizontal-mode-keys.i32le", "type": "int32-le",
         "axes": ["coordinate", "mode"], "shape": [2, nkl],
         "strides": [1, 2], "complex": False, "elementBytes": 4},
        {"path": "vertical-mode-keys.i32le", "type": "int32-le",
         "axes": ["j"], "shape": [nj], "strides": [1],
         "complex": False, "elementBytes": 4},
        {"path": "mode-group-indices.u32le", "type": "uint32-le",
         "axes": ["mode"], "shape": [nkl], "strides": [1],
         "complex": False, "elementBytes": 4},
        {"path": "group-keys.u64le", "type": "uint64-le",
         "axes": ["group"], "shape": [group_count], "strides": [1],
         "complex": False, "elementBytes": 8},
        {"path": "inverse-operators.f64le", "type": "float64-le",
         "axes": ["z", "j", "operatorFamily", "group"],
         "shape": [nz, nj, 2, group_count],
         "strides": [1, nz, nz * nj, 2 * nz * nj],
         "complex": False, "elementBytes": 8},
        {"path": "forward-operators.f64le", "type": "float64-le",
         "axes": ["j", "z", "operatorFamily", "group"],
         "shape": [nj, nz, 2, group_count],
         "strides": [1, nj, nj * nz, 2 * nj * nz],
         "complex": False, "elementBytes": 8},
        {"path": "modal-inputs.c128le",
         "type": "complex-float64-interleaved-le",
         "axes": ["j", "inputField", "mode"],
         "shape": [nj, 15, nkl], "strides": [1, nj, nj * 15],
         "complex": True, "elementBytes": 16},
        {"path": "expected-modal-targets.c128le",
         "type": "complex-float64-interleaved-le",
         "axes": ["j", "target", "mode"],
         "shape": [nj, 4, nkl], "strides": [1, nj, nj * 4],
         "complex": True, "elementBytes": 16},
    ]


def unpack(payload: bytes, code: str) -> tuple[Any, ...]:
    require(len(payload) % struct.calcsize(code) == 0,
            "payload length is not an integer number of elements")
    return struct.unpack(f"<{len(payload) // struct.calcsize(code)}{code}", payload)


def validate_and_read(directory: pathlib.Path) -> tuple[dict[str, Any], bytes,
                                                         dict[str, bytes]]:
    require(directory.is_dir(), f"fixture directory does not exist: {directory}")
    manifest_path = directory / "manifest.json"
    require(manifest_path.is_file(), f"fixture manifest is missing: {manifest_path}")
    manifest_bytes = manifest_path.read_bytes()
    try:
        manifest = mapping(json.loads(manifest_bytes.decode("utf-8")), "manifest")
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise FixtureError(f"fixture manifest is not valid UTF-8 JSON: {error}") from error

    require(manifest.get("schema") == SCHEMA, f"schema must be {SCHEMA}")
    text(manifest.get("fixtureId"), "fixtureId")
    require(manifest.get("status") == "authoritative-wvm-export",
            "fixture status must be authoritative-wvm-export")
    require(manifest.get("authoritative") is True, "fixture must be authoritative")
    require(manifest.get("numericType") == "float64", "numericType must be float64")
    require(manifest.get("byteOrder") == "little-endian",
            "byteOrder must be little-endian")

    provenance = mapping(manifest.get("provenance"), "provenance")
    require(provenance.get("repository") == WVM_REPOSITORY,
            f"provenance.repository must be {WVM_REPOSITORY}")
    commit = text(provenance.get("commit"), "provenance.commit")
    require(COMMIT_PATTERN.fullmatch(commit) is not None,
            "provenance.commit must be a lowercase full Git object id")
    tree = text(provenance.get("tree"), "provenance.tree")
    require(COMMIT_PATTERN.fullmatch(tree) is not None,
            "provenance.tree must be a lowercase full Git object id")
    require(provenance.get("dirtyTree") is False,
            "authoritative fixture requires a clean WVM tree")

    generator = mapping(manifest.get("generator"), "generator")
    generator_path = text(generator.get("path"), "generator.path")
    generator_command = text(generator.get("command"), "generator.command")
    require(generator_path == "Benchmarks/exportSpectralFluxFixture.m",
            "fixture identifies an unexpected generator")

    workload = mapping(manifest.get("workload"), "workload")
    nx = integer(workload.get("Nx"), "workload.Nx", positive=True)
    ny = integer(workload.get("Ny"), "workload.Ny", positive=True)
    nz = integer(workload.get("Nz"), "workload.Nz", positive=True)
    h = integer(workload.get("H"), "workload.H", positive=True)
    nkl = integer(workload.get("Nkl"), "workload.Nkl", positive=True)
    nj = integer(workload.get("Nj"), "workload.Nj", positive=True)
    require(integer(workload.get("fields"), "workload.fields", positive=True) == 4,
            "workload.fields must be four")
    require(nx == ny and nx % 2 == 0, "pilot fixture requires an even square grid")
    require(h == (nx // 2 + 1) * ny, "workload.H is inconsistent")
    require(nj == 2 * (nz - 1) // 3,
            "workload.Nj violates floor(2*(Nz-1)/3)")
    lengths = sequence(workload.get("Lxyz"), "workload.Lxyz")
    require(len(lengths) == 3, "workload.Lxyz must have three entries")
    lx, ly, lz = [number(value, f"workload.Lxyz[{index}]", positive=True)
                  for index, value in enumerate(lengths)]
    require(lx == ly, "pilot fixture requires Lx=Ly")
    latitude = number(workload.get("latitude"), "workload.latitude")

    retention = mapping(manifest.get("retention"), "retention")
    require(retention.get("horizontalPolicy") == "radial-two-thirds",
            "horizontal retention must be radial-two-thirds")
    require(number(retention.get("horizontalCutoffFraction"),
                   "horizontalCutoffFraction") == 2.0 / 3.0,
            "horizontal cutoff fraction must be exactly 2/3 in Float64")
    require(retention.get("verticalPolicy") == "floor(2*(Nz-1)/3)",
            "vertical retention policy is inconsistent")
    require(number(retention.get("verticalRetainedFraction"),
                   "verticalRetainedFraction") == nj / float(nz - 1),
            "vertical retained fraction must equal Nj/(Nz-1) after flooring")

    mode_order = mapping(manifest.get("modeOrder"), "modeOrder")
    require(mode_order.get("logicalAxes") == ["k", "l", "j", "field"],
            "logical mode axes are inconsistent")
    require(mode_order.get("horizontal") == "WVM radial magnitude then k then l",
            "horizontal mode order is inconsistent")
    require(mode_order.get("vertical") == "j ascending from zero",
            "vertical mode order is inconsistent")

    normalization = mapping(manifest.get("normalization"), "normalization")
    require(normalization.get("horizontalForward") == "raw FFT coefficients",
            "horizontal forward normalization is inconsistent")
    require(normalization.get("horizontalInverse") ==
            "raw inverse FFT followed by division by Nx*Ny before each physical factor",
            "horizontal inverse normalization is inconsistent")
    pointwise_scale = number(normalization.get("pointwiseScale"),
                             "normalization.pointwiseScale", positive=True)
    require(pointwise_scale == 1.0 / float((nx * ny) ** 2),
            "pointwiseScale must be 1/(Nx*Ny)^2")

    operator = mapping(manifest.get("operatorContract"), "operatorContract")
    require(operator.get("familyIds") == ["wave-f", "wave-g"],
            "operator family ids are inconsistent")
    require(operator.get("inputFieldNames") == INPUT_NAMES,
            "input field names are inconsistent")
    require(operator.get("inputFieldFamilies") == INPUT_FAMILIES,
            "input field operator mapping is inconsistent")
    require(operator.get("targetNames") == TARGET_NAMES,
            "target names are inconsistent")
    require(operator.get("targetFieldFamilies") == TARGET_FAMILIES,
            "target operator mapping is inconsistent")
    group_count = integer(operator.get("groupCount"),
                          "operatorContract.groupCount", positive=True)
    require(operator.get("groupRule") ==
            "exact integer k^2+l^2 on a square horizontal domain",
            "operator group rule is inconsistent")
    require(manifest.get("derivativeConvention") == DERIVATIVE_CONVENTION,
            "derivative convention is inconsistent")

    oracle = mapping(manifest.get("oracle"), "oracle")
    require(oracle.get("identity") ==
            "WVM MATLAB Fourier transforms plus WVM Fw/Gw projection matrices and direct -(U*qx+V*qy+W*qz) evaluation",
            "oracle identity is inconsistent")
    require(number(oracle.get("maximumScaleNormalizedErrorTolerance"),
                   "oracle maximum tolerance") == 1.0e-12,
            "oracle maximum tolerance must be 1e-12")
    require(number(oracle.get("relativeL2ErrorTolerance"),
                   "oracle L2 tolerance") == 1.0e-12,
            "oracle L2 tolerance must be 1e-12")

    declared = sequence(manifest.get("payloads"), "payloads")
    expected = expected_payloads(nx, ny, nz, nkl, nj, group_count)
    require(len(declared) == len(expected), "fixture must declare exactly eight payloads")
    payload_bytes: dict[str, bytes] = {}
    for index, (actual_value, specification) in enumerate(zip(declared, expected)):
        actual = mapping(actual_value, f"payloads[{index}]")
        path_text = text(actual.get("path"), f"payloads[{index}].path")
        require(path_text == specification["path"],
                f"payload {index} must be {specification['path']}")
        relative = pathlib.PurePosixPath(path_text)
        require(not relative.is_absolute() and ".." not in relative.parts and
                len(relative.parts) == 1,
                f"unsafe payload path: {path_text}")
        require(actual.get("elementType") == specification["type"],
                f"{path_text} elementType is inconsistent")
        require(scalar_or_list(actual.get("logicalAxes")) == specification["axes"],
                f"{path_text} logicalAxes are inconsistent")
        require(scalar_or_list(actual.get("shape")) == specification["shape"],
                f"{path_text} shape is inconsistent")
        require(scalar_or_list(actual.get("stridesElements")) == specification["strides"],
                f"{path_text} strides are inconsistent")
        require(actual.get("isComplex") is specification["complex"],
                f"{path_text} complex flag is inconsistent")
        expected_bytes = math.prod(specification["shape"]) * specification["elementBytes"]
        require(integer(actual.get("byteCount"), f"{path_text}.byteCount") == expected_bytes,
                f"{path_text} byte count is inconsistent")
        digest = text(actual.get("sha256"), f"{path_text}.sha256")
        require(SHA256_PATTERN.fullmatch(digest) is not None,
                f"{path_text} SHA-256 is malformed")
        payload_path = directory / path_text
        require(payload_path.is_file(), f"payload is missing: {payload_path}")
        content = payload_path.read_bytes()
        require(len(content) == expected_bytes,
                f"{path_text} actual byte count does not match the manifest")
        require(hashlib.sha256(content).hexdigest() == digest,
                f"{path_text} SHA-256 does not match the manifest")
        payload_bytes[path_text] = content

    expected_modes = retained_modes(nx, ny)
    require(len(expected_modes) == nkl, "Nkl does not match radial retention")
    raw_modes = unpack(payload_bytes["horizontal-mode-keys.i32le"], "i")
    actual_modes = list(zip(raw_modes[0::2], raw_modes[1::2]))
    require(len(set(actual_modes)) == nkl,
            "horizontal mode keys must be unique")
    require(set(actual_modes) == set(expected_modes),
            "horizontal mode keys do not match radial retention")
    actual_squared_keys = [k * k + l * l for k, l in actual_modes]
    require(actual_squared_keys == sorted(actual_squared_keys),
            "horizontal mode keys are not in nondecreasing radial groups")
    vertical_keys = unpack(payload_bytes["vertical-mode-keys.i32le"], "i")
    require(list(vertical_keys) == list(range(nj)),
            "vertical mode keys must be j=0..Nj-1")
    group_indices = unpack(payload_bytes["mode-group-indices.u32le"], "I")
    group_keys = unpack(payload_bytes["group-keys.u64le"], "Q")
    distinct_keys: list[int] = []
    expected_indices: list[int] = []
    for key in actual_squared_keys:
        if not distinct_keys or distinct_keys[-1] != key:
            distinct_keys.append(key)
        expected_indices.append(len(distinct_keys) - 1)
    require(list(group_keys) == distinct_keys,
            "group keys do not match exact k^2+l^2")
    require(list(group_indices) == expected_indices,
            "mode group indices do not match exact k^2+l^2")
    require(group_count == len(distinct_keys), "operator group count is inconsistent")

    for path_text in ("inverse-operators.f64le", "forward-operators.f64le"):
        values = unpack(payload_bytes[path_text], "d")
        require(all(math.isfinite(value) for value in values),
                f"{path_text} contains non-finite values")
    for path_text in ("modal-inputs.c128le", "expected-modal-targets.c128le"):
        values = unpack(payload_bytes[path_text], "d")
        require(all(math.isfinite(value) for value in values),
                f"{path_text} contains non-finite values")
    modal_values = unpack(payload_bytes["modal-inputs.c128le"], "d")
    for field in range(15):
        for j in range(nj):
            complex_index = j + nj * field
            require(modal_values[2 * complex_index + 1] == 0.0,
                    "DC modal inputs must be exactly real")

    return manifest, manifest_bytes, payload_bytes


def fixture_identity(manifest: dict[str, Any], manifest_bytes: bytes) -> str:
    digest = hashlib.sha256()
    digest.update(manifest_bytes)
    for payload in sequence(manifest["payloads"], "payloads"):
        digest.update(bytes.fromhex(text(mapping(payload, "payload")["sha256"],
                                         "payload.sha256")))
    return f"sha256:{digest.hexdigest()}"


def write_string(stream: Any, value: str) -> None:
    encoded = value.encode("utf-8")
    stream.write(struct.pack("<Q", len(encoded)))
    stream.write(encoded)


def prepare(directory: pathlib.Path, output: pathlib.Path) -> dict[str, Any]:
    manifest, manifest_bytes, payloads = validate_and_read(directory)
    require(not output.exists(), f"refusing to overwrite prepared fixture: {output}")
    require(output.parent.is_dir(), f"output parent does not exist: {output.parent}")
    workload = manifest["workload"]
    operator = manifest["operatorContract"]
    lengths = workload["Lxyz"]
    identity = fixture_identity(manifest, manifest_bytes)
    generator = manifest["generator"]
    generator_identity = f"{generator['path']}: {generator['command']}"

    with output.open("xb") as stream:
        stream.write(MAGIC)
        stream.write(struct.pack("<IIII", VERSION, ENDIAN_MARKER, 1, 0))
        stream.write(struct.pack(
            "<9Q5d",
            workload["Nx"], workload["Ny"], workload["Nz"],
            workload["Nkl"], workload["Nj"], operator["groupCount"],
            2, 15, 4,
            float(lengths[0]), float(lengths[1]), float(lengths[2]),
            float(workload["latitude"]),
            float(manifest["normalization"]["pointwiseScale"]),
        ))
        for value in (
            manifest["fixtureId"], WVM_REPOSITORY,
            manifest["provenance"]["commit"], generator_identity, identity,
            NORMALIZATION_ID, MODE_MAPPING_ID, DERIVATIVE_CONVENTION,
        ):
            write_string(stream, value)
        specifications = expected_payloads(
            workload["Nx"], workload["Ny"], workload["Nz"],
            workload["Nkl"], workload["Nj"], operator["groupCount"])
        for specification in specifications[:4]:
            stream.write(payloads[specification["path"]])
        stream.write(struct.pack("<15I", *INPUT_FAMILIES))
        stream.write(struct.pack("<4I", *TARGET_FAMILIES))
        for specification in specifications[4:]:
            stream.write(payloads[specification["path"]])
    return {
        "fixtureId": manifest["fixtureId"],
        "fixtureHash": identity,
        "waveVortexModelCommit": manifest["provenance"]["commit"],
        "preparedPath": str(output.resolve()),
        "preparedBytes": output.stat().st_size,
    }


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", required=True, type=pathlib.Path,
                        help="directory containing manifest.json and payloads")
    parser.add_argument("--output", required=True, type=pathlib.Path,
                        help="new prepared binary path; existing files are rejected")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    try:
        summary = prepare(args.fixture.resolve(), args.output.resolve())
    except (FixtureError, OSError) as error:
        print(f"prepare_spectral_flux_fixture: {error}", file=sys.stderr)
        return 2
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
