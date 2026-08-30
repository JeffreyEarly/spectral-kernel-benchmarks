#!/usr/bin/env python3
"""Validate and prepare an authoritative WVM constant-stratification fixture."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import pathlib
import re
import shutil
import struct
import sys
from typing import Any, BinaryIO


SCHEMA = "constant-stratification-flux-fixture-v1"
MAGIC = b"SKCFP001"
VERSION = 1
ENDIAN_MARKER = 0x01020304
WVM_REPOSITORY = "JeffreyEarly/wave-vortex-model"
GENERATOR_REPOSITORY = "JeffreyEarly/spectral-kernel-benchmarks"
AUDITED_WVM_COMMIT = "6ad254fb9756ac918bb72e036020d004879df1f2"
AUDITED_SOURCES = {
    "CompiledKernel/src/WVCoefficientFormulas.hpp":
        "78d647608d8919b33c278c280eb6d2b253dfb0acd8d3330141fbdc845e00864a",
    "CompiledKernel/src/WVTransformConstantStratificationKernel.cpp":
        "3e8f618fa813ca274b0c33ed3a34c023fc891ef79b0a062abdc99a967de4a3a9",
    "CompiledKernel/src/WVKernelTypes.cpp":
        "4b7cf55649b02eaae224bfaf618f756122a81c1caaff92c751b4e35f827ea1cc",
}
COMMIT_PATTERN = re.compile(r"[0-9a-f]{40}")
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
TOLERANCE = 1.0e-12
CROSS_BACKEND_TOLERANCE = 1.0e-10
NORMALIZATION_ID = (
    "raw horizontal FFT; inverse type-I factors placed in coefficient "
    "assembly; no explicit pointwise scale; forward type-I divided by "
    "Nz-1; modal projection includes 1/(Nx*Ny)"
)
MODE_MAPPING_ID = (
    "logical (k,l,j,coefficient); WVM radial magnitude then k then l; "
    "prepared payload reordered to skbench radial mode order; j fastest"
)
COEFFICIENT_CONTRACT_ID = (
    "WVM constant-stratification natural-dimensional-prescaled "
    "nonhydrostatic nonlinear flux with phase and inertial/MDA exceptions"
)


class FixtureError(ValueError):
    """Raised when a fixture violates the authoritative evidence contract."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise FixtureError(message)


def mapping(value: Any, label: str) -> dict[str, Any]:
    require(isinstance(value, dict), f"{label} must be an object")
    return value


def sequence(value: Any, label: str) -> list[Any]:
    require(isinstance(value, list), f"{label} must be an array")
    return value


def text(value: Any, label: str) -> str:
    require(isinstance(value, str) and value, f"{label} must be a nonempty string")
    require("\x00" not in value, f"{label} cannot contain NUL")
    return value


def integer(value: Any, label: str, *, positive: bool = False) -> int:
    require(isinstance(value, (int, float)) and not isinstance(value, bool),
            f"{label} must be an integer")
    require(math.isfinite(float(value)) and float(value).is_integer(),
            f"{label} must be an integer")
    result = int(value)
    if positive:
        require(result > 0, f"{label} must be positive")
    return result


def number(value: Any, label: str, *, positive: bool = False) -> float:
    require(isinstance(value, (int, float)) and not isinstance(value, bool),
            f"{label} must be numeric")
    result = float(value)
    require(math.isfinite(result), f"{label} must be finite")
    if positive:
        require(result > 0.0, f"{label} must be positive")
    return result


def dft_modes(count: int) -> list[int]:
    return list(range((count + 1) // 2)) + list(range(-(count // 2), 0))


def self_conjugate(mode: int, count: int) -> bool:
    return mode == 0 or (count % 2 == 0 and mode == -(count // 2))


def primary_mode(k: int, l: int, nx: int, ny: int) -> bool:
    l_self = self_conjugate(l, ny)
    return l > 0 or (l_self and (k > 0 or self_conjugate(k, nx)))


def retained_modes(nx: int, ny: int) -> list[tuple[int, int]]:
    require(nx == ny and nx % 2 == 0,
            "the fixture requires an even square horizontal grid")
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


def expected_payloads(nkl: int, nj: int) -> list[dict[str, Any]]:
    return [
        {"path": "horizontal-mode-keys.i32le", "type": "int32-le",
         "axes": ["coordinate", "mode"], "shape": [2, nkl],
         "strides": [1, 2], "complex": False, "elementBytes": 4},
        {"path": "modal-state.c128le",
         "type": "complex-float64-interleaved-le",
         "axes": ["j", "coefficient", "mode"], "shape": [nj, 3, nkl],
         "strides": [1, nj, 3 * nj], "complex": True, "elementBytes": 16},
        {"path": "expected-modal-flux.c128le",
         "type": "complex-float64-interleaved-le",
         "axes": ["j", "flux", "mode"], "shape": [nj, 3, nkl],
         "strides": [1, nj, 3 * nj], "complex": True, "elementBytes": 16},
    ]


def scalar_or_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else [value]


def hash_file(path: pathlib.Path) -> tuple[int, str]:
    digest = hashlib.sha256()
    byte_count = 0
    with path.open("rb") as stream:
        while chunk := stream.read(16 * 1024 * 1024):
            digest.update(chunk)
            byte_count += len(chunk)
    return byte_count, digest.hexdigest()


def validate_payloads(directory: pathlib.Path, manifest: dict[str, Any],
                      nkl: int, nj: int) -> dict[str, pathlib.Path]:
    specifications = expected_payloads(nkl, nj)
    declared = sequence(manifest.get("payloads"), "payloads")
    require(len(declared) == len(specifications),
            "fixture must contain exactly three payloads")
    paths: dict[str, pathlib.Path] = {}
    for expected, raw in zip(specifications, declared, strict=True):
        payload = mapping(raw, "payload")
        path_text = text(payload.get("path"), "payload.path")
        require(path_text == expected["path"],
                f"expected payload {expected['path']}, found {path_text}")
        relative = pathlib.PurePosixPath(path_text)
        require(not relative.is_absolute() and ".." not in relative.parts,
                "payload path must be a safe relative path")
        path = directory / pathlib.Path(*relative.parts)
        require(path.is_file(), f"fixture payload is missing: {path}")
        require(payload.get("elementType") == expected["type"],
                f"payload {path_text} has the wrong element type")
        require(scalar_or_list(payload.get("logicalAxes")) == expected["axes"],
                f"payload {path_text} has the wrong logical axes")
        require([integer(value, f"{path_text}.shape")
                 for value in scalar_or_list(payload.get("shape"))] ==
                expected["shape"], f"payload {path_text} has the wrong shape")
        require([integer(value, f"{path_text}.strides")
                 for value in scalar_or_list(payload.get("stridesElements"))] ==
                expected["strides"],
                f"payload {path_text} has the wrong strides")
        require(payload.get("isComplex") is expected["complex"],
                f"payload {path_text} has the wrong complex flag")
        expected_bytes = math.prod(expected["shape"]) * expected["elementBytes"]
        require(integer(payload.get("byteCount"), f"{path_text}.byteCount") ==
                expected_bytes, f"payload {path_text} has the wrong byte count")
        digest = text(payload.get("sha256"), f"{path_text}.sha256")
        require(SHA256_PATTERN.fullmatch(digest) is not None,
                f"payload {path_text} has an invalid SHA-256")
        byte_count, actual_digest = hash_file(path)
        require(byte_count == expected_bytes,
                f"payload {path_text} size disagrees with its manifest")
        require(actual_digest == digest,
                f"payload {path_text} SHA-256 disagrees with its manifest")
        paths[path_text] = path
    return paths


def validate_and_read(directory: pathlib.Path) -> tuple[
        dict[str, Any], bytes, dict[str, pathlib.Path], list[int]]:
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
    require(commit == AUDITED_WVM_COMMIT,
            f"WVM commit must be the audited {AUDITED_WVM_COMMIT}")
    require(COMMIT_PATTERN.fullmatch(text(provenance.get("tree"),
                                          "provenance.tree")) is not None,
            "provenance.tree must be a lowercase full Git object id")
    require(provenance.get("dirtyTree") is False,
            "authoritative fixture requires a clean WVM tree")

    generator = mapping(manifest.get("generator"), "generator")
    require(generator.get("path") ==
            "tools/exportConstantStratificationFluxFixture.m",
            "fixture identifies an unexpected generator")
    require(generator.get("repository") == GENERATOR_REPOSITORY,
            "fixture identifies the wrong generator repository")
    require(COMMIT_PATTERN.fullmatch(text(generator.get("commit"),
                                          "generator.commit")) is not None,
            "generator.commit must be a lowercase full Git object id")
    require(COMMIT_PATTERN.fullmatch(text(generator.get("tree"),
                                          "generator.tree")) is not None,
            "generator.tree must be a lowercase full Git object id")
    require(generator.get("dirtyTree") is False,
            "authoritative fixture requires a clean generator tree")
    text(generator.get("command"), "generator.command")

    sources = sequence(manifest.get("auditedSources"), "auditedSources")
    actual_sources = {
        text(mapping(item, "auditedSource").get("path"), "auditedSource.path"):
        text(mapping(item, "auditedSource").get("sha256"),
             "auditedSource.sha256") for item in sources
    }
    require(actual_sources == AUDITED_SOURCES,
            "fixture audited source hashes do not match the issue #20 contract")

    backend = mapping(manifest.get("compiledBackend"), "compiledBackend")
    module = mapping(backend.get("module"), "compiledBackend.module")
    require(module.get("identityValidated") is True,
            "compiled WVM module identity was not validated")
    module_sha = text(module.get("sha256"), "compiledBackend.module.sha256")
    require(SHA256_PATTERN.fullmatch(module_sha) is not None,
            "compiled WVM module SHA-256 is invalid")

    workload = mapping(manifest.get("workload"), "workload")
    nx = integer(workload.get("Nx"), "workload.Nx", positive=True)
    ny = integer(workload.get("Ny"), "workload.Ny", positive=True)
    nz = integer(workload.get("Nz"), "workload.Nz", positive=True)
    nkl = integer(workload.get("Nkl"), "workload.Nkl", positive=True)
    nj = integer(workload.get("Nj"), "workload.Nj", positive=True)
    require(integer(workload.get("H"), "workload.H", positive=True) ==
            (nx // 2 + 1) * ny, "workload.H is inconsistent")
    require(integer(workload.get("fields"), "workload.fields", positive=True) == 4,
            "workload.fields must be four")
    require(nx == ny and nx % 2 == 0,
            "fixture requires an even square horizontal grid")
    require(nj == 2 * (nz - 1) // 3,
            "workload.Nj violates floor(2*(Nz-1)/3)")
    lengths = sequence(workload.get("Lxyz"), "workload.Lxyz")
    require(len(lengths) == 3, "workload.Lxyz must contain three values")
    lx, ly, lz = [number(value, f"workload.Lxyz[{index}]", positive=True)
                  for index, value in enumerate(lengths)]
    require(lx == ly, "fixture requires Lx=Ly")

    physical = mapping(manifest.get("physicalConfiguration"),
                       "physicalConfiguration")
    n0 = number(physical.get("N0"), "physicalConfiguration.N0", positive=True)
    rotation = number(physical.get("rotationRate"),
                      "physicalConfiguration.rotationRate")
    latitude = number(physical.get("latitude"), "physicalConfiguration.latitude")
    gravity = number(physical.get("g"), "physicalConfiguration.g", positive=True)
    elapsed = number(physical.get("elapsedTime"),
                     "physicalConfiguration.elapsedTime")
    require(physical.get("isHydrostatic") is False,
            "fixture must be nonhydrostatic")
    require(physical.get("shouldAntialias") is True,
            "fixture must use antialiasing")
    require(number(physical.get("matlabModalNormalizationGravity"),
                   "matlabModalNormalizationGravity") == 9.81,
            "MATLAB modal normalization gravity must be 9.81")
    coriolis = 2.0 * rotation * math.sin(math.radians(latitude))
    require(n0 * n0 > coriolis * coriolis,
            "nonhydrostatic fixture requires N0 squared greater than f squared")
    del lz, gravity, elapsed

    retention = mapping(manifest.get("retention"), "retention")
    require(retention.get("horizontalPolicy") == "radial-two-thirds",
            "horizontal retention must be radial-two-thirds")
    require(number(retention.get("horizontalCutoffFraction"),
                   "horizontalCutoffFraction") == 2.0 / 3.0,
            "horizontal cutoff must be exactly 2/3")
    require(retention.get("verticalPolicy") == "floor(2*(Nz-1)/3)",
            "vertical retention policy is inconsistent")
    require(number(retention.get("verticalRetainedFraction"),
                   "verticalRetainedFraction") == nj / float(nz - 1),
            "vertical retained fraction is inconsistent")

    mode_order = mapping(manifest.get("modeOrder"), "modeOrder")
    require(mode_order.get("logicalAxes") == ["k", "l", "j", "coefficient"],
            "logical mode axes are inconsistent")
    require(mode_order.get("horizontal") == "WVM radial magnitude then k then l",
            "horizontal mode order is inconsistent")
    require(mode_order.get("vertical") == "j ascending from zero",
            "vertical mode order is inconsistent")
    require(mode_order.get("coefficientNames") == ["Ap", "Am", "A0"] and
            mode_order.get("fluxNames") == ["Fp", "Fm", "F0"],
            "coefficient or flux names are inconsistent")

    normalization = mapping(manifest.get("normalization"), "normalization")
    pointwise = number(normalization.get("pointwiseScale"),
                       "normalization.pointwiseScale", positive=True)
    require(pointwise == 1.0,
            "the exact WVM constant-stratification pointwise scale must be one")
    coefficient = mapping(manifest.get("coefficientContract"),
                          "coefficientContract")
    require(coefficient.get("identity") ==
            "WVM constant-stratification natural-dimensional-prescaled nonlinear flux",
            "coefficient contract identity is inconsistent")

    oracle = mapping(manifest.get("oracle"), "oracle")
    require(oracle.get("identity") ==
            "WVM MATLAB nonlinearFlux cross-checked against the compiled WVTransformConstantStratificationKernel nonlinearFlux",
            "oracle identity is inconsistent")
    maximum_error = number(oracle.get("maximumScaleNormalizedError"),
                           "oracle.maximumScaleNormalizedError")
    l2_error = number(oracle.get("relativeL2Error"), "oracle.relativeL2Error")
    require(number(oracle.get("maximumScaleNormalizedErrorTolerance"),
                   "oracle.maximumScaleNormalizedErrorTolerance") ==
            CROSS_BACKEND_TOLERANCE and
            number(oracle.get("relativeL2ErrorTolerance"),
                   "oracle.relativeL2ErrorTolerance") ==
            CROSS_BACKEND_TOLERANCE,
            "WVM cross-backend oracle tolerances must be 1e-10")
    require(number(oracle.get("benchmarkMaximumScaleNormalizedErrorTolerance"),
                   "oracle.benchmarkMaximumScaleNormalizedErrorTolerance") ==
            TOLERANCE and
            number(oracle.get("benchmarkRelativeL2ErrorTolerance"),
                   "oracle.benchmarkRelativeL2ErrorTolerance") == TOLERANCE,
            "benchmark-to-oracle tolerances must be 1e-12")
    require(maximum_error <= CROSS_BACKEND_TOLERANCE and
            l2_error <= CROSS_BACKEND_TOLERANCE,
            "MATLAB and compiled WVM oracle comparison failed")

    payloads = validate_payloads(directory, manifest, nkl, nj)
    raw_keys = struct.unpack(f"<{2 * nkl}i",
                             payloads["horizontal-mode-keys.i32le"].read_bytes())
    source_modes = [(raw_keys[2 * index], raw_keys[2 * index + 1])
                    for index in range(nkl)]
    require(len(set(source_modes)) == nkl,
            "fixture horizontal mode keys must be unique")
    expected_modes = retained_modes(nx, ny)
    require(len(expected_modes) == nkl and set(source_modes) == set(expected_modes),
            "fixture horizontal mode set is not the radial retained set")
    source_indices = {mode: index for index, mode in enumerate(source_modes)}
    source_for_expected = [source_indices[mode] for mode in expected_modes]

    block_bytes = nj * 3 * 16
    dc_source_index = source_indices[(0, 0)]
    with payloads["modal-state.c128le"].open("rb") as stream:
        stream.seek(dc_source_index * block_bytes)
        dc = stream.read(block_bytes)
    dc_values = struct.unpack(f"<{nj * 3 * 2}d", dc)
    for j in range(nj):
        ap = complex(dc_values[2 * j], dc_values[2 * j + 1])
        am_index = j + nj
        a0_index = j + 2 * nj
        am = complex(dc_values[2 * am_index], dc_values[2 * am_index + 1])
        require(am == ap.conjugate(), "DC Am must be the exact conjugate of Ap")
        require(dc_values[2 * a0_index + 1] == 0.0,
                "DC A0 must be exactly real")

    return manifest, manifest_bytes, payloads, source_for_expected


def fixture_identity(manifest: dict[str, Any], manifest_bytes: bytes) -> str:
    digest = hashlib.sha256()
    digest.update(manifest_bytes)
    for raw in sequence(manifest["payloads"], "payloads"):
        digest.update(bytes.fromhex(text(mapping(raw, "payload")["sha256"],
                                         "payload.sha256")))
    return f"sha256:{digest.hexdigest()}"


def write_string(stream: BinaryIO, value: str) -> None:
    encoded = value.encode("utf-8")
    stream.write(struct.pack("<Q", len(encoded)))
    stream.write(encoded)


def copy_mode_blocks(source: pathlib.Path, destination: BinaryIO,
                     source_for_expected: list[int], block_bytes: int) -> None:
    if source_for_expected == list(range(len(source_for_expected))):
        with source.open("rb") as input_stream:
            shutil.copyfileobj(input_stream, destination, 16 * 1024 * 1024)
        return
    with source.open("rb") as input_stream:
        for source_index in source_for_expected:
            input_stream.seek(source_index * block_bytes)
            block = input_stream.read(block_bytes)
            require(len(block) == block_bytes,
                    f"fixture payload {source.name} is truncated")
            destination.write(block)


def prepare(directory: pathlib.Path, output: pathlib.Path) -> dict[str, Any]:
    manifest, manifest_bytes, payloads, source_for_expected = validate_and_read(
        directory
    )
    require(not output.exists(), f"refusing to overwrite prepared fixture: {output}")
    require(output.parent.is_dir(), f"output parent does not exist: {output.parent}")
    workload = manifest["workload"]
    physical = manifest["physicalConfiguration"]
    oracle = manifest["oracle"]
    lengths = workload["Lxyz"]
    identity = fixture_identity(manifest, manifest_bytes)
    generator = manifest["generator"]
    generator_identity = f"{generator['repository']}@{generator['commit']}:{generator['path']}"
    module_sha = manifest["compiledBackend"]["module"]["sha256"]
    expected_modes = retained_modes(workload["Nx"], workload["Ny"])
    block_bytes = workload["Nj"] * 3 * 16

    with output.open("xb") as stream:
        stream.write(MAGIC)
        stream.write(struct.pack("<IIII", VERSION, ENDIAN_MARKER, 1, 0))
        stream.write(struct.pack(
            "<6Q9d2d",
            workload["Nx"], workload["Ny"], workload["Nz"],
            workload["Nkl"], workload["Nj"], 3,
            float(lengths[0]), float(lengths[1]), float(lengths[2]),
            float(physical["N0"]), float(physical["rotationRate"]),
            float(physical["latitude"]), float(physical["g"]),
            float(physical["elapsedTime"]),
            float(manifest["normalization"]["pointwiseScale"]),
            float(oracle["maximumScaleNormalizedError"]),
            float(oracle["relativeL2Error"]),
        ))
        for value in (
            manifest["fixtureId"], WVM_REPOSITORY,
            manifest["provenance"]["commit"], generator_identity, identity,
            NORMALIZATION_ID, MODE_MAPPING_ID, COEFFICIENT_CONTRACT_ID,
            module_sha,
        ):
            write_string(stream, value)
        stream.write(struct.pack(
            f"<{2 * len(expected_modes)}i",
            *(coordinate for mode in expected_modes for coordinate in mode),
        ))
        copy_mode_blocks(payloads["modal-state.c128le"], stream,
                         source_for_expected, block_bytes)
        copy_mode_blocks(payloads["expected-modal-flux.c128le"], stream,
                         source_for_expected, block_bytes)
    return {
        "fixtureId": manifest["fixtureId"],
        "fixtureHash": identity,
        "waveVortexModelCommit": manifest["provenance"]["commit"],
        "preparedPath": str(output.resolve()),
        "preparedBytes": output.stat().st_size,
        "oracleMaximumScaleNormalizedError":
            oracle["maximumScaleNormalizedError"],
        "oracleRelativeL2Error": oracle["relativeL2Error"],
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
        print(f"prepare_constant_stratification_flux_fixture: {error}",
              file=sys.stderr)
        return 2
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
