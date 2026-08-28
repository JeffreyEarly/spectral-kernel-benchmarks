#!/usr/bin/env python3
"""Build the static GitHub Pages evidence archive from published bundles."""

from __future__ import annotations

import argparse
import html
import math
import random
import re
import shutil
import statistics
from pathlib import Path
from urllib.parse import quote

from publication import PublishedBundle, load_and_validate


REPOSITORY_URL = "https://github.com/JeffreyEarly/spectral-kernel-benchmarks"
SUMMARY_SCOPES = (
    ("Raw FFT", "primitive"),
    ("WVM-compatible adapter", "adapter-total"),
    ("Retained horizontal operator", "uninstrumented-total"),
)

EXPERIMENT_PROVIDER_IDS = {
    "issue-003-fftw-production-baseline": "fftw",
    "issue-004-fftw-strategy-sweep": "fftw",
    "issue-005-vdsp-native-baseline": "accelerate-vdsp",
    "issue-006-vdsp-batching-scheduling": "accelerate-vdsp",
}


def escaped(value: object) -> str:
    return html.escape(str(value), quote=True)


def format_ms(seconds: float) -> str:
    milliseconds = 1000.0 * float(seconds)
    if milliseconds >= 100.0:
        return f"{milliseconds:.1f}"
    if milliseconds >= 1.0:
        return f"{milliseconds:.3f}"
    return f"{milliseconds:.4f}"


def format_bytes(byte_count: int) -> str:
    value = float(byte_count)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024.0 or unit == "TiB":
            return f"{value:.0f} {unit}" if unit == "B" else f"{value:.2f} {unit}"
        value /= 1024.0
    raise AssertionError("unreachable")


def format_error(value: float | None) -> str:
    return "not available" if value is None else f"{float(value):.3e}"


def display_timestamp(value: str) -> str:
    return value.replace("T", " ").replace("Z", " UTC")


def provider_name(provider: dict) -> str:
    names = {
        "fftw": "FFTW",
        "fftw-split": "FFTW split",
        "accelerate-vdsp": "Accelerate/vDSP",
        "accelerate-zgemm": "Accelerate complex zgemm",
        "accelerate-split-dgemm": "Accelerate split dgemm",
        "ordering-pack-accelerate-zgemm": "Packed Accelerate complex zgemm",
        "ordering-pack-accelerate-split-dgemm": "Packed Accelerate split dgemm",
        "ordering-no-reorder-accelerate-zgemm": "Direct WVM-order Accelerate zgemm",
        "fftw-full-2d-retained-reference": "FFTW full 2-D plus radial selection",
        "fftw-partial-column-pruned": "FFTW partial-column-pruned",
        "accelerate-vdsp-native-retained": "Accelerate/vDSP native retained split",
        "pipeline-wvm-direct": "WVM direct/no-reorder pipeline",
        "pipeline-plane-major-fused-split": "Plane-major fused-split pipeline",
        "pipeline-streaming-pruned-compact-split": "Streaming pruned compact-split pipeline",
        "fftw-explicit-dealiased-convolution": "Explicit FFTW dealiased convolution",
        "fftwpp-hybrid-hermitian-convolution": "FFTW++ implicit/hybrid Hermitian convolution",
        "fftw-explicit-streamed-wvm-advection": "Explicit FFTW streamed WVM advection",
        "fftw-explicit-parallel-target-wvm-advection": "Explicit FFTW parallel-target WVM advection",
        "fftwpp-streamed-target-wvm-advection": "FFTW++ streamed-target WVM advection",
        "fftwpp-all-target-wvm-advection": "FFTW++ all-target WVM advection",
        "fftwpp-parallel-target-wvm-advection": "FFTW++ parallel-target WVM advection",
    }
    return names.get(provider["id"], provider["id"])


def publication_badge(status: str) -> str:
    return f'<span class="status publication-{escaped(status)}">{escaped(status)}</span>'


def timing(provider: dict, scope: str, direction: str) -> dict | None:
    matches = [
        item
        for item in provider["timings"]
        if item["scope"] == scope and item["direction"] == direction
    ]
    if len(matches) > 1:
        raise ValueError(f"Multiple {scope}/{direction} timings for {provider['id']}")
    return matches[0] if matches else None


def stage_timing(provider: dict, scope: str, stage: str, direction: str) -> dict | None:
    matches = [
        item
        for item in provider["timings"]
        if item["scope"] == scope and item["stage"] == stage and item["direction"] == direction
    ]
    if len(matches) > 1:
        raise ValueError(f"Multiple {scope}/{stage}/{direction} timings for {provider['id']}")
    return matches[0] if matches else None


def coefficient_of_variation(item: dict | None) -> str:
    if item is None:
        return "not measured"
    values = [float(value) for value in item.get("samplesSeconds", [])]
    if len(values) < 2 or statistics.mean(values) == 0.0:
        return "not available"
    return f"{100.0 * statistics.stdev(values) / statistics.mean(values):.1f}%"


def bootstrap_median_interval(item: dict | None) -> tuple[float, float] | None:
    if item is None:
        return None
    values = [float(value) for value in item.get("samplesSeconds", [])]
    if len(values) < 2:
        return None
    generator = random.Random(0x5B3EC6)
    medians = sorted(
        statistics.median(generator.choices(values, k=len(values))) for _ in range(4096)
    )
    return medians[int(0.025 * (len(medians) - 1))], medians[int(0.975 * (len(medians) - 1))]


def timing_with_interval(item: dict | None) -> str:
    if item is None:
        return "not measured"
    interval = bootstrap_median_interval(item)
    if interval is None:
        return format_ms(item["medianSeconds"])
    return f'{format_ms(item["medianSeconds"])} [{format_ms(interval[0])}–{format_ms(interval[1])}]'


def maximum_correctness_error(provider: dict) -> float | None:
    values = [
        item["maximumRelativeError"]
        for item in provider["correctness"]
        if item["maximumRelativeError"] is not None
    ]
    return max(values) if values else None


def maximum_l2_error(provider: dict) -> float | None:
    values = [
        item["relativeL2Error"]
        for item in provider["correctness"]
        if item.get("relativeL2Error") is not None
    ]
    return max(values) if values else None


def summary_timing_table(result: dict) -> str:
    providers = result["providers"]
    measurements: list[tuple[str, str, tuple[str, ...], tuple[str, ...]]] = [
        (label, scope, (), ("forward", "inverse")) for label, scope in SUMMARY_SCOPES
    ]
    if any(provider["id"].startswith("pipeline-") for provider in providers):
        measurements = [
            ("Raw FFT", "primitive", ("raw FFT",), ("forward", "inverse")),
            ("Raw vertical MM", "primitive", ("raw vertical MM",), ("forward", "inverse")),
            ("Representation movement", "adapter-component", (), ("forward", "inverse")),
            ("Mode-keyed modal work", "component", ("mode-keyed modal work",), ("modal",)),
            (
                "Synthetic antialiased spectral pipeline",
                "uninstrumented-total",
                ("synthetic antialiased spectral pipeline",),
                ("round-trip",),
            ),
        ]
    elif any(provider["id"].startswith("boundary-") for provider in providers):
        measurements = [
            ("Raw horizontal transform", "primitive", ("raw FFT", "raw pruned FFT"), ("forward", "inverse")),
            ("Raw vertical MM", "primitive", ("raw vertical MM",), ("forward", "inverse")),
            ("Representation movement", "adapter-component", (), ("forward", "inverse")),
            ("Composed boundary", "uninstrumented-total", (), ("forward", "inverse")),
        ]
    elif any(
        item["scope"] == "primitive" and item["stage"] == "raw vertical GEMM"
        for provider in providers
        for item in provider["timings"]
    ):
        measurements = [("Raw vertical GEMM", "primitive", (), ("forward", "inverse"))]
    header = "".join(f'<th scope="col">{escaped(provider_name(provider))}</th>' for provider in providers)
    rows: list[str] = []
    for label, scope, stages, directions in measurements:
        for direction in directions:
            values = []
            for provider in providers:
                if stages:
                    item = next(
                        (
                            stage_timing(provider, scope, stage, direction)
                            for stage in stages
                            if stage_timing(provider, scope, stage, direction) is not None
                        ),
                        None,
                    )
                else:
                    item = timing(provider, scope, direction)
                values.append(item)
            finite = [float(item["medianSeconds"]) for item in values if item is not None]
            best = min(finite) if finite else None
            cells: list[str] = []
            for item in values:
                if item is None:
                    cells.append('<td class="muted">not measured</td>')
                    continue
                seconds = float(item["medianSeconds"])
                class_name = ' class="best"' if best is not None and seconds == best else ""
                suffix = ' <span class="best-label">fastest in this run</span>' if class_name else ""
                cells.append(f"<td{class_name}>{format_ms(seconds)}{suffix}</td>")
            rows.append(
                f'<tr><th scope="row">{escaped(label)} <span>{direction}</span></th>{"".join(cells)}</tr>'
            )
    return (
        '<div class="table-scroll"><table class="timing-table">'
        '<caption>Median steady-state time in milliseconds; lower is better.</caption>'
        f'<thead><tr><th scope="col">Measurement</th>{header}</tr></thead>'
        f'<tbody>{"".join(rows)}</tbody></table></div>'
    )


def result_summary_cards(bundle: PublishedBundle) -> str:
    result = bundle.result
    publication = bundle.publication
    workload = result["workload"]
    environment = result["environment"]
    status = result["status"]
    status_class = "passed" if status == "passed" else "failed"
    scalar_type = result.get("numericType", {}).get("id", "float64 (legacy record)")
    return f"""
      <div class="summary-grid">
        <section class="summary-card">
          <p class="eyebrow">Machine</p>
          <p class="summary-value">{escaped(environment['cpuBrand'])}</p>
          <p>{escaped(environment['machineModel'])} · {environment['performanceCores']}P + {environment['efficiencyCores']}E cores</p>
        </section>
        <section class="summary-card">
          <p class="eyebrow">Workload</p>
          <p class="summary-value">{workload['Nx']} × {workload['Ny']} × {workload['planes']} planes</p>
          <p>N<sub>z</sub>={workload['Nz']}, fields={workload['fields']}, N<sub>kl</sub>={workload['Nkl']}, N<sub>j</sub>={workload['Nj']} · {escaped(scalar_type)}</p>
        </section>
        <section class="summary-card">
          <p class="eyebrow">Evidence status</p>
          <p class="summary-value"><span class="status {status_class}">{escaped(status)}</span> {publication_badge(publication['status'])}</p>
          <p>{result['run']['samples']} samples · {result['run']['warmups']} warmups · max error {format_error(max(maximum_correctness_error(provider) or 0.0 for provider in result['providers']))}</p>
        </section>
      </div>
    """


def shell(title: str, content: str, root_prefix: str = "") -> str:
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta name="description" content="Apple Silicon FFT and spectral-kernel benchmark evidence">
  <title>{escaped(title)} · Spectral Kernel Benchmarks</title>
  <link rel="icon" href="{root_prefix}assets/favicon.svg" type="image/svg+xml">
  <link rel="stylesheet" href="{root_prefix}assets/style.css">
</head>
<body>
  <a class="skip-link" href="#content">Skip to content</a>
  <header class="site-header">
    <a class="wordmark" href="{root_prefix}index.html"><span class="mark" aria-hidden="true">∿</span> Spectral Kernel Benchmarks</a>
    <nav aria-label="Primary">
      <a href="{root_prefix}index.html#runs">Runs</a>
      <a href="{root_prefix}experiments/index.html">Experiments</a>
      <a href="{root_prefix}methods/operators-and-representations/index.html">Methods</a>
      <a href="{root_prefix}decisions/v1/index.html">Decision</a>
      <a href="{REPOSITORY_URL}">Repository</a>
    </nav>
  </header>
  <main id="content">{content}</main>
  <footer>
    <p>Generated from committed evidence. Benchmarks never run during Pages generation.</p>
    <p><a href="{REPOSITORY_URL}">JeffreyEarly/spectral-kernel-benchmarks</a></p>
  </footer>
</body>
</html>
"""


def archive(bundles: list[PublishedBundle], root_prefix: str = "") -> str:
    if not bundles:
        return '<p class="empty-state">No immutable runs are associated with this experiment yet.</p>'
    cards: list[str] = []
    for bundle in bundles:
        publication = bundle.publication
        result = bundle.result
        workload = result["workload"]
        environment = result["environment"]
        run_id = result["run"]["id"]
        cards.append(f"""
          <li>
            <a class="run-card" href="{root_prefix}runs/{quote(run_id)}/index.html">
              <span class="run-date">{escaped(display_timestamp(environment['timestampUtc']))}</span>
              <strong>{escaped(environment['cpuBrand'])}</strong>
              <span>{workload['Nx']} × {workload['Ny']}, N<sub>z</sub>={workload['Nz']}, fields={workload['fields']}</span>
              <span class="run-meta">{escaped(result['run']['profile'])} · {result['run']['samples']} samples · {publication_badge(publication['status'])}</span>
            </a>
          </li>
        """)
    return f'<ul class="run-list">{"".join(cards)}</ul>'


def experiment_cards(catalog: dict, root_prefix: str = "") -> str:
    cards = []
    for experiment in sorted(catalog["experiments"], key=lambda item: item["issue"]):
        cards.append(f"""
          <li>
            <a class="experiment-card" href="{root_prefix}experiments/{quote(experiment['id'])}/index.html">
              <span class="eyebrow">Issue #{experiment['issue']} · {escaped(experiment['phase'])}</span>
              <strong>{escaped(experiment['title'])}</strong>
              <span>{escaped(experiment['question'])}</span>
            </a>
          </li>
        """)
    return f'<ul class="experiment-list">{"".join(cards)}</ul>'


def build_index(catalog: dict, bundles: list[PublishedBundle]) -> str:
    latest = bundles[0]
    latest_result = latest.result
    latest_run_id = latest_result["run"]["id"]
    content = f"""
    <section class="hero">
      <p class="eyebrow">Apple Silicon · antialiased spectral operators</p>
      <h1>Which spectral kernels are actually fastest?</h1>
      <p class="lede">An append-only evidence archive comparing FFT providers, memory representations, data movement, and composed spectral operations for the Wave–Vortex Model workload.</p>
      <p class="notice"><strong>Current phase:</strong> the first FFTW–vDSP vertical slice is preliminary evidence, not a provider recommendation. Later experiments add new immutable runs without replacing this record.</p>
    </section>
    <section class="section" aria-labelledby="latest-heading">
      <div class="section-heading">
        <div><p class="eyebrow">Latest published record</p><h2 id="latest-heading">{escaped(display_timestamp(latest_result['environment']['timestampUtc']))}</h2></div>
        <a class="button secondary" href="runs/{quote(latest_run_id)}/index.html">Permanent run page</a>
      </div>
      {result_summary_cards(latest)}
      {summary_timing_table(latest_result)}
      <p class="method-note">These within-run comparisons retain their publication status. Only reference runs may contribute to adoption statistics on the decision page.</p>
    </section>
    <section class="section" id="runs" aria-labelledby="runs-heading">
      <p class="eyebrow">Append-only archive</p>
      <h2 id="runs-heading">Published runs</h2>
      {archive(bundles)}
    </section>
    <section class="section" aria-labelledby="experiments-heading">
      <p class="eyebrow">Evidence chain</p>
      <h2 id="experiments-heading">Experiments</h2>
      {experiment_cards(catalog)}
    </section>
    <section class="section scope" aria-labelledby="scope-heading">
      <p class="eyebrow">Scope</p>
      <h2 id="scope-heading">Benchmark the kernel graph, not the nonlinear model</h2>
      <p>The laboratory reports primitive FFT and matrix performance separately from packing, ordering, and complete operator totals. It does not reimplement WVM’s nonlinear flux calculation.</p>
      <a href="methods/operators-and-representations/index.html">Read the methods and representation contract</a>
    </section>
    """
    return shell("Results", content)


def full_timing_table(result: dict) -> str:
    rows: list[str] = []
    for provider in result["providers"]:
        for item in provider["timings"]:
            samples = len(item["samplesSeconds"])
            median = "—" if item["medianSeconds"] is None else format_ms(item["medianSeconds"])
            rows.append(
                "<tr>"
                f'<th scope="row">{escaped(provider_name(provider))}</th>'
                f"<td>{escaped(item['scope'])}</td><td>{escaped(item['stage'])}</td>"
                f"<td>{escaped(item['direction'])}</td><td>{escaped(item['state'])}</td>"
                f'<td class="numeric">{median}</td><td class="numeric">{samples}</td>'
                f'<td class="numeric">{format_bytes(item["bytesMoved"])}</td>'
                "</tr>"
            )
    return f"""
      <div class="table-scroll"><table>
        <caption>All recorded components. Times are medians in milliseconds.</caption>
        <thead><tr><th scope="col">Provider</th><th scope="col">Scope</th><th scope="col">Stage</th><th scope="col">Direction</th><th scope="col">State</th><th scope="col">ms</th><th scope="col">n</th><th scope="col">Estimated movement</th></tr></thead>
        <tbody>{''.join(rows)}</tbody>
      </table></div>
    """


def execution_contract_details(provider: dict) -> str:
    execution = provider.get("executionContract")
    if execution is None:
        return '<p class="legacy-note">This grandfathered result predates the explicit placement contract. Its algorithm identifiers and component timings are preserved, but placement must not be inferred from this record.</p>'
    rows = []
    for direction in ("forward", "inverse"):
        contract = execution[direction]
        preservation = "included" if contract["preservationIncludedInPrimitiveTiming"] else "excluded"
        rows.append(
            "<tr>"
            f'<th scope="row">{escaped(direction)}</th>'
            f'<td>{escaped(contract["nativePlacement"])}</td>'
            f'<td>{escaped(contract["adapterPlacement"])}</td>'
            f'<td>{"yes" if contract["destroysNativeInput"] else "no"}</td>'
            f'<td>{escaped(preservation)}</td>'
            f'<td>{escaped(contract["physicalExtents"])}</td>'
            f'<td>{format_bytes(contract["reusableWorkBytes"])}</td>'
            "</tr>"
        )
    return f"""
      <div class="table-scroll"><table>
        <caption>Native and adapter storage semantics. Preservation describes primitive timing.</caption>
        <thead><tr><th scope="col">Direction</th><th scope="col">Native</th><th scope="col">Adapter</th><th scope="col">Destroys native input</th><th scope="col">Preservation work</th><th scope="col">Physical extents</th><th scope="col">Reusable work</th></tr></thead>
        <tbody>{''.join(rows)}</tbody>
      </table></div>
    """


def provider_details(result: dict) -> str:
    cards: list[str] = []
    for provider in result["providers"]:
        setup = provider["setup"]
        planning = provider["planning"]
        correctness = "".join(
            f'<li><span>{escaped(item["name"])}</span><strong>max {format_error(item["maximumRelativeError"])}'
            + (f' · L2 {format_error(item["relativeL2Error"])}' if item.get("relativeL2Error") is not None else "")
            + f'</strong><span class="status {"passed" if item["passed"] else "failed"}">{"passed" if item["passed"] else "failed"}</span></li>'
            for item in provider["correctness"]
        )
        cards.append(f"""
          <article class="provider-card">
            <p class="eyebrow">{escaped(provider['id'])}</p>
            <h3>{escaped(provider_name(provider))}</h3>
            <dl class="detail-list">
              <div><dt>Algorithm</dt><dd>{escaped(provider['algorithmId'])}</dd></div>
              <div><dt>Representation</dt><dd>{escaped(provider['nativeRepresentationId'])}</dd></div>
              <div><dt>Scheduling</dt><dd>{escaped(provider['schedulingId'])}, {provider['workers']} workers</dd></div>
              <div><dt>GEMM calls per execution</dt><dd>{provider.get('gemmCallsPerExecution', 0)}</dd></div>
              <div><dt>Setup total</dt><dd>{format_ms(setup['totalSeconds'])} ms</dd></div>
              <div><dt>Planning</dt><dd>{format_ms(planning['seconds'])} ms · {escaped(planning['configuration'])}</dd></div>
              <div><dt>Persistent memory</dt><dd>{format_bytes(provider['memory']['persistentBytes'])}</dd></div>
            </dl>
            <h4>Correctness</h4>
            <ul class="correctness-list">{correctness}</ul>
            <h4>Execution and storage contract</h4>
            {execution_contract_details(provider)}
          </article>
        """)
    return f'<div class="provider-grid">{"".join(cards)}</div>'


def environment_details(result: dict) -> str:
    environment = result["environment"]
    numeric_type = result.get("numericType", {"id": "float64 (legacy record)", "scalarBits": 64})
    fields = (
        ("Run ID", result["run"]["id"]),
        ("Numeric type", f"{numeric_type['id']} · {numeric_type['scalarBits']} bits"),
        ("Timestamp", display_timestamp(environment["timestampUtc"])),
        ("Machine", f"{environment['machineModel']} · {environment['cpuBrand']}"),
        ("Cores", f"{environment['totalCores']} total · {environment['performanceCores']} performance · {environment['efficiencyCores']} efficiency"),
        ("Memory", format_bytes(environment["physicalMemoryBytes"])),
        ("Operating system", environment["operatingSystem"]),
        ("Compiler", f"{environment['compiler']} {environment['compilerVersion']}"),
        ("Flags", environment["compilerFlags"]),
        ("Benchmark commit", f"{environment['gitCommit']} ({'dirty' if environment['gitDirty'] else 'clean'})"),
    )
    entries = "".join(f"<div><dt>{escaped(label)}</dt><dd>{escaped(value)}</dd></div>" for label, value in fields)
    return f'<dl class="detail-list environment-list">{entries}</dl>'


def experiment_links(publication: dict, root_prefix: str) -> str:
    links = [
        f'<a href="{root_prefix}experiments/{quote(experiment_id)}/index.html">{escaped(experiment_id)}</a>'
        for experiment_id in publication["experiments"]
    ]
    return ", ".join(links)


def build_run_page(bundle: PublishedBundle, root_prefix: str) -> str:
    result = bundle.result
    publication = bundle.publication
    run_id = result["run"]["id"]
    artifact_root = f"{root_prefix}artifacts/{quote(run_id)}"
    content = f"""
    <section class="hero compact">
      <p class="eyebrow">Immutable benchmark run · {escaped(run_id)}</p>
      <h1>{escaped(result['environment']['cpuBrand'])}</h1>
      <p class="lede">{result['workload']['Nx']} × {result['workload']['Ny']}, N<sub>z</sub>={result['workload']['Nz']}, fields={result['workload']['fields']} · {escaped(display_timestamp(result['environment']['timestampUtc']))}</p>
      <p>{publication_badge(publication['status'])} {escaped(publication['statusReason'])}</p>
      <p>Evidence for {experiment_links(publication, root_prefix)}.</p>
      <div class="button-row"><a class="button" href="{artifact_root}/result.json" download>Download result JSON</a><a class="button secondary" href="{artifact_root}/samples.csv" download>Download samples CSV</a></div>
    </section>
    <section class="section" aria-labelledby="summary-heading">
      <p class="eyebrow">Comparison</p><h2 id="summary-heading">Headline timings</h2>
      {result_summary_cards(bundle)}
      {summary_timing_table(result)}
      <p class="method-note">Publication status and numerical status are independent. This run remains visible even if later evidence supersedes or withdraws it.</p>
    </section>
    <section class="section" aria-labelledby="provider-heading">
      <p class="eyebrow">Providers</p><h2 id="provider-heading">Setup, correctness, and storage</h2>
      {provider_details(result)}
    </section>
    <section class="section" aria-labelledby="components-heading">
      <p class="eyebrow">Component ledger</p><h2 id="components-heading">Every measured piece</h2>
      {full_timing_table(result)}
    </section>
    <section class="section" aria-labelledby="environment-heading">
      <p class="eyebrow">Reproducibility</p><h2 id="environment-heading">Environment</h2>
      {environment_details(result)}
    </section>
    """
    return shell(run_id, content, root_prefix)


def definition_list(values: dict) -> str:
    labels = {
        "unitOfWork": "Logical unit of work",
        "candidates": "Candidates",
        "workloads": "Workloads",
        "controlledVariables": "Held fixed",
        "changedVariables": "Changed",
        "timedBoundary": "Timed boundary",
        "excludedWork": "Excluded work",
        "allocationPolicy": "Allocation policy",
        "oracle": "Oracle",
        "tolerance": "Tolerance",
        "capabilityHandling": "Capability handling",
        "canSupport": "Can support",
        "cannotSupport": "Cannot support",
    }
    entries = "".join(
        f"<div><dt>{escaped(labels.get(key, key))}</dt><dd>{escaped(value)}</dd></div>"
        for key, value in values.items()
    )
    return f'<dl class="detail-list evidence-definition">{entries}</dl>'


def vdsp_batch_evidence_table(bundles: list[PublishedBundle]) -> str:
    rows: list[str] = []
    def sort_key(bundle: PublishedBundle) -> tuple[int, int, str, int]:
        provider = next(item for item in bundle.result["providers"] if item["id"] == "accelerate-vdsp")
        workload = bundle.result["workload"]
        return workload["Nx"], workload["Ny"], provider["algorithmId"], provider["workers"]

    for bundle in sorted(bundles, key=sort_key):
        result = bundle.result
        provider = next((item for item in result["providers"] if item["id"] == "accelerate-vdsp"), None)
        if provider is None:
            continue
        workload = result["workload"]
        run = result["run"]
        publication = bundle.publication
        raw_forward = timing(provider, "primitive", "forward")
        raw_inverse = timing(provider, "primitive", "inverse")
        scheduler = stage_timing(provider, "diagnostic-component", "batch scheduler empty dispatch", "shared")
        row_forward = stage_timing(provider, "primitive-component", "real row FFTs", "forward")
        row_inverse = stage_timing(provider, "primitive-component", "real row FFTs", "inverse")
        column_forward = stage_timing(
            provider, "primitive-component", "complex column FFTs and Hermitian boundaries", "forward"
        )
        column_inverse = stage_timing(
            provider, "primitive-component", "complex column FFTs and Hermitian boundaries", "inverse"
        )
        adapter_forward = timing(provider, "adapter-total", "forward")
        adapter_inverse = timing(provider, "adapter-total", "inverse")

        def pair(forward: dict | None, inverse: dict | None) -> str:
            if forward is None or inverse is None:
                return '<span class="muted">not measured</span>'
            if forward["state"] != "executed" or inverse["state"] != "executed":
                state = forward["state"] if forward["state"] == inverse["state"] else "mixed states"
                return f'<span class="muted">{escaped(state)}</span>'
            return f'{format_ms(forward["medianSeconds"])} / {format_ms(inverse["medianSeconds"])}'

        def interval_pair(forward: dict | None, inverse: dict | None) -> str:
            return f"{escaped(timing_with_interval(forward))}<br>{escaped(timing_with_interval(inverse))}"

        run_id = run["id"]
        rows.append(
            "<tr>"
            f'<td><a href="../../runs/{quote(run_id)}/index.html">{escaped(run_id)}</a><br>'
            f'<span class="muted">{run["samples"]} samples · {publication_badge(publication["status"])}</span></td>'
            f'<td class="numeric">{workload["Nx"]} × {workload["Ny"]}<br>N<sub>z</sub>={workload["Nz"]}, fields={workload["fields"]}</td>'
            f'<td>{escaped(provider["algorithmId"])}<br><span class="muted">{escaped(provider["schedulingId"])}</span></td>'
            f'<td class="numeric">{provider["workers"]}</td>'
            f'<td class="numeric">{interval_pair(raw_forward, raw_inverse)}</td>'
            f'<td class="numeric">{escaped(coefficient_of_variation(raw_forward))} / {escaped(coefficient_of_variation(raw_inverse))}</td>'
            f'<td class="numeric">{format_ms(scheduler["medianSeconds"]) if scheduler is not None else "not measured"}</td>'
            f'<td class="numeric">{pair(row_forward, row_inverse)}</td>'
            f'<td class="numeric">{pair(column_forward, column_inverse)}</td>'
            f'<td class="numeric">{pair(adapter_forward, adapter_inverse)}</td>'
            f'<td class="numeric">{format_bytes(provider["memory"]["persistentBytes"])}<br>'
            f'<span class="muted">scratch {format_bytes(provider["memory"]["scratchBytes"])}</span></td>'
            f'<td class="numeric">{format_error(maximum_correctness_error(provider))}</td>'
            "</tr>"
        )
    if not rows:
        return ""
    return (
        '<div class="table-scroll"><table class="experiment-evidence-table issue6-evidence-table">'
        '<caption>Raw FFT rows show forward then inverse median milliseconds with deterministic percentile-bootstrap 95% intervals in brackets. Raw FFT is the authoritative batch wall time. '
        'CV is the sample standard deviation divided by the sample mean. Empty-dispatch time is a non-additive scheduler diagnostic. '
        'Row and column phases include their own dispatches; adapter includes all WVM-compatible packing and conversion.</caption>'
        '<thead><tr><th scope="col">Run</th><th scope="col">Workload</th><th scope="col">Algorithm / scheduler</th>'
        '<th scope="col">Workers</th><th scope="col">Raw FFT</th><th scope="col">Raw CV</th>'
        '<th scope="col">Empty dispatch</th><th scope="col">Row phase</th><th scope="col">Column phase</th>'
        '<th scope="col">Adapter</th><th scope="col">Explicit memory</th><th scope="col">Max error</th></tr></thead>'
        f'<tbody>{"".join(rows)}</tbody></table></div>'
    )


def fftw_candidate_name(provider: dict) -> str:
    algorithm = provider["algorithmId"].removeprefix("wvm-guru64-")
    scheduling = provider["scheduling"]
    internal = scheduling["internalWorkers"]
    outer = scheduling["outerWorkers"]
    if outer == 1:
        topology = f"internal {internal}"
    elif internal == 1:
        topology = f"outer {outer}"
    else:
        topology = f"hybrid {internal}×{outer}"
    return f"{algorithm} · {topology}"


def fftw_primary_provider(result: dict) -> dict | None:
    interleaved = next(
        (item for item in result["providers"] if item["id"] == "fftw"), None
    )
    if interleaved is not None:
        return interleaved
    return next(
        (item for item in result["providers"] if item["id"] == "fftw-split"), None
    )


def fftw_candidate_signature(provider: dict) -> tuple[str, str, int, int, str]:
    scheduling = provider["scheduling"]
    return (
        provider["algorithmId"],
        provider["schedulingId"],
        int(scheduling["internalWorkers"]),
        int(scheduling["outerWorkers"]),
        provider["nativeRepresentationId"],
    )


def fftw_split_evidence_table(bundles: list[PublishedBundle]) -> str:
    rows: list[str] = []
    for bundle in bundles:
        result = bundle.result
        interleaved = next((item for item in result["providers"] if item["id"] == "fftw"), None)
        split = next((item for item in result["providers"] if item["id"] == "fftw-split"), None)
        if interleaved is None or split is None:
            continue
        workload = result["workload"]
        run = result["run"]

        def pair(provider: dict, scope: str) -> str:
            forward = timing(provider, scope, "forward")
            inverse = timing(provider, scope, "inverse")
            if forward is None or inverse is None:
                return "not measured"
            return f'{format_ms(forward["medianSeconds"])} / {format_ms(inverse["medianSeconds"])}'

        def ratio(scope: str) -> str:
            values: list[float] = []
            for direction in ("forward", "inverse"):
                split_timing = timing(split, scope, direction)
                interleaved_timing = timing(interleaved, scope, direction)
                if split_timing is None or interleaved_timing is None:
                    return "not measured"
                values.append(float(split_timing["medianSeconds"]) / float(interleaved_timing["medianSeconds"]))
            return f"{values[0]:.3f}× / {values[1]:.3f}×"

        def raw_cv(provider: dict) -> str:
            forward = timing(provider, "primitive", "forward")
            inverse = timing(provider, "primitive", "inverse")
            return f"{coefficient_of_variation(forward)} / {coefficient_of_variation(inverse)}"

        conversion_forward = stage_timing(
            split, "adapter-component", "split-to-interleaved conversion", "forward"
        )
        conversion_inverse = stage_timing(
            split, "adapter-component", "interleaved-to-split conversion", "inverse"
        )
        direct_retention = stage_timing(
            split, "operator-component", "direct split horizontal retention", "forward"
        )
        direct_embedding = stage_timing(
            split, "operator-component", "direct split horizontal embedding", "inverse"
        )
        capability = stage_timing(split, "capability", "exact WVM-order split in-place", "shared")
        capability_state = capability["state"] if capability is not None else "not recorded"
        run_id = run["id"]
        rows.append(
            "<tr>"
            f'<td><a href="../../runs/{quote(run_id)}/index.html">{escaped(run_id)}</a><br>'
            f'<span class="muted">{run["samples"]} samples · {publication_badge(bundle.publication["status"])}</span></td>'
            f'<td class="numeric">{workload["Nx"]} × {workload["Ny"]}<br>N<sub>z</sub>={workload["Nz"]}, fields={workload["fields"]}</td>'
            f'<td>{escaped(fftw_candidate_name(interleaved))}<br><span class="muted">{escaped(split["algorithmId"])}</span></td>'
            f'<td class="numeric">{escaped(pair(interleaved, "primitive"))}<br><span class="muted">interleaved</span><br>'
            f'{escaped(pair(split, "primitive"))}<br><span class="muted">split</span></td>'
            f'<td class="numeric">{escaped(ratio("primitive"))}</td>'
            f'<td class="numeric">{escaped(raw_cv(interleaved))}<br><span class="muted">interleaved</span><br>'
            f'{escaped(raw_cv(split))}<br><span class="muted">split</span></td>'
            f'<td class="numeric">{format_ms(conversion_forward["medianSeconds"]) if conversion_forward else "not measured"} / '
            f'{format_ms(conversion_inverse["medianSeconds"]) if conversion_inverse else "not measured"}</td>'
            f'<td class="numeric">{escaped(ratio("adapter-total"))}</td>'
            f'<td class="numeric">{escaped(ratio("uninstrumented-total"))}</td>'
            f'<td class="numeric">{format_ms(direct_retention["medianSeconds"]) if direct_retention else "not measured"} / '
            f'{format_ms(direct_embedding["medianSeconds"]) if direct_embedding else "not measured"}</td>'
            f'<td class="numeric">{format_ms(interleaved["setup"]["totalSeconds"])} / '
            f'{format_ms(split["setup"]["totalSeconds"])}</td>'
            f'<td class="numeric">{format_bytes(interleaved["planning"]["temporaryBytes"])} / '
            f'{format_bytes(split["planning"]["temporaryBytes"])}</td>'
            f'<td>{escaped(capability_state)}<br><span class="muted">out-of-place native paths measured</span></td>'
            f'<td class="numeric">{format_error(max(maximum_correctness_error(interleaved) or 0.0, maximum_correctness_error(split) or 0.0))}</td>'
            "</tr>"
        )
    if not rows:
        return ""
    return (
        '<h3>Paired split-versus-interleaved increment</h3>'
        '<p>This increment asks whether FFTW’s split guru64 API beats its matched interleaved API at the raw transform boundary, after WVM-compatible conversion, or when the split layout persists through retained-mode selection and embedding. Each row holds planning, alignment, worker topology, fixture, and sampling fixed while changing only the FFTW complex API and physical layout. Providers are measured as sequential blocks in one process; the ratios are matched-configuration medians, not alternating-sample confidence intervals.</p>'
        '<div class="table-scroll"><table class="experiment-evidence-table issue4-split-evidence-table">'
        '<caption>Times are forward / inverse median milliseconds. Ratio columns are split divided by interleaved, so values below one favor split for that boundary. CV is the sample standard deviation divided by the mean. Conversion is reported separately. The retained ratio compares direct persistent split selection/embedding with the existing interleaved retained operator. Setup and planning memory are interleaved / split.</caption>'
        '<thead><tr><th scope="col">Run</th><th scope="col">Workload</th><th scope="col">Matched strategy</th>'
        '<th scope="col">Raw FFT interleaved / split</th><th scope="col">Raw split / interleaved</th><th scope="col">Raw CV interleaved / split</th>'
        '<th scope="col">Conversion</th><th scope="col">Adapter ratio</th><th scope="col">Retained ratio</th>'
        '<th scope="col">Direct split select / embed</th><th scope="col">Setup</th><th scope="col">Planning memory</th><th scope="col">Exact WVM-order in-place</th>'
        '<th scope="col">Max error</th></tr></thead>'
        f'<tbody>{"".join(rows)}</tbody></table></div>'
    )


def fftw_screen_cohort(bundles: list[PublishedBundle]) -> list[PublishedBundle]:
    current = [
        bundle
        for bundle in bundles
        if bundle.publication["status"] in ("reference", "preliminary")
    ]
    if not current:
        return []
    status = "reference" if any(bundle.publication["status"] == "reference" for bundle in current) else "preliminary"
    same_status = [bundle for bundle in current if bundle.publication["status"] == status]
    latest = max(same_status, key=lambda bundle: bundle.result["environment"]["timestampUtc"])
    increment_id = latest.publication.get("incrementId")
    if increment_id:
        return [
            bundle
            for bundle in same_status
            if bundle.publication.get("incrementId") == increment_id
        ]
    commit = latest.result["environment"]["gitCommit"]
    return [
        bundle
        for bundle in same_status
        if bundle.result["environment"]["gitCommit"] == commit
    ]


def fftw_screen_classifications(bundles: list[PublishedBundle]) -> tuple[dict[str, str], list[PublishedBundle]]:
    cohort = fftw_screen_cohort(bundles)
    classifications: dict[str, str] = {}
    workload_groups: dict[
        tuple[int, int, int, int],
        dict[tuple[str, str, int, int, str], list[tuple[PublishedBundle, tuple[float, float, float]]]],
    ] = {}
    for bundle in cohort:
        result = bundle.result
        provider = fftw_primary_provider(result)
        if provider is None:
            continue
        run_id = result["run"]["id"]
        if provider["planning"].get("budgetExhausted", False):
            classifications[run_id] = "infeasible within planning budget"
            continue
        raw_forward = timing(provider, "primitive", "forward")
        raw_inverse = timing(provider, "primitive", "inverse")
        if raw_forward is None or raw_inverse is None:
            classifications[run_id] = "incomplete"
            continue
        workload = result["workload"]
        key = workload["Nx"], workload["Ny"], workload["Nz"], workload["fields"]
        objectives = (
            float(raw_forward["medianSeconds"]),
            float(raw_inverse["medianSeconds"]),
            float(provider["setup"]["totalSeconds"]),
        )
        workload_groups.setdefault(key, {}).setdefault(
            fftw_candidate_signature(provider), []
        ).append((bundle, objectives))

    for candidates in workload_groups.values():
        aggregates = [
            (
                signature,
                records,
                tuple(statistics.median(objective[index] for _, objective in records)
                      for index in range(3)),
            )
            for signature, records in candidates.items()
        ]
        for signature, records, objectives in aggregates:
            dominated = any(
                other_signature != signature
                and all(left <= right for left, right in zip(other_objectives, objectives))
                and any(left < right for left, right in zip(other_objectives, objectives))
                for other_signature, _, other_objectives in aggregates
            )
            classification = "dominated" if dominated else "Pareto"
            for bundle, _ in records:
                classifications[bundle.result["run"]["id"]] = classification
    return classifications, cohort


def fftw_strategy_evidence_table(bundles: list[PublishedBundle]) -> str:
    classifications, cohort = fftw_screen_classifications(bundles)
    cohort_ids = {bundle.result["run"]["id"] for bundle in cohort}
    rows: list[str] = []

    def sort_key(bundle: PublishedBundle) -> tuple[int, int, int, str, int, int]:
        result = bundle.result
        provider = fftw_primary_provider(result)
        if provider is None:
            return (0, 0, 9, "missing", 0, 0)
        workload = result["workload"]
        rank = {"Pareto": 0, "dominated": 1, "infeasible within planning budget": 2}.get(
            classifications.get(result["run"]["id"], "archived cohort"), 3
        )
        scheduling = provider["scheduling"]
        return (
            workload["Nx"],
            workload["Ny"],
            rank,
            provider["algorithmId"],
            scheduling["internalWorkers"],
            scheduling["outerWorkers"],
        )

    for bundle in sorted(bundles, key=sort_key):
        result = bundle.result
        provider = fftw_primary_provider(result)
        if provider is None:
            continue
        workload = result["workload"]
        run = result["run"]
        publication = bundle.publication
        raw_forward = timing(provider, "primitive", "forward")
        raw_inverse = timing(provider, "primitive", "inverse")
        adapter_forward = timing(provider, "adapter-total", "forward")
        adapter_inverse = timing(provider, "adapter-total", "inverse")
        retained_forward = timing(provider, "uninstrumented-total", "forward")
        retained_inverse = timing(provider, "uninstrumented-total", "inverse")
        scheduler = stage_timing(provider, "diagnostic-component", "batch scheduler empty dispatch", "shared")
        setup = provider["setup"]
        planning = provider["planning"]

        def interval_pair(forward: dict | None, inverse: dict | None) -> str:
            return f"{escaped(timing_with_interval(forward))}<br>{escaped(timing_with_interval(inverse))}"

        def pair(forward: dict | None, inverse: dict | None) -> str:
            if forward is None or inverse is None:
                return '<span class="muted">not measured</span>'
            return f'{format_ms(forward["medianSeconds"])} / {format_ms(inverse["medianSeconds"])}'

        scheduler_display = (
            format_ms(scheduler["medianSeconds"])
            if scheduler is not None and scheduler["state"] == "executed"
            else "elided"
        )
        time_limit = float(planning.get("timeLimitSeconds", 0.0))
        budget_display = (
            f'{time_limit:g} s/call · {"exhausted" if planning.get("budgetExhausted", False) else "not exhausted"}'
            if time_limit > 0.0
            else "unlimited"
        )
        run_id = run["id"]
        screen = classifications.get(run_id, "archived cohort" if run_id not in cohort_ids else "incomplete")
        rows.append(
            "<tr>"
            f'<td><a href="../../runs/{quote(run_id)}/index.html">{escaped(run_id)}</a><br>'
            f'<span class="muted">{run["samples"]} samples · {publication_badge(publication["status"])}</span></td>'
            f'<td class="numeric">{workload["Nx"]} × {workload["Ny"]}<br>N<sub>z</sub>={workload["Nz"]}, fields={workload["fields"]}</td>'
            f'<td>{escaped(fftw_candidate_name(provider))}<br><span class="muted">{escaped(provider["algorithmId"])}<br>'
            f'{escaped(provider["schedulingId"])}</span></td>'
            f'<td class="numeric">{interval_pair(raw_forward, raw_inverse)}</td>'
            f'<td class="numeric">{escaped(coefficient_of_variation(raw_forward))} / {escaped(coefficient_of_variation(raw_inverse))}</td>'
            f'<td class="numeric">{pair(adapter_forward, adapter_inverse)}</td>'
            f'<td class="numeric">{pair(retained_forward, retained_inverse)}</td>'
            f'<td class="numeric">{format_ms(setup["totalSeconds"])} total<br>'
            f'<span class="muted">plan {format_ms(planning["seconds"])}; wisdom '
            f'{format_ms(setup.get("wisdomGenerationSeconds", 0.0))} / '
            f'{format_ms(setup.get("wisdomImportSeconds", 0.0))}</span></td>'
            f'<td>{escaped(budget_display)}<br><span class="muted">wisdom {format_bytes(planning.get("wisdomBytes", 0))}</span></td>'
            f'<td class="numeric">{scheduler_display}</td>'
            f'<td class="numeric">plan {format_bytes(planning["temporaryBytes"])}<br>'
            f'<span class="muted">persistent {format_bytes(provider["memory"]["persistentBytes"])}; '
            f'scratch {format_bytes(provider["memory"]["scratchBytes"])}</span></td>'
            f'<td class="numeric">{format_error(maximum_correctness_error(provider))}</td>'
            f'<td>{escaped(screen)}</td>'
            "</tr>"
        )
    if not rows:
        return ""
    strategy_table = (
        '<div class="table-scroll"><table class="experiment-evidence-table issue4-evidence-table">'
        '<caption>Raw FFT rows show forward then inverse median milliseconds with deterministic percentile-bootstrap 95% intervals in brackets. '
        'CV is the sample standard deviation divided by the sample mean. Adapter and retained totals are forward / inverse medians. '
        'Wisdom setup is generation / import. Empty-dispatch time is a non-additive scheduler diagnostic. Every current-cohort candidate is shown as Pareto, dominated, or infeasible within its planning budget.</caption>'
        '<thead><tr><th scope="col">Run</th><th scope="col">Workload</th><th scope="col">Plan / alignment / wisdom / topology</th>'
        '<th scope="col">Raw FFT</th><th scope="col">Raw CV</th><th scope="col">Adapter</th><th scope="col">Retained total</th>'
        '<th scope="col">Setup</th><th scope="col">Planning budget</th><th scope="col">Empty dispatch</th>'
        '<th scope="col">Memory</th><th scope="col">Max error</th><th scope="col">Screen</th></tr></thead>'
        f'<tbody>{"".join(rows)}</tbody></table></div>'
    )
    split_table = fftw_split_evidence_table(bundles)
    if split_table:
        return (
            split_table
            + '<h3>Append-only FFTW strategy archive</h3>'
            + '<p>The earlier planning, alignment, wisdom, scheduling, split-layout, and native-order increments remain visible in full. New reference runs extend this archive without replacing their preliminary baselines.</p>'
            + strategy_table
        )
    return strategy_table


def fftw_strategy_synthesis(bundles: list[PublishedBundle]) -> str:
    classifications, cohort = fftw_screen_classifications(bundles)
    if not cohort:
        return ""
    records: list[tuple[dict, dict, str]] = []
    for bundle in cohort:
        result = bundle.result
        provider = fftw_primary_provider(result)
        if provider is not None:
            records.append((result, provider, classifications[result["run"]["id"]]))
    if not records:
        return ""

    grouped_records: dict[
        tuple[int, int, int, int, tuple[str, str, int, int, str]],
        list[tuple[dict, dict, str]],
    ] = {}
    for record in records:
        result, provider, _ = record
        workload = result["workload"]
        key = (
            workload["Nx"], workload["Ny"], workload["Nz"], workload["fields"],
            fftw_candidate_signature(provider),
        )
        grouped_records.setdefault(key, []).append(record)

    rows: list[str] = []
    workload_keys = sorted(
        {
            (result["workload"]["Nx"], result["workload"]["Ny"], result["workload"]["Nz"], result["workload"]["fields"])
            for result, _, _ in records
        }
    )
    for key in workload_keys:
        workload_records = [
            candidate_records
            for group_key, candidate_records in grouped_records.items()
            if group_key[:4] == key
        ]
        pareto = sorted(
            (candidate_records for candidate_records in workload_records
             if candidate_records[0][2] == "Pareto"),
            key=lambda candidate_records: statistics.median(
                float(record[1]["setup"]["totalSeconds"])
                for record in candidate_records
            ),
        )
        entries = []
        for candidate_records in pareto:
            provider = candidate_records[0][1]
            forward_values = [
                float(item["medianSeconds"])
                for _, candidate_provider, _ in candidate_records
                if (item := timing(candidate_provider, "primitive", "forward")) is not None
            ]
            inverse_values = [
                float(item["medianSeconds"])
                for _, candidate_provider, _ in candidate_records
                if (item := timing(candidate_provider, "primitive", "inverse")) is not None
            ]
            if not forward_values or not inverse_values:
                continue
            setup_value = statistics.median(
                float(candidate_provider["setup"]["totalSeconds"])
                for _, candidate_provider, _ in candidate_records
            )
            entries.append(
                f'<strong>{escaped(fftw_candidate_name(provider))}</strong><br>'
                f'<span class="muted">{format_ms(statistics.median(forward_values))} / '
                f'{format_ms(statistics.median(inverse_values))} ms; setup '
                f'{format_ms(setup_value)} ms; {len(candidate_records)} process run(s)</span>'
            )
        rows.append(
            "<tr>"
            f'<th scope="row">{key[0]} × {key[1]}<br><span class="muted">N<sub>z</sub>={key[2]}, fields={key[3]}</span></th>'
            f'<td>{"<hr>".join(entries)}</td>'
            "</tr>"
        )

    group_classifications = [candidate_records[0][2] for candidate_records in grouped_records.values()]
    counts = {
        name: group_classifications.count(name) for name in set(group_classifications)
    }
    max_error = max(maximum_correctness_error(provider) or 0.0 for _, provider, _ in records)
    commit = cohort[0].result["environment"]["gitCommit"]
    status = cohort[0].publication["status"]
    paired = []
    for bundle in bundles:
        if bundle.publication["status"] not in ("reference", "preliminary"):
            continue
        interleaved = next((item for item in bundle.result["providers"] if item["id"] == "fftw"), None)
        split = next((item for item in bundle.result["providers"] if item["id"] == "fftw-split"), None)
        if interleaved is not None and split is not None:
            paired.append((interleaved, split))

    split_synthesis = ""
    split_scope_note = "It excludes FFTW split APIs and most production fields/workloads."
    if paired:
        def geometric_ratio(scope: str, direction: str) -> float:
            ratios = []
            for interleaved, split in paired:
                interleaved_timing = timing(interleaved, scope, direction)
                split_timing = timing(split, scope, direction)
                if interleaved_timing is None or split_timing is None:
                    continue
                ratios.append(float(split_timing["medianSeconds"]) / float(interleaved_timing["medianSeconds"]))
            return math.exp(statistics.fmean(math.log(value) for value in ratios))

        split_error = max(maximum_correctness_error(split) or 0.0 for _, split in paired)
        split_synthesis = f"""
      <h3>Split-layout diagnostic</h3>
      <p>Across {len(paired)} matched configuration-workloads, the geometric-mean split/interleaved ratios are {geometric_ratio("primitive", "forward"):.3f}× forward and {geometric_ratio("primitive", "inverse"):.3f}× inverse for raw FFT, {geometric_ratio("adapter-total", "forward"):.3f}× and {geometric_ratio("adapter-total", "inverse"):.3f}× for the WVM-compatible adapter, and {geometric_ratio("uninstrumented-total", "forward"):.3f}× and {geometric_ratio("uninstrumented-total", "inverse"):.3f}× for the retained horizontal operator. Values below one favor split. The split maximum relative error is {format_error(split_error)}.</p>
      <p>FFTW 3.3.11 documents rank-greater-than-one guru split real transforms as out-of-place only, and both exact WVM-order in-place planners reject the tested alias. The measured new-array path therefore uses one contiguous allocation with a fixed [real][imaginary] component separation matching the planning buffers. Provider-native layouts with different strides remain a later algorithm experiment.</p>
      """
        split_scope_note = "It includes a bounded paired split/interleaved screen, but still excludes most production fields/workloads and provider-native split orders with different strides."

    native_order_synthesis = ""
    comparison_pairs: list[
        tuple[list[tuple[dict, dict, str]], list[tuple[dict, dict, str]]]
    ] = []
    for key in workload_keys:
        workload_candidates = [
            candidate_records
            for group_key, candidate_records in grouped_records.items()
            if group_key[:4] == key
        ]
        baseline = next(
            (
                candidate_records
                for candidate_records in workload_candidates
                if candidate_records[0][1]["nativeRepresentationId"] ==
                    "wvm-frequency-major-interleaved-half-spectrum"
                and candidate_records[0][1]["scheduling"]["internalWorkers"] == 1
                and candidate_records[0][1]["scheduling"]["outerWorkers"] == 12
            ),
            None,
        )
        plane_major = next(
            (
                candidate_records
                for candidate_records in workload_candidates
                if candidate_records[0][1]["nativeRepresentationId"] ==
                    "plane-major-interleaved-half-spectrum"
                and candidate_records[0][1]["scheduling"]["internalWorkers"] == 1
                and candidate_records[0][1]["scheduling"]["outerWorkers"] == 12
            ),
            None,
        )
        if baseline is not None and plane_major is not None:
            comparison_pairs.append((baseline, plane_major))

    if comparison_pairs:
        def aggregate_timing(
            candidate_records: list[tuple[dict, dict, str]], scope: str, direction: str
        ) -> float:
            values = [
                float(item["medianSeconds"])
                for _, provider, _ in candidate_records
                if (item := timing(provider, scope, direction)) is not None
            ]
            return statistics.median(values)

        def comparison(scope: str, direction: str) -> tuple[float, float, float, int]:
            ratios = [
                aggregate_timing(plane_major, scope, direction) /
                aggregate_timing(baseline, scope, direction)
                for baseline, plane_major in comparison_pairs
            ]
            return (
                math.exp(statistics.fmean(math.log(value) for value in ratios)),
                min(ratios),
                max(ratios),
                sum(value < 1.0 for value in ratios),
            )

        def comparison_text(scope: str, direction: str) -> str:
            geometric, minimum, maximum, wins = comparison(scope, direction)
            return (
                f"{geometric:.3f}× ({wins}/{len(comparison_pairs)} wins; "
                f"range {minimum:.3f}×–{maximum:.3f}×)"
            )

        native_order_synthesis = f"""
      <h3>Provider-native order result</h3>
      <p>The matched production comparison holds FFTW <code>MEASURE</code>, unaligned execution, one internal worker, and 12 persistent outer workers fixed while changing only the complete half-spectrum order. Plane-major divided by WVM frequency-major is {comparison_text("primitive", "forward")} for raw forward FFT and {comparison_text("primitive", "inverse")} for raw inverse FFT.</p>
      <p>When plane-major storage persists through mode-keyed antialiasing, the complete retained-operator ratios improve to {comparison_text("uninstrumented-total", "forward")} forward and {comparison_text("uninstrumented-total", "inverse")} inverse. In contrast, materializing the complete WVM-compatible order costs {comparison_text("adapter-total", "forward")} forward and {comparison_text("adapter-total", "inverse")} inverse. Thus the native order is an issue #7 survivor only as a persistent representation with direct retained selection/embedding; it is not a faster drop-in full-spectrum WVM adapter.</p>
      <p>The reference campaign used {len(records)} clean process runs: three independently planned processes, 21 steady-state samples per process, five finalists, and six production workloads. The raw/setup Pareto set is shape-specific and retains low-setup internal-12, outer-12, outer-16, or hybrid 4×3 plane-major policies where non-dominated. The WVM outer-12 baseline is dominated in the final raw/setup frontier but remains the matched production comparator.</p>
      """

    completion_note = (
        "This is the final issue #4 M4 Max reference Pareto set, not a WVM adoption decision. "
        "Issue #7 must compare these full-FFT survivors with the pruned retained transform, "
        "and the later combined pipeline determines adoption."
        if status == "reference"
        else (
            "This is a bounded strategy screen, not the final issue #4 Pareto set or a WVM "
            f"adoption decision. {split_scope_note}"
        )
    )
    return f"""
      <h3>Reproducible Pareto screen</h3>
      <p>The current {escaped(status)} cohort is commit <code>{escaped(commit)}</code>. Within each workload, repeated processes with the same algorithm, representation, and worker topology are combined by the median of their process medians. A candidate is Pareto when no other eligible candidate is no slower in aggregate raw forward FFT, raw inverse FFT, and total setup while being strictly better in at least one. Total setup includes allocation, planning, and wisdom generation/import.</p>
      <p>{counts.get("Pareto", 0)} workload-candidates are Pareto, {counts.get("dominated", 0)} are dominated, and {counts.get("infeasible within planning budget", 0)} are marked infeasible within the configured per-plan feasibility budget. All {len(records)} process runs passed the fixed-reference, retained-operator, and round-trip checks; the cohort maximum relative error is {format_error(max_error)}.</p>
      <div class="table-scroll"><table class="experiment-evidence-table">
        <caption>Pareto candidates ordered by total setup. Times are raw forward / inverse medians followed by total setup; all are milliseconds.</caption>
        <thead><tr><th scope="col">Workload</th><th scope="col">Non-dominated candidates</th></tr></thead>
        <tbody>{"".join(rows)}</tbody>
      </table></div>
      {native_order_synthesis}
      {split_synthesis}
      <p class="method-note">{escaped(completion_note)} Budget-limited PATIENT and EXHAUSTIVE plans remain visible as preliminary feasibility evidence and are not part of this reference cohort.</p>
    """


def vdsp_batch_synthesis(bundles: list[PublishedBundle]) -> str:
    records: list[tuple[dict, dict, dict]] = []
    for bundle in bundles:
        vdsp = next((item for item in bundle.result["providers"] if item["id"] == "accelerate-vdsp"), None)
        fftw = next((item for item in bundle.result["providers"] if item["id"] == "fftw"), None)
        if vdsp is not None and fftw is not None:
            records.append((bundle.result, vdsp, fftw))
    if not records:
        return ""

    def raw(provider: dict, direction: str) -> float:
        item = timing(provider, "primitive", direction)
        if item is None:
            raise ValueError(f"Missing primitive {direction} timing for {provider['id']}")
        return float(item["medianSeconds"])

    def candidate_name(provider: dict) -> str:
        algorithm = provider["algorithmId"]
        for name in ("direct-persistent", "direct-gcd", "separable-persistent", "separable-gcd"):
            if algorithm.endswith(name):
                return name
        return algorithm

    profiles = sorted({result["run"]["profile"] for result, _, _ in records})
    rows: list[str] = []
    candidate_algorithms = {
        vdsp["algorithmId"]
        for _, vdsp, _ in records
        if not vdsp["algorithmId"].endswith("direct-persistent")
    }
    advancing = set(candidate_algorithms)
    for profile in profiles:
        profile_records = [record for record in records if record[0]["run"]["profile"] == profile]
        baseline = [record for record in profile_records if record[1]["algorithmId"].endswith("direct-persistent")]
        alternatives = [record for record in profile_records if not record[1]["algorithmId"].endswith("direct-persistent")]
        workload = profile_records[0][0]["workload"]
        baseline_forward = min(baseline, key=lambda record: raw(record[1], "forward"))
        baseline_inverse = min(baseline, key=lambda record: raw(record[1], "inverse"))
        alternative_forward = min(alternatives, key=lambda record: raw(record[1], "forward"))
        alternative_inverse = min(alternatives, key=lambda record: raw(record[1], "inverse"))
        fftw_forward = min(profile_records, key=lambda record: raw(record[2], "forward"))
        fftw_inverse = min(profile_records, key=lambda record: raw(record[2], "inverse"))
        best_vdsp_forward = min(profile_records, key=lambda record: raw(record[1], "forward"))
        best_vdsp_inverse = min(profile_records, key=lambda record: raw(record[1], "inverse"))
        forward_improvement = 100.0 * (
            1.0 - raw(alternative_forward[1], "forward") / raw(baseline_forward[1], "forward")
        )
        inverse_improvement = 100.0 * (
            1.0 - raw(alternative_inverse[1], "inverse") / raw(baseline_inverse[1], "inverse")
        )
        for algorithm in list(advancing):
            candidate = [record for record in profile_records if record[1]["algorithmId"] == algorithm]
            improves_forward = min(raw(record[1], "forward") for record in candidate) <= 0.9 * raw(
                baseline_forward[1], "forward"
            )
            improves_inverse = min(raw(record[1], "inverse") for record in candidate) <= 0.9 * raw(
                baseline_inverse[1], "inverse"
            )
            if not (improves_forward and improves_inverse):
                advancing.remove(algorithm)

        rows.append(
            "<tr>"
            f'<th scope="row">{workload["Nx"]} × {workload["Ny"]}<br><span class="muted">N<sub>z</sub>={workload["Nz"]}, fields={workload["fields"]}</span></th>'
            f'<td class="numeric">{format_ms(raw(baseline_forward[1], "forward"))} (w{baseline_forward[1]["workers"]}) / '
            f'{format_ms(raw(baseline_inverse[1], "inverse"))} (w{baseline_inverse[1]["workers"]})</td>'
            f'<td class="numeric">{format_ms(raw(alternative_forward[1], "forward"))} '
            f'({escaped(candidate_name(alternative_forward[1]))}, w{alternative_forward[1]["workers"]}) / '
            f'{format_ms(raw(alternative_inverse[1], "inverse"))} '
            f'({escaped(candidate_name(alternative_inverse[1]))}, w{alternative_inverse[1]["workers"]})</td>'
            f'<td class="numeric">{forward_improvement:+.1f}% / {inverse_improvement:+.1f}%</td>'
            f'<td class="numeric">{raw(best_vdsp_forward[1], "forward") / raw(fftw_forward[2], "forward"):.2f}× / '
            f'{raw(best_vdsp_inverse[1], "inverse") / raw(fftw_inverse[2], "inverse"):.2f}×</td>'
            "</tr>"
        )

    advancement = (
        "New candidates advancing to the full production matrix: "
        + ", ".join(sorted(advancing))
        if advancing
        else "No new GCD or separable candidate clears the 10% advancement screen on both directions and both workloads."
    )
    return f"""
      <h3>Qualified negative feasibility conclusion</h3>
      <p>The advancement screen carries a new candidate forward only when its best worker count reduces both raw forward and inverse medians by at least 10% on both representative workloads, without moving work outside the primitive boundary. {escaped(advancement)}</p>
      <div class="table-scroll"><table class="experiment-evidence-table">
        <caption>Best median raw times in milliseconds are forward / inverse. Positive improvement means the best non-baseline candidate is faster than direct-persistent. The final column is best vDSP divided by matched best FFTW.</caption>
        <thead><tr><th scope="col">Workload</th><th scope="col">Direct-persistent best</th><th scope="col">Best alternative</th><th scope="col">Improvement</th><th scope="col">vDSP / FFTW</th></tr></thead>
        <tbody>{"".join(rows)}</tbody>
      </table></div>
      <p>Best native vDSP execution remains approximately 5.5–6.6× slower than the matched FFTW baseline on both representative horizontal sizes. Closing that gap would require an approximately 82–85% runtime reduction. Because the worker-count sweep, GCD alternative, and separable implementation did not reveal such a mechanism, the project deliberately stops this Float64 scheduling sweep instead of spending a full reference campaign on fields 1/3/4.</p>
      <p class="method-note">Direct-persistent remains the only issue #6 candidate carried forward, and only as an issue #7 guardrail: 12 workers for the representative 256² batch and 16 workers for the representative 512² batch. Issue #7 will test whether preserving native packed split storage changes the complete retained-operator conclusion on those two workloads. Broader vDSP coverage is triggered only if one workload comes within 1.25× of the best matched FFTW retained operator in both directions. GCD dispatch overhead is negligible relative to transform time but does not materially change throughput. In the separable implementation, the isolated strided-column phase dominates; its phase medians are diagnostic and non-additive.</p>
      <p class="method-note">All 48 runs remain preliminary because they use nine samples and do not satisfy the reference-depth machine-state protocol. They support an explicit early-stop feasibility decision, not reference adoption statistics, exhaustive production coverage, Float32 conclusions, or a cross-Mac default.</p>
    """


def vertical_gemm_evidence_table(bundles: list[PublishedBundle]) -> str:
    rows: list[str] = []
    for bundle in sorted(
        bundles,
        key=lambda item: (
            item.result["workload"]["Nx"],
            "k2-group-" in item.result["providers"][0]["algorithmId"],
            item.result["providers"][0]["workers"],
            item.result["run"]["id"],
        ),
    ):
        result = bundle.result
        workload = result["workload"]
        run = result["run"]
        complex_provider = next(
            item for item in result["providers"] if item["id"] == "accelerate-zgemm"
        )
        split_provider = next(
            item for item in result["providers"] if item["id"] == "accelerate-split-dgemm"
        )
        complex_forward = timing(complex_provider, "primitive", "forward")
        complex_inverse = timing(complex_provider, "primitive", "inverse")
        split_forward = timing(split_provider, "primitive", "forward")
        split_inverse = timing(split_provider, "primitive", "inverse")
        if None in (complex_forward, complex_inverse, split_forward, split_inverse):
            raise ValueError(f"Incomplete vertical GEMM primitive evidence in {run['id']}")
        forward_ratio = float(split_forward["medianSeconds"]) / float(complex_forward["medianSeconds"])
        inverse_ratio = float(split_inverse["medianSeconds"]) / float(complex_inverse["medianSeconds"])
        maximum_error = max(
            maximum_correctness_error(complex_provider) or 0.0,
            maximum_correctness_error(split_provider) or 0.0,
        )
        maximum_l2 = max(
            maximum_l2_error(complex_provider) or 0.0,
            maximum_l2_error(split_provider) or 0.0,
        )
        family_setup = stage_timing(
            complex_provider, "setup-shared-component", "logical matrix-family fixture generation", "shared"
        )
        complex_setup = stage_timing(
            complex_provider, "setup-component", "matrix preparation", "shared"
        )
        split_setup = stage_timing(
            split_provider, "setup-component", "matrix preparation", "shared"
        )
        grouping = workload.get("grouping", {})
        estimated_peak = workload.get("bytes", {}).get("verticalBenchmarkEstimatedExplicitPeak", 0)
        grouped = "k2-group-" in complex_provider["algorithmId"]
        family = "K²-grouped synthetic" if grouped else "Common DCT-II"
        group_count = int(grouping.get("verticalGroupCount", 1))
        group_columns = grouping.get("verticalGroupColumns", {})
        group_description = (
            f'{group_count} groups<br>columns min/median/max '
            f'{group_columns.get("minimum", "—")}/{group_columns.get("median", "—")}/{group_columns.get("maximum", "—")}'
            if grouped
            else "1 group"
        )
        rows.append(
            "<tr>"
            f'<td><a href="../../runs/{quote(run["id"])}/index.html">{escaped(run["id"])}</a><br>'
            f'<span class="muted">{publication_badge(bundle.publication["status"])}</span></td>'
            f'<td>{workload["Nx"]}²<br><span class="muted">N<sub>z</sub>={workload["Nz"]}, '
            f'N<sub>j</sub>={workload["Nj"]}, fields={workload["fields"]}, K={workload["Nkl"] * workload["fields"]}</span></td>'
            f'<td>{family}<br><span class="muted">{group_description}</span></td>'
            f'<td>{escaped(complex_provider["schedulingId"])}<br><span class="muted">'
            f'internal {complex_provider["scheduling"]["internalWorkers"]} × outer '
            f'{complex_provider["scheduling"]["outerWorkers"]}</span></td>'
            f'<td class="numeric">{timing_with_interval(complex_forward)} / {timing_with_interval(complex_inverse)}<br>'
            f'<span class="muted">CV {coefficient_of_variation(complex_forward)} / {coefficient_of_variation(complex_inverse)}</span></td>'
            f'<td class="numeric">{timing_with_interval(split_forward)} / {timing_with_interval(split_inverse)}<br>'
            f'<span class="muted">CV {coefficient_of_variation(split_forward)} / {coefficient_of_variation(split_inverse)}</span></td>'
            f'<td class="numeric">{forward_ratio:.3f}× / {inverse_ratio:.3f}×</td>'
            f'<td class="numeric">{format_ms(family_setup["medianSeconds"]) if family_setup else "legacy —"} / '
            f'{format_ms(complex_setup["medianSeconds"]) if complex_setup else "—"} / '
            f'{format_ms(split_setup["medianSeconds"]) if split_setup else "—"}</td>'
            f'<td class="numeric">{format_bytes(complex_provider["memory"]["persistentBytes"])} / '
            f'{format_bytes(split_provider["memory"]["persistentBytes"])}<br><span class="muted">source setup-only '
            f'{format_bytes(workload.get("bytes", {}).get("verticalMatrixFamilySource", 0))}; explicit peak '
            f'{format_bytes(estimated_peak) if estimated_peak else "legacy —"}</span></td>'
            f'<td class="numeric">{format_error(maximum_error)} / {format_error(maximum_l2)}</td>'
            "</tr>"
        )
    if not rows:
        return ""
    return f"""
      <div class="table-scroll"><table class="experiment-evidence-table">
        <caption>Primitive medians and deterministic percentile-bootstrap 95% intervals are forward / inverse in milliseconds. CV is the sample coefficient of variation. Split / complex below 1 favors the two-real-GEMM formulation. Setup is logical family generation / complex preparation / split preparation. Persistent memory is complex / split; explicit peak is the harness allocation high-water estimate, not sampled RSS. No packing or representation conversion is timed.</caption>
        <thead><tr><th scope="col">Run</th><th scope="col">Workload</th><th scope="col">Matrix family</th><th scope="col">Scheduling</th><th scope="col">Complex zgemm</th><th scope="col">Two split dgemm</th><th scope="col">Split / complex</th><th scope="col">Setup ms</th><th scope="col">Explicit memory</th><th scope="col">Max / L2 error</th></tr></thead>
        <tbody>{''.join(rows)}</tbody>
      </table></div>
    """


def vertical_gemm_synthesis(bundles: list[PublishedBundle]) -> str:
    records: list[tuple[dict, dict, dict, str]] = []
    for bundle in bundles:
        complex_provider = next(
            (item for item in bundle.result["providers"] if item["id"] == "accelerate-zgemm"), None
        )
        split_provider = next(
            (item for item in bundle.result["providers"] if item["id"] == "accelerate-split-dgemm"), None
        )
        if complex_provider is not None and split_provider is not None:
            family = "grouped" if "k2-group-" in complex_provider["algorithmId"] else "common"
            records.append((bundle.result, complex_provider, split_provider, family))
    if not records:
        return ""

    common_records = [record for record in records if record[3] == "common"]
    common_rows: list[str] = []
    for profile in sorted({result["run"]["profile"] for result, _, _, _ in common_records}):
        candidates = [record for record in common_records if record[0]["run"]["profile"] == profile]
        result = candidates[0][0]
        workload = result["workload"]
        values: list[str] = []
        for direction in ("forward", "inverse"):
            best_complex = min(
                candidates,
                key=lambda record: float(timing(record[1], "primitive", direction)["medianSeconds"]),
            )
            best_split = min(
                candidates,
                key=lambda record: float(timing(record[2], "primitive", direction)["medianSeconds"]),
            )
            complex_time = timing(best_complex[1], "primitive", direction)
            split_time = timing(best_split[2], "primitive", direction)
            ratio = float(split_time["medianSeconds"]) / float(complex_time["medianSeconds"])
            values.append(
                f'{format_ms(complex_time["medianSeconds"])} (limit {best_complex[1]["workers"]}) / '
                f'{format_ms(split_time["medianSeconds"])} (limit {best_split[2]["workers"]}) / {ratio:.3f}×'
            )
        common_rows.append(
            "<tr>"
            f'<th scope="row">{workload["Nx"]}²<br><span class="muted">N<sub>z</sub>={workload["Nz"]}, fields={workload["fields"]}</span></th>'
            f'<td class="numeric">{values[0]}</td><td class="numeric">{values[1]}</td>'
            "</tr>"
        )
    common_section = f"""
      <h3>Bounded common-matrix screen</h3>
      <p>This first issue #8 increment tests only the shared deterministic DCT-II matrix family with fields=3. Each row below reports best complex / best split / split-to-complex ratio across the isolated thread-limit runs. It establishes primitive behavior, not the complete issue #8 winner.</p>
      <div class="table-scroll"><table class="experiment-evidence-table">
        <caption>Best observed primitive medians in milliseconds. Thread settings are process-isolated <code>VECLIB_MAXIMUM_THREADS</code> limits; they are not claims about the exact worker count selected internally by Accelerate.</caption>
        <thead><tr><th scope="col">Workload</th><th scope="col">Forward: complex / split / ratio</th><th scope="col">Inverse: complex / split / ratio</th></tr></thead>
        <tbody>{''.join(common_rows)}</tbody>
      </table></div>
    """

    grouped_records = [record for record in records if record[3] == "grouped"]
    serial_grouped_records = [
        record for record in grouped_records
        if record[1]["scheduling"]["outerWorkers"] == 1
    ]
    grouped_rows: list[str] = []
    for grouped in sorted(
        serial_grouped_records,
        key=lambda record: (record[0]["workload"]["Nx"], record[1]["workers"]),
    ):
        result, complex_provider, split_provider, _ = grouped
        matched_candidates = [
            record for record in common_records
            if record[0]["run"]["profile"] == result["run"]["profile"]
            and record[1]["workers"] == complex_provider["workers"]
        ]
        if not matched_candidates:
            continue
        matched = max(
            matched_candidates,
            key=lambda record: (
                record[0]["environment"]["gitCommit"] == result["environment"]["gitCommit"],
                record[0]["environment"]["timestampUtc"],
            ),
        )
        workload = result["workload"]
        grouping = workload.get("grouping", {})
        ratios: list[str] = []
        split_vs_complex: list[str] = []
        for direction in ("forward", "inverse"):
            grouped_complex = timing(complex_provider, "primitive", direction)
            grouped_split = timing(split_provider, "primitive", direction)
            common_complex = timing(matched[1], "primitive", direction)
            common_split = timing(matched[2], "primitive", direction)
            ratios.append(
                f'{float(grouped_complex["medianSeconds"]) / float(common_complex["medianSeconds"]):.2f}× / '
                f'{float(grouped_split["medianSeconds"]) / float(common_split["medianSeconds"]):.2f}×'
            )
            split_vs_complex.append(
                f'{float(grouped_split["medianSeconds"]) / float(grouped_complex["medianSeconds"]):.3f}×'
            )
        columns = grouping.get("verticalGroupColumns", {})
        grouped_rows.append(
            "<tr>"
            f'<th scope="row">{workload["Nx"]}², limit {complex_provider["workers"]}</th>'
            f'<td class="numeric">{grouping.get("verticalGroupCount", "—")}<br>'
            f'<span class="muted">{columns.get("minimum", "—")}/{columns.get("median", "—")}/{columns.get("maximum", "—")} columns</span></td>'
            f'<td class="numeric">{ratios[0]}</td><td class="numeric">{ratios[1]}</td>'
            f'<td class="numeric">{split_vs_complex[0]} / {split_vs_complex[1]}</td>'
            "</tr>"
        )
    grouped_section = ""
    if grouped_rows:
        grouped_section = f"""
      <h3>K²-grouped matrix-family penalty</h3>
      <p>The grouped increment gives every exact integer K² group its own deterministic dense orthonormal matrix pair. The table compares each grouped run with the matched common-matrix run at the same workload and thread limit. Ratios include BLAS call overhead and the loss of large-GEMM efficiency, but still exclude producing the grouped order.</p>
      <div class="table-scroll"><table class="experiment-evidence-table">
        <caption>Grouped / common ratios are complex / split for forward and inverse; values above 1 are the primitive grouping penalty. Split / complex is measured within the grouped run.</caption>
        <thead><tr><th scope="col">Workload</th><th scope="col">Groups and min/median/max size</th><th scope="col">Forward grouped/common</th><th scope="col">Inverse grouped/common</th><th scope="col">Grouped split/complex F/I</th></tr></thead>
        <tbody>{''.join(grouped_rows)}</tbody>
      </table></div>
    """

    scheduling_rows: list[str] = []
    outer_records = [
        record for record in grouped_records
        if record[1]["scheduling"]["outerWorkers"] > 1
        and record[1]["scheduling"]["internalWorkers"] == 1
    ]
    for candidate in sorted(
        outer_records,
        key=lambda record: (
            record[0]["workload"]["Nx"],
            record[1]["algorithmId"],
            record[1]["scheduling"]["outerWorkers"],
        ),
    ):
        result, complex_provider, split_provider, _ = candidate
        baselines = [
            record for record in serial_grouped_records
            if record[0]["run"]["profile"] == result["run"]["profile"]
            and record[1]["scheduling"]["internalWorkers"] == 1
            and record[0]["environment"]["gitCommit"] == result["environment"]["gitCommit"]
        ]
        if not baselines:
            continue
        baseline = max(baselines, key=lambda record: record[0]["environment"]["timestampUtc"])
        speedups: list[str] = []
        medians: list[str] = []
        for provider_index in (1, 2):
            provider_speedups: list[str] = []
            provider_medians: list[str] = []
            for direction in ("forward", "inverse"):
                baseline_timing = timing(baseline[provider_index], "primitive", direction)
                candidate_timing = timing(candidate[provider_index], "primitive", direction)
                provider_speedups.append(
                    f'{float(baseline_timing["medianSeconds"]) / float(candidate_timing["medianSeconds"]):.2f}×'
                )
                provider_medians.append(format_ms(candidate_timing["medianSeconds"]))
            speedups.append(" / ".join(provider_speedups))
            medians.append(" / ".join(provider_medians))
        dispatch = stage_timing(
            complex_provider, "primitive-diagnostic", "empty group dispatch", "shared"
        )
        algorithm = complex_provider["algorithmId"]
        schedule = "outer-static" if "outer-static" in algorithm else "outer-dynamic"
        workload = result["workload"]
        scheduling_rows.append(
            "<tr>"
            f'<th scope="row">{workload["Nx"]}²<br><span class="muted">N<sub>z</sub>={workload["Nz"]}, '
            f'fields={workload["fields"]}</span></th>'
            f'<td>{schedule}<br><span class="muted">{complex_provider["scheduling"]["outerWorkers"]} outer × 1 BLAS</span></td>'
            f'<td class="numeric">{medians[0]}<br><span class="muted">speedup {speedups[0]}</span></td>'
            f'<td class="numeric">{medians[1]}<br><span class="muted">speedup {speedups[1]}</span></td>'
            f'<td class="numeric">{format_ms(dispatch["medianSeconds"]) if dispatch else "—"}</td>'
            "</tr>"
        )
    scheduling_section = ""
    if scheduling_rows:
        best_rows: list[str] = []
        for profile in sorted({record[0]["run"]["profile"] for record in outer_records}):
            candidates = [record for record in outer_records if record[0]["run"]["profile"] == profile]
            result = candidates[0][0]
            baselines = [
                record for record in serial_grouped_records
                if record[0]["run"]["profile"] == profile
                and record[1]["scheduling"]["internalWorkers"] == 1
                and record[0]["environment"]["gitCommit"] == result["environment"]["gitCommit"]
            ]
            if not baselines:
                continue
            baseline = max(baselines, key=lambda record: record[0]["environment"]["timestampUtc"])

            def best_cell(provider_index: int, direction: str) -> str:
                winner = min(
                    candidates,
                    key=lambda record: float(timing(record[provider_index], "primitive", direction)["medianSeconds"]),
                )
                provider = winner[provider_index]
                candidate_timing = timing(provider, "primitive", direction)
                baseline_timing = timing(baseline[provider_index], "primitive", direction)
                schedule = "static" if "outer-static" in provider["algorithmId"] else "dynamic"
                workers = provider["scheduling"]["outerWorkers"]
                speedup = float(baseline_timing["medianSeconds"]) / float(candidate_timing["medianSeconds"])
                return f'{format_ms(candidate_timing["medianSeconds"])} · {schedule}-{workers} · {speedup:.2f}×'

            workload = result["workload"]
            best_rows.append(
                "<tr>"
                f'<th scope="row">{workload["Nx"]}²<br><span class="muted">N<sub>z</sub>={workload["Nz"]}, '
                f'fields={workload["fields"]}</span></th>'
                f'<td class="numeric">{best_cell(1, "forward")}</td>'
                f'<td class="numeric">{best_cell(1, "inverse")}</td>'
                f'<td class="numeric">{best_cell(2, "forward")}</td>'
                f'<td class="numeric">{best_cell(2, "inverse")}</td>'
                "</tr>"
            )
        adjacent_scan_conclusion = (
            "Every published scheduling run reports zero exactly equivalent adjacent matrix pairs."
            if all(
                "exactly equivalent adjacent matrix pairs=0" in record[1]["planning"]["configuration"]
                for record in outer_records
            )
            else "The adjacent-equality result varies across the published scheduling runs."
        )
        complete_cohorts: list[tuple[str, list[str], list[tuple[dict, dict, dict, str]]]] = []
        for commit in {record[0]["environment"]["gitCommit"] for record in grouped_records}:
            commit_records = [
                record for record in grouped_records
                if record[0]["environment"]["gitCommit"] == commit
            ]
            complete_profiles: list[str] = []
            for profile in {record[0]["run"]["profile"] for record in commit_records}:
                profile_records = [
                    record for record in commit_records
                    if record[0]["run"]["profile"] == profile
                    and record[1]["scheduling"]["internalWorkers"] == 1
                ]
                schedules = {
                    (
                        "serial"
                        if record[1]["scheduling"]["outerWorkers"] == 1
                        else "static-12"
                        if "outer-static" in record[1]["algorithmId"]
                        and record[1]["scheduling"]["outerWorkers"] == 12
                        else "dynamic-16"
                        if "outer-dynamic" in record[1]["algorithmId"]
                        and record[1]["scheduling"]["outerWorkers"] == 16
                        else "other"
                    )
                    for record in profile_records
                }
                if {"serial", "static-12", "dynamic-16"} <= schedules:
                    complete_profiles.append(profile)
            if complete_profiles:
                complete_cohorts.append((commit, complete_profiles, commit_records))

        portability_section = ""
        if complete_cohorts:
            _, portability_profiles, portability_records = max(
                complete_cohorts,
                key=lambda cohort: max(
                    record[0]["environment"]["timestampUtc"] for record in cohort[2]
                ),
            )
            winner_counts = {"static-12": 0, "dynamic-16": 0}
            best_speedups: list[float] = []
            candidate_speedups: list[float] = []
            peak_bytes: list[int] = []
            errors: list[float] = []
            for profile in portability_profiles:
                candidates = [
                    record for record in portability_records
                    if record[0]["run"]["profile"] == profile
                    and record[1]["scheduling"]["internalWorkers"] == 1
                ]
                baseline = next(
                    record for record in candidates
                    if record[1]["scheduling"]["outerWorkers"] == 1
                )
                finalists = [
                    record for record in candidates
                    if (
                        "outer-static" in record[1]["algorithmId"]
                        and record[1]["scheduling"]["outerWorkers"] == 12
                    )
                    or (
                        "outer-dynamic" in record[1]["algorithmId"]
                        and record[1]["scheduling"]["outerWorkers"] == 16
                    )
                ]
                for provider_index in (1, 2):
                    for direction in ("forward", "inverse"):
                        baseline_seconds = float(
                            timing(baseline[provider_index], "primitive", direction)["medianSeconds"]
                        )
                        measurements = [
                            (
                                float(timing(record[provider_index], "primitive", direction)["medianSeconds"]),
                                "static-12" if "outer-static" in record[1]["algorithmId"] else "dynamic-16",
                            )
                            for record in finalists
                        ]
                        winner_seconds, winner_name = min(measurements)
                        winner_counts[winner_name] += 1
                        best_speedups.append(baseline_seconds / winner_seconds)
                        candidate_speedups.extend(
                            baseline_seconds / candidate_seconds
                            for candidate_seconds, _ in measurements
                        )
                peak_bytes.extend(
                    int(record[0]["workload"].get("bytes", {}).get(
                        "verticalBenchmarkEstimatedExplicitPeak", 0
                    ))
                    for record in candidates
                )
                errors.extend(
                    float(metric[name])
                    for record in candidates
                    for provider in record[0]["providers"]
                    for metric in provider["correctness"]
                    for name in ("maximumRelativeError", "relativeL2Error")
                    if metric.get(name) is not None
                )
            geometric_speedup = math.exp(
                sum(math.log(value) for value in best_speedups) / len(best_speedups)
            )
            portability_section = f"""
      <h4>Finalist portability across fields and vertical depth</h4>
      <p>This increment asks whether the scheduling conclusion from the earlier fields=3 screens survives changes in field count and vertical depth. It holds Float64, the synthetic K²-grouped matrices, prearranged column order, one requested BLAS thread, warmups, samples, and source commit fixed; only the named workload and outer schedule change. The timed boundary remains the complete out-of-place group loop for raw forward or inverse vertical GEMM. Matrix construction, provider preparation, scheduler construction, correctness copies, horizontal gathering, and packing are excluded.</p>
      <p>Across {len(portability_profiles)} newly completed profiles and 32 provider/direction cells, dynamic-16 is fastest in {winner_counts["dynamic-16"]} cells and static-12 in {winner_counts["static-12"]}. The best finalist improves on its same-commit serial baseline by {geometric_speedup:.2f}× geometrically, spanning {min(best_speedups):.2f}×–{max(best_speedups):.2f}×. Both finalists beat serial in {sum(value > 1.0 for value in candidate_speedups)} of {len(candidate_speedups)} individual comparisons. Reported explicit peaks span {format_bytes(min(peak_bytes))}–{format_bytes(max(peak_bytes))}; the largest observed correctness error is {format_error(max(errors))}.</p>
      <p>This advances dynamic-16 as the default scheduling finalist for the ordering/packing crossover and combined-pipeline experiments, while retaining static-12 as a close alternative. It does not establish a machine-independent winner or include the upstream data movement owned by issue #13.</p>
            """
        scheduling_section = f"""
      <h3>Persistent outer group scheduling</h3>
      <p>These candidates hold each Accelerate GEMM to one requested internal thread and distribute complete K² groups over persistent C++ workers. Weighted-static uses setup-time contiguous partitions balanced by group-column count; dynamic uses an allocation-free atomic next-group counter. Speedup is relative to the same-commit serial grouped run with one BLAS thread.</p>
      <div class="table-scroll"><table class="experiment-evidence-table">
        <caption>Candidate medians are forward / inverse milliseconds; speedups above one favor outer scheduling. Empty dispatch traverses the same group schedule without GEMM and is a non-additive diagnostic.</caption>
        <thead><tr><th scope="col">Workload</th><th scope="col">Schedule</th><th scope="col">Complex time and speedup</th><th scope="col">Split time and speedup</th><th scope="col">Empty dispatch ms</th></tr></thead>
        <tbody>{''.join(scheduling_rows)}</tbody>
      </table></div>
      <h4>Best observed candidates</h4>
      <div class="table-scroll"><table class="experiment-evidence-table">
        <caption>Each cell is median milliseconds · schedule-worker count · speedup over the same-commit serial grouped baseline.</caption>
        <thead><tr><th scope="col">Workload</th><th scope="col">Complex forward</th><th scope="col">Complex inverse</th><th scope="col">Split forward</th><th scope="col">Split inverse</th></tr></thead>
        <tbody>{''.join(best_rows)}</tbody>
      </table></div>
      {portability_section}
      <p>The installed public Accelerate CBLAS headers expose no variable-size grouped GEMM batch API. {escaped(adjacent_scan_conclusion)} Nonadjacent merging would require the reordering or block-diagonal expansion deliberately left outside this primitive experiment.</p>
        """
    return common_section + grouped_section + scheduling_section + """
      <p class="method-note">Inputs are already arranged as column-major vertical-contiguous matrices, both algorithms are out-of-place, and all buffers are persistent. Matrix expansion/transposition and scheduler construction are setup-only. Thread-stack memory remains opaque when persistent outer workers are used. Packing and horizontal ordering are deliberately excluded for later issue #13 measurement. The named fields=1/3/4 and N<sub>z</sub>=65/129/257 profiles now have a preliminary finalist screen; machine-state repeats, third-party grouped APIs, blocking, and the packing crossover remain open.</p>
    """


def ordering_packing_crossover(provider: dict, direction: str) -> tuple[int | None, float | None]:
    final_speedup: float | None = None
    crossover: int | None = None
    for reuse_count in (2, 4, 8):
        direction_id = f"{direction}-r{reuse_count}"
        repeated = stage_timing(
            provider, "reuse-total", "boundary-movement-each-use", direction_id
        )
        persistent = stage_timing(
            provider, "reuse-total", "persistent-compact-boundary-once", direction_id
        )
        if repeated is None or persistent is None:
            continue
        speedup = float(repeated["medianSeconds"]) / float(persistent["medianSeconds"])
        if crossover is None and speedup > 1.0:
            crossover = reuse_count
        if reuse_count == 8:
            final_speedup = speedup
    return crossover, final_speedup


def ordering_packing_legacy_evidence_table(bundles: list[PublishedBundle]) -> str:
    rows: list[str] = []
    for bundle in sorted(
        bundles,
        key=lambda item: (
            item.result["workload"]["Nx"],
            item.result["workload"]["Nz"],
            item.result["workload"]["fields"],
            item.result["providers"][0]["scheduling"]["outerWorkers"],
            item.result["run"]["id"],
        ),
    ):
        result = bundle.result
        workload = result["workload"]
        run = result["run"]
        peak = int(workload.get("bytes", {}).get("orderingPackingEstimatedExplicitPeak", 0))
        for provider in result["providers"]:
            direct = provider["id"] == "ordering-no-reorder-accelerate-zgemm"
            raw_forward = timing(provider, "primitive", "forward")
            raw_inverse = timing(provider, "primitive", "inverse")
            pack = timing(provider, "adapter-component", "forward")
            embed = timing(provider, "adapter-component", "inverse")
            combined_forward = timing(provider, "adapter-total", "forward")
            combined_inverse = timing(provider, "adapter-total", "inverse")
            if None in (
                raw_forward,
                raw_inverse,
                pack,
                embed,
                combined_forward,
                combined_inverse,
            ):
                raise ValueError(f"Incomplete ordering/packing evidence in {run['id']}")
            forward_crossover, forward_r8 = ordering_packing_crossover(provider, "forward")
            inverse_crossover, inverse_r8 = ordering_packing_crossover(provider, "inverse")
            forward_r8_text = "not measured" if forward_r8 is None else f"{forward_r8:.2f}×"
            inverse_r8_text = "not measured" if inverse_r8 is None else f"{inverse_r8:.2f}×"
            maximum_error = maximum_correctness_error(provider)
            maximum_l2 = maximum_l2_error(provider)
            representation = (
                "No reorder / interleaved"
                if direct
                else "Split" if "split" in provider["algorithmId"] else "Interleaved"
            )
            schedule = (
                "dynamic" if "outer-dynamic" in provider["algorithmId"] else "static"
            )
            movement_text = (
                "elided / elided<br><span class=\"muted\">0 B / 0 B</span>"
                if direct
                else f'{timing_with_interval(pack)} / {timing_with_interval(embed)}<br>'
                     f'<span class="muted">{format_bytes(pack["bytesMoved"])} / '
                     f'{format_bytes(embed["bytesMoved"])}</span>'
            )
            movement_ratio_text = (
                "0× / 0×<br><span class=\"muted\">algorithmically elided</span>"
                if direct
                else f'{float(pack["medianSeconds"]) / float(raw_forward["medianSeconds"]):.3f}× / '
                     f'{float(embed["medianSeconds"]) / float(raw_inverse["medianSeconds"]):.3f}×'
            )
            reuse_text = (
                "provider order persists<br><span class=\"muted\">R=2/4/8 measured</span>"
                if direct
                else f'R={forward_crossover if forward_crossover is not None else "none"} / '
                     f'R={inverse_crossover if inverse_crossover is not None else "none"}<br>'
                     f'<span class="muted">R8 {forward_r8_text} / {inverse_r8_text}</span>'
            )
            rows.append(
                "<tr>"
                f'<td><a href="../../runs/{quote(run["id"])}/index.html">{escaped(run["id"])}</a><br>'
                f'<span class="muted">{publication_badge(bundle.publication["status"])}</span></td>'
                f'<td>{workload["Nx"]}²<br><span class="muted">N<sub>z</sub>={workload["Nz"]}, '
                f'fields={workload["fields"]}</span></td>'
                f'<td>{representation}<br><span class="muted">{schedule}-{provider["scheduling"]["outerWorkers"]}; '
                f'{provider.get("gemmCallsPerExecution", "—")} calls/direction</span></td>'
                f'<td class="numeric">{movement_text}</td>'
                f'<td class="numeric">{timing_with_interval(raw_forward)} / {timing_with_interval(raw_inverse)}</td>'
                f'<td class="numeric">{timing_with_interval(combined_forward)} / '
                f'{timing_with_interval(combined_inverse)}</td>'
                f'<td class="numeric">{movement_ratio_text}</td>'
                f'<td class="numeric">{reuse_text}</td>'
                f'<td class="numeric">{format_bytes(provider["memory"]["persistentBytes"])}<br>'
                f'<span class="muted">peak {format_bytes(peak) if peak else "not reported"}</span></td>'
                f'<td class="numeric">{format_error(maximum_error)} / {format_error(maximum_l2)}</td>'
                "</tr>"
            )
    if not rows:
        return ""
    return f"""
      <div class="table-scroll"><table class="experiment-evidence-table">
        <caption>Forward / inverse medians with deterministic percentile-bootstrap 95% intervals. Packed movement is WVM retained gather/radial pack / zero-fill scatter/Hermitian embed; the direct algorithm elides both and retains zero-padded provider order. Primitive and combined remain separate timing series even when their boundaries coincide. Movement/primitive is not an additive reconstruction of combined time. Crossover applies only to compact storage.</caption>
        <thead><tr><th scope="col">Run</th><th scope="col">Workload</th><th scope="col">Representation and schedule</th><th scope="col">Movement ms and bytes</th><th scope="col">Primitive GEMM ms</th><th scope="col">Combined ms</th><th scope="col">Movement / primitive</th><th scope="col">Persistent crossover F/I</th><th scope="col">Memory</th><th scope="col">Max / L2 error</th></tr></thead>
        <tbody>{''.join(rows)}</tbody>
      </table></div>
    """


def ordering_packing_legacy_synthesis(bundles: list[PublishedBundle]) -> str:
    if not bundles:
        return ""
    latest_commit = max(
        {bundle.result["environment"]["gitCommit"] for bundle in bundles},
        key=lambda commit: max(
            bundle.result["environment"]["timestampUtc"]
            for bundle in bundles
            if bundle.result["environment"]["gitCommit"] == commit
        ),
    )
    cohort = [
        bundle for bundle in bundles
        if bundle.result["environment"]["gitCommit"] == latest_commit
    ]
    profiles = sorted({bundle.result["run"]["profile"] for bundle in cohort})
    if not profiles:
        return ""
    schedule_winners = {"dynamic": 0, "static": 0}
    representation_winners = {"split": 0, "interleaved": 0}
    movement_ratios: list[float] = []
    crossovers: list[int | None] = []
    peak_bytes: list[int] = []
    errors: list[float] = []
    direct_primitive_ratios: list[float] = []
    direct_combined_ratios: list[float] = []
    direct_ratios_by_fields: dict[int, list[float]] = {}
    direct_call_counts: list[int] = []
    packed_call_counts: list[int] = []
    for profile in profiles:
        candidates: list[tuple[float, dict, str]] = []
        for bundle in cohort:
            if bundle.result["run"]["profile"] != profile:
                continue
            peak_bytes.append(int(bundle.result["workload"]["bytes"].get(
                "orderingPackingEstimatedExplicitPeak", 0
            )))
            packed_providers = [
                provider for provider in bundle.result["providers"]
                if provider["id"].startswith("ordering-pack-")
            ]
            direct_provider = next(
                (
                    provider for provider in bundle.result["providers"]
                    if provider["id"] == "ordering-no-reorder-accelerate-zgemm"
                ),
                None,
            )
            packed_split = next(
                (
                    provider for provider in packed_providers
                    if provider["id"] == "ordering-pack-accelerate-split-dgemm"
                ),
                None,
            )
            for provider in packed_providers:
                for direction in ("forward", "inverse"):
                    raw = timing(provider, "primitive", direction)
                    movement = timing(provider, "adapter-component", direction)
                    combined = timing(provider, "adapter-total", direction)
                    movement_ratios.append(
                        float(movement["medianSeconds"]) / float(raw["medianSeconds"])
                    )
                    crossovers.append(ordering_packing_crossover(provider, direction)[0])
                    candidates.append((float(combined["medianSeconds"]), provider, direction))
                errors.extend(
                    float(metric[name])
                    for metric in provider["correctness"]
                    for name in ("maximumRelativeError", "relativeL2Error")
                    if metric.get(name) is not None
                )
            if direct_provider is not None and packed_split is not None:
                direct_call_counts.append(int(direct_provider.get("gemmCallsPerExecution", 0)))
                packed_call_counts.append(int(packed_split.get("gemmCallsPerExecution", 0)))
                for direction in ("forward", "inverse"):
                    direct_raw = timing(direct_provider, "primitive", direction)
                    packed_raw = timing(packed_split, "primitive", direction)
                    direct_total = timing(direct_provider, "adapter-total", direction)
                    packed_total = timing(packed_split, "adapter-total", direction)
                    direct_primitive_ratios.append(
                        float(direct_raw["medianSeconds"]) /
                        float(packed_raw["medianSeconds"])
                    )
                    direct_combined_ratios.append(
                        float(direct_total["medianSeconds"]) /
                        float(packed_total["medianSeconds"])
                    )
                    direct_ratios_by_fields.setdefault(
                        int(bundle.result["workload"]["fields"]), []
                    ).append(direct_combined_ratios[-1])
                errors.extend(
                    float(metric[name])
                    for metric in direct_provider["correctness"]
                    for name in ("maximumRelativeError", "relativeL2Error")
                    if metric.get(name) is not None
                )
        for direction in ("forward", "inverse"):
            _, winner, _ = min(
                (item for item in candidates if item[2] == direction),
                key=lambda item: item[0],
            )
            schedule_winners[
                "dynamic" if "outer-dynamic" in winner["algorithmId"] else "static"
            ] += 1
            representation_winners[
                "split" if "split" in winner["algorithmId"] else "interleaved"
            ] += 1
    sampled_crossovers = [value for value in crossovers if value is not None]
    no_reorder_section = ""
    if direct_combined_ratios:
        primitive_geomean = math.exp(
            sum(math.log(value) for value in direct_primitive_ratios) /
            len(direct_primitive_ratios)
        )
        combined_geomean = math.exp(
            sum(math.log(value) for value in direct_combined_ratios) /
            len(direct_combined_ratios)
        )
        direct_wins = sum(value < 1.0 for value in direct_combined_ratios)
        fields_summary = "; ".join(
            f"fields={fields}: {sum(value < 1.0 for value in values)}/{len(values)} wins, "
            f"{math.exp(sum(math.log(value) for value in values) / len(values)):.3f}× geometric, "
            f"{min(values):.3f}×–{max(values):.3f}× range"
            for fields, values in sorted(direct_ratios_by_fields.items())
        )
        conclusion = (
            "For this bounded cohort, avoided movement offsets the efficiency loss from the many small GEMMs."
            if combined_geomean < 1.0
            else "For this bounded cohort, avoided movement does not offset the efficiency loss from the many small GEMMs."
        )
        no_reorder_section = f"""
      <h3>Direct WVM-order no-reorder increment</h3>
      <p>This follow-on asks whether the radial gather and split conversion are worth paying for at all. The new algorithm reads each retained frequency’s contiguous N<sub>z</sub>×fields block directly from the WVM half-spectrum, applies its real K²-dependent matrix with one complex zgemm per retained frequency, keeps a zero-padded frequency-major modal representation, and reconstructs directly into persistent zero-padded WVM storage. Immutable matrix preparation, scheduler creation, and the initial zero-fill are setup-only. Hermitian boundary repair is fused into the timed kernel. The mathematical operator, fixtures, Float64 precision, thread limit, schedules, workloads, warmups, and samples match the packed split candidate in each run.</p>
      <p>Across {len(direct_combined_ratios)} same-run workload/schedule/direction comparisons, direct no-reorder wins {direct_wins}; its direct/packed-split geometric ratio is {primitive_geomean:.3f}× at the primitive boundary and {combined_geomean:.3f}× for the one-shot total. The one-shot ratios span {min(direct_combined_ratios):.3f}×–{max(direct_combined_ratios):.3f}×. By field count: {fields_summary}. Direct execution issues {min(direct_call_counts)}–{max(direct_call_counts)} GEMM calls per direction, versus {min(packed_call_counts)}–{max(packed_call_counts)} for packed split. {conclusion}</p>
      <p>This comparison still excludes raw FFT execution, nonlinear or modal physics, provider-native horizontal fusion, tiling, and 512-class scaling. A shape-specific result is evidence about this algorithm boundary, not a requirement to gather or to preserve WVM order in production.</p>
        """
    baseline_scope = (
        "Raw FFT execution, modal physics, nonlinear flux, 512-class workloads, provider-native fusion, and tiled packing remain outside the timed boundary."
        if direct_combined_ratios
        else "Raw FFT execution, modal physics, nonlinear flux, 512-class workloads, no-reorder kernels, provider-native fusion, and tiled packing remain outside the timed boundary."
    )
    baseline_conclusion = (
        "This establishes the movement denominator and the packed policies used by the direct comparison below. Earlier immutable runs remain the baseline evidence and are not rewritten."
        if direct_combined_ratios
        else "This establishes the movement denominator and policies to compare next. It cannot yet answer whether movement is worthwhile versus a correct strided/no-reorder vertical kernel because that competing algorithm has not been implemented."
    )
    return f"""
      <h3>First bounded MATLAB-style baseline</h3>
      <p>This increment measures one concrete policy, not a required mathematical order. It starts with a WVM frequency-major interleaved half-spectrum, gathers retained modes directly into the final radial/K²-grouped vertical input, executes the issue #8 static-12 or dynamic-16 kernel, and performs the reverse zero-fill scatter/Hermitian embed. It holds fixtures, logical mode keys, matrix family, precision, and one requested BLAS thread fixed. {baseline_scope}</p>
      <p>The latest same-commit cohort contains {len(cohort)} runs across {len(profiles)} profiles and {2 * len(profiles)} forward/inverse combined winners. Dynamic scheduling wins {schedule_winners["dynamic"]} cells and static {schedule_winners["static"]}; split storage wins {representation_winners["split"]} and interleaved {representation_winners["interleaved"]}. Movement costs span {min(movement_ratios):.3f}×–{max(movement_ratios):.3f}× of the separately measured primitive. Persistent compact storage first wins at a sampled reuse count in {len(sampled_crossovers)} of {len(crossovers)} representation/schedule/direction cells; {sum(value == 2 for value in sampled_crossovers)} cross at R=2. Explicit peaks span {format_bytes(min(peak_bytes))}–{format_bytes(max(peak_bytes))}, and the largest correctness error is {format_error(max(errors))}.</p>
      <p>{baseline_conclusion}</p>
      {no_reorder_section}
    """


def spectral_boundary_bundles(
    bundles: list[PublishedBundle],
) -> list[PublishedBundle]:
    return [
        bundle for bundle in bundles
        if any(
            provider["id"].startswith("boundary-")
            for provider in bundle.result["providers"]
        )
    ]


def ordering_packing_boundary_evidence_table(
    bundles: list[PublishedBundle],
) -> str:
    rows: list[str] = []
    for bundle in sorted(
        spectral_boundary_bundles(bundles),
        key=lambda item: (
            item.result["workload"]["Nx"],
            item.result["workload"]["Nz"],
            item.result["workload"]["fields"],
            item.result["providers"][0]["id"],
            item.result["providers"][0]["schedulingId"],
            item.result["run"]["id"],
        ),
    ):
        result = bundle.result
        workload = result["workload"]
        run = result["run"]
        provider = next(
            item for item in result["providers"]
            if item["id"].startswith("boundary-")
        )
        raw_fft_forward = (
            stage_timing(provider, "primitive", "raw FFT", "forward")
            or stage_timing(provider, "primitive", "raw pruned FFT", "forward")
        )
        raw_fft_inverse = (
            stage_timing(provider, "primitive", "raw FFT", "inverse")
            or stage_timing(provider, "primitive", "raw pruned FFT", "inverse")
        )
        raw_vertical_forward = stage_timing(
            provider, "primitive", "raw vertical MM", "forward"
        )
        raw_vertical_inverse = stage_timing(
            provider, "primitive", "raw vertical MM", "inverse"
        )
        movement_forward = timing(provider, "adapter-component", "forward")
        movement_inverse = timing(provider, "adapter-component", "inverse")
        total_forward = timing(provider, "uninstrumented-total", "forward")
        total_inverse = timing(provider, "uninstrumented-total", "inverse")
        required = (
            raw_fft_forward,
            raw_fft_inverse,
            raw_vertical_forward,
            raw_vertical_inverse,
            movement_forward,
            movement_inverse,
            total_forward,
            total_inverse,
        )
        if any(item is None for item in required):
            raise ValueError(f"Incomplete spectral-boundary evidence in {run['id']}")

        def movement(item: dict) -> str:
            if item["state"] == "elided":
                return "elided"
            return (
                f'{timing_with_interval(item)}<br><span class="muted">'
                f'{format_bytes(item["bytesMoved"])}</span>'
            )

        peak = int(
            workload.get("bytes", {}).get("orderingPackingEstimatedExplicitPeak", 0)
        )
        rows.append(
            "<tr>"
            f'<td><a href="../../runs/{quote(run["id"])}/index.html">'
            f'{escaped(run["id"])}</a><br><span class="muted">'
            f'{publication_badge(bundle.publication["status"])}</span></td>'
            f'<td>{workload["Nx"]}²<br><span class="muted">N<sub>z</sub>='
            f'{workload["Nz"]}, fields={workload["fields"]}</span></td>'
            f'<td>{escaped(provider["id"].removeprefix("boundary-"))}<br>'
            f'<span class="muted">{escaped(provider["schedulingId"])}</span></td>'
            f'<td class="numeric">{timing_with_interval(raw_fft_forward)} / '
            f'{timing_with_interval(raw_fft_inverse)}</td>'
            f'<td class="numeric">{timing_with_interval(raw_vertical_forward)} / '
            f'{timing_with_interval(raw_vertical_inverse)}</td>'
            f'<td class="numeric">{movement(movement_forward)} / '
            f'{movement(movement_inverse)}</td>'
            f'<td class="numeric">{timing_with_interval(total_forward)} / '
            f'{timing_with_interval(total_inverse)}</td>'
            f'<td class="numeric">{format_bytes(provider["memory"]["persistentBytes"])}'
            f'<br><span class="muted">peak {format_bytes(peak) if peak else "not reported"}</span></td>'
            f'<td class="numeric">{format_error(maximum_correctness_error(provider))} / '
            f'{format_error(maximum_l2_error(provider))}</td>'
            "</tr>"
        )
    if not rows:
        return ""
    return f"""
      <h3>Composed horizontal-to-vertical boundary evidence</h3>
      <div class="table-scroll"><table class="experiment-evidence-table">
        <caption>Forward / inverse medians with deterministic percentile-bootstrap 95% intervals. Raw horizontal transform, raw vertical MM, data movement, and the uninstrumented composed boundary remain separate. Movement may be elided or fused; component medians are diagnostic and are not summed to reconstruct the authoritative total. Full-spectrum inverse views include their required per-execution zero-padding rebuild.</caption>
        <thead><tr><th scope="col">Run</th><th scope="col">Workload</th><th scope="col">Policy and schedule</th><th scope="col">Raw horizontal ms</th><th scope="col">Raw vertical ms</th><th scope="col">Movement ms and bytes</th><th scope="col">Composed total ms</th><th scope="col">Memory</th><th scope="col">Max / L2 error</th></tr></thead>
        <tbody>{''.join(rows)}</tbody>
      </table></div>
    """


def ordering_packing_boundary_synthesis(
    bundles: list[PublishedBundle],
) -> str:
    boundary = spectral_boundary_bundles(bundles)
    if not boundary:
        return ""
    latest_increment = max(
        {
            bundle.publication.get("incrementId", "")
            for bundle in boundary
        },
        key=lambda increment: max(
            bundle.result["environment"]["timestampUtc"]
            for bundle in boundary
            if bundle.publication.get("incrementId", "") == increment
        ),
    )
    cohort = [
        bundle for bundle in boundary
        if bundle.publication.get("incrementId", "") == latest_increment
    ]
    cells: dict[tuple[str, str, str], list[float]] = {}
    policy_labels: dict[str, str] = {}
    errors: list[float] = []
    for bundle in cohort:
        provider = next(
            item for item in bundle.result["providers"]
            if item["id"].startswith("boundary-")
        )
        candidate = f'{provider["id"]};{provider["schedulingId"]}'
        policy_labels[candidate] = (
            f'{provider["id"].removeprefix("boundary-")} '
            f'({provider["schedulingId"]})'
        )
        profile = bundle.result["run"]["profile"]
        for direction in ("forward", "inverse"):
            total = timing(provider, "uninstrumented-total", direction)
            if total is not None:
                cells.setdefault((candidate, profile, direction), []).append(
                    float(total["medianSeconds"])
                )
        errors.extend(
            float(metric[name])
            for metric in provider["correctness"]
            for name in ("maximumRelativeError", "relativeL2Error")
            if metric.get(name) is not None
        )
    candidates = sorted({candidate for candidate, _, _ in cells})
    profiles = sorted({profile for _, profile, _ in cells})
    summaries: list[tuple[float, int, str, float]] = []
    paired_summaries: list[tuple[float, int, str, float]] = []
    for candidate in candidates:
        ratios: list[float] = []
        wins = 0
        for profile in profiles:
            for direction in ("forward", "inverse"):
                values = cells.get((candidate, profile, direction), [])
                peers = [
                    statistics.median(peer_values)
                    for (peer, peer_profile, peer_direction), peer_values in cells.items()
                    if peer_profile == profile and peer_direction == direction
                ]
                if not values or not peers:
                    continue
                ratio = statistics.median(values) / min(peers)
                ratios.append(ratio)
                wins += ratio <= 1.0 + 1.0e-12
        if ratios:
            summaries.append((
                math.exp(sum(math.log(value) for value in ratios) / len(ratios)),
                wins,
                candidate,
                max(ratios),
            ))
        paired_ratios: list[float] = []
        paired_wins = 0
        for profile in profiles:
            forward = cells.get((candidate, profile, "forward"), [])
            inverse = cells.get((candidate, profile, "inverse"), [])
            peer_totals = []
            for peer in candidates:
                peer_forward = cells.get((peer, profile, "forward"), [])
                peer_inverse = cells.get((peer, profile, "inverse"), [])
                if peer_forward and peer_inverse:
                    peer_totals.append(
                        statistics.median(peer_forward)
                        + statistics.median(peer_inverse)
                    )
            if not forward or not inverse or not peer_totals:
                continue
            ratio = (
                statistics.median(forward) + statistics.median(inverse)
            ) / min(peer_totals)
            paired_ratios.append(ratio)
            paired_wins += ratio <= 1.0 + 1.0e-12
        if paired_ratios:
            paired_summaries.append((
                math.exp(
                    sum(math.log(value) for value in paired_ratios)
                    / len(paired_ratios)
                ),
                paired_wins,
                candidate,
                max(paired_ratios),
            ))
    summaries.sort()
    paired_summaries.sort()
    summary_text = "; ".join(
        f"{escaped(policy_labels[candidate])}: {geometric:.3f}× geometric to the cell best, "
        f"{wins}/{2 * len(profiles)} wins, maximum {maximum:.3f}×"
        for geometric, wins, candidate, maximum in summaries
    )
    status_text = (
        "Reference runs are present and may feed the preregistered issue #9 selection."
        if any(bundle.publication["status"] == "reference" for bundle in cohort)
        else "This cohort remains preliminary; it describes the bounded screen and does not yet select issue #9 inputs."
    )
    eligible_paired = [
        item for item in paired_summaries if item[3] <= 1.10
    ]
    selected_paired: list[tuple[float, int, str, float]] = []
    if eligible_paired:
        best_paired = eligible_paired[0][0]
        selected_paired = [
            item for item in eligible_paired if item[0] <= 1.03 * best_paired
        ][:3]
    paired_text = "; ".join(
        f"{escaped(policy_labels[candidate])}: {geometric:.3f}× geometric, "
        f"{wins}/{len(profiles)} paired wins, maximum {maximum:.3f}×"
        for geometric, wins, candidate, maximum in paired_summaries
    )
    selection_text = (
        "The paired issue #9 gate selects "
        + ", ".join(
            escaped(policy_labels[candidate])
            for _, _, candidate, _ in selected_paired
        )
        + "."
        if selected_paired
        else "No candidate passes the paired issue #9 gate."
    )
    return f"""
      <h3>Composed representation-crossover increment</h3>
      <p>This increment composes the issue #7 horizontal survivors with issue #8 grouped vertical kernels. It compares WVM direct/no-reorder, historical WVM gather-to-split, pruned compact interleaved, plane-major fused split, and plane-major retained-view graphs. The mathematical boundary is horizontal forward followed by vertical forward, or vertical inverse followed by horizontal inverse. Modal work and the nonlinear flux calculation remain explicitly excluded.</p>
      <p>The authoritative totals include every per-execution movement required by each graph. In particular, direct full-spectrum inverse policies rebuild their disposable zero-padded FFTW input on every call; the earlier horizontal-only retained-view result assumed that ready view as an input and remains valid for its narrower question. Primitive FFT, primitive vertical MM, movement, setup, memory, and total timings remain separately published.</p>
      <p>For the latest increment across {len(profiles)} workload profile(s): {summary_text}. The largest correctness error is {format_error(max(errors))}. {escaped(status_text)}</p>
      <p>Issue #9 consumes one complete forward-plus-inverse representation policy, so its selection pairs those medians within each workload before applying the unchanged 3% geometric and 10% worst-workload gates. Paired results: {paired_text}. {selection_text}</p>
    """


def ordering_packing_evidence_table(bundles: list[PublishedBundle]) -> str:
    legacy = [
        bundle for bundle in bundles
        if any(
            provider["id"].startswith("ordering-")
            for provider in bundle.result["providers"]
        )
    ]
    return (
        ordering_packing_legacy_evidence_table(legacy)
        + ordering_packing_boundary_evidence_table(bundles)
    )


def ordering_packing_synthesis(bundles: list[PublishedBundle]) -> str:
    legacy = [
        bundle for bundle in bundles
        if any(
            provider["id"].startswith("ordering-")
            for provider in bundle.result["providers"]
        )
    ]
    return (
        ordering_packing_legacy_synthesis(legacy)
        + ordering_packing_boundary_synthesis(bundles)
    )


def pruned_horizontal_synthesis(bundles: list[PublishedBundle]) -> str:
    outer_increment_id = "fftw-partial-column-pruned-outer-sharding-v2"
    initial_by_topology: dict[tuple[str, int, int], dict[str, list[float]]] = {}
    outer_increment_by_topology: dict[tuple[str, int, int], dict[str, list[float]]] = {}
    errors: list[float] = []
    scratch_bytes: list[int] = []
    for bundle in bundles:
        full = next(
            (item for item in bundle.result["providers"]
             if item["id"] == "fftw-full-2d-retained-reference"),
            None,
        )
        pruned = next(
            (item for item in bundle.result["providers"]
             if item["id"] == "fftw-partial-column-pruned"),
            None,
        )
        if full is None or pruned is None:
            continue
        scheduling = pruned.get("scheduling", {})
        topology = (
            str(pruned.get("schedulingId", "unknown")),
            int(scheduling.get("internalWorkers", pruned.get("workers", 1))),
            int(scheduling.get("outerWorkers", 1)),
        )
        cohort = (
            outer_increment_by_topology
            if bundle.publication.get("incrementId") == outer_increment_id
            else initial_by_topology
        )
        for direction in ("forward", "inverse"):
            full_total = timing(full, "uninstrumented-total", direction)
            pruned_total = timing(pruned, "uninstrumented-total", direction)
            if full_total is None or pruned_total is None:
                continue
            ratio = float(pruned_total["medianSeconds"]) / float(full_total["medianSeconds"])
            cohort.setdefault(
                topology, {"forward": [], "inverse": []}
            )[direction].append(ratio)
        scratch_bytes.append(int(pruned["memory"]["scratchBytes"]))
        errors.extend(
            float(metric[name])
            for metric in pruned["correctness"]
            for name in ("maximumRelativeError", "relativeL2Error")
            if metric.get(name) is not None
        )
    if not initial_by_topology:
        return ""

    def topology_summary(ratios_by_topology, predicate) -> str:
        return "; ".join(
            f"internal={internal}, outer={outer}: " + ", ".join(
            f"{direction} {math.exp(sum(math.log(value) for value in values) / len(values)):.3f}× "
            f"({sum(value < 1.0 for value in values)}/{len(values)} wins)"
            for direction, values in directions.items()
            if values
        )
            for (scheduling_id, internal, outer), directions in sorted(
                ratios_by_topology.items(), key=lambda item: (item[0][2], item[0][1], item[0][0])
            )
            if predicate(scheduling_id, internal, outer)
        )

    initial_ratios = [
        value
        for directions in initial_by_topology.values()
        for values in directions.values()
        for value in values
    ]
    outer_control_ratios = [
        value
        for (_, _, outer), directions in outer_increment_by_topology.items()
        if outer == 1
        for values in directions.values()
        for value in values
    ]
    outer_ratios = [
        value
        for (_, _, outer), directions in outer_increment_by_topology.items()
        if outer > 1
        for values in directions.values()
        for value in values
    ]
    initial_summary = topology_summary(
        initial_by_topology, lambda _id, _internal, _outer: True
    )
    outer_control_summary = topology_summary(
        outer_increment_by_topology, lambda _id, _internal, outer: outer == 1
    )
    outer_summary = topology_summary(
        outer_increment_by_topology, lambda _id, _internal, outer: outer > 1
    )
    initial_geometric = math.exp(
        sum(math.log(value) for value in initial_ratios) / len(initial_ratios)
    )
    outer_section = ""
    if outer_ratios:
        outer_geometric = math.exp(
            sum(math.log(value) for value in outer_ratios) / len(outer_ratios)
        )
        outer_direction_best: list[str] = []
        for direction in ("forward", "inverse"):
            candidates = [
                (
                    math.exp(sum(math.log(value) for value in directions[direction]) /
                             len(directions[direction])),
                    internal,
                    outer,
                    directions[direction],
                )
                for (_, internal, outer), directions in outer_increment_by_topology.items()
                if outer > 1 and directions[direction]
            ]
            if candidates:
                geometric, internal, outer, values = min(candidates)
                outer_direction_best.append(
                    f"best {direction}: internal={internal}, outer={outer}, {geometric:.3f}× "
                    f"geometric ({sum(value < 1.0 for value in values)}/{len(values)} wins)"
                )
        fully_pruned_topologies = [
            (internal, outer)
            for (_, internal, outer), directions in outer_increment_by_topology.items()
            if outer > 1
            and all(directions[direction] for direction in ("forward", "inverse"))
            and all(
                value < 1.0
                for direction in ("forward", "inverse")
                for value in directions[direction]
            )
        ]
        if fully_pruned_topologies:
            advancing = ", ".join(
                f"internal={internal}, outer={outer}"
                for internal, outer in sorted(fully_pruned_topologies)
            )
            conclusion = (
                f"The fully pruned candidate therefore advances to issue #7 at {advancing}: "
                "each of those matched topologies wins every measured workload in both directions. "
                "This cohort does not require a full-forward/pruned-inverse hybrid, although issue #7 "
                "must retest the candidate against its broader algorithm set."
            )
        elif any(value < 1.0 for value in outer_ratios):
            conclusion = (
                "Some direction-specific cells are competitive, but no tested topology wins every "
                "workload in both directions; issue #7 should carry only the viable direction-specific "
                "path rather than a fully pruned tuple."
            )
        else:
            conclusion = (
                "No tested outer-sharded cell beats its matched full retained operator, so this scheduling path does not rescue the measured partial-column decomposition."
            )
        outer_section = f"""
      <h3>Persistent outer plane/field sharding increment</h3>
      <p>This append-only follow-on fixes FFTW internal pthreads at one and partitions planes/fields over persistent workers. The matched full reference uses the same worker count for its two-dimensional plans and radial adapters. The candidate partitions one aggregate full-row-spectrum scratch allocation into disjoint worker slices; it does not multiply aggregate scratch capacity. Empty scheduler dispatch is measured separately, and all complete calls remain out-of-place and allocation-free.</p>
      <p>The same-revision single-worker control is not pooled into the parallel cells: {outer_control_summary if outer_control_ratios else 'not published'}.</p>
      <p>Across {len(outer_ratios)} outer-sharded workload/topology/direction cells, the candidate/full retained ratio is {outer_geometric:.3f}× geometrically and spans {min(outer_ratios):.3f}×–{max(outer_ratios):.3f}×. Topologies: {outer_summary}. {'; '.join(outer_direction_best)}. {conclusion}</p>
      <p>Deeper within-column pruning, reduced aggregate scratch, split storage, generated transforms, efficiency-core-specific scheduling, and caller-visible in-place operation remain outside this increment.</p>
        """
    return f"""
      <h3>Partial-column-pruned feasibility increment</h3>
      <p>The candidate performs every real-row transform but omits complete high-k<sub>x</sub> complex-column transforms that cannot intersect the retained radial disk. It exposes compact mode-keyed output rather than a completed WVM-order half-spectrum. The same-run reference uses FFTW's optimized full two-dimensional transform followed by radial selection or embedding. Planning effort, workers, fixture, precision, workload, warmups, and samples are matched.</p>
      <p>The immutable initial internally threaded cohort contains {len(initial_ratios)} workload/topology/direction comparisons. Its candidate/full retained-operator geometric ratio is {initial_geometric:.3f}× and spans {min(initial_ratios):.3f}×–{max(initial_ratios):.3f}×. Topologies: {initial_summary}. The largest correctness error across all published increments is {format_error(max(errors))}. Aggregate candidate scratch spans {format_bytes(min(scratch_bytes))}–{format_bytes(max(scratch_bytes))} and remains full row-spectrum sized.</p>
      <p>The single-worker inverse evidence motivated the outer-sharding follow-on below; the internally threaded performance-core tuple remains a negative result and is not rewritten.</p>
      {outer_section}
    """


def pruned_horizontal_evidence_table(bundles: list[PublishedBundle]) -> str:
    rows: list[str] = []
    for bundle in bundles:
        result = bundle.result
        full = next(
            (item for item in result["providers"]
             if item["id"] == "fftw-full-2d-retained-reference"),
            None,
        )
        pruned = next(
            (item for item in result["providers"]
             if item["id"] == "fftw-partial-column-pruned"),
            None,
        )
        if full is None or pruned is None:
            continue
        workload = result["workload"]
        run = result["run"]
        values: list[str] = []
        for direction in ("forward", "inverse"):
            full_raw = timing(full, "primitive", direction)
            full_total = timing(full, "uninstrumented-total", direction)
            pruned_row = stage_timing(pruned, "primitive-component", "real row FFTs", direction)
            pruned_column = stage_timing(
                pruned, "primitive-component", "selected-kx complex column FFTs", direction
            )
            pruned_total = timing(pruned, "uninstrumented-total", direction)
            if any(item is None for item in (full_raw, full_total, pruned_row, pruned_column, pruned_total)):
                values.extend(["not measured"] * 4)
                continue
            ratio = float(pruned_total["medianSeconds"]) / float(full_total["medianSeconds"])
            values.extend(
                [
                    f"{format_ms(full_raw['medianSeconds'])} / {format_ms(full_total['medianSeconds'])}",
                    f"{format_ms(pruned_row['medianSeconds'])} / {format_ms(pruned_column['medianSeconds'])}",
                    format_ms(pruned_total["medianSeconds"]),
                    f"{ratio:.3f}×",
                ]
            )
        run_id = run["id"]
        scheduling = pruned.get("scheduling", {})
        internal_workers = int(scheduling.get("internalWorkers", pruned["workers"]))
        outer_workers = int(scheduling.get("outerWorkers", 1))
        planes = int(workload["Nz"]) * int(workload["fields"])
        maximum_shard_planes = (planes + outer_workers - 1) // outer_workers
        maximum_shard_scratch = (
            maximum_shard_planes * int(workload["Ny"]) *
            (int(workload["Nx"]) // 2 + 1) * 16
        )
        full_dispatch = stage_timing(
            full, "diagnostic-component", "batch scheduler empty dispatch", "shared"
        )
        pruned_dispatch = stage_timing(
            pruned, "diagnostic-component", "batch scheduler empty dispatch", "shared"
        )
        dispatch = "not measured"
        if full_dispatch is not None and pruned_dispatch is not None:
            dispatch = (
                f"{format_ms(full_dispatch['medianSeconds'])} / "
                f"{format_ms(pruned_dispatch['medianSeconds'])}"
            )
        rows.append(
            "<tr>"
            f'<td><a href="../../runs/{quote(run_id)}/index.html">{escaped(run_id)}</a><br>'
            f'<span class="muted">{run["samples"]} samples · {publication_badge(bundle.publication["status"])}</span></td>'
            f'<td class="numeric">{workload["Nx"]} × {workload["Ny"]}<br>N<sub>z</sub>={workload["Nz"]}, fields={workload["fields"]}</td>'
            f'<td class="numeric">internal={internal_workers}<br>outer={outer_workers}</td>'
            f'<td class="numeric">{values[0]}</td><td class="numeric">{values[1]}</td>'
            f'<td class="numeric">{values[2]}</td><td class="numeric">{values[3]}</td>'
            f'<td class="numeric">{values[4]}</td><td class="numeric">{values[5]}</td>'
            f'<td class="numeric">{values[6]}</td><td class="numeric">{values[7]}</td>'
            f'<td class="numeric">{dispatch}</td>'
            f'<td class="numeric">{format_bytes(pruned["memory"]["scratchBytes"])} / '
            f'{format_bytes(maximum_shard_scratch)}</td>'
            f'<td class="numeric">{format_error(maximum_correctness_error(pruned))}</td>'
            "</tr>"
        )
    if not rows:
        return ""
    return (
        '<div class="table-scroll"><table class="experiment-evidence-table">'
        '<caption>Medians are milliseconds. Full entries are raw 2-D FFT / retained total. Candidate components are row / selected-column diagnostics and are not added; the separately sampled total is authoritative. Ratio is candidate/full retained total. Empty dispatch is full / candidate and excludes all transform or adapter work.</caption>'
        '<thead><tr><th rowspan="2" scope="col">Run</th><th rowspan="2" scope="col">Workload</th><th rowspan="2" scope="col">Topology</th>'
        '<th colspan="4" scope="colgroup">Forward</th><th colspan="4" scope="colgroup">Inverse</th>'
        '<th rowspan="2" scope="col">Empty dispatch</th><th rowspan="2" scope="col">Scratch aggregate / max shard</th><th rowspan="2" scope="col">Max error</th></tr>'
        '<tr><th scope="col">Full raw / total</th><th scope="col">Candidate row / column</th><th scope="col">Candidate total</th><th scope="col">Ratio</th>'
        '<th scope="col">Full raw / total</th><th scope="col">Candidate row / column</th><th scope="col">Candidate total</th><th scope="col">Ratio</th></tr></thead>'
        f'<tbody>{"".join(rows)}</tbody></table></div>'
    )


RETAINED_HORIZONTAL_PROVIDER_IDS = {
    "fftw",
    "fftw-full-2d-retained-reference",
    "fftw-partial-column-pruned",
    "fftw-plane-major-retained-view",
    "fftw-plane-major-fused-retained-split",
    "fftw-partial-column-pruned-fused-split",
    "accelerate-vdsp-native-retained",
}


def retained_horizontal_candidate_name(provider: dict) -> str:
    outer = provider.get("scheduling", {}).get(
        "outerWorkers", provider.get("outerWorkers", provider.get("workers", 1))
    )
    if provider["id"] == "fftw-plane-major-retained-view":
        return f"FFTW plane-major retained view outer-{outer}"
    if provider["id"] == "fftw-plane-major-fused-retained-split":
        return f"FFTW plane-major fused split outer-{outer}"
    if provider["id"] == "fftw-partial-column-pruned-fused-split":
        return f"FFTW pruned fused split outer-{outer}"
    if provider["id"] == "fftw-partial-column-pruned":
        return f"FFTW pruned outer-{outer}"
    if provider["id"] == "fftw-full-2d-retained-reference":
        return f"FFTW WVM full outer-{outer}"
    if provider["id"] == "accelerate-vdsp-native-retained":
        return f"vDSP native retained outer-{outer}"
    if provider["nativeRepresentationId"].startswith("plane-major"):
        return f"FFTW plane-major full outer-{outer}"
    return f"FFTW WVM full outer-{outer}"


def retained_horizontal_providers(bundle: PublishedBundle) -> list[dict]:
    return [
        provider for provider in bundle.result["providers"]
        if provider["id"] in RETAINED_HORIZONTAL_PROVIDER_IDS
    ]


def retained_component(provider: dict, direction: str) -> dict | None:
    matches = [
        item for item in provider["timings"]
        if item["scope"] == "operator-component"
        and item["direction"] == direction
    ]
    if len(matches) > 1:
        raise ValueError(
            f"Multiple retained operator components for {provider['id']} {direction}"
        )
    return matches[0] if matches else None


def retained_horizontal_evidence_table(bundles: list[PublishedBundle]) -> str:
    rows: list[str] = []
    for bundle in sorted(
        bundles,
        key=lambda item: (
            item.result["workload"]["Nx"],
            item.result["workload"]["Nz"],
            item.result["workload"]["fields"],
            item.result["run"]["id"],
        ),
    ):
        result = bundle.result
        workload = result["workload"]
        for provider in retained_horizontal_providers(bundle):
            raw_forward = timing(provider, "primitive", "forward")
            raw_inverse = timing(provider, "primitive", "inverse")
            retained_forward = retained_component(provider, "forward")
            retained_inverse = retained_component(provider, "inverse")
            total_forward = timing(provider, "uninstrumented-total", "forward")
            total_inverse = timing(provider, "uninstrumented-total", "inverse")
            if total_forward is None or total_inverse is None:
                continue

            def pair(first: dict | None, second: dict | None) -> str:
                if first is None or second is None:
                    return '<span class="muted">fused or not measured</span>'
                if (
                    first.get("medianSeconds") is None
                    or second.get("medianSeconds") is None
                ):
                    return '<span class="muted">elided</span>'
                return f"{timing_with_interval(first)} / {timing_with_interval(second)}"

            setup = provider["setup"]
            memory = provider["memory"]
            explicit_memory = (
                int(memory["persistentBytes"]) + int(memory["scratchBytes"])
            )
            run_id = result["run"]["id"]
            rows.append(
                "<tr>"
                f'<td><a href="../../runs/{quote(run_id)}/index.html">{escaped(run_id)}</a><br>'
                f'<span class="muted">{publication_badge(bundle.publication["status"])} · '
                f'{result["run"]["samples"]} samples</span></td>'
                f'<td>{workload["Nx"]}²<br><span class="muted">N<sub>z</sub>={workload["Nz"]}, '
                f'fields={workload["fields"]}</span></td>'
                f'<td>{escaped(retained_horizontal_candidate_name(provider))}<br>'
                f'<span class="muted">{escaped(provider["nativeRepresentationId"])}</span></td>'
                f'<td class="numeric">{pair(raw_forward, raw_inverse)}</td>'
                f'<td class="numeric">{pair(retained_forward, retained_inverse)}</td>'
                f'<td class="numeric">{pair(total_forward, total_inverse)}</td>'
                f'<td class="numeric">{format_ms(setup["totalSeconds"])}<br>'
                f'<span class="muted">{format_bytes(explicit_memory)}</span></td>'
                f'<td class="numeric">{format_error(maximum_correctness_error(provider))}</td>'
                "</tr>"
            )
    if not rows:
        return ""
    return (
        '<div class="table-scroll"><table class="experiment-evidence-table">'
        '<caption>Forward / inverse medians and deterministic percentile-bootstrap 95% intervals are milliseconds. Raw is shown only when a standalone full provider primitive exists; the pruned transform reports its row and selected-column work as inseparable components rather than inventing a raw FFT total. Retention includes direct selection or inverse zero-fill/embed. The complete retained total is authoritative.</caption>'
        '<thead><tr><th scope="col">Run</th><th scope="col">Workload</th>'
        '<th scope="col">Algorithm / representation</th><th scope="col">Raw FFT</th>'
        '<th scope="col">Retention / embedding</th><th scope="col">Complete retained</th>'
        '<th scope="col">Setup / explicit memory</th><th scope="col">Max error</th></tr></thead>'
        f'<tbody>{"".join(rows)}</tbody></table></div>'
    )


def retained_horizontal_closeout_synthesis(
    bundles: list[PublishedBundle],
) -> str:
    screen_increment = "retained-horizontal-representation-closeout-screen-v1"
    reference_increment = "retained-horizontal-representation-closeout-reference-v1"
    screen = [
        bundle for bundle in bundles
        if bundle.publication.get("incrementId") == screen_increment
        and bundle.publication["status"] == "preliminary"
    ]
    reference = [
        bundle for bundle in bundles
        if bundle.publication.get("incrementId") == reference_increment
        and bundle.publication["status"] == "reference"
    ]
    cohort = reference or screen
    if not cohort:
        return ""

    provider_priority = [
        "fftw-plane-major-retained-view",
        "fftw-plane-major-fused-retained-split",
        "fftw-partial-column-pruned-fused-split",
        "fftw-partial-column-pruned",
        "fftw",
    ]
    records: dict[tuple[str, str, str], list[float]] = {}
    providers_by_id: dict[str, list[tuple[PublishedBundle, dict]]] = {}
    for bundle in cohort:
        provider = next(
            (
                item
                for provider_id in provider_priority
                for item in bundle.result["providers"]
                if item["id"] == provider_id
            ),
            None,
        )
        if provider is None:
            continue
        profile = bundle.result["run"]["profile"]
        providers_by_id.setdefault(provider["id"], []).append((bundle, provider))
        for direction in ("forward", "inverse"):
            total = timing(provider, "uninstrumented-total", direction)
            if total is not None:
                records.setdefault((provider["id"], profile, direction), []).append(
                    float(total["medianSeconds"])
                )

    control_ids = ("fftw", "fftw-partial-column-pruned")
    candidate_ids = [
        provider_id for provider_id in provider_priority
        if provider_id not in control_ids and provider_id in providers_by_id
    ]
    profiles = sorted({key[1] for key in records})
    rows: list[str] = []
    summaries: list[str] = []
    ratio_stats: dict[str, list[tuple[str, str, float]]] = {}
    for provider_id in candidate_ids:
        ratios: list[float] = []
        wins = 0
        for profile in profiles:
            for direction in ("forward", "inverse"):
                key = (provider_id, profile, direction)
                controls = [
                    statistics.median(records[(control_id, profile, direction)])
                    for control_id in control_ids
                    if (control_id, profile, direction) in records
                ]
                if key not in records or not controls:
                    continue
                candidate_seconds = statistics.median(records[key])
                ratio = candidate_seconds / min(controls)
                ratios.append(ratio)
                ratio_stats.setdefault(provider_id, []).append(
                    (profile, direction, ratio)
                )
                wins += int(ratio < 1.0)
                provider = providers_by_id[provider_id][0][1]
                rows.append(
                    "<tr>"
                    f'<th scope="row">{escaped(profile)}</th>'
                    f'<td>{escaped(retained_horizontal_candidate_name(provider))}</td>'
                    f'<td>{escaped(direction)}</td>'
                    f'<td class="numeric">{format_ms(candidate_seconds)}</td>'
                    f'<td class="numeric">{ratio:.3f}×</td>'
                    f'<td class="numeric">{len(records[key])}</td>'
                    "</tr>"
                )
        if ratios:
            geometric = math.exp(statistics.mean(math.log(value) for value in ratios))
            provider = providers_by_id[provider_id][0][1]
            summaries.append(
                f"{escaped(retained_horizontal_candidate_name(provider))}: "
                f"{geometric:.3f}× geometric versus the faster matched control, "
                f"{wins}/{len(ratios)} workload-direction wins"
            )

    memory_rows: list[str] = []
    for provider_id in [*control_ids, *candidate_ids]:
        if provider_id not in providers_by_id:
            continue
        bundle, provider = providers_by_id[provider_id][0]
        result = bundle.result
        workload_bytes = result["workload"]["bytes"]
        if provider_id == "fftw-plane-major-retained-view":
            boundary_bytes = int(workload_bytes["fullSpectrum"])
            boundary = "full half-spectrum plus immutable index view"
            inverse_lifetime = "ready zero-padded input is dead after inverse"
        else:
            boundary_bytes = (
                int(workload_bytes["fullSpectrum"])
                + int(workload_bytes["retainedSpectrum"])
            )
            boundary = "full scratch plus compact retained storage"
            inverse_lifetime = "caller retained input is preserved"
        memory_rows.append(
            "<tr>"
            f'<th scope="row">{escaped(retained_horizontal_candidate_name(provider))}</th>'
            f'<td>{escaped(boundary)}</td>'
            f'<td class="numeric">{format_bytes(boundary_bytes)}</td>'
            f'<td>{escaped(provider["executionContract"]["forward"]["adapterPlacement"])}</td>'
            f'<td>{escaped(inverse_lifetime)}</td>'
            f'<td class="numeric">{format_error(maximum_correctness_error(provider))}</td>'
            "</tr>"
        )

    normalization_ratios: list[float] = []
    for _, provider in providers_by_id.get(
        "fftw-plane-major-fused-retained-split", [],
    ):
        fused = next(
            (
                item for item in provider["timings"]
                if item["scope"] == "diagnostic-total"
                and item["stage"] ==
                    "retained operator with fused horizontal normalization"
            ),
            None,
        )
        separate = next(
            (
                item for item in provider["timings"]
                if item["scope"] == "diagnostic-total"
                and item["stage"] ==
                    "retained operator with separate horizontal normalization"
            ),
            None,
        )
        if fused is not None and separate is not None:
            normalization_ratios.append(
                float(fused["medianSeconds"]) /
                float(separate["medianSeconds"])
            )
    normalization_note = ""
    if normalization_ratios:
        geometric = math.exp(
            statistics.mean(math.log(value) for value in normalization_ratios)
        )
        normalization_note = (
            f" The fused/separate normalized-total geometric ratio is {geometric:.3f}× "
            f"across {len(normalization_ratios)} processes. This diagnostic does not "
            "change the unnormalized retained-operator comparison; setup-time vertical "
            "matrix scaling remains the zero-horizontal-pass policy for issue #13 to test."
        )

    status = "reference" if reference else "preliminary screen"
    handoff = ""
    if reference:
        view_regressions = [
            (profile, direction, ratio)
            for profile, direction, ratio in ratio_stats.get(
                "fftw-plane-major-retained-view", [],
            )
            if ratio >= 1.0
        ]
        regression_text = "; ".join(
            f"{profile} {direction} {ratio:.3f}×"
            for profile, direction, ratio in view_regressions
        )
        handoff = f"""
      <h4>Issue #7 handoff</h4>
      <p>The small policy set carried into issues #13 and #9 is: (1) the plane-major retained view as the conditional latency winner, with ready-inverse-representation construction charged downstream; (2) pruned outer-12 interleaved compact storage as the self-contained materialized control; and (3) full plane-major fused split only as the representation bridge to split vertical GEMM. The view loses to the faster materialized control in {len(view_regressions)} cells ({escaped(regression_text)}), so it is not a universal shape-independent winner. Fused split has no reference cell win and is not a standalone horizontal recommendation. Pruned fused split and Float64 vDSP stop as published negative evidence.</p>
        """
    return f"""
      <h3>Representation-boundary close-out</h3>
      <p>This append-only {status} increment asks whether an immutable plane-major retained view or fused compact split selection can improve on the established plane-major full and partial-column-pruned outer-12 controls. It fixes the logical radial two-thirds operator, six production workloads, Float64 precision, FFTW build and planning effort, worker topology, fixtures, and normalization. It changes only the full/pruned algorithm, retained representation, and whether selection, conversion, zero fill, embedding, and optional normalization are fused.</p>
      <p>The retained view times no forward gather. Its inverse total starts from a ready disposable zero-padded provider-order spectrum because multidimensional FFTW c2r may destroy input; producing that representation remains work for issue #13, not an elided production cost. Compact split candidates include their fused movement in the complete total. Raw primitives, movement components, complete totals, explicit representation storage, placement/liveness, and correctness remain separate below.{normalization_note}</p>
      <ul>{''.join(f'<li>{summary}</li>' for summary in summaries)}</ul>
      {handoff}
      <div class="table-scroll"><table class="experiment-evidence-table">
        <caption>Candidate complete retained median versus the faster matched plane-major full or pruned control. Values below one favor the candidate.</caption>
        <thead><tr><th scope="col">Profile</th><th scope="col">Candidate</th><th scope="col">Direction</th><th scope="col">Median</th><th scope="col">Ratio</th><th scope="col">Processes</th></tr></thead>
        <tbody>{''.join(rows)}</tbody>
      </table></div>
      <div class="table-scroll"><table class="experiment-evidence-table">
        <caption>Representation storage excludes real-grid input/output and opaque FFTW planning memory. Placement and inverse lifetime are algorithm contracts, not mathematical requirements.</caption>
        <thead><tr><th scope="col">Algorithm</th><th scope="col">Boundary storage</th><th scope="col">Bytes</th><th scope="col">Forward placement</th><th scope="col">Inverse lifetime</th><th scope="col">Max error</th></tr></thead>
        <tbody>{''.join(memory_rows)}</tbody>
      </table></div>
    """


def retained_horizontal_synthesis(bundles: list[PublishedBundle]) -> str:
    reference: dict[tuple[str, str, str], list[float]] = {}
    preliminary: dict[str, list[dict]] = {}
    for bundle in bundles:
        profile = bundle.result["run"]["profile"]
        for provider in retained_horizontal_providers(bundle):
            total_forward = timing(provider, "uninstrumented-total", "forward")
            total_inverse = timing(provider, "uninstrumented-total", "inverse")
            if total_forward is None or total_inverse is None:
                continue
            if (
                bundle.publication["status"] == "reference"
                and bundle.publication.get("incrementId") ==
                    "retained-horizontal-finalist-reference-v1"
            ):
                name = retained_horizontal_candidate_name(provider)
                reference.setdefault((profile, name, "forward"), []).append(
                    float(total_forward["medianSeconds"])
                )
                reference.setdefault((profile, name, "inverse"), []).append(
                    float(total_inverse["medianSeconds"])
                )
            elif (
                bundle.publication["status"] == "preliminary"
                and bundle.publication.get("incrementId") ==
                    "retained-horizontal-finalist-screen-v1"
            ):
                preliminary.setdefault(profile, []).append(provider)

    sections: list[str] = []
    closeout = retained_horizontal_closeout_synthesis(bundles)
    if closeout:
        sections.append(closeout)
    if reference:
        profiles = sorted({key[0] for key in reference})
        names = sorted({key[1] for key in reference})
        rows: list[str] = []
        aggregate_ratios: dict[str, list[float]] = {name: [] for name in names}
        aggregate_wins: dict[str, int] = {name: 0 for name in names}
        aggregate_cells: dict[str, int] = {name: 0 for name in names}
        for profile in profiles:
            baseline_name = "FFTW WVM full outer-12"
            if (profile, baseline_name, "forward") not in reference:
                continue
            baseline = {
                direction: statistics.median(reference[(profile, baseline_name, direction)])
                for direction in ("forward", "inverse")
            }
            for name in names:
                key_forward = (profile, name, "forward")
                key_inverse = (profile, name, "inverse")
                if key_forward not in reference or key_inverse not in reference:
                    continue
                forward = statistics.median(reference[key_forward])
                inverse = statistics.median(reference[key_inverse])
                forward_ratio = forward / baseline["forward"]
                inverse_ratio = inverse / baseline["inverse"]
                aggregate_ratios[name].extend((forward_ratio, inverse_ratio))
                aggregate_wins[name] += int(forward_ratio < 1.0) + int(inverse_ratio < 1.0)
                aggregate_cells[name] += 2
                rows.append(
                    "<tr>"
                    f'<th scope="row">{escaped(profile)}</th><td>{escaped(name)}</td>'
                    f'<td class="numeric">{format_ms(forward)} / {format_ms(inverse)}</td>'
                    f'<td class="numeric">{forward_ratio:.3f}× / {inverse_ratio:.3f}×</td>'
                    f'<td class="numeric">{len(reference[key_forward])} / '
                    f'{len(reference[key_inverse])}</td>'
                    "</tr>"
                )
        summaries = []
        for name in names:
            values = aggregate_ratios[name]
            if not values:
                continue
            geometric = math.exp(statistics.mean(math.log(value) for value in values))
            summaries.append(
                f"{escaped(name)}: {geometric:.3f}× geometric across "
                f"{aggregate_cells[name]} direction-workload cells, "
                f"{aggregate_wins[name]}/{aggregate_cells[name]} wins"
            )
        sections.append(f"""
          <h3>Matched reference finalist campaign</h3>
          <p>Each cell aggregates independently planned process medians for an identical workload, algorithm, representation, and worker topology. Ratios use the WVM-order full FFTW outer-12 retained operator as 1.0. Component medians are not added; the separately sampled complete retained total is authoritative.</p>
          <ul>{''.join(f'<li>{summary}</li>' for summary in summaries)}</ul>
          <div class="table-scroll"><table class="experiment-evidence-table">
            <caption>Complete retained forward / inverse milliseconds and candidate / WVM outer-12 ratios.</caption>
            <thead><tr><th scope="col">Profile</th><th scope="col">Candidate</th><th scope="col">Forward / inverse</th><th scope="col">Ratio</th><th scope="col">Processes F / I</th></tr></thead>
            <tbody>{''.join(rows)}</tbody>
          </table></div>
        """)

    guard_rows: list[str] = []
    expansion = False
    for profile, providers in sorted(preliminary.items()):
        vdsp = [
            provider for provider in providers
            if provider["id"] == "accelerate-vdsp-native-retained"
        ]
        fftw = [
            provider for provider in providers
            if provider["id"] != "accelerate-vdsp-native-retained"
        ]
        if not vdsp or not fftw:
            continue
        candidate = min(
            vdsp,
            key=lambda provider: float(
                timing(provider, "uninstrumented-total", "forward")["medianSeconds"]
            ),
        )
        ratios = {}
        for direction in ("forward", "inverse"):
            best = min(
                float(timing(provider, "uninstrumented-total", direction)["medianSeconds"])
                for provider in fftw
            )
            ratios[direction] = (
                float(timing(candidate, "uninstrumented-total", direction)["medianSeconds"])
                / best
            )
        qualifies = ratios["forward"] <= 1.25 and ratios["inverse"] <= 1.25
        expansion = expansion or qualifies
        guard_rows.append(
            "<tr>"
            f'<th scope="row">{escaped(profile)}</th>'
            f'<td class="numeric">{ratios["forward"]:.3f}×</td>'
            f'<td class="numeric">{ratios["inverse"]:.3f}×</td>'
            f'<td>{"expand" if qualifies else "stop after guardrail"}</td>'
            "</tr>"
        )
    if guard_rows:
        decision = (
            "At least one guard workload meets the rule, so broader vDSP coverage is enabled."
            if expansion else
            "Neither guard workload meets the rule, so vDSP remains published negative evidence and does not expand to the production matrix."
        )
        sections.append(f"""
          <h3>Bounded vDSP native-layout guardrail</h3>
          <p>The guardrail grants vDSP its favorable path: direct packed-split retention and embedding without a full WVM-order spectrum. Expansion requires one workload to be no slower than 1.25× the best matched FFTW complete retained operator in both directions. {escaped(decision)}</p>
          <div class="table-scroll"><table class="experiment-evidence-table">
            <caption>vDSP / best matched FFTW complete retained total; values below one favor vDSP.</caption>
            <thead><tr><th scope="col">Profile</th><th scope="col">Forward</th><th scope="col">Inverse</th><th scope="col">Rule result</th></tr></thead>
            <tbody>{''.join(guard_rows)}</tbody>
          </table></div>
          <p class="method-note">The vDSP guard runs remain preliminary and cannot contribute to adoption statistics. Float32 remains a separate follow-up.</p>
        """)
    return "".join(sections)


PIPELINE_PROVIDER_IDS = {
    "pipeline-wvm-direct",
    "pipeline-plane-major-fused-split",
    "pipeline-streaming-pruned-compact-split",
}


def spectral_pipeline_bundles(bundles: list[PublishedBundle]) -> list[PublishedBundle]:
    return [
        bundle for bundle in bundles
        if any(provider["id"] in PIPELINE_PROVIDER_IDS for provider in bundle.result["providers"])
    ]


def spectral_pipeline_provider(bundle: PublishedBundle) -> dict:
    return next(
        provider for provider in bundle.result["providers"]
        if provider["id"] in PIPELINE_PROVIDER_IDS
    )


def spectral_pipeline_evidence_table(bundles: list[PublishedBundle]) -> str:
    rows: list[str] = []
    for bundle in spectral_pipeline_bundles(bundles):
        result = bundle.result
        provider = spectral_pipeline_provider(bundle)
        workload = result["workload"]
        publication = bundle.publication

        def milliseconds(scope: str, stage: str, direction: str) -> str:
            item = stage_timing(provider, scope, stage, direction)
            return "not measured" if item is None else format_ms(item["medianSeconds"])

        def pair(scope: str, stage: str) -> str:
            return f"{milliseconds(scope, stage, 'forward')} / {milliseconds(scope, stage, 'inverse')}"

        if provider["id"] == "pipeline-wvm-direct":
            movement = (
                f"{milliseconds('adapter-component', 'logical retained provider-order view', 'forward')} / "
                f"{milliseconds('adapter-component', 'rebuild zero-padded inverse spectrum', 'inverse')}"
            )
        elif provider["id"] == "pipeline-streaming-pruned-compact-split":
            tiled_forward = stage_timing(
                provider, "adapter-component",
                "plane-major compact staging and blocked split transpose",
                "forward",
            )
            if tiled_forward is None:
                movement = (
                    f"{milliseconds('adapter-component', 'streamed radial direct split write', 'forward')} / "
                    f"{milliseconds('adapter-component', 'streamed split embed and zero fill', 'inverse')}"
                )
            else:
                movement = (
                    f"{format_ms(tiled_forward['medianSeconds'])} / "
                    f"{milliseconds('adapter-component', 'blocked split load, compact staging, embed, and zero fill', 'inverse')}"
                )
        else:
            movement = (
                f"{milliseconds('adapter-component', 'fused retained selection and split pack', 'forward')} / "
                f"{milliseconds('adapter-component', 'fused split embed into zeroed plane-major spectrum', 'inverse')}"
            )
        if provider["id"] == "pipeline-streaming-pruned-compact-split":
            horizontal_work = (
                f"rows {milliseconds('primitive-component', 'real row FFTs', 'forward')} / "
                f"{milliseconds('primitive-component', 'real row FFTs', 'inverse')}<br>"
                f'<span class="muted">selected columns '
                f"{milliseconds('primitive-component', 'selected-kx complex column FFTs', 'forward')} / "
                f"{milliseconds('primitive-component', 'selected-kx complex column FFTs', 'inverse')}</span>"
            )
            retained_horizontal = pair(
                "retained-operator-total",
                "streaming pruned compact split horizontal operator",
            )
        else:
            horizontal_work = pair("primitive", "raw FFT")
            retained_horizontal = pair(
                "retained-operator-total",
                "full FFT fused compact split horizontal operator",
            )
        total = milliseconds(
            "uninstrumented-total", "synthetic antialiased spectral pipeline", "round-trip"
        )
        modal = milliseconds("component", "mode-keyed modal work", "modal")
        explicit_peak = result["workload"].get("bytes", {}).get(
            "spectralPipelineEstimatedExplicitPeak"
        )
        peak_text = format_bytes(explicit_peak) if explicit_peak is not None else "not reported"
        memory = provider["memory"]
        algorithm_resident = memory.get("algorithmResidentBytes")
        benchmark_harness = memory.get("benchmarkHarnessBytes")
        observed_peak = memory.get("observedProcessHighWaterBytes")
        memory_text = (
            f"algorithm {format_bytes(algorithm_resident)}<br>"
            f'<span class="muted">harness {format_bytes(benchmark_harness)}; '
            f'observed {format_bytes(observed_peak)}; estimated {peak_text}</span>'
            if algorithm_resident and benchmark_harness and observed_peak else
            f'{format_bytes(memory["persistentBytes"])}<br>'
            f'<span class="muted">legacy estimated peak {peak_text}</span>'
        )
        round_number = publication.get("campaignRound", "—")
        run_id = result["run"]["id"]
        rows.append(
            "<tr>"
            f'<td><a href="../../runs/{quote(run_id)}/index.html">{escaped(run_id)}</a><br>'
            f'<span class="muted">round {escaped(round_number)} · {publication_badge(publication["status"])}</span></td>'
            f'<td class="numeric">{workload["Nx"]} × {workload["Ny"]}<br>N<sub>z</sub>={workload["Nz"]}, fields={workload["fields"]}</td>'
            f'<td>{escaped(provider_name(provider))}</td>'
            f'<td class="numeric">{horizontal_work}</td>'
            f'<td class="numeric">{retained_horizontal}</td>'
            f'<td class="numeric">{pair("primitive", "raw vertical MM")}</td>'
            f'<td class="numeric">{movement}</td>'
            f'<td class="numeric">{modal}</td>'
            f'<td class="numeric"><strong>{total}</strong></td>'
            f'<td class="numeric">{memory_text}</td>'
            f'<td class="numeric">{format_error(maximum_correctness_error(provider))}</td>'
            "</tr>"
        )
    if not rows:
        return ""
    return (
        '<div class="table-scroll"><table class="experiment-evidence-table">'
        '<caption>Component medians in milliseconds are forward / inverse. Modal work and the uninstrumented total are single round-trip values. Component medians are diagnostic and are not summed; the uninstrumented total is authoritative. New memory-aware runs separate algorithm-resident storage, benchmark-only overhead, conservative estimated peak, and observed process high water; older immutable runs retain their original memory fields.</caption>'
        '<thead><tr><th scope="col">Run</th><th scope="col">Workload</th><th scope="col">Algorithm graph</th>'
        '<th scope="col">Raw FFT / pruned components</th><th scope="col">Horizontal retained total</th>'
        '<th scope="col">Raw vertical MM</th><th scope="col">Movement / rebuild</th>'
        '<th scope="col">Modal work</th><th scope="col">Uninstrumented total</th><th scope="col">Memory</th><th scope="col">Max error</th></tr></thead>'
        f'<tbody>{"".join(rows)}</tbody></table></div>'
    )


def spectral_pipeline_bootstrap(profile_ratios: dict[str, list[float]]) -> tuple[float, float]:
    generator = random.Random(129)
    profiles = sorted(profile_ratios)
    draws = sorted(
        math.exp(
            sum(math.log(generator.choice(profile_ratios[profile])) for profile in profiles)
            / len(profiles)
        )
        for _ in range(20000)
    )
    return draws[int(0.025 * (len(draws) - 1))], draws[int(0.975 * (len(draws) - 1))]


def paired_round_bootstrap(ratios: list[float], seed: int) -> tuple[float, float]:
    generator = random.Random(seed)
    draws = sorted(
        statistics.median(generator.choices(ratios, k=len(ratios)))
        for _ in range(20000)
    )
    return draws[int(0.025 * (len(draws) - 1))], draws[int(0.975 * (len(draws) - 1))]


def spectral_pipeline_general_synthesis(bundles: list[PublishedBundle]) -> str:
    pipeline = spectral_pipeline_bundles(bundles)
    if not pipeline:
        return ""
    reference = [bundle for bundle in pipeline if bundle.publication["status"] == "reference"]
    candidates = reference or pipeline
    latest_increment = max(
        {bundle.publication.get("incrementId", "") for bundle in candidates},
        key=lambda increment: max(
            bundle.result["environment"]["timestampUtc"]
            for bundle in candidates
            if bundle.publication.get("incrementId", "") == increment
        ),
    )
    cohort = [
        bundle for bundle in candidates
        if bundle.publication.get("incrementId", "") == latest_increment
    ]
    cells: dict[tuple[str, str, int], float] = {}
    errors: list[float] = []
    for bundle in cohort:
        provider = spectral_pipeline_provider(bundle)
        candidate_id = bundle.publication.get("campaignCandidateId") or (
            "wvm-direct--outer-dynamic-16"
            if provider["id"] == "pipeline-wvm-direct"
            else "plane-major-fused-split--outer-dynamic-16"
        )
        round_number = int(bundle.publication.get("campaignRound", 1))
        total = stage_timing(
            provider,
            "uninstrumented-total",
            "synthetic antialiased spectral pipeline",
            "round-trip",
        )
        if total is not None:
            cells[(candidate_id, bundle.result["run"]["profile"], round_number)] = float(
                total["medianSeconds"]
            )
        error = maximum_correctness_error(provider)
        if error is not None:
            errors.append(float(error))

    baseline_id = "wvm-direct--outer-dynamic-16"
    candidate_id = "plane-major-fused-split--outer-dynamic-16"
    profiles = sorted({profile for _, profile, _ in cells})
    profile_rows: list[str] = []
    profile_ratios: dict[str, list[float]] = {}
    summary_ratios: list[float] = []
    for profile in profiles:
        rounds = sorted(
            round_number for candidate, candidate_profile, round_number in cells
            if candidate == baseline_id
            and candidate_profile == profile
            and (candidate_id, profile, round_number) in cells
        )
        if not rounds:
            continue
        baseline_values = [cells[(baseline_id, profile, round_number)] for round_number in rounds]
        candidate_values = [cells[(candidate_id, profile, round_number)] for round_number in rounds]
        paired = [candidate / baseline for candidate, baseline in zip(candidate_values, baseline_values)]
        ratio = statistics.median(candidate_values) / statistics.median(baseline_values)
        profile_ratios[profile] = paired
        summary_ratios.append(ratio)
        profile_rows.append(
            "<tr>"
            f'<th scope="row">{escaped(profile)}</th>'
            f'<td class="numeric">{format_ms(statistics.median(baseline_values))}</td>'
            f'<td class="numeric">{format_ms(statistics.median(candidate_values))}</td>'
            f'<td class="numeric">{ratio:.3f}×</td>'
            f'<td class="numeric">{len(rounds)}</td>'
            "</tr>"
        )
    if not summary_ratios:
        return ""
    geometric_ratio = math.exp(sum(math.log(value) for value in summary_ratios) / len(summary_ratios))
    maximum_ratio = max(summary_ratios)
    maximum_error = max(errors) if errors else math.inf
    is_reference = bool(reference)
    interval = spectral_pipeline_bootstrap(profile_ratios) if is_reference else None
    if is_reference:
        statistics_pass = (
            geometric_ratio <= 0.90
            and maximum_ratio <= 1.03
            and interval is not None
            and interval[1] < 1.0
            and maximum_error <= 1.0e-12
        )
        gate_text = (
            "The M4 adoption-statistics gate passes; cross-Mac replication and full-model validation remain required."
            if statistics_pass else
            "The M4 adoption-statistics gate does not pass; this pipeline is not yet an adoption candidate."
        )
        interval_text = f" The stratified paired-bootstrap 95% interval is {interval[0]:.3f}×–{interval[1]:.3f}×."
    else:
        screen_pass = (
            geometric_ratio <= 0.95
            and maximum_ratio <= 1.10
            and maximum_error <= 1.0e-12
        )
        gate_text = (
            "The bounded screen passes, so the same two graphs advance to the three-round reference campaign."
            if screen_pass else
            "The bounded screen does not pass, so the reference campaign is not triggered."
        )
        interval_text = " Confidence is deliberately deferred to the reference campaign."
    return f"""
      <h3>Synthetic antialiased spectral-pipeline campaign</h3>
      <p>This experiment composes the issue #7 horizontal representation, the issue #8 vertically truncated projection, and the issue #13 ordering policy around one deterministic representation-independent real diagonal modal operator. It starts with ready real data and ends with reconstructed real data. It explicitly excludes the WVM nonlinear flux calculation, time integration, state management, and I/O, so it tests a spectral infrastructure decision rather than full-model physics.</p>
      <p>The latest {"reference" if is_reference else "screen"} cohort uses <code>{escaped(latest_increment)}</code>. The plane-major fused-split graph is {geometric_ratio:.3f}× the WVM direct/no-reorder control geometrically across {len(summary_ratios)} production workloads; its worst workload is {maximum_ratio:.3f}×. The maximum recorded correctness error is {maximum_error:.3e}.{interval_text} {escaped(gate_text)}</p>
      <div class="table-scroll"><table class="experiment-evidence-table">
        <caption>Authoritative uninstrumented round-trip medians; candidate / control below one favors the fused-split graph.</caption>
        <thead><tr><th scope="col">Profile</th><th scope="col">WVM direct (ms)</th><th scope="col">Fused split (ms)</th><th scope="col">Ratio</th><th scope="col">Rounds</th></tr></thead>
        <tbody>{"".join(profile_rows)}</tbody>
      </table></div>
      <p class="method-note">Raw FFT, raw forward and inverse vertical MM, representation movement, modal work, setup, and memory remain separately reported below. Component medians are diagnostic; the uninstrumented total is authoritative because cache effects and fused work prevent additive interpretation.</p>
    """


def spectral_pipeline_large_f4_synthesis(bundles: list[PublishedBundle]) -> str:
    pipeline = [
        bundle for bundle in spectral_pipeline_bundles(bundles)
        if bundle.publication.get("incrementId", "").startswith(
            "synthetic-spectral-pipeline-large-f4-"
        )
    ]
    if not pipeline:
        return ""
    reference = [bundle for bundle in pipeline if bundle.publication["status"] == "reference"]
    candidates = reference or pipeline
    latest_increment = max(
        {bundle.publication.get("incrementId", "") for bundle in candidates},
        key=lambda increment: max(
            bundle.result["environment"]["timestampUtc"]
            for bundle in candidates
            if bundle.publication.get("incrementId", "") == increment
        ),
    )
    cohort = [
        bundle for bundle in candidates
        if bundle.publication.get("incrementId", "") == latest_increment
    ]
    baseline_id = "wvm-direct--outer-dynamic-16"
    candidate_id = "plane-major-fused-split--outer-dynamic-16"
    cells: dict[tuple[str, str, int], dict[str, float]] = {}
    errors: list[float] = []
    for bundle in cohort:
        provider = spectral_pipeline_provider(bundle)
        campaign_candidate = bundle.publication.get("campaignCandidateId") or (
            baseline_id if provider["id"] == "pipeline-wvm-direct" else candidate_id
        )
        total = stage_timing(
            provider, "uninstrumented-total",
            "synthetic antialiased spectral pipeline", "round-trip",
        )
        memory = provider["memory"]
        if total is None or not all(memory.get(key) for key in (
            "algorithmResidentBytes", "benchmarkHarnessBytes",
            "estimatedProcessPeakBytes", "observedProcessHighWaterBytes",
        )):
            continue
        cells[(
            campaign_candidate,
            bundle.result["run"]["profile"],
            int(bundle.publication.get("campaignRound", 1)),
        )] = {
            "seconds": float(total["medianSeconds"]),
            "resident": float(memory["algorithmResidentBytes"]),
            "harness": float(memory["benchmarkHarnessBytes"]),
            "estimated": float(memory["estimatedProcessPeakBytes"]),
            "observed": float(memory["observedProcessHighWaterBytes"]),
        }
        error = maximum_correctness_error(provider)
        if error is not None:
            errors.append(float(error))

    profiles = sorted({profile for _, profile, _ in cells})
    ratios: list[float] = []
    resident_ratios: list[float] = []
    observed_ratios: list[float] = []
    profile_round_ratios: dict[str, list[float]] = {}
    rows: list[str] = []
    for profile in profiles:
        rounds = sorted(
            round_number for candidate, candidate_profile, round_number in cells
            if candidate == baseline_id
            and candidate_profile == profile
            and (candidate_id, profile, round_number) in cells
        )
        if not rounds:
            continue
        baseline = [cells[(baseline_id, profile, round_number)] for round_number in rounds]
        candidate = [cells[(candidate_id, profile, round_number)] for round_number in rounds]
        time_ratio = (
            statistics.median(item["seconds"] for item in candidate) /
            statistics.median(item["seconds"] for item in baseline)
        )
        resident_ratio = (
            statistics.median(item["resident"] for item in candidate) /
            statistics.median(item["resident"] for item in baseline)
        )
        observed_ratio = (
            statistics.median(item["observed"] for item in candidate) /
            statistics.median(item["observed"] for item in baseline)
        )
        profile_round_ratios[profile] = [
            candidate_item["seconds"] / baseline_item["seconds"]
            for candidate_item, baseline_item in zip(candidate, baseline)
        ]
        ratios.append(time_ratio)
        resident_ratios.append(resident_ratio)
        observed_ratios.append(observed_ratio)
        rows.append(
            "<tr>"
            f'<th scope="row">{escaped(profile)}</th>'
            f'<td class="numeric">{format_ms(statistics.median(item["seconds"] for item in baseline))}</td>'
            f'<td class="numeric">{format_ms(statistics.median(item["seconds"] for item in candidate))}</td>'
            f'<td class="numeric">{time_ratio:.3f}×</td>'
            f'<td class="numeric">{format_bytes(int(statistics.median(item["resident"] for item in baseline)))} / '
            f'{format_bytes(int(statistics.median(item["resident"] for item in candidate)))}<br>'
            f'<span class="muted">candidate/control {resident_ratio:.3f}×</span></td>'
            f'<td class="numeric">{format_bytes(int(statistics.median(item["observed"] for item in baseline)))} / '
            f'{format_bytes(int(statistics.median(item["observed"] for item in candidate)))}<br>'
            f'<span class="muted">candidate/control {observed_ratio:.3f}×</span></td>'
            f'<td class="numeric">{len(rounds)}</td>'
            "</tr>"
        )
    if not ratios:
        return ""
    geometric_ratio = math.exp(sum(math.log(value) for value in ratios) / len(ratios))
    resident_geometric = math.exp(
        sum(math.log(value) for value in resident_ratios) / len(resident_ratios)
    )
    observed_geometric = math.exp(
        sum(math.log(value) for value in observed_ratios) / len(observed_ratios)
    )
    maximum_ratio = max(ratios)
    maximum_error = max(errors) if errors else math.inf
    is_reference = bool(reference)
    interval = spectral_pipeline_bootstrap(profile_round_ratios) if is_reference else None
    if is_reference:
        strict_regression = maximum_ratio <= 1.03
        single_policy_winner = bool(
            geometric_ratio <= 0.90 and maximum_ratio <= 1.05
            and interval is not None and interval[1] < 1.0
            and maximum_error <= 1.0e-12
        )
        tie = bool(
            0.95 <= geometric_ratio <= 1.05
            or (interval is not None and interval[0] <= 1.0 <= interval[1])
        )
        if single_policy_winner and strict_regression:
            classification = "fused-split performance win"
        elif single_policy_winner:
            classification = "fused-split overall winner with a documented smallest-case regression"
        elif tie and resident_geometric <= 0.95:
            classification = "timing tie with a fused-split memory advantage"
        else:
            classification = "single-policy result inconclusive"
        confidence = (
            f" The stratified paired-bootstrap 95% interval is "
            f"{interval[0]:.3f}×–{interval[1]:.3f}×."
        )
    else:
        classification = "preliminary correctness/capability screen"
        confidence = " Confidence and the final classification are deferred to reference depth."
    return f"""
      <h3>Large four-field nonhydrostatic cohort</h3>
      <p>This append-only increment does not replace the six-workload mixed-field result above. It reruns fields=4 workloads from 256² through 1024² on one clean commit and treats memory capacity as an algorithm result. Algorithm-resident storage includes the ready input/output boundary, provider matrices and operands, transform/modal buffers, weights, and required scratch. Benchmark-only oracle storage remains separate, while observed process high water captures opaque libraries and setup peaks.</p>
      <p>The latest {"reference" if is_reference else "screen"} cohort uses <code>{escaped(latest_increment)}</code>. Fused split is {geometric_ratio:.3f}× the WVM control geometrically in time, with a {maximum_ratio:.3f}× worst workload. Its algorithm-resident ratio is {resident_geometric:.3f}× and observed high-water ratio is {observed_geometric:.3f}×. Maximum correctness error is {maximum_error:.3e}.{confidence} Classification: <strong>{escaped(classification)}</strong>.</p>
      <p class="method-note">Production integration selects one spectral policy; size-dependent algorithm dispatch is not recommended. The original 3% worst-case threshold remains visible as a diagnostic. For this cohort, the 4.0% regression at 256² is accepted because the single fused-split policy is faster overall, materially faster for both larger target workloads, and lower in algorithm-resident memory.</p>
      <div class="table-scroll"><table class="experiment-evidence-table">
        <caption>Authoritative round-trip time and memory. Memory columns are WVM control / fused split; ratios below one favor fused split.</caption>
        <thead><tr><th scope="col">Profile</th><th scope="col">WVM direct (ms)</th><th scope="col">Fused split (ms)</th><th scope="col">Time ratio</th><th scope="col">Algorithm resident</th><th scope="col">Observed high water</th><th scope="col">Rounds</th></tr></thead>
        <tbody>{"".join(rows)}</tbody>
      </table></div>
      <p class="method-note">The 512²/Nz=513/fields=4 and 1024²/Nz=257/fields=4 capacity cases remain deferred. An algorithm that alone satisfies the declared memory rule will be reported as a capacity result, not assigned a fabricated speed ratio.</p>
    """


NATIVE_BRIDGE_INCREMENT = (
    "synthetic-spectral-pipeline-three-way-native-control-reference-v1"
)
NATIVE_BRIDGE_CANDIDATES = (
    "wvm-direct--outer-dynamic-16",
    "plane-major-fused-split--outer-dynamic-16",
    "streaming-pruned-tiled-16--outer-dynamic-16",
)


def native_bridge_stage_pair(provider: dict, candidate_id: str) -> tuple[dict, dict]:
    if candidate_id == NATIVE_BRIDGE_CANDIDATES[0]:
        scope = "primitive"
        stage = "raw FFT"
    else:
        scope = "retained-operator-total"
        stage = (
            "streaming pruned compact split horizontal operator"
            if candidate_id == NATIVE_BRIDGE_CANDIDATES[2]
            else "full FFT fused compact split horizontal operator"
        )
    forward = stage_timing(provider, scope, stage, "forward")
    inverse = stage_timing(provider, scope, stage, "inverse")
    if forward is None or inverse is None:
        raise ValueError(f"native bridge run lacks retained-horizontal timing: {candidate_id}")
    return forward, inverse


def native_bridge_movement_pair(provider: dict, candidate_id: str) -> tuple[dict, dict]:
    if candidate_id == NATIVE_BRIDGE_CANDIDATES[0]:
        stages = (
            "logical retained provider-order view",
            "rebuild zero-padded inverse spectrum",
        )
    elif candidate_id == NATIVE_BRIDGE_CANDIDATES[1]:
        stages = (
            "fused retained selection and split pack",
            "fused split embed into zeroed plane-major spectrum",
        )
    else:
        stages = (
            "plane-major compact staging and blocked split transpose",
            "blocked split load, compact staging, embed, and zero fill",
        )
    forward = stage_timing(provider, "adapter-component", stages[0], "forward")
    inverse = stage_timing(provider, "adapter-component", stages[1], "inverse")
    if forward is None or inverse is None:
        raise ValueError(f"native bridge run lacks movement timing: {candidate_id}")
    return forward, inverse


def native_bridge_cells(
    bundles: list[PublishedBundle],
) -> dict[tuple[str, str, int], dict]:
    cells: dict[tuple[str, str, int], dict] = {}
    cohort = [
        bundle for bundle in spectral_pipeline_bundles(bundles)
        if bundle.publication.get("incrementId") == NATIVE_BRIDGE_INCREMENT
        and bundle.publication["status"] == "reference"
    ]
    for bundle in cohort:
        candidate_id = bundle.publication.get("campaignCandidateId")
        if candidate_id not in NATIVE_BRIDGE_CANDIDATES:
            continue
        provider = spectral_pipeline_provider(bundle)
        horizontal_forward, horizontal_inverse = native_bridge_stage_pair(
            provider, candidate_id,
        )
        movement_forward, movement_inverse = native_bridge_movement_pair(
            provider, candidate_id,
        )
        vertical_forward = stage_timing(
            provider, "primitive", "raw vertical MM", "forward",
        )
        vertical_inverse = stage_timing(
            provider, "primitive", "raw vertical MM", "inverse",
        )
        modal = stage_timing(provider, "component", "mode-keyed modal work", "modal")
        total = stage_timing(
            provider, "uninstrumented-total",
            "synthetic antialiased spectral pipeline", "round-trip",
        )
        if vertical_forward is None or vertical_inverse is None or modal is None or total is None:
            raise ValueError(f"native bridge run lacks a required component: {candidate_id}")
        planning_seconds = sum(
            float(item["medianSeconds"])
            for item in provider["timings"]
            if item["scope"] == "setup-component"
            and "planning" in item["stage"].lower()
        )
        memory = provider["memory"]
        cells[(
            candidate_id,
            bundle.result["run"]["profile"],
            int(bundle.publication["campaignRound"]),
        )] = {
            "runId": bundle.result["run"]["id"],
            "total": float(total["medianSeconds"]),
            "horizontalForward": float(horizontal_forward["medianSeconds"]),
            "horizontalInverse": float(horizontal_inverse["medianSeconds"]),
            "verticalForward": float(vertical_forward["medianSeconds"]),
            "verticalInverse": float(vertical_inverse["medianSeconds"]),
            "movementForward": float(movement_forward["medianSeconds"]),
            "movementInverse": float(movement_inverse["medianSeconds"]),
            "modal": float(modal["medianSeconds"]),
            "setup": float(provider["setup"]["totalSeconds"]),
            "planning": planning_seconds,
            "resident": float(memory["algorithmResidentBytes"]),
            "observed": float(memory["observedProcessHighWaterBytes"]),
            "scratch": float(memory["scratchBytes"]),
            "error": float(maximum_correctness_error(provider) or math.inf),
            "forwardPlacement": provider["executionContract"]["forward"]["nativePlacement"],
            "inversePlacement": provider["executionContract"]["inverse"]["nativePlacement"],
            "outputRepresentation": provider["executionContract"]["forward"]
            ["nativeOutputRepresentationId"],
        }
    return cells


def native_bridge_pair_summary(
    cells: dict[tuple[str, str, int], dict],
    numerator_id: str,
    denominator_id: str,
) -> dict:
    profiles = sorted({profile for _, profile, _ in cells})
    profile_ratios: dict[str, list[float]] = {}
    median_ratios: list[float] = []
    resident_ratios: list[float] = []
    observed_ratios: list[float] = []
    for profile in profiles:
        rounds = sorted(
            round_number for candidate, candidate_profile, round_number in cells
            if candidate == denominator_id and candidate_profile == profile
            and (numerator_id, profile, round_number) in cells
        )
        if not rounds:
            continue
        numerator = [cells[(numerator_id, profile, round_number)] for round_number in rounds]
        denominator = [cells[(denominator_id, profile, round_number)] for round_number in rounds]
        profile_ratios[profile] = [
            left["total"] / right["total"]
            for left, right in zip(numerator, denominator)
        ]
        median_ratios.append(
            statistics.median(item["total"] for item in numerator)
            / statistics.median(item["total"] for item in denominator)
        )
        resident_ratios.append(
            statistics.median(item["resident"] for item in numerator)
            / statistics.median(item["resident"] for item in denominator)
        )
        observed_ratios.append(
            statistics.median(item["observed"] for item in numerator)
            / statistics.median(item["observed"] for item in denominator)
        )
    if not median_ratios:
        return {}
    return {
        "geometricTime": math.exp(
            sum(math.log(value) for value in median_ratios) / len(median_ratios)
        ),
        "maximumTime": max(median_ratios),
        "interval": spectral_pipeline_bootstrap(profile_ratios),
        "geometricResident": math.exp(
            sum(math.log(value) for value in resident_ratios) / len(resident_ratios)
        ),
        "geometricObserved": math.exp(
            sum(math.log(value) for value in observed_ratios) / len(observed_ratios)
        ),
    }


def spectral_pipeline_native_bridge_synthesis(bundles: list[PublishedBundle]) -> str:
    cells = native_bridge_cells(bundles)
    expected_cells = 3 * 3 * 3
    if len(cells) != expected_cells:
        return ""
    profiles = sorted({profile for _, profile, _ in cells})
    tile_to_wvm = native_bridge_pair_summary(
        cells, NATIVE_BRIDGE_CANDIDATES[2], NATIVE_BRIDGE_CANDIDATES[0],
    )
    tile_to_fused = native_bridge_pair_summary(
        cells, NATIVE_BRIDGE_CANDIDATES[2], NATIVE_BRIDGE_CANDIDATES[1],
    )
    labels = {
        NATIVE_BRIDGE_CANDIDATES[0]: "WVM direct",
        NATIVE_BRIDGE_CANDIDATES[1]: "Fused split",
        NATIVE_BRIDGE_CANDIDATES[2]: "Streaming tile 16",
    }
    total_rows: list[str] = []
    component_rows: list[str] = []
    contract_rows: list[str] = []
    for profile in profiles:
        medians = {
            candidate_id: {
                key: statistics.median(
                    item[key] for (candidate, candidate_profile, _), item in cells.items()
                    if candidate == candidate_id and candidate_profile == profile
                )
                for key in (
                    "total", "horizontalForward", "horizontalInverse",
                    "verticalForward", "verticalInverse", "movementForward",
                    "movementInverse", "modal", "setup", "planning", "resident",
                    "observed", "scratch", "error",
                )
            }
            for candidate_id in NATIVE_BRIDGE_CANDIDATES
        }
        total_rows.append(
            "<tr>"
            f'<th scope="row">{escaped(profile)}</th>'
            + "".join(
                f'<td class="numeric">{format_ms(medians[candidate]["total"])}</td>'
                for candidate in NATIVE_BRIDGE_CANDIDATES
            )
            + f'<td class="numeric">{medians[NATIVE_BRIDGE_CANDIDATES[2]]["total"] / medians[NATIVE_BRIDGE_CANDIDATES[0]]["total"]:.3f}×</td>'
            + f'<td class="numeric">{medians[NATIVE_BRIDGE_CANDIDATES[2]]["total"] / medians[NATIVE_BRIDGE_CANDIDATES[1]]["total"]:.3f}×</td>'
            + "</tr>"
        )
        for candidate_id in NATIVE_BRIDGE_CANDIDATES:
            item = medians[candidate_id]
            representative = next(
                value for (candidate, candidate_profile, _), value in cells.items()
                if candidate == candidate_id and candidate_profile == profile
            )
            component_rows.append(
                "<tr>"
                f'<th scope="row">{escaped(profile)}<br><span class="muted">{escaped(labels[candidate_id])}</span></th>'
                f'<td class="numeric">{format_ms(item["horizontalForward"])} / {format_ms(item["horizontalInverse"])}</td>'
                f'<td class="numeric">{format_ms(item["verticalForward"])} / {format_ms(item["verticalInverse"])}</td>'
                f'<td class="numeric">{format_ms(item["movementForward"])} / {format_ms(item["movementInverse"])}</td>'
                f'<td class="numeric">{format_ms(item["modal"])}</td>'
                f'<td class="numeric"><strong>{format_ms(item["total"])}</strong></td>'
                "</tr>"
            )
            contract_rows.append(
                "<tr>"
                f'<th scope="row">{escaped(profile)}<br><span class="muted">{escaped(labels[candidate_id])}</span></th>'
                f'<td class="numeric">{format_ms(item["setup"])}<br><span class="muted">planning {format_ms(item["planning"])}</span></td>'
                f'<td>{escaped(representative["forwardPlacement"])} / {escaped(representative["inversePlacement"])}</td>'
                f'<td>{escaped(representative["outputRepresentation"])}</td>'
                f'<td class="numeric">{format_bytes(int(item["resident"]))}<br><span class="muted">scratch {format_bytes(int(item["scratch"]))}</span></td>'
                f'<td class="numeric">{format_bytes(int(item["observed"]))}</td>'
                f'<td class="numeric">{item["error"]:.3e}</td>'
                "</tr>"
            )
    return f"""
      <h3>Three-way native-control M4 reference</h3>
      <p>This final same-commit campaign asks two deployment questions. For a persistent compiled engine, all three representations are eligible. For a MATLAB-owned compiled core, MATLAB-visible WVM frequency-major interleaved arrays remain authoritative, so only WVM direct is eligible. The latter is therefore the best measured native-boundary algorithm, not the fastest unrestricted pipeline.</p>
      <p>Streaming tile 16 is {tile_to_wvm["geometricTime"]:.3f}× WVM direct geometrically in authoritative total time, with a stratified paired-bootstrap 95% interval of {tile_to_wvm["interval"][0]:.3f}×–{tile_to_wvm["interval"][1]:.3f}× and a {tile_to_wvm["maximumTime"]:.3f}× worst profile. Against fused split it is {tile_to_fused["geometricTime"]:.3f}× ({tile_to_fused["interval"][0]:.3f}×–{tile_to_fused["interval"][1]:.3f}×), with a {tile_to_fused["maximumTime"]:.3f}× worst profile. Its algorithm-resident memory is {tile_to_wvm["geometricResident"]:.3f}× WVM direct and {tile_to_fused["geometricResident"]:.3f}× fused split.</p>
      <p><strong>Persistent compiled-engine selection:</strong> <code>{NATIVE_BRIDGE_CANDIDATES[2]}</code>. <strong>MATLAB-owned WVM-native boundary selection:</strong> <code>{NATIVE_BRIDGE_CANDIDATES[0]}</code>. One policy is used at all three sizes; no size-dependent dispatch or further tuning entered this campaign.</p>
      <div class="table-scroll"><table class="experiment-evidence-table">
        <caption>Authoritative three-round medians. Ratios below one favor streaming tile 16.</caption>
        <thead><tr><th scope="col">Profile</th><th scope="col">WVM direct</th><th scope="col">Fused split</th><th scope="col">Streaming tile 16</th><th scope="col">Tile / WVM</th><th scope="col">Tile / fused</th></tr></thead>
        <tbody>{"".join(total_rows)}</tbody>
      </table></div>
      <h4>Component ledger</h4>
      <div class="table-scroll"><table class="experiment-evidence-table">
        <caption>Three-round component medians in milliseconds. Paired values are forward / inverse. The horizontal column is raw FFT for WVM direct and the complete retained operator for the compact graphs. Components are diagnostic and do not sum to the uninstrumented total.</caption>
        <thead><tr><th scope="col">Profile and graph</th><th scope="col">Horizontal transform</th><th scope="col">Raw vertical MM</th><th scope="col">Movement / rebuild</th><th scope="col">Modal work</th><th scope="col">Authoritative total</th></tr></thead>
        <tbody>{"".join(component_rows)}</tbody>
      </table></div>
      <h4>Setup, placement, and memory</h4>
      <div class="table-scroll"><table class="experiment-evidence-table">
        <caption>Setup and planning are outside steady-state totals. Placement is forward / inverse native placement. No steady-state timed-loop allocation is permitted.</caption>
        <thead><tr><th scope="col">Profile and graph</th><th scope="col">Setup</th><th scope="col">Placement</th><th scope="col">Forward output representation</th><th scope="col">Algorithm resident</th><th scope="col">Observed high water</th><th scope="col">Max error</th></tr></thead>
        <tbody>{"".join(contract_rows)}</tbody>
      </table></div>
      <p class="method-note">The synthetic operator starts from ready real fields and ends with reconstructed real fields. It excludes the nonlinear flux calculation, time stepping, state management, and I/O. Cross-Mac replication and full WVM integration validation remain required before a general-Mac default recommendation.</p>
    """


def spectral_pipeline_synthesis(bundles: list[PublishedBundle]) -> str:
    pipeline = spectral_pipeline_bundles(bundles)
    original = [
        bundle for bundle in pipeline
        if bundle.publication.get("incrementId", "") in {
            "synthetic-spectral-pipeline-screen-v1",
            "synthetic-spectral-pipeline-reference-v1",
        }
    ]
    sections = [spectral_pipeline_general_synthesis(original or pipeline)]
    large_f4 = spectral_pipeline_large_f4_synthesis(pipeline)
    if large_f4:
        sections.append(large_f4)
    native_bridge = spectral_pipeline_native_bridge_synthesis(pipeline)
    if native_bridge:
        sections.append(native_bridge)
    return "".join(sections)


def streaming_pruned_pipeline_synthesis(
    bundles: list[PublishedBundle],
) -> str:
    cohort = [
        bundle for bundle in spectral_pipeline_bundles(bundles)
        if bundle.publication.get("incrementId") ==
        "streaming-pruned-compact-split-screen-v1"
    ]
    if not cohort:
        return ""
    baseline_id = "plane-major-fused-split--outer-dynamic-16"
    candidate_id = "streaming-pruned-compact-split--outer-dynamic-16"
    cells: dict[tuple[str, str], dict[str, float]] = {}
    errors: list[float] = []
    for bundle in cohort:
        provider = spectral_pipeline_provider(bundle)
        candidate = bundle.publication.get("campaignCandidateId")
        if candidate not in {baseline_id, candidate_id}:
            continue
        total = stage_timing(
            provider, "uninstrumented-total",
            "synthetic antialiased spectral pipeline", "round-trip",
        )
        horizontal_stage = (
            "streaming pruned compact split horizontal operator"
            if candidate == candidate_id
            else "full FFT fused compact split horizontal operator"
        )
        horizontal_forward = stage_timing(
            provider, "retained-operator-total", horizontal_stage, "forward",
        )
        horizontal_inverse = stage_timing(
            provider, "retained-operator-total", horizontal_stage, "inverse",
        )
        memory = provider["memory"]
        if total is None or horizontal_forward is None or horizontal_inverse is None:
            continue
        cells[(candidate, bundle.result["run"]["profile"])] = {
            "total": float(total["medianSeconds"]),
            "horizontalForward": float(horizontal_forward["medianSeconds"]),
            "horizontalInverse": float(horizontal_inverse["medianSeconds"]),
            "resident": float(memory["algorithmResidentBytes"]),
            "scratch": float(memory["scratchBytes"]),
            "estimated": float(memory["estimatedProcessPeakBytes"]),
            "observed": float(memory["observedProcessHighWaterBytes"]),
        }
        error = maximum_correctness_error(provider)
        if error is not None:
            errors.append(float(error))

    profiles = sorted({profile for _, profile in cells})
    time_ratios: list[float] = []
    large_ratios: list[float] = []
    resident_ratios: list[float] = []
    rows: list[str] = []
    for profile in profiles:
        baseline = cells.get((baseline_id, profile))
        candidate = cells.get((candidate_id, profile))
        if baseline is None or candidate is None:
            continue
        time_ratio = candidate["total"] / baseline["total"]
        resident_ratio = candidate["resident"] / baseline["resident"]
        time_ratios.append(time_ratio)
        resident_ratios.append(resident_ratio)
        if profile in {
            "wvm-current-512-nz257-f4",
            "wvm-large-1024-nz129-f4",
        }:
            large_ratios.append(time_ratio)
        rows.append(
            "<tr>"
            f'<th scope="row">{escaped(profile)}</th>'
            f'<td class="numeric">{format_ms(baseline["horizontalForward"])} / '
            f'{format_ms(candidate["horizontalForward"])}<br>'
            f'<span class="muted">inverse {format_ms(baseline["horizontalInverse"])} / '
            f'{format_ms(candidate["horizontalInverse"])}</span></td>'
            f'<td class="numeric">{format_ms(baseline["total"])} / '
            f'{format_ms(candidate["total"])}<br>'
            f'<span class="muted">candidate/control {time_ratio:.3f}×</span></td>'
            f'<td class="numeric">{format_bytes(int(baseline["resident"]))} / '
            f'{format_bytes(int(candidate["resident"]))}<br>'
            f'<span class="muted">candidate/control {resident_ratio:.3f}×</span></td>'
            f'<td class="numeric">{format_bytes(int(baseline["scratch"]))} / '
            f'{format_bytes(int(candidate["scratch"]))}</td>'
            f'<td class="numeric">{format_bytes(int(baseline["observed"]))} / '
            f'{format_bytes(int(candidate["observed"]))}</td>'
            "</tr>"
        )
    if not time_ratios or len(time_ratios) != 3 or len(large_ratios) != 2:
        return ""
    geometric_ratio = math.exp(
        sum(math.log(value) for value in time_ratios) / len(time_ratios)
    )
    large_ratio = math.exp(
        sum(math.log(value) for value in large_ratios) / len(large_ratios)
    )
    resident_ratio = math.exp(
        sum(math.log(value) for value in resident_ratios) / len(resident_ratios)
    )
    maximum_error = max(errors) if errors else math.inf
    time_gate = large_ratio <= 0.97 and maximum_error <= 1.0e-12
    memory_gate = (
        resident_ratio <= 0.90 and geometric_ratio <= 1.02
        and maximum_error <= 1.0e-12
    )
    if time_gate:
        classification = "advances on large-case time"
    elif memory_gate:
        classification = "advances on memory within the time bound"
    else:
        classification = "negative preliminary screen"
    initial_screen = f"""
      <h3>Streaming pruned-to-compact-split preliminary screen</h3>
      <p>This non-blocking experiment asks whether issue #12's outer-12 partial-column pruning can survive the final issue #9 representation boundary. The candidate holds one half-spectrum plane per worker, executes every real row transform, skips inactive complex columns, writes retained values directly to compact radial split arrays, and reverses the process one plane at a time. No batch-sized candidate half-spectrum is materialized.</p>
      <p>Across the three uniform fields=4 workloads, the candidate/control complete-pipeline ratio is {geometric_ratio:.3f}× geometrically; the two large decision cases are {large_ratio:.3f}×. Algorithm-resident memory is {resident_ratio:.3f}×. Maximum correctness error is {maximum_error:.3e}. Classification: <strong>{escaped(classification)}</strong>.</p>
      <div class="table-scroll"><table class="experiment-evidence-table">
        <caption>Control / streaming candidate medians. Horizontal values are forward on the first line and inverse on the second; the complete uninstrumented total is authoritative.</caption>
        <thead><tr><th scope="col">Profile</th><th scope="col">Retained horizontal</th><th scope="col">Complete pipeline</th><th scope="col">Algorithm resident</th><th scope="col">Scratch</th><th scope="col">Observed high water</th></tr></thead>
        <tbody>{"".join(rows)}</tbody>
      </table></div>
      <p class="method-note">This is preliminary screen evidence, not a replacement for the issue #9 M4 decision. One uniform algorithm is evaluated across sizes; no size-dependent dispatch is proposed. Issue #17 implicit and hybrid dealiased convolutions remain outside this experiment.</p>
    """
    return (
        initial_screen
        + streaming_pruned_locality_synthesis(bundles)
        + streaming_pruned_reference_synthesis(bundles)
    )


def streaming_pruned_locality_synthesis(
    bundles: list[PublishedBundle],
) -> str:
    increment_id = "streaming-pruned-compact-split-locality-screen-v1"
    cohort = [
        bundle for bundle in spectral_pipeline_bundles(bundles)
        if bundle.publication.get("incrementId") == increment_id
    ]
    if not cohort:
        return ""
    baseline_id = "plane-major-fused-split--outer-dynamic-16"
    direct_id = "streaming-pruned-direct-1--outer-dynamic-16"
    tiled_ids = [
        f"streaming-pruned-tiled-{tile_width}--outer-dynamic-16"
        for tile_width in (4, 8, 16)
    ]
    candidate_ids = [baseline_id, direct_id, *tiled_ids]
    cells: dict[tuple[str, str], dict[str, float]] = {}
    errors: list[float] = []
    for bundle in cohort:
        candidate = bundle.publication.get("campaignCandidateId")
        if candidate not in candidate_ids:
            continue
        provider = spectral_pipeline_provider(bundle)
        total = stage_timing(
            provider, "uninstrumented-total",
            "synthetic antialiased spectral pipeline", "round-trip",
        )
        horizontal_stage = (
            "full FFT fused compact split horizontal operator"
            if candidate == baseline_id else
            "streaming pruned compact split horizontal operator"
        )
        horizontal_forward = stage_timing(
            provider, "retained-operator-total", horizontal_stage, "forward",
        )
        horizontal_inverse = stage_timing(
            provider, "retained-operator-total", horizontal_stage, "inverse",
        )
        if total is None or horizontal_forward is None or horizontal_inverse is None:
            continue
        memory = provider["memory"]
        cells[(candidate, bundle.result["run"]["profile"])] = {
            "total": float(total["medianSeconds"]),
            "horizontalForward": float(horizontal_forward["medianSeconds"]),
            "horizontalInverse": float(horizontal_inverse["medianSeconds"]),
            "resident": float(memory["algorithmResidentBytes"]),
            "scratch": float(memory["scratchBytes"]),
            "estimated": float(memory["estimatedProcessPeakBytes"]),
            "observed": float(memory["observedProcessHighWaterBytes"]),
        }
        error = maximum_correctness_error(provider)
        if error is not None:
            errors.append(float(error))

    profiles = [
        "wvm-current-256-nz129-f4",
        "wvm-current-512-nz257-f4",
        "wvm-large-1024-nz129-f4",
    ]
    large_profiles = set(profiles[1:])
    if any(
        (candidate, profile) not in cells
        for candidate in candidate_ids
        for profile in profiles
    ):
        return ""

    def geometric(values: list[float]) -> float:
        return math.exp(sum(math.log(value) for value in values) / len(values))

    summaries: dict[str, dict[str, float]] = {}
    for candidate in candidate_ids:
        summaries[candidate] = {
            "allBaseline": geometric([
                cells[(candidate, profile)]["total"] /
                cells[(baseline_id, profile)]["total"]
                for profile in profiles
            ]),
            "largeBaseline": geometric([
                cells[(candidate, profile)]["total"] /
                cells[(baseline_id, profile)]["total"]
                for profile in profiles if profile in large_profiles
            ]),
            "allDirect": geometric([
                cells[(candidate, profile)]["total"] /
                cells[(direct_id, profile)]["total"]
                for profile in profiles
            ]),
            "largeDirect": geometric([
                cells[(candidate, profile)]["total"] /
                cells[(direct_id, profile)]["total"]
                for profile in profiles if profile in large_profiles
            ]),
            "resident": geometric([
                cells[(candidate, profile)]["resident"] /
                cells[(baseline_id, profile)]["resident"]
                for profile in profiles
            ]),
        }
    selected_id = min(
        (
            candidate for candidate in tiled_ids
            if summaries[candidate]["resident"] <= 0.90
        ),
        key=lambda candidate: (
            summaries[candidate]["largeBaseline"],
            summaries[candidate]["allBaseline"],
        ),
    )
    selected = summaries[selected_id]
    selected_tile = selected_id.split("tiled-", 1)[1].split("--", 1)[0]
    maximum_error = max(errors) if errors else math.inf
    candidate_rows = []
    for candidate in [direct_id, *tiled_ids]:
        summary = summaries[candidate]
        tile_label = (
            "1 · direct" if candidate == direct_id
            else candidate.split("tiled-", 1)[1].split("--", 1)[0]
        )
        candidate_rows.append(
            "<tr>"
            f'<th scope="row">{escaped(tile_label)}</th>'
            f'<td class="numeric">{summary["allBaseline"]:.3f}×</td>'
            f'<td class="numeric">{summary["largeBaseline"]:.3f}×</td>'
            f'<td class="numeric">{summary["allDirect"]:.3f}×</td>'
            f'<td class="numeric">{summary["largeDirect"]:.3f}×</td>'
            f'<td class="numeric">{summary["resident"]:.3f}×</td>'
            "</tr>"
        )
    profile_rows = []
    for profile in profiles:
        baseline = cells[(baseline_id, profile)]
        direct = cells[(direct_id, profile)]
        selected_cell = cells[(selected_id, profile)]
        profile_rows.append(
            "<tr>"
            f'<th scope="row">{escaped(profile)}</th>'
            f'<td class="numeric">{format_ms(baseline["total"])} / '
            f'{format_ms(direct["total"])} / '
            f'<strong>{format_ms(selected_cell["total"])}</strong></td>'
            f'<td class="numeric">{format_ms(selected_cell["horizontalForward"])} / '
            f'{format_ms(selected_cell["horizontalInverse"])}</td>'
            f'<td class="numeric">{format_bytes(int(selected_cell["resident"]))}<br>'
            f'<span class="muted">{selected_cell["resident"] / baseline["resident"]:.3f}× fused split</span></td>'
            f'<td class="numeric">{format_bytes(int(selected_cell["scratch"]))}</td>'
            f'<td class="numeric">{format_bytes(int(selected_cell["observed"]))}</td>'
            "</tr>"
        )
    return f"""
      <h3>Cache-local compact-tile optimization screen</h3>
      <p>The direct streaming screen exposed page-strided split access as the large-size bottleneck. This append-only follow-up keeps one FFT half-spectrum plane per worker, stages only retained coefficients in a bounded plane-major compact tile, and uses a 32-mode cache-blocked transpose so final split writes and reads are contiguous. Fixed tile widths 4, 8, and 16 are evaluated as uniform policies; no workload-size dispatch is allowed.</p>
      <p>Tile {escaped(selected_tile)} is the preregistered uniform winner. It is {selected["allBaseline"]:.3f}× fused split geometrically across all three workloads and {selected["largeBaseline"]:.3f}× across the two large cases. Relative to direct streaming it is {selected["allDirect"]:.3f}× overall and {selected["largeDirect"]:.3f}× on the large cases. Algorithm-resident memory remains {selected["resident"]:.3f}× fused split, and maximum correctness error is {maximum_error:.3e}. It therefore advances to a separate rotated reference campaign.</p>
      <div class="table-scroll"><table class="experiment-evidence-table">
        <caption>Uniform candidate geometric ratios; values below one are better. Memory is algorithm-resident storage relative to fused split.</caption>
        <thead><tr><th scope="col">Tile width</th><th scope="col">All / fused</th><th scope="col">Large / fused</th><th scope="col">All / direct</th><th scope="col">Large / direct</th><th scope="col">Memory / fused</th></tr></thead>
        <tbody>{"".join(candidate_rows)}</tbody>
      </table></div>
      <div class="table-scroll"><table class="experiment-evidence-table">
        <caption>Fused split / direct streaming / selected tile-{escaped(selected_tile)} authoritative totals in milliseconds. Horizontal values are selected-tile forward / inverse.</caption>
        <thead><tr><th scope="col">Profile</th><th scope="col">Complete pipeline</th><th scope="col">Selected horizontal</th><th scope="col">Selected algorithm resident</th><th scope="col">Selected scratch</th><th scope="col">Selected observed high water</th></tr></thead>
        <tbody>{"".join(profile_rows)}</tbody>
      </table></div>
      <p class="method-note">These are clean one-round preliminary runs with one warmup and five samples per isolated process. The earlier direct-streaming result remains visible above. Tile selection is not reference inference; only a later rotated campaign may change the M4 adoption statistics.</p>
    """


def streaming_pruned_reference_synthesis(
    bundles: list[PublishedBundle],
) -> str:
    increment_id = "streaming-pruned-compact-split-reference-v1"
    cohort = [
        bundle for bundle in spectral_pipeline_bundles(bundles)
        if bundle.publication.get("incrementId") == increment_id
        and bundle.publication["status"] == "reference"
    ]
    if not cohort:
        return ""
    baseline_id = "plane-major-fused-split--outer-dynamic-16"
    candidate_id = "streaming-pruned-tiled-16--outer-dynamic-16"
    cells: dict[tuple[str, str, int], dict[str, float]] = {}
    errors: list[float] = []
    for bundle in cohort:
        candidate = bundle.publication.get("campaignCandidateId")
        if candidate not in {baseline_id, candidate_id}:
            continue
        provider = spectral_pipeline_provider(bundle)
        horizontal_stage = (
            "full FFT fused compact split horizontal operator"
            if candidate == baseline_id else
            "streaming pruned compact split horizontal operator"
        )

        def required(scope: str, stage: str, direction: str) -> float:
            item = stage_timing(provider, scope, stage, direction)
            if item is None:
                raise ValueError(
                    f"reference provider {provider['id']} lacks {scope}/{stage}/{direction}"
                )
            return float(item["medianSeconds"])

        memory = provider["memory"]
        cells[(
            candidate,
            bundle.result["run"]["profile"],
            int(bundle.publication.get("campaignRound", 1)),
        )] = {
            "horizontalForward": required(
                "retained-operator-total", horizontal_stage, "forward",
            ),
            "horizontalInverse": required(
                "retained-operator-total", horizontal_stage, "inverse",
            ),
            "verticalForward": required(
                "primitive", "raw vertical MM", "forward",
            ),
            "modalWork": required(
                "component", "mode-keyed modal work", "modal",
            ),
            "verticalInverse": required(
                "primitive", "raw vertical MM", "inverse",
            ),
            "total": required(
                "uninstrumented-total",
                "synthetic antialiased spectral pipeline", "round-trip",
            ),
            "resident": float(memory["algorithmResidentBytes"]),
            "harness": float(memory["benchmarkHarnessBytes"]),
            "estimated": float(memory["estimatedProcessPeakBytes"]),
            "scratch": float(memory["scratchBytes"]),
            "observed": float(memory["observedProcessHighWaterBytes"]),
        }
        error = maximum_correctness_error(provider)
        if error is not None:
            errors.append(float(error))

    profiles = [
        "wvm-current-256-nz129-f4",
        "wvm-current-512-nz257-f4",
        "wvm-large-1024-nz129-f4",
    ]
    profile_ratios: dict[str, list[float]] = {}
    summary_ratios: list[float] = []
    resident_ratios: list[float] = []
    observed_ratios: list[float] = []
    total_rows: list[str] = []
    component_rows: list[str] = []
    memory_rows: list[str] = []
    for profile_index, profile in enumerate(profiles):
        rounds = [
            round_number for round_number in (1, 2, 3)
            if (baseline_id, profile, round_number) in cells
            and (candidate_id, profile, round_number) in cells
        ]
        if len(rounds) != 3:
            return ""
        baseline = [cells[(baseline_id, profile, round_number)] for round_number in rounds]
        candidate = [cells[(candidate_id, profile, round_number)] for round_number in rounds]
        paired = [
            candidate_item["total"] / baseline_item["total"]
            for candidate_item, baseline_item in zip(candidate, baseline)
        ]
        profile_ratios[profile] = paired
        total_ratio = (
            statistics.median(item["total"] for item in candidate)
            / statistics.median(item["total"] for item in baseline)
        )
        resident_ratio = (
            statistics.median(item["resident"] for item in candidate)
            / statistics.median(item["resident"] for item in baseline)
        )
        observed_ratio = (
            statistics.median(item["observed"] for item in candidate)
            / statistics.median(item["observed"] for item in baseline)
        )
        summary_ratios.append(total_ratio)
        resident_ratios.append(resident_ratio)
        observed_ratios.append(observed_ratio)
        profile_interval = paired_round_bootstrap(paired, 129 + profile_index)
        total_rows.append(
            "<tr>"
            f'<th scope="row">{escaped(profile)}</th>'
            f'<td class="numeric">{format_ms(statistics.median(item["total"] for item in baseline))}</td>'
            f'<td class="numeric">{format_ms(statistics.median(item["total"] for item in candidate))}</td>'
            f'<td class="numeric"><strong>{total_ratio:.3f}×</strong></td>'
            f'<td class="numeric">{profile_interval[0]:.3f}×–{profile_interval[1]:.3f}×</td>'
            "</tr>"
        )

        def pair_component(key: str) -> tuple[float, float, float]:
            baseline_value = statistics.median(item[key] for item in baseline)
            candidate_value = statistics.median(item[key] for item in candidate)
            return baseline_value, candidate_value, candidate_value / baseline_value

        horizontal_forward = pair_component("horizontalForward")
        horizontal_inverse = pair_component("horizontalInverse")
        vertical_forward = pair_component("verticalForward")
        vertical_inverse = pair_component("verticalInverse")
        modal = pair_component("modalWork")
        component_rows.append(
            "<tr>"
            f'<th scope="row">{escaped(profile)}</th>'
            f'<td class="numeric">{format_ms(horizontal_forward[0])} / {format_ms(horizontal_forward[1])}<br>'
            f'<span class="muted">{horizontal_forward[2]:.3f}×</span></td>'
            f'<td class="numeric">{format_ms(horizontal_inverse[0])} / {format_ms(horizontal_inverse[1])}<br>'
            f'<span class="muted">{horizontal_inverse[2]:.3f}×</span></td>'
            f'<td class="numeric">{format_ms(vertical_forward[0])} / {format_ms(vertical_forward[1])}<br>'
            f'<span class="muted">{vertical_forward[2]:.3f}×</span></td>'
            f'<td class="numeric">{format_ms(modal[0])} / {format_ms(modal[1])}<br>'
            f'<span class="muted">{modal[2]:.3f}×</span></td>'
            f'<td class="numeric">{format_ms(vertical_inverse[0])} / {format_ms(vertical_inverse[1])}<br>'
            f'<span class="muted">{vertical_inverse[2]:.3f}×</span></td>'
            "</tr>"
        )
        memory_rows.append(
            "<tr>"
            f'<th scope="row">{escaped(profile)}</th>'
            f'<td class="numeric">{format_bytes(int(statistics.median(item["resident"] for item in baseline)))} / '
            f'{format_bytes(int(statistics.median(item["resident"] for item in candidate)))}<br>'
            f'<span class="muted">{resident_ratio:.3f}×</span></td>'
            f'<td class="numeric">{format_bytes(int(statistics.median(item["scratch"] for item in baseline)))} / '
            f'{format_bytes(int(statistics.median(item["scratch"] for item in candidate)))}</td>'
            f'<td class="numeric">{format_bytes(int(statistics.median(item["estimated"] for item in baseline)))} / '
            f'{format_bytes(int(statistics.median(item["estimated"] for item in candidate)))}</td>'
            f'<td class="numeric">{format_bytes(int(statistics.median(item["observed"] for item in baseline)))} / '
            f'{format_bytes(int(statistics.median(item["observed"] for item in candidate)))}<br>'
            f'<span class="muted">{observed_ratio:.3f}×</span></td>'
            "</tr>"
        )

    geometric_ratio = math.exp(
        sum(math.log(value) for value in summary_ratios) / len(summary_ratios)
    )
    geometric_resident = math.exp(
        sum(math.log(value) for value in resident_ratios) / len(resident_ratios)
    )
    geometric_observed = math.exp(
        sum(math.log(value) for value in observed_ratios) / len(observed_ratios)
    )
    interval = spectral_pipeline_bootstrap(profile_ratios)
    maximum_ratio = max(summary_ratios)
    maximum_error = max(errors) if errors else math.inf
    gate_passed = bool(
        geometric_ratio <= 0.90
        and maximum_ratio <= 1.03
        and geometric_resident <= 0.90
        and interval[1] < 1.0
        and maximum_error <= 1.0e-12
    )
    conclusion = (
        "The M4 fixed-policy reference gate passes. Tile-16 streaming compact split "
        "supersedes fused split as the synthetic-pipeline winner; cross-Mac replication "
        "and full-model validation remain required."
        if gate_passed else
        "The fixed-policy reference gate does not pass; fused split remains the M4 winner."
    )
    return f"""
      <h3>Fixed tile-16 reference campaign</h3>
      <p>After the append-only locality screen selected tile 16, this campaign froze that one policy before inference and compared it only with the same-commit fused-split control. Three independently planned processes per cell used three warmups and 21 measured samples. Candidate and profile order rotated by round; no size-dependent selection or dispatch was permitted.</p>
      <p>Fixed tile-16 streaming is {geometric_ratio:.3f}× fused split geometrically in authoritative total time, with a stratified paired-bootstrap 95% interval of {interval[0]:.3f}×–{interval[1]:.3f}×. Its worst profile remains a win at {maximum_ratio:.3f}×. Algorithm-resident memory is {geometric_resident:.3f}× and observed process high water is {geometric_observed:.3f}×. Maximum correctness error is {maximum_error:.3e}. <strong>{escaped(conclusion)}</strong></p>
      <div class="table-scroll"><table class="experiment-evidence-table">
        <caption>Authoritative uninstrumented totals. Each workload ratio uses the median of three process medians; intervals bootstrap the three paired round ratios.</caption>
        <thead><tr><th scope="col">Profile</th><th scope="col">Fused split (ms)</th><th scope="col">Tile 16 (ms)</th><th scope="col">Ratio</th><th scope="col">Paired 95% interval</th></tr></thead>
        <tbody>{"".join(total_rows)}</tbody>
      </table></div>
      <div class="table-scroll"><table class="experiment-evidence-table">
        <caption>Component medians are fused split / tile 16 with candidate/control below. They remain diagnostic and are not summed; the uninstrumented total above is authoritative.</caption>
        <thead><tr><th scope="col">Profile</th><th scope="col">Horizontal forward</th><th scope="col">Horizontal inverse</th><th scope="col">Raw vertical forward</th><th scope="col">Modal work</th><th scope="col">Raw vertical inverse</th></tr></thead>
        <tbody>{"".join(component_rows)}</tbody>
      </table></div>
      <div class="table-scroll"><table class="experiment-evidence-table">
        <caption>Memory is fused split / tile 16. Algorithm resident is the decision measure; scratch, conservative explicit peak, and observed process high water remain separate.</caption>
        <thead><tr><th scope="col">Profile</th><th scope="col">Algorithm resident</th><th scope="col">Scratch</th><th scope="col">Estimated explicit peak</th><th scope="col">Observed high water</th></tr></thead>
        <tbody>{"".join(memory_rows)}</tbody>
      </table></div>
      <p class="method-note">The speedup resides in the retained horizontal operator: vertical GEMM and modal-work medians are effectively unchanged. The macOS allocator interposer verifies zero steady-state allocation on the same clean source commit. Primitive full-FFT and primitive GEMM results remain separately published in their foundational experiments.</p>
    """


def dealiased_convolution_reference_synthesis(
    bundles: list[PublishedBundle],
) -> str:
    cohort = [
        bundle for bundle in bundles
        if bundle.publication.get("incrementId") ==
        "wvm-derived-horizontal-advection-reference-v1"
    ]
    if not cohort:
        return ""
    baseline_id = "explicit-parallel"
    candidate_id = "fftwpp-parallel"
    cells: dict[tuple[str, str, int], dict[str, float]] = {}
    errors: list[float] = []
    for bundle in cohort:
        campaign_candidate = bundle.publication.get("campaignCandidateId")
        round_number = bundle.publication.get("campaignRound")
        if campaign_candidate not in {baseline_id, candidate_id}:
            continue
        if not isinstance(round_number, int):
            continue
        provider = bundle.result["providers"][0]
        total = stage_timing(
            provider, "uninstrumented-total",
            "WVM-like four-target horizontal advection", "forward",
        )
        if total is None:
            continue
        memory = provider["memory"]
        cells[(
            campaign_candidate, bundle.result["run"]["profile"], round_number,
        )] = {
            "total": float(total["medianSeconds"]),
            "planning": float(provider["planning"]["seconds"]),
            "resident": float(memory["algorithmResidentBytes"]),
            "harness": float(memory["benchmarkHarnessBytes"]),
            "estimated": float(memory["estimatedProcessPeakBytes"]),
            "observed": float(memory["observedProcessHighWaterBytes"]),
        }
        error = maximum_correctness_error(provider)
        if error is not None:
            errors.append(float(error))

    profiles = sorted({profile for _, profile, _ in cells})
    if len(profiles) != 3:
        return ""
    rows: list[str] = []
    memory_rows: list[str] = []
    profile_ratios: dict[str, list[float]] = {}
    summary_ratios: list[float] = []
    resident_ratios: list[float] = []
    for profile_index, profile in enumerate(profiles):
        baseline = {
            round_number: cells[(baseline_id, profile, round_number)]
            for round_number in range(1, 4)
            if (baseline_id, profile, round_number) in cells
        }
        candidate = {
            round_number: cells[(candidate_id, profile, round_number)]
            for round_number in range(1, 4)
            if (candidate_id, profile, round_number) in cells
        }
        if len(baseline) != 3 or len(candidate) != 3:
            return ""
        ratios = [
            candidate[round_number]["total"] /
            baseline[round_number]["total"]
            for round_number in range(1, 4)
        ]
        profile_ratios[profile] = ratios
        baseline_total = statistics.median(
            baseline[round_number]["total"] for round_number in range(1, 4)
        )
        candidate_total = statistics.median(
            candidate[round_number]["total"] for round_number in range(1, 4)
        )
        ratio = candidate_total / baseline_total
        summary_ratios.append(ratio)
        lower, upper = paired_round_bootstrap(ratios, 129 + profile_index)
        baseline_resident = statistics.median(
            baseline[round_number]["resident"] for round_number in range(1, 4)
        )
        candidate_resident = statistics.median(
            candidate[round_number]["resident"] for round_number in range(1, 4)
        )
        resident_ratio = candidate_resident / baseline_resident
        resident_ratios.append(resident_ratio)
        rows.append(
            "<tr>"
            f'<th scope="row">{escaped(profile)}</th>'
            f'<td class="numeric">{format_ms(baseline_total)}</td>'
            f'<td class="numeric">{format_ms(candidate_total)}</td>'
            f'<td class="numeric">{ratio:.3f}×</td>'
            f'<td class="numeric">{lower:.3f}×–{upper:.3f}×</td>'
            f'<td class="numeric">{format_ms(statistics.median(baseline[round_number]["planning"] for round_number in range(1, 4)))}</td>'
            f'<td class="numeric">{format_ms(statistics.median(candidate[round_number]["planning"] for round_number in range(1, 4)))}</td>'
            "</tr>"
        )
        memory_rows.append(
            "<tr>"
            f'<th scope="row">{escaped(profile)}</th>'
            f'<td class="numeric">{format_bytes(int(baseline_resident))}</td>'
            f'<td class="numeric">{format_bytes(int(candidate_resident))}</td>'
            f'<td class="numeric">{resident_ratio:.3f}×</td>'
            f'<td class="numeric">{format_bytes(int(statistics.median(baseline[round_number]["harness"] for round_number in range(1, 4))))} / '
            f'{format_bytes(int(statistics.median(candidate[round_number]["harness"] for round_number in range(1, 4))))}</td>'
            f'<td class="numeric">{format_bytes(int(statistics.median(baseline[round_number]["observed"] for round_number in range(1, 4))))} / '
            f'{format_bytes(int(statistics.median(candidate[round_number]["observed"] for round_number in range(1, 4))))}</td>'
            "</tr>"
        )

    geometric_time = math.exp(statistics.fmean(
        math.log(value) for value in summary_ratios
    ))
    geometric_resident = math.exp(statistics.fmean(
        math.log(value) for value in resident_ratios
    ))
    lower, upper = spectral_pipeline_bootstrap(profile_ratios)
    maximum_ratio = max(summary_ratios)
    maximum_error = max(errors) if errors else math.inf
    adoption_passed = bool(
        geometric_time <= 0.90 and maximum_ratio <= 1.03
        and upper < 1.0 and geometric_resident <= 0.80
        and maximum_error <= 1.0e-12
    )
    conclusion = (
        "The campaign is reference-quality, and the fixed FFTW++ four-target policy "
        "passes the M4 adoption gate. Full-flux integration and cross-Mac replication remain required."
        if adoption_passed else
        "The campaign is reference-quality, but the fixed FFTW++ candidate does not meet the M4 adoption threshold."
    )
    return f"""
      <h3>Rotated finalist-only M4 reference-depth campaign</h3>
      <p>Each finalist runs alone after an independent correctness oracle is destroyed and FFTW wisdom is cleared. Three process rounds rotate both candidate and workload order; every process uses three warmups and 21 samples. The authoritative boundary remains 15 ready compact spectra to four ready compact outputs.</p>
      <p>FFTW++ is {geometric_time:.3f}× explicit FFTW geometrically, with a stratified paired-bootstrap 95% interval of {lower:.3f}×–{upper:.3f}× and a worst profile of {maximum_ratio:.3f}×. Counted algorithm-resident storage is {geometric_resident:.3f}× geometrically and maximum mode-keyed error is {maximum_error:.3e}. <strong>{escaped(conclusion)}</strong></p>
      <div class="table-scroll"><table class="experiment-evidence-table">
        <caption>Median authoritative totals and candidate/control inference. Planning is outside the timed boundary and shown separately.</caption>
        <thead><tr><th scope="col">Profile</th><th scope="col">Explicit FFTW (ms)</th><th scope="col">FFTW++ (ms)</th><th scope="col">Ratio</th><th scope="col">Paired 95% interval</th><th scope="col">Explicit planning (ms)</th><th scope="col">FFTW++ planning (ms)</th></tr></thead>
        <tbody>{"".join(rows)}</tbody>
      </table></div>
      <div class="table-scroll"><table class="experiment-evidence-table">
        <caption>Memory is explicit / FFTW++ where paired. Algorithm resident is the preregistered decision measure; harness-only storage and process high water remain separate.</caption>
        <thead><tr><th scope="col">Profile</th><th scope="col">Explicit resident</th><th scope="col">FFTW++ resident</th><th scope="col">Ratio</th><th scope="col">Harness explicit / FFTW++</th><th scope="col">Observed high water explicit / FFTW++</th></tr></thead>
        <tbody>{"".join(memory_rows)}</tbody>
      </table></div>
      <p class="method-note">The same-commit allocator interposer verifies zero steady-state application allocations. This reference-depth conclusion covers only the isolated WVM-derived horizontal quadratic operator on the measured M4 Max. It does not include vertical transforms, coefficient accumulation, the complete nonlinear flux, Float32, or a general-Mac recommendation.</p>
    """


def dealiased_convolution_synthesis(bundles: list[PublishedBundle]) -> str:
    feasibility_rows: list[str] = []
    advection_rows: list[str] = []
    projections: list[str] = []
    advection_time_ratios: list[float] = []
    advection_memory_ratios: list[float] = []
    advection_errors: list[float] = []
    reference_section = dealiased_convolution_reference_synthesis(bundles)
    ordered = sorted(
        bundles,
        key=lambda item: (
            item.result["workload"]["Nx"],
            item.result["providers"][0]["algorithmId"],
        ),
    )
    for bundle in ordered:
        result = bundle.result
        providers = {provider["id"]: provider for provider in result["providers"]}
        baseline = providers.get("fftw-explicit-dealiased-convolution")
        candidate = providers.get("fftwpp-hybrid-hermitian-convolution")
        if baseline is not None and candidate is not None:
            baseline_total = stage_timing(
                baseline, "uninstrumented-total",
                "dealiased four-field convolution", "forward",
            )
            candidate_total = stage_timing(
                candidate, "uninstrumented-total",
                "dealiased four-field convolution", "forward",
            )
            if baseline_total is not None and candidate_total is not None:
                products = int(candidate["algorithmId"].rsplit("p", 1)[1])
                baseline_memory = int(baseline["memory"].get(
                    "algorithmResidentBytes", baseline["memory"]["persistentBytes"],
                ))
                candidate_memory = int(candidate["memory"].get(
                    "algorithmResidentBytes", candidate["memory"]["persistentBytes"],
                ))
                feasibility_rows.append(
                    "<tr>"
                    f'<td>{result["workload"]["Nx"]} × {result["workload"]["Ny"]}</td>'
                    f'<td class="numeric">{products}</td>'
                    f'<td class="numeric">{format_ms(baseline_total["medianSeconds"])}</td>'
                    f'<td class="numeric">{format_ms(candidate_total["medianSeconds"])}</td>'
                    f'<td class="numeric">{candidate_total["medianSeconds"] / baseline_total["medianSeconds"]:.3f}×</td>'
                    f'<td class="numeric">{format_bytes(baseline_memory)}</td>'
                    f'<td class="numeric">{format_bytes(candidate_memory)}</td>'
                    f'<td class="numeric">{candidate_memory / baseline_memory:.3f}×</td>'
                    f'<td class="numeric">{format_error(maximum_correctness_error(candidate))}</td>'
                    "</tr>"
                )
                if result["workload"]["Nx"] == 512:
                    projections.append(
                        "<tr>"
                        f'<td class="numeric">{products}</td>'
                        f'<td class="numeric">{format_bytes(4 * baseline_memory)}</td>'
                        f'<td class="numeric">{format_bytes(4 * candidate_memory)}</td>'
                        f'<td class="numeric">{candidate_memory / baseline_memory:.3f}×</td>'
                        "</tr>"
                    )

        advection_ids = (
            "fftw-explicit-streamed-wvm-advection",
            "fftw-explicit-parallel-target-wvm-advection",
            "fftwpp-streamed-target-wvm-advection",
            "fftwpp-all-target-wvm-advection",
            "fftwpp-parallel-target-wvm-advection",
        )
        if not all(identifier in providers for identifier in advection_ids):
            continue
        advection = [providers[identifier] for identifier in advection_ids]
        totals = [
            stage_timing(
                provider, "uninstrumented-total",
                "WVM-like four-target horizontal advection", "forward",
            )
            for provider in advection
        ]
        if any(item is None for item in totals):
            continue
        serial_explicit, parallel_explicit, serial_implicit, all_target, parallel_implicit = advection
        parallel_explicit_memory = int(parallel_explicit["memory"].get(
            "algorithmResidentBytes", parallel_explicit["memory"]["persistentBytes"],
        ))
        parallel_implicit_memory = int(parallel_implicit["memory"].get(
            "algorithmResidentBytes", parallel_implicit["memory"]["persistentBytes"],
        ))
        time_ratio = (
            totals[4]["medianSeconds"] / totals[1]["medianSeconds"]
        )
        memory_ratio = parallel_implicit_memory / parallel_explicit_memory
        error = max(maximum_correctness_error(provider) for provider in advection)
        advection_time_ratios.append(time_ratio)
        advection_memory_ratios.append(memory_ratio)
        advection_errors.append(error)
        advection_rows.append(
            "<tr>"
            f'<td>{result["workload"]["Nx"]} × {result["workload"]["Ny"]}</td>'
            f'<td class="numeric">{format_ms(totals[0]["medianSeconds"])}</td>'
            f'<td class="numeric">{format_ms(totals[1]["medianSeconds"])}</td>'
            f'<td class="numeric">{format_ms(totals[2]["medianSeconds"])}</td>'
            f'<td class="numeric">{format_ms(totals[3]["medianSeconds"])}</td>'
            f'<td class="numeric">{format_ms(totals[4]["medianSeconds"])}</td>'
            f'<td class="numeric">{time_ratio:.3f}×</td>'
            f'<td class="numeric">{format_bytes(parallel_explicit_memory)}</td>'
            f'<td class="numeric">{format_bytes(parallel_implicit_memory)}</td>'
            f'<td class="numeric">{memory_ratio:.3f}×</td>'
            f'<td class="numeric">{format_error(error)}</td>'
            "</tr>"
        )
    if not feasibility_rows and not advection_rows and not reference_section:
        return ""
    feasibility = ""
    projection = ""
    if feasibility_rows:
        feasibility = f"""
      <h3>Independent-product feasibility screen</h3>
      <p>The original increment computes a synthetic quadratic convolution, not WVM’s nonlinear flux. It uses four compact radial Hermitian inputs and either four or twelve independent products. It remains the amortization and FFTW++ capability baseline; it is not rewritten as a WVM expression benchmark.</p>
      <div class="table-scroll"><table class="experiment-evidence-table">
        <caption>Preliminary single-thread M4 screen. Ratios below one favor FFTW++; the caller-visible compact total is authoritative.</caption>
        <thead><tr><th scope="col">Grid</th><th scope="col">Products</th><th scope="col">Explicit (ms)</th><th scope="col">FFTW++ (ms)</th><th scope="col">Time ratio</th><th scope="col">Explicit memory</th><th scope="col">FFTW++ memory</th><th scope="col">Memory ratio</th><th scope="col">Max error</th></tr></thead>
        <tbody>{"".join(feasibility_rows)}</tbody>
      </table></div>
        """
    if projections:
        projection = f"""
      <h3>Analytical 1024² memory projection</h3>
      <p>The same fixed topology has quadratic storage in horizontal size. The table scales each measured 512² algorithm-resident ledger by four; it is a capacity projection, not a 1024² timing result or a substitute for measuring allocator and plan overhead.</p>
      <div class="table-scroll"><table class="experiment-evidence-table">
        <thead><tr><th scope="col">Products</th><th scope="col">Explicit FFTW</th><th scope="col">FFTW++</th><th scope="col">Ratio</th></tr></thead>
        <tbody>{"".join(projections)}</tbody>
      </table></div>
        """
    advection_section = ""
    if advection_rows:
        geometric_time = math.exp(statistics.fmean(
            math.log(value) for value in advection_time_ratios
        ))
        geometric_memory = math.exp(statistics.fmean(
            math.log(value) for value in advection_memory_ratios
        ))
        maximum_error = max(advection_errors)
        advection_section = f"""
      <h3>WVM-derived four-target horizontal-advection screen</h3>
      <p>The follow-up audits WVM’s nonhydrostatic compiled flux and isolates its shared horizontal quadratic map. Ready spectra for <em>U</em>, <em>V</em>, <em>W</em> and the three derivatives of each of four targets produce four outputs, −(<em>Uq</em><sub>x</sub> + <em>Vq</em><sub>y</sub> + <em>Wq</em><sub>z</sub>). Vertical reconstruction and projection, phase evolution, coefficient accumulation, and the rest of the nonlinear flux remain excluded.</p>
      <p>The fixed primary comparison uses persistent four-target scheduling on both sides. Across the published cohort, FFTW++ is {geometric_time:.3f}× the explicit FFTW time and {geometric_memory:.3f}× its counted algorithm-resident storage geometrically; maximum mode-keyed error is {maximum_error:.3e}. This is a preliminary advancement screen, not a reference result or full-flux claim.</p>
      <div class="table-scroll"><table class="experiment-evidence-table">
        <caption>Authoritative ready-compact-input to ready-compact-output medians. The primary ratio compares the two persistent four-target schedules; ratios below one favor FFTW++.</caption>
        <thead><tr><th scope="col">Grid</th><th scope="col">Explicit serial (ms)</th><th scope="col">Explicit parallel (ms)</th><th scope="col">FFTW++ streamed (ms)</th><th scope="col">FFTW++ all-target (ms)</th><th scope="col">FFTW++ parallel (ms)</th><th scope="col">Primary time ratio</th><th scope="col">Explicit parallel memory</th><th scope="col">FFTW++ parallel memory</th><th scope="col">Memory ratio</th><th scope="col">Max error</th></tr></thead>
        <tbody>{"".join(advection_rows)}</tbody>
      </table></div>
      <p class="method-note">All five candidates use one topology uniformly across 256², 512², and 1024². The all-target FFTW++ path fixes the centered transform to <em>m=N</em>: optimizer-selected two-loop input rotation is not valid for this non-permutation-equivariant expression map. The allocator interposer verifies zero application allocations after warmup.</p>
        """
    closing_note = (
        "The reference campaign can motivate bounded full-model integration, but "
        "it cannot establish complete WVM nonlinear-flux performance, Float32 "
        "behavior, a full timestep result, or a general-Mac recommendation."
        if reference_section else
        "These results can promote a fixed topology into rotated reference depth. "
        "They cannot establish complete WVM nonlinear-flux performance, Float32 "
        "behavior, a full timestep result, or a general-Mac recommendation."
    )
    return f"""
      <h2>Bounded feasibility synthesis</h2>
      <p>Each increment declares its mathematical expression map, topology, timed boundary, and exclusions. New WVM-derived evidence extends the original feasibility result without clobbering it.</p>
      {feasibility}
      <p>The public FFTW++ optimizer is usable with four inputs and four outputs. A single twelve-output application (B &gt; A) crashes reproducibly during optimization, so the correct twelve-product candidate consists of three independently planned four-output applications and includes the required input restoration. That limitation remains visible rather than being silently treated as one fused call.</p>
      {projection}
      {advection_section}
      {reference_section}
      <p class="method-note">{escaped(closing_note)}</p>
    """


def vertically_batched_advection_reference_synthesis(
    bundles: list[PublishedBundle],
) -> str:
    cohort = [
        bundle for bundle in bundles
        if bundle.publication.get("incrementId") ==
        "vertically-batched-advection-reference-v1"
        and bundle.publication.get("status") == "reference"
    ]
    if not cohort:
        return ""
    baseline_id = "explicit-parallel"
    candidate_id = "fftwpp-parallel"
    profile_order = (
        "wvm-current-256-nz129-f4",
        "wvm-current-512-nz257-f4",
        "wvm-large-512-nz513-f4",
        "wvm-large-1024-nz129-f4",
    )
    cells: dict[tuple[str, str, int], dict[str, float]] = {}
    errors: list[float] = []
    for bundle in cohort:
        campaign_candidate = bundle.publication.get("campaignCandidateId")
        round_number = bundle.publication.get("campaignRound")
        if campaign_candidate not in {baseline_id, candidate_id}:
            continue
        if not isinstance(round_number, int):
            continue
        provider = bundle.result["providers"][0]
        total = stage_timing(
            provider, "uninstrumented-total",
            "vertically batched WVM-derived advection pipeline", "forward",
        )
        inverse_vertical = stage_timing(
            provider, "primitive", "raw inverse vertical GEMM (15 fields)",
            "inverse",
        )
        horizontal = stage_timing(
            provider, "component",
            "vertically batched horizontal advection including level movement",
            "horizontal",
        )
        movement = stage_timing(
            provider, "adapter-component",
            "all-level split/field-major packing and projected-output scatter",
            "horizontal",
        )
        forward_vertical = stage_timing(
            provider, "primitive", "raw forward vertical GEMM (4 fields)",
            "forward",
        )
        matrix_setup = stage_timing(
            provider, "setup-component", "directional vertical matrix preparation",
            "shared",
        )
        horizontal_setup = stage_timing(
            provider, "setup-component",
            "horizontal planning and persistent scheduler setup", "shared",
        )
        required = (
            total, inverse_vertical, horizontal, movement, forward_vertical,
            matrix_setup, horizontal_setup,
        )
        if any(item is None for item in required):
            continue
        memory = provider["memory"]
        cells[(
            campaign_candidate, bundle.result["run"]["profile"], round_number,
        )] = {
            "total": float(total["medianSeconds"]),
            "inverse": float(inverse_vertical["medianSeconds"]),
            "horizontal": float(horizontal["medianSeconds"]),
            "movement": float(movement["medianSeconds"]),
            "forward": float(forward_vertical["medianSeconds"]),
            "matrixSetup": float(matrix_setup["medianSeconds"]),
            "horizontalSetup": float(horizontal_setup["medianSeconds"]),
            "resident": float(memory["algorithmResidentBytes"]),
            "scratch": float(memory["scratchBytes"]),
            "harness": float(memory["benchmarkHarnessBytes"]),
            "estimated": float(memory["estimatedProcessPeakBytes"]),
            "observed": float(memory["observedProcessHighWaterBytes"]),
        }
        error = maximum_correctness_error(provider)
        if error is not None:
            errors.append(float(error))

    rows: list[str] = []
    component_rows: list[str] = []
    memory_rows: list[str] = []
    profile_ratios: dict[str, list[float]] = {}
    summary_ratios: list[float] = []
    resident_ratios: list[float] = []
    for profile_index, profile in enumerate(profile_order):
        baseline = {
            round_number: cells[(baseline_id, profile, round_number)]
            for round_number in range(1, 4)
            if (baseline_id, profile, round_number) in cells
        }
        candidate = {
            round_number: cells[(candidate_id, profile, round_number)]
            for round_number in range(1, 4)
            if (candidate_id, profile, round_number) in cells
        }
        if len(baseline) != 3 or len(candidate) != 3:
            return ""
        ratios = [
            candidate[round_number]["total"] /
            baseline[round_number]["total"]
            for round_number in range(1, 4)
        ]
        profile_ratios[profile] = ratios
        baseline_total = statistics.median(
            baseline[round_number]["total"] for round_number in range(1, 4)
        )
        candidate_total = statistics.median(
            candidate[round_number]["total"] for round_number in range(1, 4)
        )
        ratio = candidate_total / baseline_total
        summary_ratios.append(ratio)
        lower, upper = paired_round_bootstrap(ratios, 181 + profile_index)
        baseline_resident = statistics.median(
            baseline[round_number]["resident"] for round_number in range(1, 4)
        )
        candidate_resident = statistics.median(
            candidate[round_number]["resident"] for round_number in range(1, 4)
        )
        resident_ratio = candidate_resident / baseline_resident
        resident_ratios.append(resident_ratio)
        rows.append(
            "<tr>"
            f'<th scope="row">{escaped(profile)}</th>'
            f'<td class="numeric">{format_ms(baseline_total)}</td>'
            f'<td class="numeric">{format_ms(candidate_total)}</td>'
            f'<td class="numeric">{ratio:.3f}×</td>'
            f'<td class="numeric">{lower:.3f}×–{upper:.3f}×</td>'
            f'<td class="numeric">{format_ms(statistics.median(baseline[round_number]["matrixSetup"] for round_number in range(1, 4)))} / '
            f'{format_ms(statistics.median(candidate[round_number]["matrixSetup"] for round_number in range(1, 4)))}</td>'
            f'<td class="numeric">{format_ms(statistics.median(baseline[round_number]["horizontalSetup"] for round_number in range(1, 4)))} / '
            f'{format_ms(statistics.median(candidate[round_number]["horizontalSetup"] for round_number in range(1, 4)))}</td>'
            "</tr>"
        )

        def paired_component(key: str) -> str:
            baseline_value = statistics.median(
                baseline[round_number][key] for round_number in range(1, 4)
            )
            candidate_value = statistics.median(
                candidate[round_number][key] for round_number in range(1, 4)
            )
            return f"{format_ms(baseline_value)} / {format_ms(candidate_value)}"

        component_rows.append(
            "<tr>"
            f'<th scope="row">{escaped(profile)}</th>'
            f'<td class="numeric">{paired_component("inverse")}</td>'
            f'<td class="numeric">{paired_component("horizontal")}</td>'
            f'<td class="numeric">{paired_component("movement")}</td>'
            f'<td class="numeric">{paired_component("forward")}</td>'
            "</tr>"
        )
        memory_rows.append(
            "<tr>"
            f'<th scope="row">{escaped(profile)}</th>'
            f'<td class="numeric">{format_bytes(int(baseline_resident))} / '
            f'{format_bytes(int(candidate_resident))}</td>'
            f'<td class="numeric">{resident_ratio:.3f}×</td>'
            f'<td class="numeric">{format_bytes(int(statistics.median(baseline[round_number]["scratch"] for round_number in range(1, 4))))} / '
            f'{format_bytes(int(statistics.median(candidate[round_number]["scratch"] for round_number in range(1, 4))))}</td>'
            f'<td class="numeric">{format_bytes(int(statistics.median(baseline[round_number]["estimated"] for round_number in range(1, 4))))} / '
            f'{format_bytes(int(statistics.median(candidate[round_number]["estimated"] for round_number in range(1, 4))))}</td>'
            f'<td class="numeric">{format_bytes(int(statistics.median(baseline[round_number]["observed"] for round_number in range(1, 4))))} / '
            f'{format_bytes(int(statistics.median(candidate[round_number]["observed"] for round_number in range(1, 4))))}</td>'
            "</tr>"
        )

    geometric_time = math.exp(statistics.fmean(
        math.log(value) for value in summary_ratios
    ))
    geometric_resident = math.exp(statistics.fmean(
        math.log(value) for value in resident_ratios
    ))
    lower, upper = spectral_pipeline_bootstrap(profile_ratios)
    maximum_ratio = max(summary_ratios)
    maximum_error = max(errors) if errors else math.inf
    adoption_passed = bool(
        geometric_time <= 0.90 and maximum_ratio <= 1.03
        and upper < 1.0 and geometric_resident <= 0.80
        and maximum_error <= 1.0e-12
    )
    conclusion = (
        "The fixed FFTW++ composition passes the M4 adoption-statistics gate; cross-Mac replication and a bounded model prototype remain required."
        if adoption_passed else
        "The campaign is reference-quality, but the fixed FFTW++ composition does not meet the M4 adoption threshold."
    )
    return f"""
      <h2>Four-workload composed M4 reference campaign</h2>
      <p>Three isolated process rounds rotate both fixed candidates and all four production workloads. Each process uses three warmups and 21 samples after the same-commit allocator test. The exact boundary remains 15 ready retained and vertically truncated modal inputs through vertical reconstruction, 129–513 streamed physical levels and four horizontal advective expressions, to four ready modal outputs.</p>
      <p>FFTW++ is {geometric_time:.3f}× explicit geometrically, with a stratified paired-bootstrap 95% interval of {lower:.3f}×–{upper:.3f}× and a worst workload of {maximum_ratio:.3f}×. Algorithm-resident storage is {geometric_resident:.3f}× geometrically and maximum mode-keyed error is {maximum_error:.3e}. <strong>{escaped(conclusion)}</strong></p>
      <div class="table-scroll"><table class="experiment-evidence-table">
        <caption>Authoritative composed medians and paired inference. Setup columns are explicit / FFTW++ and remain outside the timed boundary.</caption>
        <thead><tr><th scope="col">Profile</th><th scope="col">Explicit total (ms)</th><th scope="col">FFTW++ total (ms)</th><th scope="col">Ratio</th><th scope="col">Paired 95% interval</th><th scope="col">Matrix setup</th><th scope="col">Horizontal planning</th></tr></thead>
        <tbody>{"".join(rows)}</tbody>
      </table></div>
      <div class="table-scroll"><table class="experiment-evidence-table">
        <caption>Component medians are explicit / FFTW++ in milliseconds. Horizontal includes movement; movement is repeated only as a diagnostic and is not added to the authoritative total.</caption>
        <thead><tr><th scope="col">Profile</th><th scope="col">Inverse vertical</th><th scope="col">Horizontal + movement</th><th scope="col">Movement alone</th><th scope="col">Forward vertical</th></tr></thead>
        <tbody>{"".join(component_rows)}</tbody>
      </table></div>
      <div class="table-scroll"><table class="experiment-evidence-table">
        <caption>Memory is explicit / FFTW++ where paired. Algorithm resident is the decision measure; scratch, estimated live peak and process high water remain separate.</caption>
        <thead><tr><th scope="col">Profile</th><th scope="col">Algorithm resident</th><th scope="col">Ratio</th><th scope="col">Scratch</th><th scope="col">Estimated peak</th><th scope="col">Observed high water</th></tr></thead>
        <tbody>{"".join(memory_rows)}</tbody>
      </table></div>
      <p class="method-note">No size-dependent dispatch is permitted. Primitive vertical GEMM and horizontal evidence remain independently published in issues #8 and #17. This synthetic composition still excludes phase evolution, coefficient accumulation, remaining nonlinear-flux bookkeeping, the complete nonlinear flux, and time integration.</p>
    """


def vertically_batched_advection_synthesis(bundles: list[PublishedBundle]) -> str:
    reference_section = vertically_batched_advection_reference_synthesis(bundles)
    increment_id = "vertically-batched-advection-first-composition-v1"
    candidate_bundles: dict[str, PublishedBundle] = {}
    for bundle in sorted(bundles, key=lambda item: item.result["run"]["id"]):
        if bundle.publication.get("incrementId") != increment_id:
            continue
        candidate_id = bundle.publication.get("campaignCandidateId")
        if candidate_id in ("explicit-parallel", "fftwpp-parallel"):
            candidate_bundles[candidate_id] = bundle
    if not all(candidate in candidate_bundles for candidate in ("explicit-parallel", "fftwpp-parallel")):
        return reference_section

    candidates = (
        ("explicit-parallel", "Explicit FFTW"),
        ("fftwpp-parallel", "FFTW++"),
    )
    evidence: dict[str, dict] = {}
    timing_rows: list[str] = []
    memory_rows: list[str] = []
    for candidate_id, label in candidates:
        bundle = candidate_bundles[candidate_id]
        provider = bundle.result["providers"][0]
        inverse_vertical = stage_timing(
            provider, "primitive", "raw inverse vertical GEMM (15 fields)", "inverse",
        )
        horizontal = stage_timing(
            provider, "component",
            "vertically batched horizontal advection including level movement", "horizontal",
        )
        movement = stage_timing(
            provider, "adapter-component",
            "all-level split/field-major packing and projected-output scatter", "horizontal",
        )
        forward_vertical = stage_timing(
            provider, "primitive", "raw forward vertical GEMM (4 fields)", "forward",
        )
        total = stage_timing(
            provider, "uninstrumented-total",
            "vertically batched WVM-derived advection pipeline", "forward",
        )
        if any(item is None for item in (
            inverse_vertical, horizontal, movement, forward_vertical, total,
        )):
            return ""
        memory = provider["memory"]
        error = maximum_correctness_error(provider)
        evidence[candidate_id] = {
            "total": total["medianSeconds"],
            "resident": int(memory["algorithmResidentBytes"]),
            "error": error,
        }
        run_id = bundle.result["run"]["id"]
        timing_rows.append(
            "<tr>"
            f'<td><a href="../../runs/{quote(run_id)}/index.html">{escaped(label)}</a></td>'
            f'<td class="numeric">{format_ms(inverse_vertical["medianSeconds"])}</td>'
            f'<td class="numeric">{format_ms(horizontal["medianSeconds"])}</td>'
            f'<td class="numeric">{format_ms(movement["medianSeconds"])}</td>'
            f'<td class="numeric">{format_ms(forward_vertical["medianSeconds"])}</td>'
            f'<td class="numeric">{format_ms(total["medianSeconds"])}</td>'
            f'<td class="numeric">{format_error(error)}</td>'
            "</tr>"
        )
        memory_rows.append(
            "<tr>"
            f"<td>{escaped(label)}</td>"
            f'<td class="numeric">{format_bytes(int(memory["algorithmResidentBytes"]))}</td>'
            f'<td class="numeric">{format_bytes(int(memory["scratchBytes"]))}</td>'
            f'<td class="numeric">{format_bytes(int(memory["benchmarkHarnessBytes"]))}</td>'
            f'<td class="numeric">{format_bytes(int(memory["estimatedProcessPeakBytes"]))}</td>'
            f'<td class="numeric">{format_bytes(int(memory["observedProcessHighWaterBytes"]))}</td>'
            "</tr>"
        )

    explicit = evidence["explicit-parallel"]
    fftwpp = evidence["fftwpp-parallel"]
    time_ratio = fftwpp["total"] / explicit["total"]
    memory_ratio = fftwpp["resident"] / explicit["resident"]
    maximum_error = max(
        error for error in (explicit["error"], fftwpp["error"]) if error is not None
    )
    continuation_passed = bool(
        maximum_error <= 1.0e-12
        and (time_ratio <= 0.98 or (memory_ratio <= 0.80 and time_ratio <= 1.05))
    )
    continuation = (
        "The preliminary continuation gate passes, so a multi-workload reference campaign is warranted."
        if continuation_passed else
        "The preliminary continuation gate does not pass."
    )
    return f"""
      <h2>First vertically batched composition</h2>
      <p>This increment composes the fixed issue #17 horizontal finalists with the same directional split K²-grouped vertical reconstruction and projection. Its exact boundary is 15 ready retained and vertically truncated modal inputs through 129 streamed physical levels to four ready modal outputs at 256²/N<sub>z</sub>=129. Phase evolution, coefficient accumulation, the remaining nonlinear-flux bookkeeping, and the complete nonlinear flux remain excluded.</p>
      <p>FFTW++ is {time_ratio:.3f}× explicit in the authoritative composed total, while its counted algorithm-resident storage is {memory_ratio:.3f}×. Maximum mode-keyed error is {maximum_error:.3e}. <strong>{escaped(continuation)}</strong> This one-process, three-sample result is preliminary and does not meet or test the separate 0.9000 multi-workload adoption threshold.</p>
      <div class="table-scroll"><table class="experiment-evidence-table">
        <caption>Component medians and the authoritative uninstrumented total, in milliseconds. The horizontal component includes level movement; movement is also timed alone as a diagnostic and must not be added to it again.</caption>
        <thead><tr><th scope="col">Candidate</th><th scope="col">Inverse vertical</th><th scope="col">Horizontal + movement</th><th scope="col">Movement alone</th><th scope="col">Forward vertical</th><th scope="col">Total</th><th scope="col">Max error</th></tr></thead>
        <tbody>{"".join(timing_rows)}</tbody>
      </table></div>
      <div class="table-scroll"><table class="experiment-evidence-table">
        <caption>Memory ledgers distinguish algorithm state from benchmark-only oracle and fixture storage.</caption>
        <thead><tr><th scope="col">Candidate</th><th scope="col">Algorithm resident</th><th scope="col">Scratch</th><th scope="col">Harness only</th><th scope="col">Estimated peak</th><th scope="col">Observed high water</th></tr></thead>
        <tbody>{"".join(memory_rows)}</tbody>
      </table></div>
      <p class="method-note">Both candidates run in isolated processes with one warmup and three samples, outer-dynamic-12 vertical scheduling, and zero warmed steady-state application allocations. Approximately 169–171 ms is movement alone, so the isolated horizontal speed and memory advantage is largely diluted by level packing/scatter and vertical storage. Observed process high water includes the independent correctness oracle that ran earlier in each process; estimated peak is the fairer live benchmark ledger. Component medians need not sum to the separately measured total.</p>
      {reference_section}
    """


def experiment_evidence_table(experiment: dict, bundles: list[PublishedBundle]) -> str:
    if experiment["id"] == "issue-009-combined-spectral-pipeline":
        return spectral_pipeline_evidence_table(bundles)
    if experiment["id"] == "issue-016-streaming-pruned-compact-split":
        return spectral_pipeline_evidence_table(bundles)
    if experiment["id"] == "issue-004-fftw-strategy-sweep":
        return fftw_strategy_evidence_table(bundles)
    if experiment["id"] == "issue-006-vdsp-batching-scheduling":
        return vdsp_batch_evidence_table(bundles)
    if experiment["id"] == "issue-008-vertical-projection-gemm":
        return vertical_gemm_evidence_table(bundles)
    if experiment["id"] == "issue-013-ordering-packing-crossover":
        return ordering_packing_evidence_table(bundles)
    if experiment["id"] == "issue-012-pruned-horizontal-transforms":
        return pruned_horizontal_evidence_table(bundles)
    if experiment["id"] == "issue-007-retained-horizontal-algorithms":
        return retained_horizontal_evidence_table(bundles)
    provider_id = EXPERIMENT_PROVIDER_IDS.get(experiment["id"])
    if provider_id is None or not bundles:
        return ""
    rows: list[str] = []
    for bundle in bundles:
        result = bundle.result
        provider = next((item for item in result["providers"] if item["id"] == provider_id), None)
        if provider is None:
            continue
        workload = result["workload"]
        run = result["run"]
        publication = bundle.publication
        raw_forward = timing(provider, "primitive", "forward")
        raw_inverse = timing(provider, "primitive", "inverse")
        adapter_forward = timing(provider, "adapter-total", "forward")
        adapter_inverse = timing(provider, "adapter-total", "inverse")
        retained_forward = timing(provider, "uninstrumented-total", "forward")
        retained_inverse = timing(provider, "uninstrumented-total", "inverse")

        def pair(forward: dict | None, inverse: dict | None) -> str:
            if forward is None or inverse is None:
                return '<span class="muted">not measured</span>'
            return f'{format_ms(forward["medianSeconds"])} / {format_ms(inverse["medianSeconds"])}'

        explicit_memory = format_bytes(provider["memory"]["persistentBytes"])
        scratch_memory = format_bytes(provider["memory"]["scratchBytes"])
        run_id = run["id"]
        rows.append(
            "<tr>"
            f'<td><a href="../../runs/{quote(run_id)}/index.html">{escaped(run_id)}</a><br>'
            f'<span class="muted">{run["samples"]} samples · {publication_badge(publication["status"])}</span></td>'
            f'<td class="numeric">{workload["Nx"]} × {workload["Ny"]}<br>N<sub>z</sub>={workload["Nz"]}, fields={workload["fields"]}</td>'
            f'<td>{escaped(provider["algorithmId"])}</td>'
            f'<td class="numeric">{provider["workers"]}</td>'
            f'<td class="numeric">{pair(raw_forward, raw_inverse)}</td>'
            f'<td class="numeric">{pair(adapter_forward, adapter_inverse)}</td>'
            f'<td class="numeric">{pair(retained_forward, retained_inverse)}</td>'
            f'<td class="numeric">{explicit_memory}<br><span class="muted">scratch {scratch_memory}</span></td>'
            f'<td class="numeric">{format_error(maximum_correctness_error(provider))}</td>'
            "</tr>"
        )
    if not rows:
        return ""
    return (
        '<div class="table-scroll"><table class="experiment-evidence-table">'
        '<caption>Medians in milliseconds are forward / inverse. Primitive FFT excludes packing and conversion; adapter includes the WVM-compatible mapping; retained total also includes horizontal selection or embedding. Memory is explicit persistent provider storage plus separately reported scratch.</caption>'
        '<thead><tr><th scope="col">Run</th><th scope="col">Workload</th><th scope="col">Algorithm</th>'
        '<th scope="col">Workers</th><th scope="col">Primitive FFT</th><th scope="col">Adapter</th>'
        '<th scope="col">Retained total</th><th scope="col">Explicit memory</th><th scope="col">Max error</th></tr></thead>'
        f'<tbody>{"".join(rows)}</tbody></table></div>'
    )


def build_experiment_page(experiment: dict, bundles: list[PublishedBundle]) -> str:
    experiment_id = experiment["id"]
    related = [bundle for bundle in bundles if experiment_id in bundle.publication["experiments"]]
    prior_links = [
        f'<a href="../../experiments/{quote(prior)}/index.html">{escaped(prior)}</a>'
        for prior in experiment["priorExperiments"]
    ]
    downstream_links = [
        f'<a href="{REPOSITORY_URL}/issues/{issue}">#{issue}</a>'
        for issue in experiment["downstreamIssues"]
    ]
    reference_count = sum(bundle.publication["status"] == "reference" for bundle in related)
    evidence_statement = (
        f"{reference_count} reference run(s) currently contribute to comparisons."
        if reference_count
        else "No reference run has been published. Planned, preliminary, negative, and capability evidence remains visible below."
    )
    evidence_table = experiment_evidence_table(experiment, related)
    if experiment_id == "issue-009-combined-spectral-pipeline":
        synthesis = spectral_pipeline_synthesis(related)
    elif experiment_id == "issue-016-streaming-pruned-compact-split":
        synthesis = streaming_pruned_pipeline_synthesis(related)
    elif experiment_id == "issue-017-implicit-hybrid-dealiased-convolution":
        synthesis = dealiased_convolution_synthesis(related)
    elif experiment_id == "issue-018-vertically-batched-advection-pipeline":
        synthesis = vertically_batched_advection_synthesis(related)
    elif experiment_id == "issue-004-fftw-strategy-sweep":
        synthesis = fftw_strategy_synthesis(related)
    elif experiment_id == "issue-006-vdsp-batching-scheduling":
        synthesis = vdsp_batch_synthesis(related)
    elif experiment_id == "issue-008-vertical-projection-gemm":
        synthesis = vertical_gemm_synthesis(related)
    elif experiment_id == "issue-013-ordering-packing-crossover":
        synthesis = ordering_packing_synthesis(related)
    elif experiment_id == "issue-012-pruned-horizontal-transforms":
        synthesis = pruned_horizontal_synthesis(related)
    elif experiment_id == "issue-007-retained-horizontal-algorithms":
        synthesis = retained_horizontal_synthesis(related)
    else:
        synthesis = ""
    content = f"""
    <section class="hero compact">
      <p class="eyebrow">Experiment · issue #{experiment['issue']} · {escaped(experiment['phase'])}</p>
      <h1>{escaped(experiment['title'])}</h1>
      <p class="lede">{escaped(experiment['question'])}</p>
      <a class="button secondary" href="{REPOSITORY_URL}/issues/{experiment['issue']}">Open tracker issue</a>
    </section>
    <section class="section" aria-labelledby="position-heading">
      <p class="eyebrow">Position in the evidence chain</p><h2 id="position-heading">Baseline and consumers</h2>
      <p>{escaped(experiment['baseline'])}</p>
      <dl class="detail-list evidence-definition">
        <div><dt>Prior experiments</dt><dd>{', '.join(prior_links) if prior_links else 'Foundational experiment; no prior benchmark page.'}</dd></div>
        <div><dt>Downstream issues</dt><dd>{', '.join(downstream_links) if downstream_links else 'Non-blocking follow-up; no current downstream issue.'}</dd></div>
      </dl>
    </section>
    <section class="section" aria-labelledby="definition-heading">
      <p class="eyebrow">Benchmark definition</p><h2 id="definition-heading">What this experiment measures</h2>
      {definition_list(experiment['definition'])}
    </section>
    <section class="section" aria-labelledby="correctness-heading">
      <p class="eyebrow">Correctness and interpretation</p><h2 id="correctness-heading">What the result can establish</h2>
      {definition_list(experiment['correctness'])}
    </section>
    <section class="section" aria-labelledby="publication-heading">
      <p class="eyebrow">Publication deliverable</p><h2 id="publication-heading">Permanent evidence</h2>
      <p>Stable experiment ID: <code>{escaped(experiment_id)}</code>. Required result tables: {escaped(', '.join(experiment['requiredTables']))}.</p>
      <p>{escaped(evidence_statement)}</p>
      {synthesis}
      {evidence_table}
      <h3>Immutable run archive</h3>
      {archive(related, '../../')}
    </section>
    """
    return shell(experiment["title"], content, "../../")


def build_experiment_index(catalog: dict) -> str:
    content = f"""
    <section class="hero compact">
      <p class="eyebrow">Issue-level publication</p>
      <h1>Experiments</h1>
      <p class="lede">Each page states one performance question, its baseline, controlled and changed variables, timed boundary, exclusions, correctness oracle, and role in later decisions.</p>
    </section>
    <section class="section" aria-labelledby="all-experiments-heading">
      <h2 id="all-experiments-heading">Evidence chain</h2>
      {experiment_cards(catalog, '../')}
    </section>
    """
    return shell("Experiments", content, "../")


def inline_markdown(value: str) -> str:
    rendered = escaped(value)
    rendered = re.sub(r"`([^`]+)`", r"<code>\1</code>", rendered)
    rendered = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r'<a href="\2">\1</a>', rendered)
    rendered = re.sub(r"\$([^$]+)\$", r'<code class="math-inline">\1</code>', rendered)
    return rendered


def render_markdown(source: str) -> str:
    lines = source.splitlines()
    blocks: list[str] = []
    index = 0
    while index < len(lines):
        line = lines[index]
        if not line.strip():
            index += 1
            continue
        if line.startswith("$$"):
            math_lines = [line[2:]] if line.strip() != "$$" else []
            index += 1
            while index < len(lines) and not lines[index].startswith("$$"):
                math_lines.append(lines[index])
                index += 1
            if index < len(lines):
                ending = lines[index][2:]
                if ending:
                    math_lines.append(ending)
                index += 1
            blocks.append(f'<pre class="math-block"><code>{escaped(chr(10).join(math_lines).strip())}</code></pre>')
            continue
        heading = re.match(r"^(#{1,4})\s+(.+)$", line)
        if heading:
            level = len(heading.group(1))
            blocks.append(f"<h{level}>{inline_markdown(heading.group(2))}</h{level}>")
            index += 1
            continue
        if line.startswith("- "):
            items = []
            while index < len(lines) and lines[index].startswith("- "):
                items.append(f"<li>{inline_markdown(lines[index][2:])}</li>")
                index += 1
            blocks.append(f'<ul>{"".join(items)}</ul>')
            continue
        if re.match(r"^\d+\. ", line):
            items = []
            while index < len(lines) and re.match(r"^\d+\. ", lines[index]):
                items.append(f"<li>{inline_markdown(re.sub(r'^\d+\. ', '', lines[index]))}</li>")
                index += 1
            blocks.append(f'<ol>{"".join(items)}</ol>')
            continue
        paragraph = [line.strip()]
        index += 1
        while index < len(lines) and lines[index].strip() and not re.match(r"^(#{1,4})\s+|- |\d+\. |\$\$", lines[index]):
            paragraph.append(lines[index].strip())
            index += 1
        blocks.append(f'<p>{inline_markdown(" ".join(paragraph))}</p>')
    return "\n".join(blocks)


def build_methods_page(methods_source: Path) -> str:
    rendered = render_markdown(methods_source.read_text(encoding="utf-8"))
    content = f"""
    <section class="hero compact">
      <p class="eyebrow">Versioned methodology</p>
      <h1>Operators and representations</h1>
      <p class="lede">The mathematical result is independent of physical order, packing, placement, or fusion. Those are measured algorithm choices.</p>
    </section>
    <article class="section prose">{rendered}</article>
    """
    return shell("Operators and representations", content, "../../")


def build_decision_page(catalog: dict, bundles: list[PublishedBundle]) -> str:
    reference_bundles = [bundle for bundle in bundles if bundle.publication["status"] == "reference"]
    experiment_rows = []
    for experiment in sorted(catalog["experiments"], key=lambda item: item["issue"]):
        related = [bundle for bundle in bundles if experiment["id"] in bundle.publication["experiments"]]
        status_counts = {status: 0 for status in ("reference", "preliminary", "superseded", "withdrawn")}
        for bundle in related:
            status_counts[bundle.publication["status"]] += 1
        counts = ", ".join(f"{count} {status}" for status, count in status_counts.items() if count) or "no runs"
        experiment_rows.append(
            "<tr>"
            f'<th scope="row"><a href="../../experiments/{quote(experiment["id"])}/index.html">#{experiment["issue"]} {escaped(experiment["title"])}</a></th>'
            f"<td>{escaped(experiment['phase'])}</td><td>{escaped(counts)}</td>"
            "</tr>"
        )
    bridge_cells = native_bridge_cells(bundles)
    bridge_complete = len(bridge_cells) == 3 * 3 * 3
    if bridge_complete:
        tile_to_wvm = native_bridge_pair_summary(
            bridge_cells, NATIVE_BRIDGE_CANDIDATES[2], NATIVE_BRIDGE_CANDIDATES[0],
        )
        tile_to_fused = native_bridge_pair_summary(
            bridge_cells, NATIVE_BRIDGE_CANDIDATES[2], NATIVE_BRIDGE_CANDIDATES[1],
        )
        maximum_error = max(item["error"] for item in bridge_cells.values())
        heading = "M4 deployment decision recorded"
        readiness = (
            "The same reference campaign supports two boundary-specific selections. "
            "Cross-Mac replication and full-model validation remain open before a "
            "general-Mac default recommendation."
        )
        decision_sections = f"""
        <section class="section" aria-labelledby="selection-heading">
          <p class="eyebrow">M4 selections</p><h2 id="selection-heading">Choose by ownership boundary</h2>
          <div class="table-scroll"><table>
            <thead><tr><th scope="col">Deployment contract</th><th scope="col">Selected tuple</th><th scope="col">Interpretation</th></tr></thead>
            <tbody>
              <tr><th scope="row">Persistent compiled spectral engine</th><td><code>{NATIVE_BRIDGE_CANDIDATES[2]}</code><br>FFTW partial row/selected-column transform · compact radial split · grouped split dgemm · outer-dynamic-16</td><td>Fastest unrestricted complete synthetic operator and lowest algorithm-resident memory of the three finalists.</td></tr>
              <tr><th scope="row">MATLAB + compiled core with WVM-native arrays</th><td><code>{NATIVE_BRIDGE_CANDIDATES[0]}</code><br>FFTW full WVM order · frequency-major interleaved · direct per-mode zgemm · outer-dynamic-16</td><td>Best measured eligible algorithm when MATLAB-visible WVM frequency-major interleaved arrays must remain authoritative; not the fastest unrestricted pipeline.</td></tr>
            </tbody>
          </table></div>
          <p>Streaming tile 16 is {tile_to_wvm["geometricTime"]:.3f}× WVM direct geometrically in authoritative time (95% paired interval {tile_to_wvm["interval"][0]:.3f}×–{tile_to_wvm["interval"][1]:.3f}×; worst profile {tile_to_wvm["maximumTime"]:.3f}×). It is {tile_to_fused["geometricTime"]:.3f}× fused split ({tile_to_fused["interval"][0]:.3f}×–{tile_to_fused["interval"][1]:.3f}×; worst profile {tile_to_fused["maximumTime"]:.3f}×). Algorithm-resident memory is {tile_to_wvm["geometricResident"]:.3f}× WVM direct and {tile_to_fused["geometricResident"]:.3f}× fused split. Maximum correctness error is {maximum_error:.3e}.</p>
          <p class="method-note">These are single-policy selections across 256²/Nz=129/F4, 512²/Nz=257/F4, and 1024²/Nz=129/F4. No size-dependent dispatch was allowed. See the <a href="../../experiments/issue-009-combined-spectral-pipeline/index.html">issue #9 experiment page</a> for totals, components, placement, setup, memory, and immutable runs.</p>
        </section>
        <section class="section" aria-labelledby="layers-heading">
          <p class="eyebrow">Layered conclusion</p><h2 id="layers-heading">What each benchmark established</h2>
          <ul>
            <li><strong>Primitive horizontal transform:</strong> FFTW remains the Float64 full-transform provider baseline; vDSP's negative Float64 evidence remains visible rather than being generalized to Float32.</li>
            <li><strong>Primitive vertical projection:</strong> grouped split real GEMMs with outer-dynamic-16 are the persistent compact-layout kernel; WVM direct zgemm remains the native interleaved control.</li>
            <li><strong>Movement and representation:</strong> compact tile-16 staging preserves partial-column pruning without materializing the batch-sized full spectrum. Optional packing, not a mathematical gather requirement, is part of the selected algorithm.</li>
            <li><strong>Complete pipeline:</strong> the uninstrumented ready-real-to-reconstructed-real total is authoritative. Component medians diagnose the result but are not summed.</li>
          </ul>
          <p>The benchmark explicitly excludes the nonlinear flux calculation, time integration, state management, and I/O. The recorded result is therefore a spectral-infrastructure decision for this M4, not a complete WVM performance claim.</p>
        </section>
        """
    else:
        heading = "Adoption decision not yet ready"
        readiness = (
            "Reference evidence exists, but the decision remains open until the issue #11 adoption gates are evaluated."
            if reference_bundles
            else "No reference runs exist yet, so no adoption statistics or provider recommendation can be produced."
        )
        decision_sections = ""
    content = f"""
    <section class="hero compact">
      <p class="eyebrow">Decision record · v1</p>
      <h1>{escaped(heading)}</h1>
      <p class="lede">{escaped(readiness)}</p>
      <a class="button secondary" href="{REPOSITORY_URL}/issues/11">Open decision issue #11</a>
    </section>
    {decision_sections}
    <section class="section" aria-labelledby="policy-heading">
      <p class="eyebrow">Evidence policy</p><h2 id="policy-heading">What enters the decision</h2>
      <p>Only reference runs contribute to adoption statistics. Preliminary, superseded, withdrawn, negative, and unsupported evidence stays linked from its experiment page with its explanation.</p>
      <p>The report separates primitive FFT, primitive GEMM, movement/adapters, setup/memory, and uninstrumented pipeline conclusions.</p>
    </section>
    <section class="section" aria-labelledby="evidence-heading">
      <p class="eyebrow">Current archive</p><h2 id="evidence-heading">Experiment status</h2>
      <div class="table-scroll"><table>
        <thead><tr><th scope="col">Experiment</th><th scope="col">Phase</th><th scope="col">Published evidence</th></tr></thead>
        <tbody>{''.join(experiment_rows)}</tbody>
      </table></div>
    </section>
    """
    return shell("v1 decision", content, "../../")


def clean_output(output_dir: Path, repository_root: Path, results_dir: Path) -> None:
    resolved = output_dir.resolve()
    forbidden = {Path("/").resolve(), Path.home().resolve(), repository_root.resolve(), results_dir.resolve()}
    if resolved in forbidden or resolved in repository_root.resolve().parents or results_dir.resolve() in resolved.parents:
        raise ValueError(f"Refusing to replace unsafe output directory: {resolved}")
    if resolved.exists():
        shutil.rmtree(resolved)
    resolved.mkdir(parents=True)


def write_page(path: Path, contents: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(contents, encoding="utf-8")


def build_site(results_dir: Path, output_dir: Path) -> None:
    repository_root = Path(__file__).resolve().parents[1]
    results_dir = results_dir.resolve()
    output_dir = output_dir.resolve()
    catalog, bundles = load_and_validate(results_dir)
    if not bundles:
        raise ValueError(f"No published runs found in {results_dir}")
    clean_output(output_dir, repository_root, results_dir)

    (output_dir / "assets").mkdir()
    (output_dir / "artifacts").mkdir()
    (output_dir / "results").mkdir()
    (output_dir / "schema").mkdir()
    shutil.copyfile(repository_root / "site" / "style.css", output_dir / "assets" / "style.css")
    shutil.copyfile(repository_root / "site" / "favicon.svg", output_dir / "assets" / "favicon.svg")
    for schema_path in sorted((repository_root / "schema").glob("*.json")):
        shutil.copyfile(schema_path, output_dir / "schema" / schema_path.name)
    shutil.copyfile(results_dir / "catalog.json", output_dir / "catalog.json")
    (output_dir / ".nojekyll").write_text("", encoding="utf-8")
    write_page(output_dir / "index.html", build_index(catalog, bundles))

    for bundle in bundles:
        run_id = bundle.publication["id"]
        artifact_dir = output_dir / "artifacts" / run_id
        artifact_dir.mkdir(parents=True)
        shutil.copyfile(bundle.result_path, artifact_dir / "result.json")
        shutil.copyfile(bundle.samples_path, artifact_dir / "samples.csv")
        write_page(output_dir / "runs" / run_id / "index.html", build_run_page(bundle, "../../"))
        if bundle.publication["grandfathered"]:
            shutil.copyfile(bundle.result_path, output_dir / "results" / bundle.result_path.name)
            shutil.copyfile(bundle.samples_path, output_dir / "results" / bundle.samples_path.name)
        for legacy_url in bundle.publication["legacyUrls"]:
            legacy_path = output_dir / legacy_url.removeprefix("/")
            if legacy_path.suffix != ".html":
                raise ValueError(f"Legacy page URL must end in .html: {legacy_url}")
            write_page(legacy_path, build_run_page(bundle, "../"))

    write_page(output_dir / "experiments" / "index.html", build_experiment_index(catalog))
    for experiment in catalog["experiments"]:
        write_page(
            output_dir / "experiments" / experiment["id"] / "index.html",
            build_experiment_page(experiment, bundles),
        )
    write_page(
        output_dir / "methods" / "operators-and-representations" / "index.html",
        build_methods_page(repository_root / "docs" / "benchmark-contract.md"),
    )
    write_page(output_dir / "decisions" / "v1" / "index.html", build_decision_page(catalog, bundles))


def main() -> int:
    repository_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=repository_root / "results" / "published")
    parser.add_argument("--output", type=Path, default=repository_root / "_site")
    arguments = parser.parse_args()
    build_site(arguments.results, arguments.output)
    print(f"Built {arguments.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
