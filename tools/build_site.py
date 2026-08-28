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
    scopes = SUMMARY_SCOPES
    if any(
        item["scope"] == "primitive" and item["stage"] == "raw vertical GEMM"
        for provider in providers
        for item in provider["timings"]
    ):
        scopes = (("Raw vertical GEMM", "primitive"),)
    header = "".join(f'<th scope="col">{escaped(provider_name(provider))}</th>' for provider in providers)
    rows: list[str] = []
    for label, scope in scopes:
        for direction in ("forward", "inverse"):
            values = [timing(provider, scope, direction) for provider in providers]
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
            f'<li><span>{escaped(item["name"])}</span><strong>{format_error(item["maximumRelativeError"])}</strong><span class="status {"passed" if item["passed"] else "failed"}">{"passed" if item["passed"] else "failed"}</span></li>'
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
    commit = latest.result["environment"]["gitCommit"]
    return [
        bundle
        for bundle in same_status
        if bundle.result["environment"]["gitCommit"] == commit
    ]


def fftw_screen_classifications(bundles: list[PublishedBundle]) -> tuple[dict[str, str], list[PublishedBundle]]:
    cohort = fftw_screen_cohort(bundles)
    classifications: dict[str, str] = {}
    workload_groups: dict[tuple[int, int, int, int], list[tuple[PublishedBundle, tuple[float, float, float]]]] = {}
    for bundle in cohort:
        result = bundle.result
        provider = next((item for item in result["providers"] if item["id"] == "fftw"), None)
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
        workload_groups.setdefault(key, []).append((bundle, objectives))

    for records in workload_groups.values():
        for bundle, objectives in records:
            dominated = any(
                other is not bundle
                and all(left <= right for left, right in zip(other_objectives, objectives))
                and any(left < right for left, right in zip(other_objectives, objectives))
                for other, other_objectives in records
            )
            classifications[bundle.result["run"]["id"]] = "dominated" if dominated else "Pareto"
    return classifications, cohort


def fftw_strategy_evidence_table(bundles: list[PublishedBundle]) -> str:
    classifications, cohort = fftw_screen_classifications(bundles)
    cohort_ids = {bundle.result["run"]["id"] for bundle in cohort}
    rows: list[str] = []

    def sort_key(bundle: PublishedBundle) -> tuple[int, int, int, str, int, int]:
        result = bundle.result
        provider = next(item for item in result["providers"] if item["id"] == "fftw")
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
        provider = next((item for item in result["providers"] if item["id"] == "fftw"), None)
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
            + '<h3>Append-only interleaved strategy archive</h3>'
            + '<p>The earlier planning, alignment, wisdom, and scheduling screen remains visible in full. The interleaved partner from each newer paired run extends that archive without replacing the original cohort.</p>'
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
        provider = next((item for item in result["providers"] if item["id"] == "fftw"), None)
        if provider is not None:
            records.append((result, provider, classifications[result["run"]["id"]]))
    if not records:
        return ""

    rows: list[str] = []
    workload_keys = sorted(
        {
            (result["workload"]["Nx"], result["workload"]["Ny"], result["workload"]["Nz"], result["workload"]["fields"])
            for result, _, _ in records
        }
    )
    for key in workload_keys:
        workload_records = [
            record
            for record in records
            if (
                record[0]["workload"]["Nx"],
                record[0]["workload"]["Ny"],
                record[0]["workload"]["Nz"],
                record[0]["workload"]["fields"],
            ) == key
        ]
        pareto = sorted(
            (record for record in workload_records if record[2] == "Pareto"),
            key=lambda record: float(record[1]["setup"]["totalSeconds"]),
        )
        entries = []
        for _, provider, _ in pareto:
            forward = timing(provider, "primitive", "forward")
            inverse = timing(provider, "primitive", "inverse")
            if forward is None or inverse is None:
                continue
            entries.append(
                f'<strong>{escaped(fftw_candidate_name(provider))}</strong><br>'
                f'<span class="muted">{format_ms(forward["medianSeconds"])} / '
                f'{format_ms(inverse["medianSeconds"])} ms; setup {format_ms(provider["setup"]["totalSeconds"])} ms</span>'
            )
        rows.append(
            "<tr>"
            f'<th scope="row">{key[0]} × {key[1]}<br><span class="muted">N<sub>z</sub>={key[2]}, fields={key[3]}</span></th>'
            f'<td>{"<hr>".join(entries)}</td>'
            "</tr>"
        )

    counts = {name: list(classifications.values()).count(name) for name in set(classifications.values())}
    max_error = max(maximum_correctness_error(provider) or 0.0 for _, provider, _ in records)
    commit = cohort[0].result["environment"]["gitCommit"]
    status = cohort[0].publication["status"]
    paired = []
    for bundle in cohort:
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
    return f"""
      <h3>Reproducible Pareto screen</h3>
      <p>The current {escaped(status)} cohort is commit <code>{escaped(commit)}</code>. Within each workload, a candidate is Pareto when no other eligible candidate is no slower in raw forward FFT, raw inverse FFT, and total setup while being strictly better in at least one. Total setup includes allocation, planning, and wisdom generation/import. Memory is reported but is identical within each workload in this increment; nine-sample CVs are reported rather than used as a hard gate.</p>
      <p>{counts.get("Pareto", 0)} workload-candidates are Pareto, {counts.get("dominated", 0)} are dominated, and {counts.get("infeasible within planning budget", 0)} are marked infeasible within the configured per-plan feasibility budget. All {len(records)} runs passed the fixed-reference and round-trip checks; the cohort maximum relative error is {format_error(max_error)}.</p>
      <div class="table-scroll"><table class="experiment-evidence-table">
        <caption>Pareto candidates ordered by total setup. Times are raw forward / inverse medians followed by total setup; all are milliseconds.</caption>
        <thead><tr><th scope="col">Workload</th><th scope="col">Non-dominated candidates</th></tr></thead>
        <tbody>{"".join(rows)}</tbody>
      </table></div>
      {split_synthesis}
      <p class="method-note">This is a bounded strategy screen, not the final issue #4 Pareto set or a WVM adoption decision. {escaped(split_scope_note)} Budget-limited PATIENT and EXHAUSTIVE plans remain valid measured transforms, but this experiment cannot claim that their requested search completed.</p>
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
      <h3>Diagnostic conclusion</h3>
      <p>The advancement screen carries a new candidate forward only when its best worker count reduces both raw forward and inverse medians by at least 10% on both representative workloads, without moving work outside the primitive boundary. {escaped(advancement)}</p>
      <div class="table-scroll"><table class="experiment-evidence-table">
        <caption>Best median raw times in milliseconds are forward / inverse. Positive improvement means the best non-baseline candidate is faster than direct-persistent. The final column is best vDSP divided by matched best FFTW.</caption>
        <thead><tr><th scope="col">Workload</th><th scope="col">Direct-persistent best</th><th scope="col">Best alternative</th><th scope="col">Improvement</th><th scope="col">vDSP / FFTW</th></tr></thead>
        <tbody>{"".join(rows)}</tbody>
      </table></div>
      <p class="method-note">Direct-persistent therefore remains the only issue #6 candidate carried forward: 12 workers for the representative 256² batch and 16 workers for the representative 512² batch. GCD dispatch overhead is negligible relative to transform time but does not materially change throughput. In the separable implementation, the isolated strided-column phase dominates; its phase medians are diagnostic and non-additive. These preliminary results do not replace the full fields 1/3/4 workload matrix or reference-depth machine-state protocol.</p>
    """


def vertical_gemm_evidence_table(bundles: list[PublishedBundle]) -> str:
    rows: list[str] = []
    for bundle in sorted(
        bundles,
        key=lambda item: (
            item.result["workload"]["Nx"],
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
        rows.append(
            "<tr>"
            f'<td><a href="../../runs/{quote(run["id"])}/index.html">{escaped(run["id"])}</a><br>'
            f'<span class="muted">{publication_badge(bundle.publication["status"])}</span></td>'
            f'<td>{workload["Nx"]}²<br><span class="muted">N<sub>z</sub>={workload["Nz"]}, '
            f'N<sub>j</sub>={workload["Nj"]}, fields={workload["fields"]}, K={workload["Nkl"] * workload["fields"]}</span></td>'
            f'<td class="numeric">{complex_provider["workers"]}</td>'
            f'<td class="numeric">{timing_with_interval(complex_forward)} / {timing_with_interval(complex_inverse)}<br>'
            f'<span class="muted">CV {coefficient_of_variation(complex_forward)} / {coefficient_of_variation(complex_inverse)}</span></td>'
            f'<td class="numeric">{timing_with_interval(split_forward)} / {timing_with_interval(split_inverse)}<br>'
            f'<span class="muted">CV {coefficient_of_variation(split_forward)} / {coefficient_of_variation(split_inverse)}</span></td>'
            f'<td class="numeric">{forward_ratio:.3f}× / {inverse_ratio:.3f}×</td>'
            f'<td class="numeric">{format_bytes(complex_provider["memory"]["persistentBytes"])} / '
            f'{format_bytes(split_provider["memory"]["persistentBytes"])}</td>'
            f'<td class="numeric">{format_error(maximum_error)} / {format_error(maximum_l2)}</td>'
            "</tr>"
        )
    if not rows:
        return ""
    return f"""
      <div class="table-scroll"><table class="experiment-evidence-table">
        <caption>Primitive medians and deterministic percentile-bootstrap 95% intervals are forward / inverse in milliseconds. CV is the sample coefficient of variation. Split / complex below 1 favors the two-real-GEMM formulation. Persistent memory is complex / split. No packing or representation conversion is timed.</caption>
        <thead><tr><th scope="col">Run</th><th scope="col">Workload</th><th scope="col">VECLIB thread limit</th><th scope="col">Complex zgemm</th><th scope="col">Two split dgemm</th><th scope="col">Split / complex</th><th scope="col">Explicit memory</th><th scope="col">Max / L2 error</th></tr></thead>
        <tbody>{''.join(rows)}</tbody>
      </table></div>
    """


def vertical_gemm_synthesis(bundles: list[PublishedBundle]) -> str:
    records: list[tuple[dict, dict, dict]] = []
    for bundle in bundles:
        complex_provider = next(
            (item for item in bundle.result["providers"] if item["id"] == "accelerate-zgemm"), None
        )
        split_provider = next(
            (item for item in bundle.result["providers"] if item["id"] == "accelerate-split-dgemm"), None
        )
        if complex_provider is not None and split_provider is not None:
            records.append((bundle.result, complex_provider, split_provider))
    if not records:
        return ""

    rows: list[str] = []
    for profile in sorted({result["run"]["profile"] for result, _, _ in records}):
        candidates = [record for record in records if record[0]["run"]["profile"] == profile]
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
        rows.append(
            "<tr>"
            f'<th scope="row">{workload["Nx"]}²<br><span class="muted">N<sub>z</sub>={workload["Nz"]}, fields={workload["fields"]}</span></th>'
            f'<td class="numeric">{values[0]}</td><td class="numeric">{values[1]}</td>'
            "</tr>"
        )
    return f"""
      <h3>Bounded common-matrix screen</h3>
      <p>This first issue #8 increment tests only the shared deterministic DCT-II matrix family with fields=3. Each row below reports best complex / best split / split-to-complex ratio across the isolated thread-limit runs. It establishes primitive behavior, not the complete issue #8 winner.</p>
      <div class="table-scroll"><table class="experiment-evidence-table">
        <caption>Best observed primitive medians in milliseconds. Thread settings are process-isolated <code>VECLIB_MAXIMUM_THREADS</code> limits; they are not claims about the exact worker count selected internally by Accelerate.</caption>
        <thead><tr><th scope="col">Workload</th><th scope="col">Forward: complex / split / ratio</th><th scope="col">Inverse: complex / split / ratio</th></tr></thead>
        <tbody>{''.join(rows)}</tbody>
      </table></div>
      <p class="method-note">Inputs are already arranged as column-major vertical-contiguous matrices, both algorithms are out-of-place, and all buffers are persistent. Matrix expansion/transposition is setup-only. Packing and horizontal ordering are deliberately excluded for later issue #13 measurement. Grouped K² matrix families, fields 1/4, N<sub>z</sub>=257, blocking, and full reference-depth sampling remain open.</p>
    """


def experiment_evidence_table(experiment: dict, bundles: list[PublishedBundle]) -> str:
    if experiment["id"] == "issue-004-fftw-strategy-sweep":
        return fftw_strategy_evidence_table(bundles)
    if experiment["id"] == "issue-006-vdsp-batching-scheduling":
        return vdsp_batch_evidence_table(bundles)
    if experiment["id"] == "issue-008-vertical-projection-gemm":
        return vertical_gemm_evidence_table(bundles)
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
    if experiment_id == "issue-004-fftw-strategy-sweep":
        synthesis = fftw_strategy_synthesis(related)
    elif experiment_id == "issue-006-vdsp-batching-scheduling":
        synthesis = vdsp_batch_synthesis(related)
    elif experiment_id == "issue-008-vertical-projection-gemm":
        synthesis = vertical_gemm_synthesis(related)
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
    readiness = (
        "Reference evidence exists, but the decision remains open until the issue #11 adoption gates are evaluated."
        if reference_bundles
        else "No reference runs exist yet, so no adoption statistics or provider recommendation can be produced."
    )
    content = f"""
    <section class="hero compact">
      <p class="eyebrow">Decision record · v1</p>
      <h1>Adoption decision not yet ready</h1>
      <p class="lede">{escaped(readiness)}</p>
      <a class="button secondary" href="{REPOSITORY_URL}/issues/11">Open decision issue #11</a>
    </section>
    <section class="section" aria-labelledby="policy-heading">
      <p class="eyebrow">Evidence policy</p><h2 id="policy-heading">What enters the decision</h2>
      <p>Only reference runs contribute to adoption statistics. Preliminary, superseded, withdrawn, negative, and unsupported evidence stays linked from its experiment page with its explanation.</p>
      <p>The final report will separate primitive FFT, primitive GEMM, movement/adapters, setup/memory, and uninstrumented pipeline conclusions.</p>
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
