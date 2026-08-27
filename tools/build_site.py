#!/usr/bin/env python3
"""Build the static GitHub Pages evidence archive from published bundles."""

from __future__ import annotations

import argparse
import html
import re
import shutil
from pathlib import Path
from urllib.parse import quote

from publication import PublishedBundle, load_and_validate


REPOSITORY_URL = "https://github.com/JeffreyEarly/spectral-kernel-benchmarks"
SUMMARY_SCOPES = (
    ("Raw FFT", "primitive"),
    ("WVM-compatible adapter", "adapter-total"),
    ("Retained horizontal operator", "uninstrumented-total"),
)


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
    names = {"fftw": "FFTW", "accelerate-vdsp": "Accelerate/vDSP"}
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


def maximum_correctness_error(provider: dict) -> float | None:
    values = [
        item["maximumRelativeError"]
        for item in provider["correctness"]
        if item["maximumRelativeError"] is not None
    ]
    return max(values) if values else None


def summary_timing_table(result: dict) -> str:
    providers = result["providers"]
    header = "".join(f'<th scope="col">{escaped(provider_name(provider))}</th>' for provider in providers)
    rows: list[str] = []
    for label, scope in SUMMARY_SCOPES:
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
