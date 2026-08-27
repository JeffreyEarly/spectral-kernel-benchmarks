#!/usr/bin/env python3
"""Build the static GitHub Pages dashboard from published benchmark bundles."""

from __future__ import annotations

import argparse
import html
import json
import shutil
from pathlib import Path
from urllib.parse import quote


REPOSITORY_URL = "https://github.com/JeffreyEarly/spectral-kernel-benchmarks"
SCHEMA_NAME = "spectral-kernel-benchmark-v1"
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


def validate_result(path: Path, result: dict) -> None:
    required = ("schema", "status", "run", "workload", "environment", "providers")
    missing = [key for key in required if key not in result]
    if missing:
        raise ValueError(f"{path}: missing required keys: {', '.join(missing)}")
    if result["schema"] != SCHEMA_NAME:
        raise ValueError(f"{path}: unsupported schema {result['schema']!r}")
    if not result["providers"]:
        raise ValueError(f"{path}: provider list is empty")
    for provider in result["providers"]:
        for key in ("id", "timings", "correctness", "setup", "planning"):
            if key not in provider:
                raise ValueError(f"{path}: provider is missing {key!r}")


def load_results(results_dir: Path) -> list[tuple[Path, dict]]:
    loaded: list[tuple[Path, dict]] = []
    json_paths = sorted(results_dir.glob("*.json"))
    json_stems = {path.stem for path in json_paths}
    orphaned_csv = [path.name for path in results_dir.glob("*.csv") if path.stem not in json_stems]
    if orphaned_csv:
        raise ValueError(f"Published CSV bundles lack matching JSON: {', '.join(sorted(orphaned_csv))}")
    for path in json_paths:
        if not path.with_suffix(".csv").is_file():
            raise ValueError(f"{path}: matching sample CSV is missing")
        with path.open(encoding="utf-8") as stream:
            result = json.load(stream)
        validate_result(path, result)
        loaded.append((path, result))
    if not loaded:
        raise ValueError(f"No published JSON bundles found in {results_dir}")
    return sorted(loaded, key=lambda item: item[1]["environment"]["timestampUtc"], reverse=True)


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
                suffix = ' <span class="best-label">fastest</span>' if class_name else ""
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


def result_summary_cards(result: dict) -> str:
    workload = result["workload"]
    environment = result["environment"]
    status = result["status"]
    status_class = "passed" if status == "passed" else "failed"
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
          <p>N<sub>z</sub>={workload['Nz']}, fields={workload['fields']}, N<sub>kl</sub>={workload['Nkl']}, N<sub>j</sub>={workload['Nj']}</p>
        </section>
        <section class="summary-card">
          <p class="eyebrow">Run quality</p>
          <p class="summary-value"><span class="status {status_class}">{escaped(status)}</span></p>
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
  <meta name="description" content="Apple Silicon FFT and spectral-kernel benchmark results">
  <title>{escaped(title)} · Spectral Kernel Benchmarks</title>
  <link rel="icon" href="{root_prefix}assets/favicon.svg" type="image/svg+xml">
  <link rel="stylesheet" href="{root_prefix}assets/style.css">
</head>
<body>
  <a class="skip-link" href="#content">Skip to results</a>
  <header class="site-header">
    <a class="wordmark" href="{root_prefix}index.html"><span class="mark" aria-hidden="true">∿</span> Spectral Kernel Benchmarks</a>
    <nav aria-label="Primary">
      <a href="{root_prefix}index.html#runs">Runs</a>
      <a href="{REPOSITORY_URL}">Repository</a>
    </nav>
  </header>
  <main id="content">{content}</main>
  <footer>
    <p>Generated from versioned benchmark bundles. Performance is never measured in CI.</p>
    <p><a href="{REPOSITORY_URL}">JeffreyEarly/spectral-kernel-benchmarks</a></p>
  </footer>
</body>
</html>
"""


def archive(result_items: list[tuple[Path, dict]], root_prefix: str = "") -> str:
    cards: list[str] = []
    for path, result in result_items:
        workload = result["workload"]
        environment = result["environment"]
        cards.append(f"""
          <li>
            <a class="run-card" href="{root_prefix}runs/{quote(path.stem)}.html">
              <span class="run-date">{escaped(display_timestamp(environment['timestampUtc']))}</span>
              <strong>{escaped(environment['cpuBrand'])}</strong>
              <span>{workload['Nx']} × {workload['Ny']}, N<sub>z</sub>={workload['Nz']}, fields={workload['fields']}</span>
              <span class="run-meta">{escaped(result['run']['profile'])} · {result['run']['samples']} samples · {escaped(result['status'])}</span>
            </a>
          </li>
        """)
    return f'<ul class="run-list">{"".join(cards)}</ul>'


def build_index(result_items: list[tuple[Path, dict]]) -> str:
    latest_path, latest = result_items[0]
    content = f"""
    <section class="hero">
      <p class="eyebrow">Apple Silicon · double precision · antialiased operators</p>
      <h1>Which spectral kernels are actually fastest?</h1>
      <p class="lede">A reproducible comparison of FFT providers, memory representations, data movement, and composed spectral operations for the Wave–Vortex Model workload.</p>
      <p class="notice"><strong>Current phase:</strong> first FFTW–vDSP vertical slice. These measurements establish a baseline; they are not yet a general provider recommendation.</p>
    </section>
    <section class="section" aria-labelledby="latest-heading">
      <div class="section-heading">
        <div><p class="eyebrow">Latest published run</p><h2 id="latest-heading">{escaped(display_timestamp(latest['environment']['timestampUtc']))}</h2></div>
        <a class="button secondary" href="runs/{quote(latest_path.stem)}.html">Full run details</a>
      </div>
      {result_summary_cards(latest)}
      {summary_timing_table(latest)}
      <p class="method-note">Raw calls exclude packing and conversion. Adapter timings include the WVM-compatible full-spectrum boundary. Retained-operator totals also include horizontal selection or embedding.</p>
    </section>
    <section class="section" id="runs" aria-labelledby="runs-heading">
      <p class="eyebrow">Archive</p>
      <h2 id="runs-heading">Published runs</h2>
      {archive(result_items)}
    </section>
    <section class="section scope" aria-labelledby="scope-heading">
      <p class="eyebrow">Scope</p>
      <h2 id="scope-heading">Benchmark the kernel graph, not the nonlinear model</h2>
      <p>The laboratory reports primitive FFT and matrix performance separately from packing, ordering, and complete operator totals. It does not reimplement WVM’s nonlinear flux calculation.</p>
      <a href="{REPOSITORY_URL}/blob/main/docs/benchmark-contract.md">Read the benchmark contract</a>
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
          </article>
        """)
    return f'<div class="provider-grid">{"".join(cards)}</div>'


def environment_details(result: dict) -> str:
    environment = result["environment"]
    fields = (
        ("Run ID", result["run"]["id"]),
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


def build_run_page(path: Path, result: dict) -> str:
    csv_path = path.with_suffix(".csv")
    csv_link = (
        f'<a class="button secondary" href="../results/{quote(csv_path.name)}" download>Download samples CSV</a>'
        if csv_path.exists()
        else ""
    )
    content = f"""
    <section class="hero compact">
      <p class="eyebrow">Published benchmark run</p>
      <h1>{escaped(result['environment']['cpuBrand'])}</h1>
      <p class="lede">{result['workload']['Nx']} × {result['workload']['Ny']}, N<sub>z</sub>={result['workload']['Nz']}, fields={result['workload']['fields']} · {escaped(display_timestamp(result['environment']['timestampUtc']))}</p>
      <div class="button-row"><a class="button" href="../results/{quote(path.name)}" download>Download result JSON</a>{csv_link}</div>
    </section>
    <section class="section" aria-labelledby="summary-heading">
      <p class="eyebrow">Comparison</p><h2 id="summary-heading">Headline timings</h2>
      {result_summary_cards(result)}
      {summary_timing_table(result)}
    </section>
    <section class="section" aria-labelledby="provider-heading">
      <p class="eyebrow">Providers</p><h2 id="provider-heading">Setup and correctness</h2>
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
    return shell(result["run"]["id"], content, "../")


def clean_output(output_dir: Path, repository_root: Path, results_dir: Path) -> None:
    resolved = output_dir.resolve()
    forbidden = {Path("/").resolve(), Path.home().resolve(), repository_root.resolve(), results_dir.resolve()}
    if resolved in forbidden or resolved in repository_root.resolve().parents or results_dir.resolve() in resolved.parents:
        raise ValueError(f"Refusing to replace unsafe output directory: {resolved}")
    if resolved.exists():
        shutil.rmtree(resolved)
    resolved.mkdir(parents=True)


def build_site(results_dir: Path, output_dir: Path) -> None:
    repository_root = Path(__file__).resolve().parents[1]
    results_dir = results_dir.resolve()
    output_dir = output_dir.resolve()
    result_items = load_results(results_dir)
    clean_output(output_dir, repository_root, results_dir)

    (output_dir / "assets").mkdir()
    (output_dir / "runs").mkdir()
    (output_dir / "results").mkdir()
    (output_dir / "schema").mkdir()
    shutil.copyfile(repository_root / "site" / "style.css", output_dir / "assets" / "style.css")
    shutil.copyfile(repository_root / "site" / "favicon.svg", output_dir / "assets" / "favicon.svg")
    shutil.copyfile(repository_root / "schema" / "spectral-kernel-benchmark-v1.schema.json", output_dir / "schema" / "spectral-kernel-benchmark-v1.schema.json")
    (output_dir / ".nojekyll").write_text("", encoding="utf-8")
    (output_dir / "index.html").write_text(build_index(result_items), encoding="utf-8")

    for source_path, result in result_items:
        shutil.copyfile(source_path, output_dir / "results" / source_path.name)
        csv_path = source_path.with_suffix(".csv")
        if csv_path.exists():
            shutil.copyfile(csv_path, output_dir / "results" / csv_path.name)
        run_html = build_run_page(source_path, result)
        (output_dir / "runs" / f"{source_path.stem}.html").write_text(run_html, encoding="utf-8")


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
