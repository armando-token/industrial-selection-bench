"""Tests for demo report integrity adhering to MEGAPLAN.md §25 and G0/G1 freeze requirements.

Verifies:
1. ReportBuilder.create_demo_data() does not claim "catálogo real" and uses synthetic fixtures.
2. generate_reports(demo=True) includes explicit DEMO / SYNTHETIC / NON-OFFICIAL badges and disclaimers.
3. runs/latest_report/report.md does not contain "catálogo real" anywhere in text.
4. structured_jev is never reported with fabricated metrics as an official benchmark.
"""

from __future__ import annotations

from pathlib import Path
import re
import pytest

from tests.conftest import requires_runs

from industrial_lab.report.build import (
    EngineMetrics,
    ReportBuilder,
    ReportData,
    generate_reports,
)


def test_create_demo_data_no_real_catalog_claim() -> None:
    """Verifies that ReportBuilder.create_demo_data() does not claim 'catálogo real'."""
    data = ReportBuilder.create_demo_data()

    # Metadata assertions
    assert data.metadata.get("is_demo") is True
    assert data.metadata.get("data_origin") == "synthetic_fixture"
    assert "real" not in str(data.metadata.get("catalog", "")).lower()

    # Inspect all metadata values
    for k, v in data.metadata.items():
        assert "catálogo real" not in str(v).lower()
        assert "catalogo real" not in str(v).lower()

    # Render Markdown and HTML
    builder = ReportBuilder(data)
    md = builder.generate_markdown()
    html = builder.generate_html()

    # Verify "catálogo real" / "catalogo real" is NEVER present in rendered reports
    assert "catálogo real" not in md.lower()
    assert "catalogo real" not in md.lower()
    assert "catálogo real" not in html.lower()
    assert "catalogo real" not in html.lower()

    # Verify explicit synthetic fixture catalog statement
    assert "sintético de prueba" in md.lower() or "synthetic" in md.lower()
    assert "sintético de prueba" in html.lower() or "synthetic" in html.lower()


def test_generate_reports_demo_mode_badges_and_disclaimers(tmp_path: Path) -> None:
    """Verifies generate_reports(demo=True) includes explicit DEMO / SYNTHETIC / NON-OFFICIAL badges/disclaimers."""
    out_dir = tmp_path / "test_demo_report"
    md_file, html_file = generate_reports(output_dir=out_dir, demo=True)

    assert md_file.exists()
    assert html_file.exists()

    md_text = md_file.read_text(encoding="utf-8")
    html_text = html_file.read_text(encoding="utf-8")

    # 1. Verify absence of "catálogo real"
    assert "catálogo real" not in md_text.lower()
    assert "catalogo real" not in md_text.lower()
    assert "catálogo real" not in html_text.lower()
    assert "catalogo real" not in html_text.lower()

    # 2. Verify explicit badges: DEMO / SYNTHETIC / NON-OFFICIAL
    assert "DEMO / SYNTHETIC / NON-OFFICIAL" in md_text
    assert "DEMO / SYNTHETIC / NON-OFFICIAL" in html_text

    # 3. Verify prominent disclaimer warning banner
    assert "AVISO: DATOS SINTÉTICOS DE DEMOSTRACIÓN" in md_text
    assert "NO CONSTITUYE BENCHMARK OFICIAL NI CORRIDA REAL" in md_text

    assert "AVISO: DATOS SINTÉTICOS DE DEMOSTRACIÓN" in html_text
    assert "NO CONSTITUYE BENCHMARK OFICIAL NI CORRIDA REAL" in html_text

    # 4. Verify data_origin identifier
    assert "synthetic_fixture" in md_text
    assert "synthetic_fixture" in html_text


@requires_runs
def test_runs_latest_report_does_not_contain_catalogo_real() -> None:
    """Verifies that runs/latest_report/report.md does not contain 'catálogo real'."""
    # Find runs/latest_report/report.md relative to test location or current working dir
    repo_root = Path(__file__).resolve().parent.parent
    report_md_path = repo_root / "runs" / "latest_report" / "report.md"

    if not report_md_path.exists():
        # Fallback to local runs directory in working path
        report_md_path = Path("runs/latest_report/report.md")

    assert report_md_path.exists(), f"Expected report file at {report_md_path}"

    content = report_md_path.read_text(encoding="utf-8")

    # Strict check: "catálogo real" / "catalogo real" MUST NOT appear anywhere
    assert "catálogo real" not in content.lower()
    assert "catalogo real" not in content.lower()

    # Must contain clear non-official / synthetic indicators
    assert "DEMO" in content
    assert "SYNTHETIC" in content
    assert "NON-OFFICIAL" in content

    # Check html if present as well
    report_html_path = report_md_path.with_suffix(".html")
    if report_html_path.exists():
        html_content = report_html_path.read_text(encoding="utf-8")
        assert "catálogo real" not in html_content.lower()
        assert "catalogo real" not in html_content.lower()
        assert "DEMO / SYNTHETIC / NON-OFFICIAL" in html_content


def test_structured_jev_never_reported_as_official_with_fabricated_metrics(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verifies structured_jev has zero fabricated success metrics in demo data or rendered reports."""
    demo_data = ReportBuilder.create_demo_data()
    jev_metrics = demo_data.engines.get("structured_jev")
    assert jev_metrics is not None

    # 1. In demo data, structured_jev MUST have zero cases, zero successes, and zero invented latencies/costs
    assert jev_metrics.total_cases == 0, f"Expected 0 cases for blocked structured_jev, got {jev_metrics.total_cases}"
    assert jev_metrics.successful_tasks == 0, f"Expected 0 successes for blocked structured_jev, got {jev_metrics.successful_tasks}"
    assert jev_metrics.success_rate_pct == 0.0
    assert len(jev_metrics.latencies_ms) == 0, f"Expected empty latencies for blocked structured_jev, got {jev_metrics.latencies_ms}"
    assert jev_metrics.total_cost_usd is None or jev_metrics.total_cost_usd == 0.0
    assert "BLOCKED" in jev_metrics.cost_status or "BLOQUEADO" in jev_metrics.display_name or "BLOCKED" in jev_metrics.display_name

    # Metadata must record that System A is BLOCKED (§0.1 #7)
    assert demo_data.metadata.get("system_a_status") == "BLOCKED (§0.1 #7)"

    # 2. Rendered report must state System A is blocked and JEV cannot be substituted or mocked
    builder = ReportBuilder(demo_data)
    md = builder.generate_markdown()
    html = builder.generate_html()

    assert "BLOQUEADO (§0.1 #7)" in md or "BLOCKED (§0.1 #7)" in md
    assert "TYPESAFE_API_KEY" in md
    assert "NUNCA debe ser sustituido" in md or "NUNCA sustituir" in md

    assert "BLOQUEADO (§0.1 #7)" in html or "BLOCKED (§0.1 #7)" in html
    assert "TYPESAFE_API_KEY" in html

    # 3. ABSOLUTE BAN: Zero fabricated success numbers, percentages, or narratives
    banned_metrics = [
        "87.5",
        "42/48",
        "42 / 48",
        "81.3",
        "pipeline estructurado de a registró",
        "pipeline estructurado de a alcanzo",
        "1,240 ms",
    ]
    for b in banned_metrics:
        assert b not in md.lower(), f"Banned metric/phrase '{b}' found in demo Markdown!"
        assert b not in html.lower(), f"Banned metric/phrase '{b}' found in demo HTML!"

    # 4. In table rows, structured_jev row MUST NOT contain any success percentage or CI
    for line in md.splitlines():
        if "structured_jev" in line and line.strip().startswith("| **"):
            assert "%" not in line, f"Found success percentage in structured_jev table row: {line}"
            assert not re.search(r"\[\s*\d+\.?\d*%\s*-\s*\d+\.?\d*%\s*\]", line), f"Found CI brackets in structured_jev table row: {line}"
            assert "BLOCKED" in line or "BLOQUEADO" in line, f"Expected BLOCKED indicator in row: {line}"

    # Verify that in a non-demo, official report context, ReportBuilder does not fabricate metrics
    monkeypatch.delenv("LAB_DRY_RUN", raising=False)
    empty_official_data = ReportBuilder.create_empty_data(run_id="official_run_01")
    empty_official_data.metadata["data_origin"] = "official"
    official_builder = ReportBuilder(empty_official_data)
    assert official_builder.is_demo is False

    official_md = official_builder.generate_markdown()
    assert "42/48" not in official_md
    assert "87.5" not in official_md
    assert "A: structured_jev" in official_md
    assert "0/0" in official_md or "(Sin datos de ejecución)" in official_md


def test_runs_latest_report_strictly_forbids_fabricated_metrics() -> None:
    """Verifies runs/latest_report has zero fabricated metrics (87.5%, 42/48, etc.) and zero 'catálogo real'."""
    repo_root = Path(__file__).resolve().parent.parent
    latest_md_path = repo_root / "runs" / "latest_report" / "report.md"
    latest_html_path = repo_root / "runs" / "latest_report" / "report.html"

    if not latest_md_path.exists():
        pytest.skip("runs/latest_report/report.md does not exist yet")

    md_content = latest_md_path.read_text(encoding="utf-8")
    assert "catálogo real" not in md_content.lower()
    assert "catalogo real" not in md_content.lower()
    assert "87.5" not in md_content
    assert "42/48" not in md_content
    assert "81.3" not in md_content
    assert "pipeline estructurado de a registró" not in md_content.lower()

    if latest_html_path.exists():
        html_content = latest_html_path.read_text(encoding="utf-8")
        assert "catálogo real" not in html_content.lower()
        assert "catalogo real" not in html_content.lower()
        assert "87.5" not in html_content
        assert "42/48" not in html_content
        assert "81.3" not in html_content

