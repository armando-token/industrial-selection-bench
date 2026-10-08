"""Tests for report generation module adhering to MEGAPLAN.md §25."""

from pathlib import Path
import pytest

from industrial_lab.observability.spans import TelemetryRecord
from industrial_lab.report.build import (
    EngineMetrics,
    ReportBuilder,
    ReportData,
    generate_reports,
    wilson_score_interval,
)


def test_wilson_score_interval() -> None:
    """Test confidence interval computation."""
    low, high = wilson_score_interval(42, 48)
    assert 70.0 < low < 85.0
    assert 85.0 < high <= 98.0

    # Boundary: 0 / 10
    low_zero, high_zero = wilson_score_interval(0, 10)
    assert low_zero == 0.0
    assert high_zero > 0.0


def test_report_builder_markdown_generation() -> None:
    """Test Markdown report contains required tables and insights."""
    data = ReportBuilder.create_demo_data()
    builder = ReportBuilder(data)
    md = builder.generate_markdown()

    # Check primary table header from §25.1
    assert "motor | casos | éxitos completos | falsas aprobaciones | abstención excesiva | p50 ms | p95 ms | costo total | USD/tarea correcta | preparación" in md

    # Check motor names
    assert "structured_jev" in md
    assert "rag_llm" in md
    assert "scrape_llm" in md

    # Check all §25.3 Insights sections
    assert "## 25.3 Insights Obligatorios (§25.3)" in md
    assert "1. ¿Qué motor ganó cada dimensión y con qué incertidumbre?" in md
    assert "2. ¿Qué parte del ahorro provino de preparar documentos, de JEV o de evitar generación?" in md
    assert "3. ¿Qué costo tuvo convertir manuales a hechos y revisarlos?" in md
    assert "4. ¿Cuántos casos requirieron reglas, JEV, texto recuperado o scraping?" in md
    assert "5. ¿La precisión de A vino de revisiones humanas que B/C no tenían?" in md
    assert "6. ¿Qué errores de JEV fueron corregidos por reglas? ¿Qué errores no pudieron corregirse?" in md
    assert "7. ¿Qué pasó con C cuando se habilitó cache documental?" in md
    assert "8. ¿El catálogo de tres productos era suficiente para poner a prueba selección semántica?" in md
    assert "9. ¿Cómo cambia la conclusión al mantener el mismo estado técnico y reemplazar JEV por LLM?" in md

    # Check hackathon narrative §25.4
    assert "## 25.4 Narrativa para el Hackathon" in md


def test_report_builder_html_generation() -> None:
    """Test HTML report structure, styling, and elements."""
    data = ReportBuilder.create_demo_data()
    builder = ReportBuilder(data)
    html = builder.generate_html()

    assert "<!DOCTYPE html>" in html
    assert "Industrial Selection Lab — Benchmark Report" in html
    assert "kpi-card" in html
    assert "structured_jev" in html
    assert "25.1 Tabla Principal de Resultados" in html
    assert "25.3 Insights Obligatorios" in html


def test_generate_reports_file_saving(tmp_path: Path) -> None:
    """Test generating and saving report.md and report.html to disk."""
    out_dir = tmp_path / "custom_report"
    md_file, html_file = generate_reports(output_dir=out_dir, demo=True)

    assert md_file.exists()
    assert html_file.exists()
    assert md_file.name == "report.md"
    assert html_file.name == "report.html"
    assert len(md_file.read_text(encoding="utf-8")) > 500
    assert len(html_file.read_text(encoding="utf-8")) > 1000


def test_report_strict_integrity(tmp_path: Path) -> None:
    """Test strict report integrity: no 'catálogo real', explicit demo banners, and blocked System A."""
    out_dir = tmp_path / "integrity_report"
    md_file, html_file = generate_reports(output_dir=out_dir, demo=True)

    md = md_file.read_text(encoding="utf-8")
    html = html_file.read_text(encoding="utf-8")

    # 1. MUST NOT contain 'catálogo real' anywhere
    assert "catálogo real" not in md.lower()
    assert "catálogo real" not in html.lower()

    # 2. Prominent demo labels and banners
    assert "[DEMO / SYNTHETIC FIXTURE / NON-OFFICIAL]" in md
    assert "[DEMO / SYNTHETIC FIXTURE / NON-OFFICIAL]" in html
    assert "[DEMO / SYNTHETIC / NON-OFFICIAL]" in md
    assert "[DEMO / SYNTHETIC / NON-OFFICIAL]" in html
    assert "AVISO: DATOS SINTÉTICOS DE DEMOSTRACIÓN — NO CONSTITUYE BENCHMARK OFICIAL NI CORRIDA REAL" in md
    assert "AVISO: DATOS SINTÉTICOS DE DEMOSTRACIÓN — NO CONSTITUYE BENCHMARK OFICIAL NI CORRIDA REAL" in html
    assert "CATÁLOGO SINTÉTICO DE PRUEBA (3 EQUIPOS FIXTURECORP: P1, P2, P3)" in md
    assert "CATÁLOGO SINTÉTICO DE PRUEBA (3 EQUIPOS FIXTURECORP: P1, P2, P3)" in html

    # 3. System A blocked status
    assert "BLOCKED (§0.1 #7)" in md
    assert "BLOCKED (§0.1 #7)" in html
    assert "TYPESAFE_API_KEY" in md
    assert "TYPESAFE_API_KEY" in html

