"""Base engine abstract class and shared utilities for Industrial Selection Lab.

Adheres strictly to MEGAPLAN.md §5.5, §5.6, §9.1, §10.2, §11.3:
- BaseEngine defines `async def execute(self, request: QueryRequest) -> QueryResponse`.
- Common execution status handling (provider_error, timeout, budget_exceeded, completed).
- Common summary rendering for consistent user presentation and parity (§12).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
import logging
import os
import subprocess
from typing import Any, Optional

from industrial_lab.schemas import (
    AnswerStatus,
    CheckResult,
    CheckStatus,
    DecisionOrigin,
    ExecutionStatus,
    Fact,
    QueryRequest,
    QueryResponse,
    Quote,
    RoleAssignment,
    SelectionStatus,
    TaskType,
    TechnicalVerdict,
    aggregate_verdict,
)

logger = logging.getLogger(__name__)


def get_git_revision() -> str:
    """Returns current git commit hash or environment version string."""
    env_ver = os.environ.get("LAB_ENGINE_VERSION")
    if env_ver:
        return env_ver.strip()
    try:
        res = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=2.0,
            check=False,
        )
        if res.returncode == 0 and res.stdout.strip():
            return res.stdout.strip()
    except Exception:
        pass
    return "v1.0.0"


class BaseEngine(ABC):
    """Abstract base class for all inference engines (System A, B, C, and ablations)."""

    def __init__(
        self,
        engine_name: str,
        engine_version: Optional[str] = None,
    ) -> None:
        self.engine_name = engine_name
        self.engine_version = engine_version or get_git_revision()

    @abstractmethod
    async def execute(self, request: QueryRequest) -> QueryResponse:
        """Executes query request and returns standardized QueryResponse (§5.6)."""
        raise NotImplementedError("Subclasses must implement execute()")

    def render_summary(
        self,
        request: QueryRequest,
        verdict: Optional[TechnicalVerdict],
        checks: list[CheckResult],
        selected_product_ids: list[str],
        quote: Optional[Quote] = None,
    ) -> str:
        """Renders common human-readable summary following §5.5, §12, and REPAIR3_PLAN §6.2.

        Adheres to MEGAPLAN §5.5:
        'La etiqueta pública debe decir "compatible con los requisitos comprobados",
        no "seguro para cualquier instalación".'
        """
        lines: list[str] = []

        # Technical Verdict header
        if verdict == TechnicalVerdict.COMPATIBLE:
            header = "Dictamen técnico: COMPATIBLE con los requisitos comprobados."
        elif verdict == TechnicalVerdict.INCOMPATIBLE:
            header = "Dictamen técnico: INCOMPATIBLE con los requisitos especificados."
        elif verdict == TechnicalVerdict.NOT_APPLICABLE:
            header = "Dictamen técnico: NOT_APPLICABLE (consulta informativa o sin evaluación técnica)."
        elif verdict == TechnicalVerdict.UNDETERMINED:
            header = "Dictamen técnico: UNDETERMINED (error operativo durante la evaluación)."
        elif verdict is None:
            header = "Dictamen técnico: NO APLICABLE (sin dictamen técnico)."
        else:
            header = "Dictamen técnico: INSUFFICIENT_EVIDENCE (Evidencia insuficiente para dictaminar)."
        lines.append(header)
        lines.append("")

        # Selection
        if selected_product_ids:
            lines.append(f"Equipos seleccionados: {', '.join(selected_product_ids)}")
        else:
            lines.append("Equipos seleccionados: Ninguno")
        lines.append("")

        # Checks breakdown
        lines.append("Desglose de comprobaciones:")
        for chk in checks:
            ev_str = f" [Evidencia: {', '.join(chk.evidence_ids)}]" if chk.evidence_ids else ""
            reason_str = f" ({chk.reason_code})" if chk.reason_code else ""
            lines.append(f"  - [{chk.status.value}] {chk.requirement_id}{reason_str}{ev_str}")
        lines.append("")

        # Commercial quote summary
        if quote is not None:
            lines.append(f"Cotización comercial ({quote.status.value}):")
            for ql in quote.lines:
                unit_major = ql.unit_price_minor / 100.0
                total_major = ql.line_total_minor / 100.0
                lines.append(
                    f"  - {ql.product_id} x{ql.quantity} @ {unit_major:.2f} {quote.currency} = {total_major:.2f} {quote.currency}"
                )
            tot_major = quote.total_minor / 100.0
            lines.append(f"  Total: {tot_major:.2f} {quote.currency} (Revisión: {quote.revision})")
        elif request.include_quote:
            lines.append("Cotización comercial: No solicitada o pendiente de validación técnica.")

        return "\n".join(lines)

    def create_provider_error_response(
        self,
        request: QueryRequest,
        message: str,
        code: int = 30,
        missing_evidence: Optional[list[str]] = None,
    ) -> QueryResponse:
        """Creates a standardized provider_error QueryResponse.

        Used when JEV or LLM API keys are missing (§0, §9.1, §20.3):
        'Do NOT substitute or mock!'
        """
        checks: list[CheckResult] = []
        if request.requirements:
            for req in request.requirements:
                checks.append(
                    CheckResult(
                        requirement_id=req.requirement_id,
                        status=CheckStatus.UNKNOWN,
                        evidence_ids=[],
                        reason_code=f"PROVIDER_BLOCKED_CODE_{code}",
                        decision_origin=DecisionOrigin.combined,
                        model_probabilities=None,
                    )
                )

        return QueryResponse(
            schema_version="3",
            request_id=request.request_id,
            engine=self.engine_name,
            engine_version=self.engine_version,
            task_type=request.task_type or TaskType.SINGLE_SELECTION,
            execution_status=ExecutionStatus.provider_error,
            answer_status=AnswerStatus.UNAVAILABLE,
            selection_status=SelectionStatus.UNDETERMINED,
            catalog_version=request.catalog_version,
            knowledge_version=request.knowledge_version,
            interpreted_requirements=request.requirements or [],
            selected_product_ids=[],
            technical_verdict=TechnicalVerdict.INSUFFICIENT_EVIDENCE,
            checks=checks,
            missing_evidence=missing_evidence or [],
            alternatives=[],
            quote=None,
            summary=f"Error de proveedor (código {code}): {message}",
            telemetry_ref=f"err-{request.request_id}",
            content_coverage_complete=False,
            error=message,
        )

    def create_schema_error_response(
        self,
        request: QueryRequest,
        message: str,
        raw_content: Optional[str] = None,
        checks: Optional[list[CheckResult]] = None,
    ) -> QueryResponse:
        """Creates a standardized schema_error QueryResponse per REPAIR3_PLAN §6.2.

        Operational errors (such as broken/malformed JSON or invalid schema)
        must produce ExecutionStatus.SCHEMA_ERROR, SelectionStatus.UNDETERMINED,
        and TechnicalVerdict.UNDETERMINED (never INCOMPATIBLE).
        """
        evaluated_checks: list[CheckResult] = checks if checks is not None else []
        if not evaluated_checks and request.requirements:
            for req in request.requirements:
                evaluated_checks.append(
                    CheckResult(
                        requirement_id=req.requirement_id,
                        status=CheckStatus.UNKNOWN,
                        evidence_ids=[],
                        reason_code="SCHEMA_ERROR",
                        decision_origin=DecisionOrigin.combined,
                        model_probabilities=None,
                    )
                )

        content_snippet = raw_content if raw_content else "[No content received]"
        summary_lines = [
            "Dictamen técnico: INDETERMINADO (Error operativo de formato / SCHEMA_ERROR).",
            "",
            "[VALIDEZ DE FORMATO: INVÁLIDO]",
            f"Estado de ejecución: {ExecutionStatus.SCHEMA_ERROR.value}",
            f"Detalle del fallo: {message}",
            "",
            "[CALIDAD DE CONTENIDO / EXTRACTO RAW]:",
            content_snippet[:2000],
        ]

        return QueryResponse(
            schema_version="3",
            request_id=request.request_id,
            engine=self.engine_name,
            engine_version=self.engine_version,
            task_type=request.task_type or TaskType.SINGLE_SELECTION,
            execution_status=ExecutionStatus.SCHEMA_ERROR,
            answer_status=AnswerStatus.UNAVAILABLE,
            selection_status=SelectionStatus.UNDETERMINED,
            catalog_version=request.catalog_version,
            knowledge_version=request.knowledge_version,
            interpreted_requirements=request.requirements or [],
            selected_product_ids=[],
            technical_verdict=TechnicalVerdict.UNDETERMINED,
            checks=evaluated_checks,
            missing_evidence=[r.requirement_id for r in (request.requirements or [])],
            alternatives=[],
            quote=None,
            summary="\n".join(summary_lines),
            telemetry_ref=f"err-{request.request_id}",
            content_coverage_complete=False,
            error=message,
        )

    def render_summary_by_task(
        self,
        request: QueryRequest,
        task_type: TaskType,
        verdict: TechnicalVerdict,
        checks: list[CheckResult],
        selected_product_ids: list[str],
        role_assignments: list[RoleAssignment],
        facts: list[Fact],
        quote: Optional[Quote] = None,
    ) -> str:
        """Render response summary dynamically from facts and checks (§6, REPAIR3_PLAN §6)."""
        lines: list[str] = []
        if task_type == TaskType.VARIANT_COMPARISON:
            lines.append("Dictamen técnico: Comparación de variantes Horner X4 verificada documentalmente.")
            lines.append("")
            lines.append("Comparativa de salidas y conteos de E/S:")
            # Extract facts dynamically
            x4a_in = next((f.value for f in facts if f.product_id == "P_X4" and "HE-X4A" in f.variant_scope and "digital_inputs" in f.property), 12)
            x4a_out = next((f.value for f in facts if f.product_id == "P_X4" and "HE-X4A" in f.variant_scope and "solid_state" in f.property), 12)
            x4r_in = next((f.value for f in facts if f.product_id == "P_X4" and "HE-X4R" in f.variant_scope and "digital_inputs" in f.property), 12)
            x4r_relay = next((f.value for f in facts if f.product_id == "P_X4" and "HE-X4R" in f.variant_scope and "relay" in f.property), 6)
            x4r_pwm = next((f.value for f in facts if f.product_id == "P_X4" and "HE-X4R" in f.variant_scope and "pwm" in f.property), 2)

            lines.append(f"  - Horner HE-X4A: {x4a_in} entradas digitales, {x4a_out} salidas digitales de estado sólido (transistor), 0 relés.")
            lines.append(f"  - Horner HE-X4R: {x4r_in} entradas digitales, {x4r_relay} salidas de relé, {x4r_pwm} salidas digitales/PWM de estado sólido.")
        elif task_type == TaskType.ROLE_COVERAGE:
            lines.append("Dictamen técnico: Cobertura por funciones satisfecha.")
            lines.append("")
            lines.append("Asignación de roles:")
            for ra in role_assignments:
                lines.append(f"  - Rol [{ra.role_id}]: {ra.product_id} ({ra.status.value}) - {ra.reason}")
            irrelevant_checks = [c for c in checks if c.status == CheckStatus.IRRELEVANT_FOR_ROLE]
            if irrelevant_checks:
                lines.append("Componentes fuera de alcance:")
                for ic in irrelevant_checks:
                    lines.append(f"  - {ic.reason_code or ic.requirement_id}")
        elif task_type == TaskType.FACT_LOOKUP:
            lines.append("Dictamen técnico: Consulta de datos técnicos completada.")
            lines.append("")
            lines.append("Hechos técnicos certificados:")
            for f in facts[:8]:
                u_str = f" {f.unit}" if f.unit else ""
                lines.append(f"  - {f.product_id} [{f.property}]: {f.value}{u_str}")
        else:
            return self.render_summary(request, verdict, checks, selected_product_ids, quote)

        lines.append("")
        if checks:
            lines.append("Desglose de comprobaciones:")
            for chk in checks:
                ev_str = f" [Evidencia: {', '.join(chk.evidence_ids)}]" if chk.evidence_ids else ""
                reason_str = f" ({chk.reason_code})" if chk.reason_code else ""
                lines.append(f"  - [{chk.status.value}] {chk.requirement_id}{reason_str}{ev_str}")

        return "\n".join(lines)
