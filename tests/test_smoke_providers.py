"""Ultra-light optional live provider smokes.

Default CI: offline only (no network). Enable live with:
  RUN_LIVE_PROVIDER_SMOKE=1 pytest -m live_providers

Constraints: max_tokens<=8 for LLM; single tiny JEV question. Never print secrets.
"""

from __future__ import annotations

import os

import pytest

live_enabled = os.environ.get("RUN_LIVE_PROVIDER_SMOKE", "").strip() in {"1", "true", "TRUE", "yes"}


pytestmark = pytest.mark.live_providers


def test_smoke_providers_offline_cli() -> None:
    from click.testing import CliRunner
    from industrial_lab.cli import ExitCode, cli

    result = CliRunner().invoke(cli, ["smoke-providers"])
    assert result.exit_code == ExitCode.SUCCESS
    assert "OFFLINE MODE" in result.output


@pytest.mark.skipif(not live_enabled, reason="live provider smoke disabled (set RUN_LIVE_PROVIDER_SMOKE=1)")
@pytest.mark.asyncio
async def test_live_jev_and_mantle_ping() -> None:
    """One JEV ping + one Mantle ping (max_tokens=8). Skipped without env flag."""
    from industrial_lab.adapters.jev import JevAdapter
    from industrial_lab.adapters.llm import LLMAdapter

    jev_key = os.environ.get("TYPESAFE_API_KEY", "").strip()
    llm_key = os.environ.get("LLM_API_KEY", "").strip() or os.environ.get("AWS_BEARER_TOKEN_BEDROCK", "").strip()
    if not jev_key and not llm_key:
        pytest.skip("No provider keys in environment")

    if jev_key:
        adapter = JevAdapter(timeout=20.0)
        result = await adapter.judge_state(
            "Cable conductor material is copper; cross-section 2.5 mm2.",
            {
                "q1": {
                    "type": "choice",
                    "instructions": "Is the conductor material copper? Use only the state. Do not invent missing specs.",
                    "criteria": {
                        "supported": "Evidence supports that the conductor is copper.",
                        "contradicted": "Evidence contradicts that the conductor is copper.",
                        "insufficient": "Evidence is insufficient to decide.",
                    },
                }
            },
        )
        assert result is not None
        statuses = (result.get("telemetry") or {}).get("http_statuses") or []
        assert not statuses or statuses[-1] == 200

    if llm_key:
        llm = LLMAdapter(timeout=20.0, max_retries=0)
        resp = await llm.chat(
            [{"role": "user", "content": "Reply with one word: ok"}],
            max_tokens=8,
            temperature=0.0,
        )
        assert (resp.content or "").strip()
        statuses = (resp.telemetry or {}).get("http_statuses") or []
        assert not statuses or statuses[-1] == 200
