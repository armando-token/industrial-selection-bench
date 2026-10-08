"""Inference engines for Industrial Selection Lab.

Adheres to MEGAPLAN.md §2.1, §2.2, §9, §10, and §11:
- BaseEngine: Abstract base class for all engines
- StructuredJevEngine: System A (structured knowledge + TypeSafe JEV)
- RagLlmEngine: System B (hybrid retrieval + structured LLM output)
- ScrapeLlmEngine: System C (online scraping agent with explicit tools)
- StructuredLlmEngine: Ablation H4 (replaces JEV with LLM)
- StructuredRulesEngine: Ablation H6 (pure deterministic rules)
- RagLlmGuardedEngine: Ablation (System B with deterministic guardrails)
"""

from __future__ import annotations

from industrial_lab.engines.ablations import (
    RagLlmGuardedEngine,
    StructuredLlmEngine,
    StructuredRulesEngine,
)
from industrial_lab.engines.base import BaseEngine
from industrial_lab.engines.rag_llm import RagLlmEngine
from industrial_lab.engines.scrape_llm import ScrapeLlmEngine
from industrial_lab.engines.structured_jev import StructuredJevEngine

__all__ = [
    "BaseEngine",
    "StructuredJevEngine",
    "RagLlmEngine",
    "ScrapeLlmEngine",
    "StructuredLlmEngine",
    "StructuredRulesEngine",
    "RagLlmGuardedEngine",
]
