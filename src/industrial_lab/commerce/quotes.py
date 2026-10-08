"""Quote helper and commercial integration for inference engines.

Adheres strictly to MEGAPLAN.md §6.1, §6.2, and §6.3:
- Pure integer arithmetic in minor currency units (no float sums).
- Interacts with Shop Simulator via POST /api/quotes or computes atomically via fixtures.
- Sets QuoteStatus.requires_technical_review if technical_verdict is INCOMPATIBLE or INSUFFICIENT_EVIDENCE.
- Sets QuoteStatus.unavailable if any requested item lacks stock.
- Preserves scenario revision and simulated expiration dates.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import logging
import os
from typing import Any, Mapping, Optional, Sequence, Union

import httpx

from industrial_lab.schemas import (
    Quote,
    QuoteLine,
    QuoteStatus,
    TechnicalVerdict,
)
from industrial_lab.shop import fixtures

logger = logging.getLogger(__name__)

DEFAULT_SHOP_BASE_URL = os.environ.get("SHOP_BASE_URL", "http://127.0.0.1:8081")


class QuoteService:
    """Service to generate consistent commercial quotes for engines."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        http_client: Optional[httpx.AsyncClient] = None,
        timeout: float = 5.0,
    ) -> None:
        self.base_url = (base_url or os.environ.get("SHOP_BASE_URL") or DEFAULT_SHOP_BASE_URL).rstrip("/")
        self._http_client = http_client
        self.timeout = timeout

    async def create_quote(
        self,
        product_quantities: Mapping[str, int] | Sequence[Union[dict[str, Any], tuple[str, int]]],
        technical_verdict: Optional[TechnicalVerdict] = None,
        scenario_id: Optional[str] = None,
    ) -> Quote:
        """Creates a commercial quote for given products and quantities.

        Attempts HTTP call to shop simulator POST /api/quotes first.
        If unavailable or fails, computes directly and atomically from fixtures.
        """
        # Normalize items to list of (product_id, quantity)
        normalized_lines: list[tuple[str, int]] = []
        if isinstance(product_quantities, Mapping):
            for pid, qty in product_quantities.items():
                if int(qty) > 0:
                    normalized_lines.append((str(pid), int(qty)))
        elif isinstance(product_quantities, Sequence):
            for item in product_quantities:
                if isinstance(item, Mapping):
                    pid = str(item["product_id"])
                    qty = int(item.get("quantity", 1))
                elif isinstance(item, (tuple, list)):
                    pid = str(item[0])
                    qty = int(item[1])
                else:
                    continue
                if qty > 0:
                    normalized_lines.append((pid, qty))

        if not normalized_lines:
            scenario = fixtures.load_scenario(scenario_id)
            now = datetime.now(timezone.utc)
            return Quote(
                status=QuoteStatus.preliminary,
                lines=[],
                subtotal_minor=0,
                tax_minor=0,
                total_minor=0,
                currency=scenario.get("currency", "EUR"),
                revision=scenario.get("revision", "rev-empty"),
                expires_at_simulated=(now + timedelta(days=7)).isoformat(),
            )

        # 1. Attempt HTTP call if possible
        try:
            http_quote = await self._call_shop_api(normalized_lines)
            if http_quote is not None:
                return self._apply_verdict_guard(http_quote, technical_verdict)
        except Exception as exc:
            logger.debug("Shop API HTTP call failed (%s); falling back to direct fixtures computation.", exc)

        # 2. Compute atomically from fixtures
        computed = self._compute_from_fixtures(normalized_lines, scenario_id=scenario_id)
        return self._apply_verdict_guard(computed, technical_verdict)

    async def _call_shop_api(self, lines: list[tuple[str, int]]) -> Optional[Quote]:
        """POST /api/quotes to the shop simulator."""
        endpoint = f"{self.base_url}/api/quotes"
        payload = {"lines": [{"product_id": pid, "quantity": qty} for pid, qty in lines]}

        client = self._http_client
        owns_client = client is None
        if owns_client:
            client = httpx.AsyncClient(timeout=self.timeout)

        try:
            resp = await client.post(endpoint, json=payload)
            if resp.status_code != 200:
                logger.debug("Shop API returned status %d: %s", resp.status_code, resp.text)
                return None

            data = resp.json()
            quote_lines: list[QuoteLine] = []
            for item in data.get("lines", []):
                quote_lines.append(
                    QuoteLine(
                        product_id=str(item["product_id"]),
                        quantity=int(item["quantity"]),
                        unit_price_minor=int(item["unit_price_minor"]),
                        line_total_minor=int(item["line_total_minor"]),
                    )
                )

            # Map status
            api_status = str(data.get("status", "valid")).lower()
            avail_status = str(data.get("availability_status", "available")).lower()
            if api_status == "unavailable" or avail_status == "unavailable":
                q_status = QuoteStatus.unavailable
            else:
                q_status = QuoteStatus.preliminary

            return Quote(
                status=q_status,
                lines=quote_lines,
                subtotal_minor=int(data.get("subtotal_minor", 0)),
                tax_minor=int(data.get("tax_minor", 0)),
                total_minor=int(data.get("total_minor", 0)),
                currency=str(data.get("currency", "EUR")),
                revision=str(data.get("revision", "rev-unknown")),
                expires_at_simulated=str(data.get("expires_at", "")),
            )
        finally:
            if owns_client and client is not None:
                await client.aclose()

    def _compute_from_fixtures(
        self,
        lines: list[tuple[str, int]],
        scenario_id: Optional[str] = None,
    ) -> Quote:
        """Atomic computation from shop fixtures with integer minor units."""
        scenario = fixtures.load_scenario(scenario_id)
        revision = str(scenario.get("revision", "rev-default"))
        currency = str(scenario.get("currency", "EUR"))

        quote_lines: list[QuoteLine] = []
        subtotal_minor = 0
        all_in_stock = True

        for pid, qty in lines:
            comm = fixtures.get_commerce(pid, scenario_id=scenario_id)
            if comm is None:
                # If product not found in scenario, use placeholder unit price 0 and mark unavailable
                unit_price = 0
                stock = 0
            else:
                unit_price = int(comm.get("price_minor", 0))
                stock = int(comm.get("stock_available", 0))

            line_total = unit_price * qty
            subtotal_minor += line_total

            if stock < qty:
                all_in_stock = False

            quote_lines.append(
                QuoteLine(
                    product_id=pid,
                    quantity=qty,
                    unit_price_minor=unit_price,
                    line_total_minor=line_total,
                )
            )

        tax_minor = 0
        total_minor = subtotal_minor + tax_minor

        now = datetime.now(timezone.utc)
        expires_at = (now + timedelta(days=7)).isoformat()

        status = QuoteStatus.preliminary if all_in_stock else QuoteStatus.unavailable

        return Quote(
            status=status,
            lines=quote_lines,
            subtotal_minor=subtotal_minor,
            tax_minor=tax_minor,
            total_minor=total_minor,
            currency=currency,
            revision=revision,
            expires_at_simulated=expires_at,
        )

    def _apply_verdict_guard(
        self,
        quote: Quote,
        verdict: Optional[TechnicalVerdict],
    ) -> Quote:
        """Enforces MEGAPLAN §6.2:

        'Si hay incompatibilidad o evidencia insuficiente, puede mostrarse un listado
        de precios, pero quote.status debe ser requires_technical_review; no anunciar
        conjunto validado.'
        """
        if verdict in (TechnicalVerdict.INCOMPATIBLE, TechnicalVerdict.INSUFFICIENT_EVIDENCE):
            # If stock was already unavailable, keep unavailable or update to requires_technical_review
            return Quote(
                status=QuoteStatus.requires_technical_review,
                lines=quote.lines,
                subtotal_minor=quote.subtotal_minor,
                tax_minor=quote.tax_minor,
                total_minor=quote.total_minor,
                currency=quote.currency,
                revision=quote.revision,
                expires_at_simulated=quote.expires_at_simulated,
            )
        return quote


# Standalone helper function
async def fetch_or_compute_quote(
    product_ids: list[str],
    quantities: Optional[dict[str, int]] = None,
    technical_verdict: Optional[TechnicalVerdict] = None,
    scenario_id: Optional[str] = None,
    base_url: Optional[str] = None,
) -> Quote:
    """Convenience helper to compute or fetch a Quote for selected product IDs."""
    qty_map: dict[str, int] = {}
    for pid in product_ids:
        q = quantities.get(pid, 1) if quantities else 1
        qty_map[pid] = qty_map.get(pid, 0) + q

    service = QuoteService(base_url=base_url)
    return await service.create_quote(
        product_quantities=qty_map,
        technical_verdict=technical_verdict,
        scenario_id=scenario_id,
    )
