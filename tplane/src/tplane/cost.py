"""Compute one unit's cost line from resources held and a price table; usd is absent unless every used resource is priced."""

from __future__ import annotations

import math
from dataclasses import dataclass, fields
from typing import Final

from tplane.schema import CostLine

SECONDS_PER_HOUR: Final[int] = 60 * 60
TOKENS_PER_MILLION: Final[int] = 1_000_000


class PriceError(ValueError):
    """Every price is a non-negative number of US dollars per unit or absent."""


@dataclass(frozen=True, kw_only=True)
class PriceTable:
    """Unit prices in US dollars; an absent price means that resource cannot be priced."""

    cpu_usd_per_hour: float | None = None
    gpu_usd_per_hour: float | None = None
    usd_per_million_tokens_in: float | None = None
    usd_per_million_tokens_out: float | None = None

    def __post_init__(self) -> None:
        for field in fields(self):
            value = getattr(self, field.name)
            if value is not None and not (math.isfinite(value) and value >= 0):
                raise PriceError(f"{field.name} must be a finite non-negative number, got {value}")


NO_PRICES: Final[PriceTable] = PriceTable()


def cost_line(
    *, cpu_seconds: float, gpu_seconds: float, tokens_in: int, tokens_out: int, prices: PriceTable
) -> CostLine:
    """Price the held resources: usd = sum(quantity * price / per_unit) over used resources, or None if one is unpriced."""
    components: tuple[tuple[float, float | None, int], ...] = (
        (cpu_seconds, prices.cpu_usd_per_hour, SECONDS_PER_HOUR),
        (gpu_seconds, prices.gpu_usd_per_hour, SECONDS_PER_HOUR),
        (tokens_in, prices.usd_per_million_tokens_in, TOKENS_PER_MILLION),
        (tokens_out, prices.usd_per_million_tokens_out, TOKENS_PER_MILLION),
    )
    used = [
        (quantity, price, per_unit) for quantity, price, per_unit in components if quantity != 0
    ]
    priced = [
        (quantity, price, per_unit) for quantity, price, per_unit in used if price is not None
    ]
    usd = (
        sum((quantity * price / per_unit for quantity, price, per_unit in priced), 0.0)
        if len(priced) == len(used)
        else None
    )
    return CostLine(
        cpu_seconds=cpu_seconds,
        gpu_seconds=gpu_seconds,
        tokens_in=tokens_in,
        tokens_out=tokens_out,
        usd=usd,
    )
