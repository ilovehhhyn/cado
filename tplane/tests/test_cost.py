"""A cost line prices every used resource exactly, and usd is absent whenever a used resource has no price."""

from __future__ import annotations

import pytest

from tplane.cost import NO_PRICES, PriceError, PriceTable, cost_line


def test_full_price_table_prices_every_component_exactly() -> None:
    prices = PriceTable(
        cpu_usd_per_hour=3.6,
        gpu_usd_per_hour=7.2,
        usd_per_million_tokens_in=1.0,
        usd_per_million_tokens_out=2.0,
    )

    line = cost_line(
        cpu_seconds=1_800.0, gpu_seconds=900.0, tokens_in=500_000, tokens_out=250_000, prices=prices
    )

    assert line.usd == pytest.approx(4.6, rel=0.0, abs=1e-12)


def test_usd_is_absent_when_a_used_resource_is_unpriced() -> None:
    prices = PriceTable(cpu_usd_per_hour=1.0)

    line = cost_line(cpu_seconds=10.0, gpu_seconds=5.0, tokens_in=0, tokens_out=0, prices=prices)

    assert line.usd is None
    assert line.gpu_seconds == 5.0


def test_unused_resource_without_a_price_does_not_block_usd() -> None:
    prices = PriceTable(cpu_usd_per_hour=3_600.0)

    line = cost_line(cpu_seconds=2.0, gpu_seconds=0.0, tokens_in=0, tokens_out=0, prices=prices)

    assert line.usd == 2.0


def test_no_prices_gives_usd_none_and_keeps_quantities() -> None:
    line = cost_line(cpu_seconds=1.0, gpu_seconds=1.0, tokens_in=1, tokens_out=1, prices=NO_PRICES)

    assert line.usd is None
    assert (line.cpu_seconds, line.gpu_seconds, line.tokens_in, line.tokens_out) == (1.0, 1.0, 1, 1)


def test_negative_price_is_rejected() -> None:
    with pytest.raises(
        PriceError, match="gpu_usd_per_hour must be a finite non-negative number, got -1.0"
    ):
        PriceTable(gpu_usd_per_hour=-1.0)


def test_a_non_finite_price_is_rejected_at_construction() -> None:
    for value in (float("nan"), float("inf")):
        with pytest.raises(
            PriceError, match="cpu_usd_per_hour must be a finite non-negative number"
        ):
            PriceTable(cpu_usd_per_hour=value)
