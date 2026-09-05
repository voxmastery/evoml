"""Cost model. Every component is computed separately and never merged.

Applied per side (buy and sell each pay all four):
- LP fee: lp_fee_rate * gross transaction value
- Slippage: gross * (gross / (pool_liquidity + gross))   (constant-product approx)
- Priority fee: flat USD per transaction
- India TDS (Section 194S): tds_rate * gross transaction value
"""
from __future__ import annotations

from .models import FeeBreakdown


def slippage_fraction(size_usd: float, pool_liquidity_usd: float) -> float:
    if size_usd <= 0:
        return 0.0
    if pool_liquidity_usd <= 0:
        # No measurable pool: model as total loss of the trade to impact.
        return 1.0
    return size_usd / (pool_liquidity_usd + size_usd)


def compute_fees(
    size_usd: float,
    pool_liquidity_usd: float,
    lp_fee_rate: float,
    priority_fee_usd: float,
    tds_rate: float,
) -> FeeBreakdown:
    if size_usd < 0:
        raise ValueError("size_usd must be non-negative")
    return FeeBreakdown(
        lp=size_usd * lp_fee_rate,
        slippage=size_usd * slippage_fraction(size_usd, pool_liquidity_usd),
        priority=priority_fee_usd,
        tds=size_usd * tds_rate,
    )
