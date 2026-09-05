"""Token catalog: candidates ranked by transparent trading-quality metrics.

The score is a display/ranking heuristic — the LLM picker sees every metric
and makes its own choice; the random control ignores them entirely.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from .config import Settings
from .indicators import one_minute_closes, realized_vol_pct
from .models import TokenSnapshot


@dataclass(frozen=True)
class CatalogEntry:
    mint: str
    symbol: str
    price_usd: float
    liquidity_usd: float
    volume_h24: float
    price_change_h1: float
    price_change_h24: float
    volatility_pct: float      # stddev of 1-min returns, %
    est_cost_pct: float        # est. full round-trip cost for our size, %
    score: float
    rank: int = 0
    buys_m5: int = 0
    sells_m5: int = 0
    vol_m5: float = 0.0
    chg_m5: float = 0.0


def est_round_trip_cost_pct(settings: Settings, liquidity_usd: float,
                            size_usd: float) -> float:
    if size_usd <= 0:
        return 100.0
    if liquidity_usd <= 0:
        return 100.0
    per_side = (settings.lp_fee_rate
                + size_usd / (liquidity_usd + size_usd)
                + settings.tds_rate
                + settings.priority_fee_usd / size_usd)
    return 2.0 * per_side * 100.0


MIN_HEALTHY_LIQUIDITY_USD = 20_000.0


def score_entry(turnover: float, chg_h1: float, cost_pct: float,
                volatility_pct: float, liquidity_usd: float) -> float:
    """Higher = more scalpable. Rewards traded volume relative to pool depth
    (log-scaled so microcap churn doesn't dominate) and per-minute movement;
    punishes momentum extremes, round-trip cost, and shallow pools."""
    turnover_score = min(math.log10(1.0 + max(0.0, turnover)), 3.0) / 3.0 * 30.0
    vol_score = min(volatility_pct, 5.0) / 5.0 * 20.0             # 0..20
    momentum_pen = min(abs(chg_h1), 30.0) / 30.0 * 10.0           # 0..10
    cost_pen = min(cost_pct, 10.0) / 10.0 * 40.0                  # 0..40
    depth_pen = 30.0 * max(0.0, 1.0 - liquidity_usd / MIN_HEALTHY_LIQUIDITY_USD)
    return turnover_score + vol_score - momentum_pen - cost_pen - depth_pen


def build_catalog(settings: Settings, snapshots: list[TokenSnapshot],
                  history_by_mint: dict[str, list[tuple[float, float]]],
                  size_usd: float) -> list[CatalogEntry]:
    entries = []
    for s in snapshots:
        turnover = s.volume_h24 / s.liquidity_usd if s.liquidity_usd > 0 else 0.0
        vol = realized_vol_pct(one_minute_closes(history_by_mint.get(s.mint, [])))
        cost = est_round_trip_cost_pct(settings, s.liquidity_usd, size_usd)
        entries.append(CatalogEntry(
            mint=s.mint, symbol=s.symbol, price_usd=s.price_usd,
            liquidity_usd=s.liquidity_usd, volume_h24=s.volume_h24,
            price_change_h1=s.price_change_h1,
            price_change_h24=s.price_change_h24,
            volatility_pct=vol, est_cost_pct=cost,
            score=score_entry(turnover, s.price_change_h1, cost, vol,
                              s.liquidity_usd),
            buys_m5=s.buys_m5, sells_m5=s.sells_m5,
            vol_m5=s.vol_m5, chg_m5=s.chg_m5,
        ))
    entries.sort(key=lambda e: e.score, reverse=True)
    return [CatalogEntry(**{**e.__dict__, "rank": i + 1})
            for i, e in enumerate(entries)]
