"""Micro-structure features from raw 5-second feed ticks.

This is the finest resolution the public data supports: the feed polls every
5 seconds, so "the live moment" here is a 5-second tick, not a millisecond.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from .catalog import CatalogEntry, est_round_trip_cost_pct
from .config import Settings


@dataclass(frozen=True)
class MicroFeatures:
    ret_30s: float       # % return over the last ~30 seconds
    ret_60s: float
    ret_300s: float
    uptick_ratio: float  # share of positive ticks over the last ~2 minutes
    micro_vol: float     # stddev of per-tick % returns over ~5 minutes
    ticks: int


def _return_over(ticks: list[tuple[float, float]], window_s: float) -> float:
    now_ts, now_price = ticks[-1]
    target = now_ts - window_s
    base = None
    for ts, price in ticks:
        if ts >= target:
            base = price
            break
    if base is None or base <= 0:
        return 0.0
    return (now_price - base) / base * 100.0


def micro_features(ticks: list[tuple[float, float]]) -> MicroFeatures | None:
    if len(ticks) < 6:
        return None
    recent = ticks[-60:]  # ~5 minutes of 5s ticks
    deltas = [(b[1] - a[1]) / a[1] for a, b in zip(recent, recent[1:])
              if a[1] > 0]
    nonzero = [d for d in deltas[-24:] if d != 0.0]
    uptick = (sum(1 for d in nonzero if d > 0) / len(nonzero)
              if nonzero else 0.5)
    vol = 0.0
    if len(deltas) >= 3:
        mean = sum(deltas) / len(deltas)
        vol = math.sqrt(sum((d - mean) ** 2 for d in deltas)
                        / (len(deltas) - 1)) * 100.0
    return MicroFeatures(
        ret_30s=_return_over(ticks, 30.0),
        ret_60s=_return_over(ticks, 60.0),
        ret_300s=_return_over(ticks, 300.0),
        uptick_ratio=uptick,
        micro_vol=vol,
        ticks=len(ticks),
    )


def live_score(f: MicroFeatures, liquidity_usd: float,
               cost_pct: float) -> float:
    """Heuristic "worth watching right now": recent movement, scaled up when
    ticks push persistently one way, minus cost and shallow-pool penalties."""
    activity = abs(f.ret_60s) + abs(f.ret_300s) / 2.0
    persistence = abs(f.uptick_ratio - 0.5) * 2.0          # 0..1
    score = activity * (0.5 + 0.5 * persistence)
    score -= cost_pct * 0.3
    if liquidity_usd < 20_000:
        score -= 5.0
    return score


def live_rank(settings: Settings, snapshots: list,
              ticks_by_mint: dict[str, list[tuple[float, float]]],
              size_usd: float) -> list[dict]:
    """Rank every tracked token by live micro-activity. Pure data, no LLM."""
    rows = []
    for s in snapshots:
        if s.price_usd <= 0 or s.liquidity_usd <= 0:
            continue
        f = micro_features(ticks_by_mint.get(s.mint, []))
        if f is None:
            continue
        cost = est_round_trip_cost_pct(settings, s.liquidity_usd, size_usd)
        rows.append({
            "mint": s.mint, "symbol": s.symbol, "price_usd": s.price_usd,
            "liquidity_usd": s.liquidity_usd,
            "ret_30s": f.ret_30s, "ret_60s": f.ret_60s,
            "ret_300s": f.ret_300s, "uptick_ratio": f.uptick_ratio,
            "micro_vol": f.micro_vol,
            "est_cost_pct": cost,
            "live_score": live_score(f, s.liquidity_usd, cost),
        })
    rows.sort(key=lambda r: r["live_score"], reverse=True)
    for i, r in enumerate(rows):
        r["rank"] = i + 1
    return rows


def micro_prompt_lines(catalog: list[CatalogEntry],
                       ticks_by_mint: dict[str, list[tuple[float, float]]],
                       top_n: int = 5) -> str:
    lines = []
    for e in catalog[:top_n]:
        f = micro_features(ticks_by_mint.get(e.mint, []))
        if f is None:
            lines.append(f"{e.symbol}: no tick data yet")
            continue
        lines.append(
            f"{e.symbol}: 30s {f.ret_30s:+.2f}% | 60s {f.ret_60s:+.2f}% | "
            f"5m {f.ret_300s:+.2f}% | upticks {f.uptick_ratio * 100:.0f}% | "
            f"tick-vol {f.micro_vol:.3f}%"
        )
    return "\n".join(lines)
