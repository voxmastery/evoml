"""Small pure indicator functions over cached price history."""
from __future__ import annotations

import math


def one_minute_closes(prices: list[tuple[float, float]]) -> list[float]:
    """Collapse (ts, price) ticks into per-minute closing prices, in order."""
    closes: dict[int, float] = {}
    for ts, price in prices:
        closes[int(ts // 60)] = price
    return [closes[m] for m in sorted(closes)]


def ema(values: list[float], period: int) -> float | None:
    if len(values) < period or period <= 0:
        return None
    k = 2.0 / (period + 1)
    result = sum(values[:period]) / period
    for v in values[period:]:
        result = v * k + result * (1 - k)
    return result


def rsi(values: list[float], period: int = 14) -> float | None:
    if len(values) < period + 1:
        return None
    gains, losses = [], []
    for prev, cur in zip(values[-period - 1:-1], values[-period:]):
        change = cur - prev
        gains.append(max(0.0, change))
        losses.append(max(0.0, -change))
    avg_gain = sum(gains) / period
    avg_loss = sum(losses) / period
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - 100.0 / (1.0 + rs)


def realized_vol_pct(closes: list[float]) -> float:
    """Sample stddev of per-period % returns. 0 when not enough data."""
    if len(closes) < 3:
        return 0.0
    rets = [(b - a) / a * 100.0 for a, b in zip(closes, closes[1:]) if a > 0]
    if len(rets) < 2:
        return 0.0
    mean = sum(rets) / len(rets)
    var = sum((r - mean) ** 2 for r in rets) / (len(rets) - 1)
    return math.sqrt(var)
