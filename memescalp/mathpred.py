"""Classical-mathematics forecaster arm — no LLM, no randomness.

Lineage of every rule (deliberately old, deliberately simple):
- Ornstein-Uhlenbeck mean reversion (1930, statistical physics): a price
  stretched far from its short-term mean tends to relax back. Measured as a
  Bollinger-style z-score against the 20-minute mean.
- Lo & MacKinlay variance ratio (1988): VR(2) > 1 means returns positively
  autocorrelate (trending regime), VR(2) < 1 means they reverse.
- EMA crossover momentum (classic technical rule): EMA5 vs EMA20 direction.
- Wen, Bouri, Xu & Zhao (2022) find crypto intraday returns show BOTH
  momentum and reversal depending on regime — hence: variance ratio picks
  the regime, then momentum or reversion supplies the call.
"""
from __future__ import annotations

import math
import time

from .catalog import CatalogEntry
from .indicators import ema
from .predictor import Prediction

MIN_CLOSES = 25
OU_Z_FADE = 2.0          # |z| beyond this: bet on reversion to the mean
VR_TREND = 1.05          # VR(2) above this: trending regime
VR_REVERT = 0.95         # VR(2) below this: reverting regime
MIN_CALL_STRENGTH = 0.15  # weaker signals abstain instead of calling


def variance_ratio_2(closes: list[float]) -> float | None:
    """Lo-MacKinlay VR(2) on log returns; 1.0 = random walk."""
    rets = [math.log(b / a) for a, b in zip(closes, closes[1:])
            if a > 0 and b > 0]
    if len(rets) < 20:
        return None
    mean = sum(rets) / len(rets)
    var1 = sum((r - mean) ** 2 for r in rets) / (len(rets) - 1)
    if var1 == 0:
        return None
    r2 = [rets[i] + rets[i + 1] for i in range(0, len(rets) - 1, 2)]
    mean2 = sum(r2) / len(r2)
    var2 = sum((r - mean2) ** 2 for r in r2) / max(1, len(r2) - 1)
    return var2 / (2.0 * var1)


def zscore_20(closes: list[float]) -> float | None:
    window = closes[-20:]
    if len(window) < 20:
        return None
    mean = sum(window) / len(window)
    var = sum((c - mean) ** 2 for c in window) / (len(window) - 1)
    std = math.sqrt(var)
    if std == 0:
        return None
    return (closes[-1] - mean) / std


def signal_for(closes: list[float]) -> tuple[str, float, str] | None:
    """Return (direction, strength 0..1, rule) or None if not enough data."""
    if len(closes) < MIN_CLOSES:
        return None
    z = zscore_20(closes)
    vr = variance_ratio_2(closes)
    e5, e20 = ema(closes, 5), ema(closes, 20)
    if z is None or vr is None or e5 is None or e20 is None:
        return None

    # 1. Ornstein-Uhlenbeck fade: extreme stretch beats everything else.
    if abs(z) >= OU_Z_FADE:
        direction = "DOWN" if z > 0 else "UP"
        return direction, min(abs(z) / 4.0, 1.0), f"OU-reversion z={z:+.2f}"

    momentum = (e5 - e20) / e20 if e20 > 0 else 0.0
    # 2. Trending regime: follow the EMA crossover.
    if vr >= VR_TREND and momentum != 0.0:
        direction = "UP" if momentum > 0 else "DOWN"
        strength = min(abs(momentum) * 100.0, 1.0) * min(vr - 1.0, 1.0) * 4.0
        return direction, min(strength, 1.0), (
            f"momentum VR2={vr:.2f} ema-gap={momentum * 100:+.2f}%")
    # 3. Reverting regime: fade the most recent move.
    if vr <= VR_REVERT and closes[-1] != closes[-2]:
        direction = "DOWN" if closes[-1] > closes[-2] else "UP"
        return direction, min((1.0 - vr) * 2.0, 1.0) * 0.6, (
            f"reversal VR2={vr:.2f}")
    # 4. Random-walk regime (Bachelier 1900): weakest lean, last 5-min drift.
    if len(closes) >= 6 and closes[-6] > 0:
        drift = closes[-1] - closes[-6]
        if drift != 0.0:
            return ("UP" if drift > 0 else "DOWN"), 0.05, (
                f"drift (VR2={vr:.2f} ~ random walk)")
    return None


def math_prediction(catalog: list[CatalogEntry],
                    closes_by_mint: dict[str, list[float]],
                    horizon_minutes: float) -> Prediction:
    """Pick the candidate with the strongest classical signal and call it."""
    best = None
    lines = []
    for entry in catalog:
        sig = signal_for(closes_by_mint.get(entry.mint, []))
        if sig is None:
            lines.append(f"{entry.symbol}: insufficient history")
            continue
        direction, strength, rule = sig
        lines.append(f"{entry.symbol}: {direction} strength={strength:.2f} [{rule}]")
        if best is None or strength > best[1]:
            best = (entry, strength, direction, rule)

    now = time.time()
    if best is None or best[1] < MIN_CALL_STRENGTH:
        # Selective prediction: no qualifying signal -> abstain, don't guess.
        reason = ("no candidate had enough history" if best is None else
                  f"best signal strength {best[1]:.2f} < {MIN_CALL_STRENGTH}")
        return Prediction(
            ts=now, arm="math", mint="", symbol="SKIP", direction="SKIP",
            confidence=0.0, horizon_end=now + horizon_minutes * 60.0,
            price_at=0.0, prompt="classical signals:\n" + "\n".join(lines),
            response=f"SKIP [{reason}]",
            model="ou1930+vr1988+ema", backend="math",
        )

    entry, strength, direction, rule = best
    return Prediction(
        ts=now, arm="math", mint=entry.mint, symbol=entry.symbol,
        direction=direction, confidence=min(0.5 + 0.35 * strength, 0.85),
        horizon_end=now + horizon_minutes * 60.0, price_at=entry.price_usd,
        prompt="classical signals:\n" + "\n".join(lines),
        response=f"{entry.symbol} {direction} [{rule}]",
        model="ou1930+vr1988+ema", backend="math",
    )
