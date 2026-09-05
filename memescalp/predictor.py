"""Directional prediction experiment: Claude forecasts UP/DOWN over a fixed
horizon; the control arm flips coins. Nothing is bought or sold — predictions
are logged, then resolved against the actual price after the horizon."""
from __future__ import annotations

import json
import logging
import random
import re
import time
from dataclasses import dataclass

from .catalog import CatalogEntry
from .indicators import ema, realized_vol_pct, rsi

log = logging.getLogger(__name__)

CHART_TOP_N = 5

PREDICT_PROMPT = """\
You are a market forecaster in a prediction-skill experiment on Solana
memecoins. Nothing is traded; only your directional accuracy is scored.

From the catalog below, choose the ONE token you have the strongest
directional view on, and predict whether its price will be HIGHER (UP) or
LOWER (DOWN) than the current price in exactly {horizon:.0f} minutes.

Candidate catalog (live data as of {now_iso} UTC), ranked by a heuristic
scalpability score. turnover = 24h volume / pool liquidity; vol_1m = stddev
of 1-minute returns:
{table}

Live 1-minute closing prices (oldest -> newest) for the top candidates:
{charts}

Live tick microstructure (5-second feed, freshest data available):
{micro}

Market regime backdrop:
{regime}

Your own recent track record in this experiment (learn from it):
{track}

Your evolving playbook — lessons you distilled from your own past results:
{playbook}

A correct prediction means the sign of the move matches your call — magnitude
does not matter. The experiment's target is a HIGH hit rate on the calls you
do make: abstaining is free and never counts against you, a wrong call does.
Call a direction ONLY when you genuinely see better-than-60% odds; if every
chart looks like noise, skip this round.

Respond with ONLY a JSON object, no other text. Either:
{{"direction": "SKIP", "reasoning": "<why no call this round>"}}
or:
{{"mint": "<mint address from the catalog>", "symbol": "<its symbol>", \
"direction": "UP" or "DOWN", "confidence": 0.5-1.0, "reasoning": "<2-3 sentences>"}}
"""


@dataclass(frozen=True)
class Prediction:
    ts: float
    arm: str
    mint: str
    symbol: str
    direction: str        # "UP" | "DOWN"  ("" when the model's pick unusable)
    confidence: float
    horizon_end: float
    price_at: float
    prompt: str
    response: str
    model: str
    backend: str


def _chart_block(entry: CatalogEntry, closes: list[float]) -> str:
    closes = closes[-40:]
    if len(closes) < 3:
        return f"{entry.symbol}: insufficient price history yet"
    e5, e20 = ema(closes, 5), ema(closes, 20)
    trend = ("up" if e5 > e20 else "down") if e5 is not None and e20 is not None else "n/a"
    r = rsi(closes)
    return (f"{entry.symbol}: " + ", ".join(f"{c:.8g}" for c in closes)
            + f"\n  (trend {trend}, RSI14 {f'{r:.0f}' if r is not None else 'n/a'},"
              f" vol_1m {realized_vol_pct(closes):.2f}%)")


def build_predict_prompt(catalog: list[CatalogEntry],
                         closes_by_mint: dict[str, list[float]],
                         horizon_minutes: float, now: float,
                         micro: str = "n/a", regime: str = "n/a",
                         track: str = "n/a", playbook: str = "n/a") -> str:
    lines = []
    for e in catalog:
        line = (f"- #{e.rank} {e.symbol} | mint: {e.mint} | price ${e.price_usd:.8g} | "
                f"liquidity ${e.liquidity_usd:,.0f} | 24h volume ${e.volume_h24:,.0f} | "
                f"5m {e.chg_m5:+.2f}% | 1h {e.price_change_h1:+.2f}% | "
                f"24h {e.price_change_h24:+.2f}% | "
                f"vol_1m {e.volatility_pct:.2f}% | score {e.score:.1f}")
        if e.buys_m5 or e.sells_m5:
            line += (f" | flow5m {e.buys_m5}buys/{e.sells_m5}sells"
                     f" (${e.vol_m5:,.0f})")
        lines.append(line)
    charts = "\n".join(
        _chart_block(e, closes_by_mint.get(e.mint, []))
        for e in catalog[:CHART_TOP_N]
    )
    return PREDICT_PROMPT.format(
        horizon=horizon_minutes,
        now_iso=time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(now)),
        table="\n".join(lines),
        charts=charts,
        micro=micro,
        regime=regime,
        track=track,
        playbook=playbook,
    )


def parse_prediction(response: str, catalog: list[CatalogEntry]):
    """Return (entry|None, direction, confidence) — direction "SKIP" for a
    deliberate abstain — or None when the response is unusable."""
    by_mint = {e.mint: e for e in catalog}
    match = re.search(r"\{.*\}", response, re.DOTALL)
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    direction = str(data.get("direction", "")).upper()
    if direction == "SKIP":
        return None, "SKIP", 0.0
    mint = data.get("mint", "")
    if mint not in by_mint or direction not in ("UP", "DOWN"):
        return None
    conf = data.get("confidence", 0.5)
    conf = float(conf) if isinstance(conf, (int, float)) else 0.5
    return by_mint[mint], direction, min(1.0, max(0.0, conf))


async def llm_prediction(backend, model: str, catalog: list[CatalogEntry],
                         closes_by_mint: dict[str, list[float]],
                         horizon_minutes: float,
                         micro: str = "n/a", regime: str = "n/a",
                         track: str = "n/a", playbook: str = "n/a") -> Prediction:
    now = time.time()
    prompt = build_predict_prompt(catalog, closes_by_mint, horizon_minutes,
                                  now, micro=micro, regime=regime, track=track,
                                  playbook=playbook)
    response = await backend.complete(prompt)
    parsed = parse_prediction(response, catalog)
    if parsed is None:
        return Prediction(ts=now, arm="llm", mint="", symbol="NONE",
                          direction="", confidence=0.0,
                          horizon_end=now + horizon_minutes * 60.0,
                          price_at=0.0, prompt=prompt, response=response,
                          model=model, backend=backend.name)
    entry, direction, conf = parsed
    if direction == "SKIP":
        return Prediction(ts=now, arm="llm", mint="", symbol="SKIP",
                          direction="SKIP", confidence=0.0,
                          horizon_end=now + horizon_minutes * 60.0,
                          price_at=0.0, prompt=prompt, response=response,
                          model=model, backend=backend.name)
    return Prediction(
        ts=now, arm="llm", mint=entry.mint, symbol=entry.symbol,
        direction=direction, confidence=conf,
        horizon_end=now + horizon_minutes * 60.0,
        price_at=entry.price_usd, prompt=prompt, response=response,
        model=model, backend=backend.name,
    )


def random_prediction(catalog: list[CatalogEntry], horizon_minutes: float,
                      rng: random.Random | None = None) -> Prediction:
    rng = rng or random
    entry = rng.choice(catalog)
    direction = rng.choice(["UP", "DOWN"])
    now = time.time()
    return Prediction(
        ts=now, arm="random", mint=entry.mint, symbol=entry.symbol,
        direction=direction, confidence=0.5,
        horizon_end=now + horizon_minutes * 60.0,
        price_at=entry.price_usd,
        prompt=f"random.choice over {len(catalog)} candidates + coin flip",
        response=f"{entry.symbol} {direction}",
        model="none", backend="random",
    )


def score_resolution(direction: str, price_at: float,
                     price_end: float) -> tuple[float, bool]:
    """Return (raw_return_pct, correct). A flat price counts as incorrect —
    the forecaster claimed a direction and none materialized."""
    if price_at <= 0:
        return 0.0, False
    ret = (price_end - price_at) / price_at * 100.0
    signed = ret if direction == "UP" else -ret
    return ret, signed > 0.0
