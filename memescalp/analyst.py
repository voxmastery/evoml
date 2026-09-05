"""Chart analyst: Claude reads the held coin's live price series and predicts
whether to keep holding or exit. Predictions only ever drive PAPER exits."""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass

from .indicators import ema, realized_vol_pct, rsi
from .models import Position

ANALYSIS_PROMPT = """\
You are managing an OPEN paper-trading position in a Solana memecoin and must
decide whether to HOLD or EXIT right now, based on the live chart below.

Position: {symbol} | entered {minutes_held:.0f} min ago | entry cost ${entry_cost:.2f}
Unrealized PnL if exited now (net of ALL costs): ${unrealized:+.2f}
Profit target: +${target:.2f} net (exits automatically when reached)
Stop mode: {stop_mode}

Last {n} one-minute closes (oldest -> newest):
{closes}

Indicators (1-minute closes):
- last price: {last:.8g} | vs 5-min ago: {chg5:+.2f}% | vs 30-min ago: {chg30:+.2f}%
- EMA5: {ema5} | EMA20: {ema20} | trend: {trend}
- RSI14: {rsi14}
- 1-min volatility: {vol:.2f}% | session high: {hi:.8g} | session low: {lo:.8g}

Exiting costs ~1.5-3% in fees, so exit only when the chart genuinely argues the
position deteriorates from here; otherwise hold for the profit target.

Respond with ONLY a JSON object, no other text:
{{"action": "HOLD" or "EXIT", "confidence": 0.0-1.0, "reasoning": "<2-3 sentences>"}}
"""


@dataclass(frozen=True)
class Analysis:
    ts: float
    arm: str
    mint: str
    symbol: str
    action: str          # "HOLD" | "EXIT"
    confidence: float
    prompt: str
    response: str
    model: str
    backend: str


def build_analysis_prompt(position: Position, closes: list[float],
                          unrealized: float, target: float, stop_mode: str,
                          now: float | None = None) -> str:
    now = time.time() if now is None else now
    closes = closes[-60:]
    fmt = lambda v: f"{v:.8g}" if v is not None else "n/a"
    last = closes[-1]
    chg = lambda n: ((last - closes[-n - 1]) / closes[-n - 1] * 100.0
                     if len(closes) > n and closes[-n - 1] > 0 else 0.0)
    e5, e20 = ema(closes, 5), ema(closes, 20)
    trend = ("up" if e5 > e20 else "down") if e5 is not None and e20 is not None \
        else "n/a"
    return ANALYSIS_PROMPT.format(
        symbol=position.symbol,
        minutes_held=(now - position.entry_ts) / 60.0,
        entry_cost=position.entry_cost_usd,
        unrealized=unrealized, target=target, stop_mode=stop_mode,
        n=len(closes),
        closes=", ".join(f"{c:.8g}" for c in closes),
        last=last, chg5=chg(5), chg30=chg(30),
        ema5=fmt(e5), ema20=fmt(e20), trend=trend,
        rsi14=fmt(rsi(closes)), vol=realized_vol_pct(closes),
        hi=max(closes), lo=min(closes),
    )


def parse_analysis(response: str) -> tuple[str, float]:
    """Return (action, confidence). Anything unparseable is a safe HOLD."""
    match = re.search(r"\{.*\}", response, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group(0))
            action = str(data.get("action", "")).upper()
            if action in ("HOLD", "EXIT"):
                conf = data.get("confidence", 0.5)
                conf = float(conf) if isinstance(conf, (int, float)) else 0.5
                return action, min(1.0, max(0.0, conf))
        except (json.JSONDecodeError, TypeError, ValueError):
            pass
    return "HOLD", 0.0


async def analyze_position(backend, model: str, position: Position,
                           closes: list[float], unrealized: float,
                           target: float, stop_mode: str) -> Analysis:
    prompt = build_analysis_prompt(position, closes, unrealized, target,
                                   stop_mode)
    response = await backend.complete(prompt)
    action, confidence = parse_analysis(response)
    return Analysis(
        ts=time.time(), arm="llm", mint=position.mint, symbol=position.symbol,
        action=action, confidence=confidence, prompt=prompt, response=response,
        model=model, backend=backend.name,
    )
