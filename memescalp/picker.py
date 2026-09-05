"""Coin pickers: the LLM hypothesis arm and the random control arm."""
from __future__ import annotations

import json
import logging
import random
import re
import time

from .models import Decision

log = logging.getLogger(__name__)

PROMPT_TEMPLATE = """\
You are picking ONE Solana memecoin for a 30-minute paper-trading window in a
simulation. The strategy will buy it with the full paper balance (~$25) and
scalp a small fixed profit target, re-entering after each exit. While the
position is open you will also be asked periodically whether to hold or exit
based on the live chart.

Candidate catalog (live data as of {now_iso} UTC), ranked by a heuristic
scalpability score (turnover + short-term volatility - momentum extremes -
round-trip cost). The score is advisory — make your own call from the metrics:
{table}

est_cost = estimated full round-trip cost for our position size, as % of the
position. vol_1m = stddev of 1-minute returns. turnover = 24h volume / pool
liquidity.

Pick exactly one token from the catalog. Favor pools deep enough that a ~$25
position exits cleanly, with enough per-minute movement to reach the target.

Respond with ONLY a JSON object, no other text:
{{"mint": "<mint address from the list>", "symbol": "<its symbol>", "reasoning": "<2-3 sentences>"}}
"""


def build_prompt(candidates: list, now: float) -> str:
    lines = []
    for c in candidates:
        line = (f"- #{getattr(c, 'rank', '?')} {c.symbol} | mint: {c.mint} | "
                f"price ${c.price_usd:.8g} | liquidity ${c.liquidity_usd:,.0f} | "
                f"24h volume ${c.volume_h24:,.0f} | "
                f"1h {c.price_change_h1:+.2f}% | 24h {c.price_change_h24:+.2f}%")
        if hasattr(c, "score"):
            line += (f" | turnover {c.volume_h24 / c.liquidity_usd:.1f}x"
                     if c.liquidity_usd > 0 else " | turnover n/a")
            line += (f" | vol_1m {c.volatility_pct:.2f}% | "
                     f"est_cost {c.est_cost_pct:.2f}% | score {c.score:.1f}")
        lines.append(line)
    return PROMPT_TEMPLATE.format(
        now_iso=time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(now)),
        table="\n".join(lines),
    )


def parse_pick(response: str, candidates: list):
    """Extract the chosen token from the model's response; None if unusable."""
    by_mint = {c.mint: c for c in candidates}
    match = re.search(r"\{.*\}", response, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group(0))
            mint = data.get("mint", "")
            if mint in by_mint:
                return by_mint[mint]
        except (json.JSONDecodeError, AttributeError, TypeError):
            pass
    # Last resort: a candidate mint quoted anywhere in the text.
    for mint, snap in by_mint.items():
        if mint in response:
            return snap
    return None


async def llm_pick(
    backend, model: str, candidates: list,
    window_start: float, window_end: float,
) -> Decision:
    """Ask Claude for a pick. Prompt and response are logged verbatim."""
    prompt = build_prompt(candidates, window_start)
    response = await backend.complete(prompt)
    chosen = parse_pick(response, candidates)
    return Decision(
        ts=time.time(), arm="llm",
        window_start=window_start, window_end=window_end,
        mint=chosen.mint if chosen else "",
        symbol=chosen.symbol if chosen else "NONE",
        direction="long" if chosen else "none",
        prompt=prompt, response=response,
        model=model, backend=backend.name,
    )


def random_pick(
    candidates: list, window_start: float, window_end: float,
    rng: random.Random | None = None,
) -> Decision:
    """Control arm: uniform random choice over the identical candidate list."""
    rng = rng or random
    chosen = rng.choice(candidates)
    return Decision(
        ts=time.time(), arm="random",
        window_start=window_start, window_end=window_end,
        mint=chosen.mint, symbol=chosen.symbol, direction="long",
        prompt=f"random.choice over {len(candidates)} candidates: "
               + ", ".join(c.symbol for c in candidates),
        response=f"picked {chosen.symbol} ({chosen.mint})",
        model="none", backend="random",
    )
