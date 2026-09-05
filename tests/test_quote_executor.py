"""Quote-based fill math, exercised with canned Jupiter quote payloads."""
import asyncio
import dataclasses

import pytest

from memescalp.executor import (
    ExecutionError, buy_swap_units, quote_buy, quote_sell, quote_sell_net,
)
from memescalp.jupiter import JupiterQuoter, QuoteError
from memescalp.models import TokenSnapshot
from memescalp.strategy import TradingArm
from tests.test_strategy import FakeFeed, decision, make_arm


def snap(price=0.02, decimals=6):
    return TokenSnapshot(mint="MINT1", symbol="TEST", price_usd=price,
                         liquidity_usd=100_000.0, decimals=decimals)


def buy_quote(in_usdc, out_tokens, decimals=6, impact="0.05"):
    return {"inAmount": str(int(round(in_usdc * 1e6))),
            "outAmount": str(int(round(out_tokens * 10**decimals))),
            "priceImpactPct": impact}


def sell_quote(out_usdc, impact="0.02"):
    return {"inAmount": "0",
            "outAmount": str(int(round(out_usdc * 1e6))),
            "priceImpactPct": impact}


def test_buy_swap_units_reserves_priority_and_tds(settings):
    units = buy_swap_units(settings, 22.99)
    assert units == int(round((22.99 - 0.04 - 0.2299) * 1e6))


def test_buy_swap_units_rejects_dust(settings):
    with pytest.raises(ExecutionError):
        buy_swap_units(settings, 0.04)


def test_quote_buy_math(settings):
    balance = 22.99
    q = buy_quote(in_usdc=22.7201, out_tokens=1000.0)  # mid value $20 @ 0.02
    fill, pos = quote_buy(settings, "llm", snap(), balance, "no_stop", q)
    assert fill.size_usd == pytest.approx(balance)
    assert fill.balance_after == pytest.approx(0.0)
    assert fill.token_qty == pytest.approx(1000.0)
    assert fill.fees.priority == 0.04
    assert fill.fees.tds == pytest.approx(0.2299)
    assert fill.fees.slippage == pytest.approx(22.7201 * 0.05)
    # lp is the remainder of the real swap cost after the impact share.
    assert fill.fees.lp == pytest.approx(22.7201 - 20.0 - 22.7201 * 0.05)
    assert pos.entry_cost_usd == pytest.approx(balance)


def test_quote_buy_requires_decimals(settings):
    with pytest.raises(ExecutionError):
        quote_buy(settings, "llm", snap(decimals=0), 22.99, "no_stop",
                  buy_quote(22.72, 1000.0))


def test_quote_sell_net_and_pnl(settings):
    balance = 22.99
    _, pos = quote_buy(settings, "llm", snap(), balance, "no_stop",
                       buy_quote(22.7201, 1000.0))
    q = sell_quote(out_usdc=25.0)
    net = quote_sell_net(settings, q)
    assert net == pytest.approx(25.0 - 0.04 - 0.25)
    fill = quote_sell(settings, "llm", pos, snap(), q, balance_usd=0.0)
    assert fill.realized_pnl == pytest.approx(net - balance)
    assert fill.balance_after == pytest.approx(net)
    assert fill.fees.tds == pytest.approx(0.25)


class FakeQuoter(JupiterQuoter):
    """Returns canned quotes; keyed by output mint direction."""

    def __init__(self):  # no HTTP client
        self.buy_response = None
        self.sell_response = None

    async def close(self):
        pass

    async def quote(self, input_mint, output_mint, amount, slippage_bps=100):
        resp = self.sell_response if output_mint.startswith("EPjF") \
            else self.buy_response
        if resp is None:
            raise QuoteError("no canned quote")
        return resp


def test_strategy_quote_mode_enters_and_exits_at_target(settings, tmp_path):
    settings = dataclasses.replace(settings, fill_model="quote")
    feed = FakeFeed()
    feed.prices["M1"] = TokenSnapshot(mint="M1", symbol="TEST", price_usd=0.02,
                                      liquidity_usd=100_000.0, decimals=6)
    quoter = FakeQuoter()
    db, arm = make_arm(settings, tmp_path, feed, quoter=quoter)
    arm.apply_decision(decision("M1"))

    quoter.buy_response = buy_quote(in_usdc=22.7201, out_tokens=1100.0)
    asyncio.run(arm.tick())
    assert arm.position is not None
    assert arm.position.token_qty == pytest.approx(1100.0)

    # Sell quote below target: hold.
    quoter.sell_response = sell_quote(out_usdc=23.0)
    asyncio.run(arm.tick())
    assert arm.position is not None

    # Sell quote clears the +$1.50 net target: exit.
    quoter.sell_response = sell_quote(out_usdc=25.5)
    asyncio.run(arm.tick())
    assert arm.position is None
    sell = db.fills("llm")[-1]
    assert sell["note"] == "target"
    assert sell["realized_pnl"] >= settings.target_profit_usd


def test_strategy_quote_mode_holds_on_quote_failure(settings, tmp_path):
    settings = dataclasses.replace(settings, fill_model="quote")
    feed = FakeFeed()
    feed.prices["M1"] = TokenSnapshot(mint="M1", symbol="TEST", price_usd=0.02,
                                      liquidity_usd=100_000.0, decimals=6)
    quoter = FakeQuoter()  # every quote raises QuoteError
    db, arm = make_arm(settings, tmp_path, feed, quoter=quoter)
    arm.apply_decision(decision("M1"))
    asyncio.run(arm.tick())
    assert arm.position is None       # stayed flat instead of crashing
    assert db.fills("llm") == []
