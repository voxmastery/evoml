import pytest

from memescalp.executor import (
    ExecutionError, sell_proceeds, simulate_buy, simulate_sell,
)
from memescalp.models import TokenSnapshot


def snap(price=0.001, liq=50_000.0):
    return TokenSnapshot(mint="MINT1", symbol="TEST", price_usd=price,
                         liquidity_usd=liq)


def test_buy_spends_full_balance_and_fees_reduce_tokens(settings):
    balance = settings.start_balance_usd
    fill, pos = simulate_buy(settings, "llm", snap(), balance, "no_stop")
    assert fill.side == "buy"
    assert fill.size_usd == pytest.approx(balance)
    assert fill.balance_after == pytest.approx(0.0)
    net = balance - fill.fees.total
    assert fill.token_qty == pytest.approx(net / 0.001)
    assert pos.entry_cost_usd == pytest.approx(balance)
    # Effective price is worse than the feed price because of costs.
    assert fill.exec_price > fill.feed_price


def test_round_trip_at_flat_price_loses_exactly_the_fees(settings):
    balance = settings.start_balance_usd
    buy, pos = simulate_buy(settings, "llm", snap(), balance, "no_stop")
    sell = simulate_sell(settings, "llm", pos, snap(), buy.balance_after)
    total_fees = buy.fees.total + sell.fees.total
    assert sell.realized_pnl == pytest.approx(-total_fees, rel=1e-9)
    assert sell.balance_after == pytest.approx(balance - total_fees, rel=1e-9)


def test_profit_when_price_rises_enough(settings):
    balance = settings.start_balance_usd
    buy, pos = simulate_buy(settings, "llm", snap(price=0.001), balance, "no_stop")
    up = snap(price=0.0012)  # +20%
    net, _ = sell_proceeds(settings, pos, up)
    sell = simulate_sell(settings, "llm", pos, up, buy.balance_after)
    assert sell.realized_pnl == pytest.approx(net - balance)
    assert sell.realized_pnl > 0


def test_buy_rejected_when_balance_below_priority_fee(settings):
    with pytest.raises(ExecutionError):
        simulate_buy(settings, "llm", snap(), 0.03, "no_stop")


def test_buy_rejected_on_zero_price(settings):
    with pytest.raises(ExecutionError):
        simulate_buy(settings, "llm", snap(price=0.0), 23.0, "no_stop")


def test_sell_rejects_mismatched_token(settings):
    _, pos = simulate_buy(settings, "llm", snap(), 23.0, "no_stop")
    other = TokenSnapshot(mint="OTHER", symbol="X", price_usd=1.0,
                          liquidity_usd=1000.0)
    with pytest.raises(ExecutionError):
        simulate_sell(settings, "llm", pos, other, 0.0)
