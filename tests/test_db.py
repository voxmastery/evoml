import time

from memescalp.db import Database
from memescalp.models import Decision, FeeBreakdown, Fill


def fill(side="buy", trade_id="t1", balance_after=0.0, pnl=None, ts=None):
    return Fill(
        ts=ts or time.time(), arm="llm", trade_id=trade_id, side=side,
        mint="M1", symbol="TEST", feed_price=0.001, exec_price=0.00105,
        size_usd=23.0, token_qty=21000.0,
        fees=FeeBreakdown(lp=0.0575, slippage=0.01, priority=0.04, tds=0.23),
        realized_pnl=pnl, balance_after=balance_after, stop_mode="no_stop",
    )


def test_fills_round_trip_all_fee_components(tmp_path):
    db = Database(tmp_path / "t.db")
    db.insert_fill(fill())
    row = db.fills("llm")[0]
    assert row["fee_lp"] == 0.0575
    assert row["fee_slippage"] == 0.01
    assert row["fee_priority"] == 0.04
    assert row["fee_tds"] == 0.23
    assert row["realized_pnl"] is None


def test_open_position_detected_from_last_fill(tmp_path):
    db = Database(tmp_path / "t.db")
    assert db.open_position("llm") is None
    db.insert_fill(fill(side="buy"))
    pos = db.open_position("llm")
    assert pos is not None and pos.mint == "M1"
    db.insert_fill(fill(side="sell", balance_after=22.7, pnl=-0.3))
    assert db.open_position("llm") is None
    assert db.latest_balance("llm", 99.0) == 22.7


def test_latest_balance_default_when_empty(tmp_path):
    db = Database(tmp_path / "t.db")
    assert db.latest_balance("llm", 22.99) == 22.99


def test_decision_logged_verbatim(tmp_path):
    db = Database(tmp_path / "t.db")
    d = Decision(ts=1.0, arm="llm", window_start=1.0, window_end=1801.0,
                 mint="M1", symbol="TEST", direction="long",
                 prompt="PROMPT\nwith newlines", response='{"mint": "M1"}',
                 model="claude-opus-5", backend="claude_cli")
    db.insert_decision(d)
    row = db.decisions()[0]
    assert row["prompt"] == "PROMPT\nwith newlines"
    assert row["response"] == '{"mint": "M1"}'


def test_price_and_liquidity_cache(tmp_path):
    db = Database(tmp_path / "t.db")
    db.insert_price(1.0, "M1", "TEST", 0.001, "jupiter")
    db.insert_price(2.0, "M1", "TEST", 0.002, "jupiter")
    db.insert_liquidity(1.0, "M1", "TEST", 50_000.0, 1e6, 1.0, -2.0)
    assert db.latest_price("M1") == (2.0, 0.002)
    assert db.latest_liquidity("M1") == 50_000.0
