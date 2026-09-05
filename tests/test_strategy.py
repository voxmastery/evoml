import asyncio
import dataclasses
import time

from memescalp.csvlog import CsvMirror
from memescalp.db import Database
from memescalp.models import Decision, TokenSnapshot
from memescalp.strategy import TradingArm


class FakeFeed:
    def __init__(self):
        self.prices = {}

    def set(self, mint, price, liq=50_000.0, symbol="TEST"):
        self.prices[mint] = TokenSnapshot(mint=mint, symbol=symbol,
                                          price_usd=price, liquidity_usd=liq)

    def snapshot(self, mint):
        return self.prices.get(mint)


def decision(mint, symbol="TEST", arm="llm"):
    now = time.time()
    return Decision(ts=now, arm=arm, window_start=now, window_end=now + 1800,
                    mint=mint, symbol=symbol, direction="long", prompt="p",
                    response="r", model="m", backend="b")


def make_arm(settings, tmp_path, feed, quoter=None):
    db = Database(tmp_path / "s.db")
    csv = CsvMirror(tmp_path / "csv")
    return db, TradingArm(settings, db, csv, feed, "llm", quoter=quoter)


def tick(arm, ts=None):
    asyncio.run(arm.tick(ts))


def test_enters_on_pick_and_holds_until_target(settings, tmp_path):
    feed = FakeFeed()
    feed.set("M1", 0.001)
    db, arm = make_arm(settings, tmp_path, feed)
    arm.apply_decision(decision("M1"))

    tick(arm)
    assert arm.position is not None
    assert arm.balance == 0.0

    # Flat price: costs mean we are below target — no exit.
    tick(arm)
    assert arm.position is not None

    # Big pump: unrealized net PnL clears the +$1.50 target.
    feed.set("M1", 0.0015)
    tick(arm)
    assert arm.position is None
    fills = db.fills("llm")
    assert [f["side"] for f in fills] == ["buy", "sell"]
    assert fills[-1]["note"] == "target"
    assert fills[-1]["realized_pnl"] >= settings.target_profit_usd


def test_reenters_after_exit(settings, tmp_path):
    feed = FakeFeed()
    feed.set("M1", 0.001)
    db, arm = make_arm(settings, tmp_path, feed)
    arm.apply_decision(decision("M1"))
    tick(arm)
    feed.set("M1", 0.0015)
    tick(arm)          # exits at target
    tick(arm)          # same pick still active -> re-enters
    assert arm.position is not None
    assert len(db.fills("llm")) == 3


def test_no_stop_mode_holds_through_drawdown(settings, tmp_path):
    feed = FakeFeed()
    feed.set("M1", 0.001)
    db, arm = make_arm(settings, tmp_path, feed)
    arm.apply_decision(decision("M1"))
    tick(arm)
    feed.set("M1", 0.0002)  # -80%
    tick(arm)
    assert arm.position is not None  # no stop: keeps holding
    assert db.fills("llm")[0]["stop_mode"] == "no_stop"


def test_stop_loss_mode_exits_and_is_flagged(settings, tmp_path):
    settings = dataclasses.replace(settings, stop_loss_pct=5.0)
    feed = FakeFeed()
    feed.set("M1", 0.001)
    db, arm = make_arm(settings, tmp_path, feed)
    arm.apply_decision(decision("M1"))
    tick(arm)
    feed.set("M1", 0.0009)  # -10%: past the 5% stop
    tick(arm)
    assert arm.position is None
    sell = db.fills("llm")[-1]
    assert sell["note"] == "stop"
    assert sell["stop_mode"] == "sl_5pct"
    assert sell["realized_pnl"] < 0


def test_rotates_when_new_window_picks_other_token(settings, tmp_path):
    feed = FakeFeed()
    feed.set("M1", 0.001, symbol="AAA")
    feed.set("M2", 0.02, symbol="BBB")
    db, arm = make_arm(settings, tmp_path, feed)
    arm.apply_decision(decision("M1", "AAA"))
    tick(arm)
    arm.apply_decision(decision("M2", "BBB"))
    tick(arm)   # exits M1 (rotate)
    tick(arm)   # enters M2 on the next tick
    fills = db.fills("llm")
    assert [f["side"] for f in fills] == ["buy", "sell", "buy"]
    assert fills[1]["note"] == "rotate"
    assert arm.position.mint == "M2"


def test_resume_restores_balance_and_open_position(settings, tmp_path):
    feed = FakeFeed()
    feed.set("M1", 0.001)
    db, arm = make_arm(settings, tmp_path, feed)
    arm.apply_decision(decision("M1"))
    tick(arm)
    qty = arm.position.token_qty

    # Simulate a process restart: a new arm built on the same database.
    arm2 = TradingArm(settings, db, CsvMirror(tmp_path / "csv"), feed, "llm")
    assert arm2.position is not None
    assert arm2.position.token_qty == qty
    assert arm2.balance == 0.0
    assert arm2.pick_mint == "M1"


def test_arm_halts_when_busted(settings, tmp_path):
    feed = FakeFeed()
    feed.set("M1", 0.001)
    db, arm = make_arm(settings, tmp_path, feed)
    arm.balance = 0.5  # below tradable minimum
    arm.apply_decision(decision("M1"))
    tick(arm)
    assert arm.busted
    assert db.fills("llm") == []
