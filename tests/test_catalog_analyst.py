"""Catalog metrics/ranking, indicators, and analyst parsing + predict-exit."""
import asyncio
import time

import pytest

from memescalp.analyst import build_analysis_prompt, parse_analysis
from memescalp.catalog import build_catalog, est_round_trip_cost_pct
from memescalp.indicators import ema, one_minute_closes, realized_vol_pct, rsi
from memescalp.models import Position, TokenSnapshot
from tests.test_strategy import FakeFeed, decision, make_arm, tick


def snap(mint, liq, vol24=1e6, chg1=0.0):
    return TokenSnapshot(mint=mint, symbol=mint[:4], price_usd=0.01,
                         liquidity_usd=liq, volume_h24=vol24,
                         price_change_h1=chg1, decimals=6)


def test_one_minute_closes_buckets_and_orders():
    prices = [(0.0, 1.0), (30.0, 1.1), (61.0, 1.2), (150.0, 1.5)]
    assert one_minute_closes(prices) == [1.1, 1.2, 1.5]


def test_ema_and_rsi_bounds():
    up = list(range(1, 40))
    assert ema(up, 5) is not None and ema(up, 5) > ema(up, 20)
    assert rsi([float(x) for x in up]) == 100.0
    assert rsi([1.0, 2.0]) is None
    assert 0.0 <= rsi([10, 9, 11, 8, 12, 7, 13, 6, 14, 5, 15, 4, 16, 3, 17, 2]) <= 100.0


def test_realized_vol_zero_without_data():
    assert realized_vol_pct([1.0]) == 0.0
    assert realized_vol_pct([1.0, 1.0, 1.0, 1.0]) == 0.0


def test_est_cost_reflects_pool_depth(settings):
    deep = est_round_trip_cost_pct(settings, 300_000, 23.0)
    thin = est_round_trip_cost_pct(settings, 2_000, 23.0)
    assert thin > deep
    # Deep pool: 2 * (0.25% lp + ~0% impact + 1% tds + 0.17% priority) ≈ 2.9%
    assert 2.0 < deep < 4.0
    assert est_round_trip_cost_pct(settings, 0, 23.0) == 100.0


def test_catalog_ranks_deep_active_pools_higher(settings):
    snaps = [snap("THIN_POOL_MINT", liq=2_000, vol24=2e6),
             snap("DEEP_POOL_MINT", liq=300_000, vol24=5e6)]
    cat = build_catalog(settings, snaps, {}, 23.0)
    assert cat[0].mint == "DEEP_POOL_MINT"
    assert cat[0].rank == 1 and cat[1].rank == 2
    assert cat[0].score > cat[1].score


def test_analysis_prompt_contains_chart_and_indicators():
    pos = Position(trade_id="t", mint="M1", symbol="TEST", token_qty=100.0,
                   entry_cost_usd=22.99, entry_ts=time.time() - 600,
                   stop_mode="no_stop")
    closes = [1.0 + i * 0.01 for i in range(40)]
    p = build_analysis_prompt(pos, closes, unrealized=-0.42, target=1.5,
                              stop_mode="no_stop")
    assert "TEST" in p and "$-0.42" in p
    assert "EMA5" in p and "RSI14" in p
    assert "1.39" in p  # newest close present


def test_parse_analysis_actions():
    assert parse_analysis('{"action": "EXIT", "confidence": 0.8, "reasoning": "x"}') == ("EXIT", 0.8)
    assert parse_analysis('text {"action": "hold", "confidence": 0.6} more')[0] == "HOLD"
    assert parse_analysis("garbage") == ("HOLD", 0.0)          # safe default
    assert parse_analysis('{"action": "SELL EVERYTHING"}') == ("HOLD", 0.0)
    assert parse_analysis('{"action": "EXIT", "confidence": 7}') == ("EXIT", 1.0)


def test_predict_exit_fires_on_next_tick(settings, tmp_path):
    feed = FakeFeed()
    feed.set("M1", 0.001)
    db, arm = make_arm(settings, tmp_path, feed)
    arm.apply_decision(decision("M1"))
    tick(arm)
    assert arm.position is not None

    arm.request_exit("predict")
    tick(arm)
    assert arm.position is None
    sell = db.fills("llm")[-1]
    assert sell["note"] == "predict"
    assert arm.pending_exit is None  # cleared after the exit executes


def test_request_exit_ignored_when_flat(settings, tmp_path):
    feed = FakeFeed()
    feed.set("M1", 0.001)
    db, arm = make_arm(settings, tmp_path, feed)
    arm.request_exit("predict")
    assert arm.pending_exit is None
