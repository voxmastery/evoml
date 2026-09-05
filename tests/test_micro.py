"""Micro-structure features and the live hot-board ranking."""
import pytest

from memescalp.micro import live_rank, live_score, micro_features, micro_prompt_lines
from memescalp.models import TokenSnapshot
from tests.test_predict import catalog


def ticks_ramp(n=60, start=1000.0, step_pct=0.001, dt=5.0):
    out, price = [], 1.0
    for i in range(n):
        out.append((start + i * dt, price))
        price *= 1 + step_pct
    return out


def test_micro_features_on_steady_ramp():
    f = micro_features(ticks_ramp())
    assert f is not None
    assert f.ret_30s > 0 and f.ret_60s > f.ret_30s / 2
    assert f.ret_300s > f.ret_60s
    assert f.uptick_ratio == 1.0
    assert f.micro_vol >= 0.0


def test_micro_features_insufficient_ticks():
    assert micro_features([(0, 1.0), (5, 1.01)]) is None


def test_live_score_prefers_active_deep_cheap():
    f_hot = micro_features(ticks_ramp(step_pct=0.01))
    f_flat = micro_features([(i * 5.0, 1.0) for i in range(60)])
    hot = live_score(f_hot, 300_000, 2.9)
    flat = live_score(f_flat, 300_000, 2.9)
    assert hot > flat
    # Shallow pool takes a hard penalty.
    assert live_score(f_hot, 2_000, 2.9) < live_score(f_hot, 300_000, 2.9)


def test_live_rank_orders_and_annotates(settings):
    snaps = [
        TokenSnapshot(mint="HOT", symbol="HOT", price_usd=1.0,
                      liquidity_usd=200_000),
        TokenSnapshot(mint="FLAT", symbol="FLAT", price_usd=1.0,
                      liquidity_usd=200_000),
    ]
    ticks = {"HOT": ticks_ramp(step_pct=0.01), "FLAT": [(i * 5.0, 1.0) for i in range(60)]}
    board = live_rank(settings, snaps, ticks, 23.0)
    assert [r["symbol"] for r in board] == ["HOT", "FLAT"]
    assert board[0]["rank"] == 1
    assert board[0]["live_score"] > board[1]["live_score"]


def test_micro_prompt_lines_cover_top_candidates():
    ticks = {"MINT_A": ticks_ramp()}
    text = micro_prompt_lines(catalog(), ticks)
    assert "AAA:" in text and "upticks" in text
    assert "BBB: no tick data yet" in text
