"""Prediction experiment: parsing, scoring, resolution queries, pass/fail."""
import time

import pytest

from memescalp.catalog import CatalogEntry
from memescalp.db import Database
from memescalp.metrics import (
    evaluate_predictions, prediction_curves, prediction_stats,
)
from memescalp.predictor import (
    Prediction, build_predict_prompt, parse_prediction, random_prediction,
    score_resolution,
)


def entry(mint="MINT_A", symbol="AAA", rank=1):
    return CatalogEntry(mint=mint, symbol=symbol, price_usd=0.01,
                        liquidity_usd=100_000.0, volume_h24=1e6,
                        price_change_h1=1.0, price_change_h24=-2.0,
                        volatility_pct=0.5, est_cost_pct=2.9, score=10.0,
                        rank=rank)


def catalog():
    return [entry(), entry("MINT_B", "BBB", 2)]


def prediction(arm="llm", mint="MINT_A", direction="UP", ts=None,
               horizon_end=None, price_at=0.01):
    ts = ts if ts is not None else time.time()
    return Prediction(ts=ts, arm=arm, mint=mint, symbol=mint[:4],
                      direction=direction, confidence=0.7,
                      horizon_end=horizon_end if horizon_end is not None
                      else ts + 1800, price_at=price_at,
                      prompt="p", response="r", model="m", backend="b")


def test_prompt_contains_catalog_and_charts():
    closes = {"MINT_A": [1.0, 1.01, 1.02, 1.03, 1.04]}
    p = build_predict_prompt(catalog(), closes, 30.0, 1_700_000_000.0)
    assert "MINT_A" in p and "MINT_B" in p
    assert "AAA: 1, 1.01" in p
    assert "BBB: insufficient price history yet" in p
    assert "30 minutes" in p


def test_parse_prediction_valid_and_invalid():
    ok = parse_prediction(
        '{"mint": "MINT_B", "symbol": "BBB", "direction": "down", "confidence": 0.8}',
        catalog())
    assert ok is not None and ok[0].mint == "MINT_B" and ok[1] == "DOWN"
    assert parse_prediction('{"mint": "MINT_B", "direction": "SIDEWAYS"}', catalog()) is None
    assert parse_prediction('{"mint": "UNKNOWN", "direction": "UP"}', catalog()) is None
    assert parse_prediction("no json", catalog()) is None


def test_parse_prediction_skip_is_deliberate_abstain():
    parsed = parse_prediction('{"direction": "SKIP", "reasoning": "pure noise"}',
                              catalog())
    assert parsed == (None, "SKIP", 0.0)


def test_score_resolution_directions():
    ret, correct = score_resolution("UP", 100.0, 105.0)
    assert ret == pytest.approx(5.0) and correct
    ret, correct = score_resolution("DOWN", 100.0, 105.0)
    assert not correct
    ret, correct = score_resolution("DOWN", 100.0, 90.0)
    assert ret == pytest.approx(-10.0) and correct
    assert score_resolution("UP", 100.0, 100.0)[1] is False  # flat = wrong
    assert score_resolution("UP", 0.0, 100.0) == (0.0, False)


def test_db_resolution_flow(settings):
    db = Database(settings.db_path)
    now = 1000.0
    pid = db.insert_prediction(prediction(ts=now, horizon_end=now + 1800))
    # not due yet
    assert db.due_unresolved(now + 100) == []
    due = db.due_unresolved(now + 1801)
    assert len(due) == 1 and due[0]["id"] == pid

    db.insert_price(now + 1795, "MINT_A", "AAA", 0.0099, "jupiter")
    db.insert_price(now + 1805, "MINT_A", "AAA", 0.0107, "jupiter")
    hit = db.price_at_or_after("MINT_A", now + 1800)
    assert hit == (now + 1805, 0.0107)

    db.insert_resolution(pid, now + 1810, 0.0107, 7.0, True, "resolved")
    assert db.due_unresolved(now + 4000) == []
    rows = db.resolved_predictions("llm")
    assert len(rows) == 1 and rows[0]["correct"] == 1
    ledger = db.prediction_ledger()
    assert ledger[0]["status"] == "resolved"


def test_parse_failed_predictions_never_resolve(settings):
    db = Database(settings.db_path)
    p = prediction(direction="")  # unusable model output
    db.insert_prediction(p)
    assert db.due_unresolved(p.horizon_end + 10_000) == []


def resolved_row(correct, direction="UP", ret=1.0, ts=0.0, conf=0.7):
    return {"status": "resolved", "correct": correct, "direction": direction,
            "return_pct": ret if direction == "UP" else -ret,
            "confidence": conf, "resolved_ts": ts}


def test_prediction_stats_and_curves():
    rows = [resolved_row(True, ts=1), resolved_row(False, ts=2),
            resolved_row(True, "DOWN", 2.0, ts=3),
            {"status": "void", "correct": 0, "direction": "UP",
             "return_pct": 0.0, "confidence": 0.5, "resolved_ts": 4}]
    s = prediction_stats(rows)
    assert s.resolved == 3 and s.correct == 2 and s.voids == 1
    assert s.accuracy == pytest.approx(2 / 3)
    curves = prediction_curves(rows)
    assert curves["accuracy"][-1][1] == pytest.approx(200 / 3)
    assert len(curves["cum_return"]) == 3


def test_evaluate_predictions_pass_and_fail():
    day = 86400.0
    llm = [resolved_row(i % 3 != 0, ts=i) for i in range(250)]     # ~66.8%
    rnd = [resolved_row(i % 2 == 0, ts=i) for i in range(250)]     # 50%
    r = evaluate_predictions(llm, rnd, 0.0, 200, 14, 1.64, now=15 * day)
    assert r.skilled and r.enough_predictions and r.enough_days
    assert r.beats_control and r.overall

    early = evaluate_predictions(llm, rnd, 0.0, 200, 14, 1.64, now=2 * day)
    assert not early.enough_days and not early.overall

    same = evaluate_predictions(rnd, rnd, 0.0, 200, 14, 1.64, now=15 * day)
    assert not same.skilled and not same.beats_control and not same.overall


def test_random_prediction_shape():
    import random as _r
    p = random_prediction(catalog(), 30.0, rng=_r.Random(7))
    assert p.arm == "random" and p.direction in ("UP", "DOWN")
    assert p.mint in ("MINT_A", "MINT_B")
    assert p.horizon_end > p.ts


def test_capital_progression_compounds_and_maps_ids():
    from memescalp.metrics import capital_progression
    rows = [
        {"status": "resolved", "direction": "UP", "return_pct": 10.0,
         "resolved_ts": 1.0, "prediction_id": 11},
        {"status": "resolved", "direction": "DOWN", "return_pct": -5.0,
         "resolved_ts": 2.0, "prediction_id": 12},   # DOWN call, price fell: gain
        {"status": "void", "direction": "UP", "return_pct": 0.0,
         "resolved_ts": 3.0, "prediction_id": 13},
        {"status": "resolved", "direction": "UP", "return_pct": -20.0,
         "resolved_ts": 4.0, "prediction_id": 14},
    ]
    prog = capital_progression(rows, 100.0)
    assert prog["by_id"][11]["usd_pnl"] == pytest.approx(10.0)
    assert prog["by_id"][12]["usd_pnl"] == pytest.approx(5.5)      # 110 * 5%
    assert 13 not in prog["by_id"]                                 # void skipped
    assert prog["by_id"][14]["usd_pnl"] == pytest.approx(-23.1)    # 115.5 * -20%
    assert prog["final"] == pytest.approx(92.4)
    assert [round(c, 1) for _, c in prog["curve"]] == [110.0, 115.5, 92.4]


def test_kelly_progression_warmup_and_edge_gating():
    from memescalp.metrics import kelly_progression
    # 10 warmup wins (no bets), then wins with a hot trailing record -> bets.
    rows = []
    for i in range(10):
        rows.append({"status": "resolved", "direction": "UP", "return_pct": 2.0,
                     "resolved_ts": float(i), "prediction_id": i, "correct": 1})
    rows.append({"status": "resolved", "direction": "UP", "return_pct": 2.0,
                 "resolved_ts": 10.0, "prediction_id": 10, "correct": 1})
    prog = kelly_progression(rows, 100.0)
    # Warmup: zero fraction, capital untouched.
    assert all(prog["by_id"][i]["fraction"] == 0.0 for i in range(10))
    assert prog["by_id"][10]["fraction"] == 0.25          # perfect record -> cap
    assert prog["by_id"][10]["usd_pnl"] == pytest.approx(100.0 * 0.25 * 0.02)
    assert prog["final"] == pytest.approx(100.5)


def test_kelly_bets_zero_without_edge():
    from memescalp.metrics import kelly_progression
    rows = []
    for i in range(40):  # alternating results: trailing p = 0.5 -> f = 0
        rows.append({"status": "resolved", "direction": "UP", "return_pct": 5.0,
                     "resolved_ts": float(i), "prediction_id": i,
                     "correct": i % 2})
    prog = kelly_progression(rows, 100.0)
    assert prog["final"] == pytest.approx(100.0)          # never bet


def test_calibration_brier_and_bins():
    from memescalp.metrics import calibration
    rows = [
        {"status": "resolved", "confidence": 0.6, "correct": 1, "return_pct": 1.0},
        {"status": "resolved", "confidence": 0.6, "correct": 0, "return_pct": -1.0},
        {"status": "resolved", "confidence": 0.8, "correct": 1, "return_pct": 2.0},
        {"status": "void", "confidence": 0.9, "correct": 0, "return_pct": 0.0},
    ]
    cal = calibration(rows)
    assert cal["n"] == 3
    expected = ((0.6 - 1) ** 2 + (0.6 - 0) ** 2 + (0.8 - 1) ** 2) / 3
    assert cal["brier"] == pytest.approx(expected)
    labels = {b["label"]: b for b in cal["bins"]}
    assert labels["55-65%"]["n"] == 2
    assert labels["55-65%"]["accuracy"] == pytest.approx(0.5)
    assert labels["75%+"]["accuracy"] == 1.0
    assert calibration([])["brier"] is None


def test_flat_resolutions_are_void_not_wrong():
    from memescalp.metrics import kelly_progression, prediction_stats
    rows = [
        resolved_row(True, ts=1),
        {"status": "resolved", "correct": 0, "direction": "UP",
         "return_pct": 0.0, "confidence": 0.6, "resolved_ts": 2,
         "prediction_id": 2},  # stale flat quote: must not count as wrong
        resolved_row(False, ts=3),
    ]
    s = prediction_stats(rows)
    assert s.resolved == 2 and s.correct == 1 and s.voids == 1
    assert s.accuracy == pytest.approx(0.5)
    prog = kelly_progression(rows, 100.0)
    assert prog["final"] == pytest.approx(100.0)


def test_prompt_includes_flow_regime_and_track():
    from dataclasses import replace
    cat = [replace(entry(), buys_m5=12, sells_m5=3, vol_m5=5000.0, chg_m5=1.2)]
    p = build_predict_prompt(cat, {}, 2.0, 1_700_000_000.0,
                             micro="AAA: 30s +1%", regime="SOL $200 | 1h +0.5%",
                             track="AAA UP -> CORRECT (+2%)")
    assert "flow5m 12buys/3sells" in p
    assert "SOL $200" in p
    assert "track record" in p and "CORRECT (+2%)" in p


def test_earn_stats_math():
    from memescalp.metrics import earn_stats
    good = [resolved_row(i % 5 != 0, ts=i) for i in range(100)]   # 80% acc
    e = earn_stats(good, calls_per_hour=12.0, fee=0.01)
    assert e["ready"] and e["accuracy"] == pytest.approx(0.8)
    assert e["edge_per_call"] == pytest.approx(0.59)
    assert e["kelly_fraction"] == pytest.approx(0.295)
    hourly_frac = 0.295 * 0.59 * 12.0
    assert e["usd_per_hour"]["23"] == pytest.approx(23 * hourly_frac)
    assert e["bankroll_for_60_per_hour"] == pytest.approx(60 / hourly_frac)

    coinflip = [resolved_row(i % 2 == 0, ts=i) for i in range(100)]
    e2 = earn_stats(coinflip, calls_per_hour=12.0)
    assert e2["kelly_fraction"] == 0.0
    assert e2["bankroll_for_60_per_hour"] is None   # no size fixes no skill
    assert earn_stats(good[:10], 12.0) == {"n": 10, "ready": False}
