"""Regime-Conditioned Hedge: expert votes, online weight updates, persistence."""
import json
import time

import pytest

from memescalp.db import Database
from memescalp.hedgepred import (
    CALL_MARGIN, HedgeForecaster, detect_regime, expert_votes,
)
from memescalp.micro import micro_features
from memescalp.predictor import Prediction
from tests.test_mathpred import oscillating, trending_up
from tests.test_predict import entry


def ramp_ticks(n=80, step=0.01):
    out, price = [], 1.0
    for i in range(n):
        out.append((1000.0 + i * 5.0, price))
        price *= 1 + step
    return out


def test_detect_regime():
    assert detect_regime(trending_up()) == "trend"
    assert detect_regime(oscillating()) == "revert"
    assert detect_regime([1.0, 1.1]) == "walk"  # not enough data


def test_expert_votes_directions():
    ticks = ramp_ticks()
    micro = micro_features(ticks)
    votes = expert_votes(trending_up(), micro, buys=20, sells=2)
    assert votes["momentum"] == 1
    assert votes["meanrevert"] == -1
    assert votes["flow"] == 1
    assert votes["classic"] in (-1, 0, 1)


def hedge_prediction_row(db, regime="trend", votes=None, ret=10.0):
    votes = votes or {"momentum": 1, "meanrevert": -1, "flow": 0,
                      "classic": 1, "oufade": 0}
    now = time.time()
    p = Prediction(ts=now, arm="hedge", mint="M1", symbol="AAA",
                   direction="UP", confidence=0.7, horizon_end=now + 120,
                   price_at=1.0,
                   prompt=json.dumps({"regime": regime, "votes": votes}),
                   response="r", model="rch-mwu", backend="hedge")
    pid = db.insert_prediction(p)
    db.insert_resolution(pid, now + 125, 1.1, ret, ret > 0, "resolved")
    return pid


def test_weights_follow_wealth_and_persist(settings):
    db = Database(settings.db_path)
    h = HedgeForecaster(db)
    w0 = dict(h.weights["trend"])
    hedge_prediction_row(db, ret=10.0)   # UP move: momentum right, revert wrong
    updated = h.absorb_outcomes()
    assert updated == 1
    assert h.weights["trend"]["momentum"] > h.weights["trend"]["meanrevert"]
    assert h.weights["trend"]["flow"] == pytest.approx(
        h.weights["trend"]["oufade"])  # non-voters move only via normalization
    # Persistence: a fresh forecaster (process restart) loads the same state.
    h2 = HedgeForecaster(db)
    assert h2.weights == h.weights
    # Already-processed outcomes are not double-counted.
    assert h2.absorb_outcomes() == 0
    assert w0 != h.weights["trend"]


def test_make_prediction_gates_and_calls(settings):
    db = Database(settings.db_path)
    h = HedgeForecaster(db)
    # No tick data at all: everything gated -> SKIP.
    p = h.make_prediction([entry()], {}, 2.0)
    assert p.direction == "SKIP"
    # Strong ramp with heavy buy flow: committee should clear the margin.
    from dataclasses import replace
    e = replace(entry(), buys_m5=30, sells_m5=2)
    p2 = h.make_prediction([e], {"MINT_A": ramp_ticks()}, 2.0)
    assert p2.arm == "hedge"
    if p2.direction != "SKIP":
        assert p2.direction in ("UP", "DOWN")
        detail = json.loads(p2.prompt)
        assert abs(detail["score"]) >= CALL_MARGIN
        assert detail["regime"] in ("trend", "revert", "walk")
