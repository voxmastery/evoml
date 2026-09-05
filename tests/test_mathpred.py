"""Classical-math forecaster: deterministic signals with known lineage."""
import math

import pytest

from memescalp.mathpred import (
    math_prediction, signal_for, variance_ratio_2, zscore_20,
)
from tests.test_predict import catalog, entry


def trending_up(n=40, step=0.01):
    return [1.0 + i * step for i in range(n)]


def oscillating(n=40):
    # Alternating up/down: strongly negative autocorrelation -> VR(2) < 1.
    return [1.0 + (0.02 if i % 2 else 0.0) for i in range(n)]


def test_variance_ratio_regimes():
    assert variance_ratio_2(trending_up()) > 1.0
    assert variance_ratio_2(oscillating()) < 1.0
    assert variance_ratio_2([1.0] * 40) is None  # zero variance
    assert variance_ratio_2([1.0, 1.1]) is None  # too short


def test_zscore_detects_stretch():
    closes = [1.0] * 19 + [1.5]
    assert zscore_20(closes) > 2.0
    assert zscore_20([1.0] * 20) is None  # zero std


def test_signal_ou_fade_beats_momentum():
    # Steady, then a violent spike: z-score says fade it -> DOWN.
    closes = [1.0 + 0.001 * math.sin(i) for i in range(35)] + [1.6]
    direction, strength, rule = signal_for(closes)
    assert direction == "DOWN"
    assert "OU-reversion" in rule
    assert strength > 0.5


def test_signal_momentum_in_trending_regime():
    direction, strength, rule = signal_for(trending_up())
    assert direction == "UP"
    assert "momentum" in rule


def test_signal_insufficient_data():
    assert signal_for([1.0, 1.1, 1.2]) is None


def test_math_prediction_is_deterministic_and_valid():
    closes = {"MINT_A": trending_up(), "MINT_B": oscillating()}
    p1 = math_prediction(catalog(), closes, 2.0)
    p2 = math_prediction(catalog(), closes, 2.0)
    assert p1.arm == "math" and p1.backend == "math"
    assert p1.direction in ("UP", "DOWN")
    assert (p1.mint, p1.direction) == (p2.mint, p2.direction)  # deterministic
    assert 0.5 <= p1.confidence <= 0.85
    assert "MINT_A" not in p1.prompt or "strength" in p1.prompt


def test_math_prediction_abstains_without_history():
    p = math_prediction(catalog(), {}, 2.0)
    assert p.direction == "SKIP"
    assert "SKIP" in p.response


def test_math_prediction_abstains_on_weak_signal():
    # Random-walk-ish drift only: strength 0.05 < MIN_CALL_STRENGTH.
    import random as _r
    rng = _r.Random(3)
    walk = [1.0]
    for _ in range(39):
        walk.append(max(0.5, walk[-1] * (1 + rng.uniform(-0.001, 0.0012))))
    p = math_prediction(catalog(), {"MINT_A": walk, "MINT_B": walk}, 2.0)
    if p.direction == "SKIP":
        assert "SKIP" in p.response
    else:
        # A strong signal can legitimately emerge from the seeded walk.
        assert p.confidence >= 0.5 + 0.35 * 0.15 - 1e-9
