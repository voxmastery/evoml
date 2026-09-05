import pytest

from memescalp.metrics import arm_stats, edge_z_score, equity_curve, evaluate


def sell(pnl, ts=0.0, balance=23.0):
    return {"side": "sell", "realized_pnl": pnl, "ts": ts, "fee_lp": 0.05,
            "fee_slippage": 0.01, "fee_priority": 0.04, "fee_tds": 0.2,
            "balance_after": balance, "size_usd": 23.0}


def buy(ts=0.0):
    return {"side": "buy", "realized_pnl": None, "ts": ts, "fee_lp": 0.05,
            "fee_slippage": 0.01, "fee_priority": 0.04, "fee_tds": 0.2,
            "balance_after": 0.0, "size_usd": 23.0}


def test_arm_stats_basic():
    fills = [buy(), sell(1.5), buy(), sell(-2.0), buy(), sell(1.6)]
    s = arm_stats(fills)
    assert s.round_trips == 3
    assert s.net_pnl == pytest.approx(1.1)
    assert s.win_rate == pytest.approx(2 / 3)
    assert s.avg_win == pytest.approx(1.55)
    assert s.avg_loss == pytest.approx(-2.0)
    assert s.largest_loss == -2.0
    assert s.fees_tds == pytest.approx(0.2 * 6)
    assert s.fees_total == pytest.approx((0.05 + 0.01 + 0.04 + 0.2) * 6)


def test_pass_requires_all_four_criteria():
    day = 86400.0
    strategy = []
    control = []
    for i in range(250):
        ts = i * (day / 20)
        # Alternate values so per-trade variance is non-zero for the z-test.
        strategy += [buy(ts), sell(1.5 if i % 2 else 1.3, ts + 1)]
        control += [buy(ts), sell(-0.5 if i % 2 else -0.7, ts + 1)]
    first_ts = 0.0

    # 15 days elapsed: everything satisfied -> PASS
    result = evaluate(strategy, control, first_ts, 200, 14, 1.64,
                      now=15 * day)
    assert result.profitable and result.enough_trades and result.enough_days
    assert result.beats_control
    assert result.overall

    # Only 5 days elapsed -> FAIL on duration alone
    early = evaluate(strategy, control, first_ts, 200, 14, 1.64, now=5 * day)
    assert not early.enough_days
    assert not early.overall


def test_fail_when_not_above_control():
    day = 86400.0
    fills = []
    for i in range(250):
        fills += [buy(i), sell(1.0, i + 0.5)]
    # Identical arms: zero edge, z == 0 -> beats_control is False.
    result = evaluate(fills, fills, 0.0, 200, 14, 1.64, now=20 * day)
    assert result.profitable and result.enough_trades and result.enough_days
    assert not result.beats_control
    assert not result.overall


def test_edge_z_zero_with_too_few_trades():
    assert edge_z_score([buy(), sell(1.0)], [buy(), sell(0.5)]) == 0.0


def test_equity_curve_tracks_round_trips():
    fills = [buy(ts=1), sell(0.5, ts=2, balance=23.5),
             buy(ts=3), sell(-1.0, ts=4, balance=22.5)]
    curve = equity_curve(fills, 23.0)
    assert curve[1] == (2, 23.5)
    assert curve[3] == (4, 22.5)
    # Buy points keep the curve continuous at entry cost.
    assert curve[0] == (1, 23.0)
