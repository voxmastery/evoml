import pytest

from memescalp.fees import compute_fees, slippage_fraction


def test_slippage_scales_with_size_vs_liquidity():
    assert slippage_fraction(100, 9900) == pytest.approx(0.01)
    assert slippage_fraction(23, 100_000) == pytest.approx(23 / 100_023)


def test_slippage_zero_for_zero_size():
    assert slippage_fraction(0, 50_000) == 0.0


def test_slippage_total_loss_when_no_liquidity():
    assert slippage_fraction(23, 0) == 1.0


def test_fee_components_are_separate_and_correct():
    fees = compute_fees(
        size_usd=100.0, pool_liquidity_usd=99_900.0,
        lp_fee_rate=0.0025, priority_fee_usd=0.04, tds_rate=0.01,
    )
    assert fees.lp == pytest.approx(0.25)
    assert fees.slippage == pytest.approx(100 * 100 / 100_000)
    assert fees.priority == 0.04
    assert fees.tds == pytest.approx(1.0)
    assert fees.total == pytest.approx(0.25 + 0.1 + 0.04 + 1.0)


def test_negative_size_rejected():
    with pytest.raises(ValueError):
        compute_fees(-1, 1000, 0.0025, 0.04, 0.01)
