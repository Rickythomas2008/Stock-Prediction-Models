from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stockpred.portfolio import (
    Allocation,
    apply_caps,
    equal_weight,
    inverse_volatility,
    min_variance,
    to_lots,
    weights_from_allocation,
)


@pytest.fixture
def returns() -> pd.DataFrame:
    rng = np.random.default_rng(11)
    return pd.DataFrame(
        rng.normal(0.0, [0.010, 0.020, 0.015, 0.030], size=(250, 4)),
        columns=["600519", "000001", "300750", "600000"],
    )


def test_equal_weight_sums_to_one() -> None:
    weights = equal_weight(["a", "b", "c"])
    assert weights.sum() == pytest.approx(1.0)
    assert np.allclose(weights.to_numpy(), 1 / 3)


def test_equal_weight_needs_symbols() -> None:
    with pytest.raises(ValueError, match="at least one symbol"):
        equal_weight([])


def test_inverse_volatility_favours_low_vol(returns: pd.DataFrame) -> None:
    weights = inverse_volatility(returns)
    assert weights.sum() == pytest.approx(1.0)
    assert weights["600519"] > weights["600000"]


def test_min_variance_is_long_only_and_normalised(returns: pd.DataFrame) -> None:
    weights = min_variance(returns)
    assert (weights >= 0).all()
    assert weights.sum() == pytest.approx(1.0, abs=1e-6)


def test_min_variance_prefers_the_quiet_asset(returns: pd.DataFrame) -> None:
    weights = min_variance(returns)
    assert weights["600519"] > weights["600000"]


def test_apply_caps_limits_single_name() -> None:
    capped = apply_caps(pd.Series({"a": 0.7, "b": 0.2, "c": 0.1}), max_position=0.4)
    assert capped.max() <= 0.4 + 1e-9
    assert capped.sum() <= 1.0 + 1e-9


def test_apply_caps_rejects_bad_cap() -> None:
    with pytest.raises(ValueError, match="max_position"):
        apply_caps(pd.Series({"a": 1.0}), max_position=0.0)


def test_apply_caps_rejects_all_zero() -> None:
    with pytest.raises(ValueError, match="all weights are zero"):
        apply_caps(pd.Series({"a": 0.0, "b": 0.0}))


def test_to_lots_uses_whole_lots() -> None:
    allocation = to_lots(
        equal_weight(["000001", "600000"]),
        capital=100_000.0,
        prices={"000001": 12.3, "600000": 8.0},
    )
    assert all(shares % 100 == 0 for shares in allocation.shares.values())
    assert allocation.invested + allocation.leftover_cash == pytest.approx(100_000.0)


def test_to_lots_drops_unaffordable_name() -> None:
    allocation = to_lots(
        equal_weight(["600519", "000001"]),
        capital=100_000.0,
        prices={"600519": 1700.0, "000001": 12.3},
    )
    assert "600519" in allocation.dropped
    assert 600519 not in allocation.shares


def test_to_lots_never_overspends() -> None:
    allocation = to_lots(
        pd.Series({"a": 0.5, "b": 0.5}),
        capital=10_000.0,
        prices={"a": 33.33, "b": 47.77},
    )
    assert allocation.invested <= 10_000.0 + 1e-9
    assert allocation.leftover_cash >= 0.0


def test_to_lots_missing_price_is_dropped() -> None:
    allocation = to_lots(pd.Series({"a": 1.0}), capital=10_000.0, prices={})
    assert allocation.dropped == ("a",)
    assert allocation.invested == 0.0


def test_to_lots_rejects_bad_capital() -> None:
    with pytest.raises(ValueError, match="capital"):
        to_lots(pd.Series({"a": 1.0}), capital=0.0, prices={"a": 10.0})


def test_allocation_invested_pct() -> None:
    allocation = Allocation(shares={"a": 100}, invested=1_000.0, leftover_cash=1_000.0, dropped=())
    assert allocation.invested_pct == pytest.approx(0.5)


def test_weights_from_allocation_values() -> None:
    allocation = Allocation(shares={"a": 200, "b": 100}, invested=0.0, leftover_cash=0.0, dropped=())
    weights = weights_from_allocation(allocation, {"a": 10.0, "b": 50.0})
    assert weights["a"] == pytest.approx(2_000.0)
    assert weights["b"] == pytest.approx(5_000.0)
