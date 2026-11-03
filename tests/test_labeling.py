from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stockpred.eval.labeling import (
    Label,
    assert_labels_honour_purge,
    meta_labels,
    sample_uniqueness,
    triple_barrier_labels,
)


def _trend(rate: float, n: int = 30, start: float = 100.0) -> pd.Series:
    return pd.Series([start * (1.0 + rate) ** i for i in range(n)])


def test_uptrend_hits_upper_barrier() -> None:
    result = triple_barrier_labels(_trend(0.01), upper_pct=0.02, lower_pct=0.02, horizon=5)
    assert result.distribution()["upper"] == 25
    assert result.distribution()["lower"] == 0


def test_downtrend_hits_lower_barrier() -> None:
    result = triple_barrier_labels(_trend(-0.01), upper_pct=0.02, lower_pct=0.02, horizon=5)
    assert result.distribution()["lower"] == 25
    assert result.distribution()["upper"] == 0


def test_flat_series_never_touches_a_barrier() -> None:
    result = triple_barrier_labels(pd.Series([100.0] * 30), horizon=5)
    assert result.distribution() == {"upper": 0, "lower": 0, "vertical": 30}


def test_barrier_width_controls_sensitivity() -> None:
    """Wider barriers take longer to touch; the last bars stay unresolved either way."""
    gentle = triple_barrier_labels(_trend(0.01), upper_pct=0.05, lower_pct=0.05, horizon=10)
    tight = triple_barrier_labels(_trend(0.01), upper_pct=0.01, lower_pct=0.01, horizon=10)
    gentle_touch = gentle.touch_bar[gentle.touch_bar > 0].mean()
    tight_touch = tight.touch_bar[tight.touch_bar > 0].mean()
    assert gentle_touch > tight_touch


def test_tail_bars_are_vertical() -> None:
    result = triple_barrier_labels(_trend(0.05), horizon=5, upper_pct=0.02, lower_pct=0.02)
    assert list(result.labels.iloc[-5:]) == [int(Label.VERTICAL)] * 5
    assert list(result.touch_bar.iloc[-5:]) == [-1] * 5


def test_touch_bar_records_first_hit() -> None:
    prices = pd.Series([100.0, 101.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0])
    result = triple_barrier_labels(prices, upper_pct=0.02, lower_pct=0.02, horizon=5)
    assert result.labels.iloc[0] == int(Label.UPPER)
    assert result.touch_bar.iloc[0] == 2


def test_horizon_is_the_purge_window() -> None:
    result = triple_barrier_labels(_trend(0.01), horizon=7)
    assert result.horizon == 7
    assert result.purge_bars == 7


def test_label_horizon_leaking_into_test_window_is_rejected() -> None:
    result = triple_barrier_labels(_trend(0.01), horizon=5)
    with pytest.raises(ValueError, match="leaks into the test window"):
        assert_labels_honour_purge(result, train_end=20, test_start=22)


def test_purged_split_passes() -> None:
    result = triple_barrier_labels(_trend(0.01), horizon=5)
    assert_labels_honour_purge(result, train_end=20, test_start=30)


def test_inverted_split_is_rejected() -> None:
    result = triple_barrier_labels(_trend(0.01), horizon=5)
    with pytest.raises(ValueError, match="must be before"):
        assert_labels_honour_purge(result, train_end=30, test_start=20)


def test_uniqueness_equals_one_without_overlap() -> None:
    result = triple_barrier_labels(_trend(0.01), horizon=1)
    assert sample_uniqueness(result).iloc[:5].to_list() == pytest.approx([1.0] * 5)


def test_uniqueness_drops_with_overlap() -> None:
    result = triple_barrier_labels(_trend(0.01), horizon=5)
    uniqueness = sample_uniqueness(result)
    assert uniqueness.iloc[0] < 1.0
    assert uniqueness.max() <= 1.0
    assert uniqueness.min() > 0.0


def test_meta_labels_flag_correct_direction() -> None:
    assert meta_labels([1, 1, -1], returns=[0.01, -0.02, 0.01]).to_list() == [1, 0, 0]


def test_meta_labels_length_mismatch_rejected() -> None:
    with pytest.raises(ValueError, match="same length"):
        meta_labels([1, 1], returns=[0.01])


def test_invalid_parameters_rejected() -> None:
    with pytest.raises(ValueError, match="horizon"):
        triple_barrier_labels(_trend(0.01), horizon=0)
    with pytest.raises(ValueError, match="positive"):
        triple_barrier_labels(_trend(0.01), upper_pct=0.0)
    with pytest.raises(ValueError, match="at least"):
        triple_barrier_labels(pd.Series([1.0, 2.0]), horizon=5)


def test_volatility_scaled_barriers_adapt() -> None:
    rng = np.random.default_rng(0)
    calm = pd.Series(100 * np.cumprod(1 + rng.normal(0, 0.002, 120)))
    wild = pd.Series(100 * np.cumprod(1 + rng.normal(0, 0.02, 120)))
    calm_labels = triple_barrier_labels(calm, horizon=5, volatility_scaled=True, upper_pct=0.02, lower_pct=0.02)
    wild_labels = triple_barrier_labels(wild, horizon=5, volatility_scaled=True, upper_pct=0.02, lower_pct=0.02)

    calm_touched = 120 - calm_labels.distribution()["vertical"]
    wild_touched = 120 - wild_labels.distribution()["vertical"]
    assert calm_touched > 0 and wild_touched > 0
