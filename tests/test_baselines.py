from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stockpred.models.baselines import (
    DriftForecaster,
    MovingAverageForecaster,
    NaiveForecaster,
    SeasonalNaiveForecaster,
    default_baselines,
    evaluate_forecasters,
)


@pytest.fixture
def series() -> pd.Series:
    return pd.Series(np.linspace(10.0, 20.0, 60))


def test_naive_repeats_last_value(series: pd.Series) -> None:
    forecast = NaiveForecaster().fit(series).predict(3)
    assert list(forecast) == pytest.approx([series.iloc[-1]] * 3)


def test_drift_extends_the_trend(series: pd.Series) -> None:
    forecast = DriftForecaster().fit(series).predict(2)
    expected_step = (series.iloc[-1] - series.iloc[0]) / (len(series) - 1)
    assert forecast[0] == pytest.approx(series.iloc[-1] + expected_step)
    assert forecast[1] == pytest.approx(series.iloc[-1] + 2 * expected_step)


def test_seasonal_naive_tiles_the_tail(series: pd.Series) -> None:
    forecast = SeasonalNaiveForecaster(season=5).fit(series).predict(7)
    assert list(forecast[:5]) == pytest.approx(list(series.iloc[-5:]))
    assert forecast[5] == pytest.approx(series.iloc[-5])


def test_moving_average_uses_window(series: pd.Series) -> None:
    forecast = MovingAverageForecaster(window=4).fit(series).predict(2)
    assert forecast[0] == pytest.approx(series.iloc[-4:].mean())


def test_predict_before_fit_raises() -> None:
    with pytest.raises(RuntimeError, match="fit before predict"):
        NaiveForecaster().predict(1)


def test_fit_on_empty_series_raises() -> None:
    with pytest.raises(ValueError):
        NaiveForecaster().fit(pd.Series(dtype=float))


def test_too_short_series_for_window_raises() -> None:
    with pytest.raises(ValueError, match="need at least"):
        MovingAverageForecaster(window=10).fit(pd.Series([1.0, 2.0]))


def test_evaluate_forecasters_ranks_by_mae(series: pd.Series) -> None:
    table = evaluate_forecasters(series, default_baselines(), horizon=1, origins=10)

    assert list(table.columns) == ["model", "mae", "horizon", "origins"]
    assert table["mae"].is_monotonic_increasing
    assert table["mae"].notna().all()
    assert set(table["model"]) == {"naive", "drift", "moving_average", "seasonal_naive"}


def test_naive_beats_drift_on_a_flat_series() -> None:
    flat = pd.Series(np.full(40, 12.0))
    table = evaluate_forecasters(
        flat, [NaiveForecaster(), DriftForecaster()], horizon=1, origins=5
    )
    assert table.iloc[0]["model"] == "naive"
    assert table.iloc[0]["mae"] == pytest.approx(0.0)


def test_rolling_origin_uses_only_history(series: pd.Series) -> None:
    """A shifted series must change nothing before the shift point."""
    short = series.iloc[:20]
    table_short = evaluate_forecasters(short, [NaiveForecaster()], horizon=1, origins=5)
    table_long = evaluate_forecasters(
        series.iloc[:25], [NaiveForecaster()], horizon=1, origins=5
    )
    assert table_short["origins"].iloc[0] == table_long["origins"].iloc[0]
