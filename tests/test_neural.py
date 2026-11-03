"""Contract tests for the neural adapter and the baseline-vs-neural comparison.

Kept cheap on purpose (few steps, short series, few origins): these guard the
protocol, not model skill.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stockpred.models import compare as compare_module
from stockpred.models.baselines import NaiveForecaster
from stockpred.models.compare import compare_models
from stockpred.models.neural import NeuralForecastAdapter


@pytest.fixture
def series() -> pd.Series:
    rng = np.random.default_rng(3)
    values = 12.0 * np.cumprod(1.0 + rng.normal(0.0005, 0.015, 80))
    return pd.Series(values, index=pd.date_range("2020-01-01", periods=80, freq="B"))


def test_adapter_unknown_model_rejected() -> None:
    with pytest.raises(ValueError, match="supported"):
        NeuralForecastAdapter("transformer_9000")


def test_adapter_requires_enough_observations() -> None:
    adapter = NeuralForecastAdapter("nhits", horizon=1)
    with pytest.raises(ValueError, match="at least 10"):
        adapter.fit(pd.Series(np.arange(6, dtype=float)))


def test_adapter_returns_exact_horizon(series: pd.Series) -> None:
    adapter = NeuralForecastAdapter("nhits", horizon=1, max_steps=5).fit(series)
    forecast = adapter.predict(1)

    assert adapter.name == "neural_nhits"
    assert isinstance(forecast, np.ndarray)
    assert forecast.shape == (1,)
    assert forecast.dtype == float
    assert np.isfinite(forecast).all()


def test_adapter_is_deterministic(series: pd.Series) -> None:
    first = NeuralForecastAdapter("nhits", horizon=1, max_steps=5, seed=7).fit(series).predict(1)
    second = NeuralForecastAdapter("nhits", horizon=1, max_steps=5, seed=7).fit(series).predict(1)

    assert np.allclose(first, second)


def test_compare_includes_baselines_and_neural(series: pd.Series) -> None:
    result = compare_models(series, origins=3, neural_max_steps=5)

    assert set(result.table["kind"]) == {"baseline", "neural"}
    assert result.winner in set(result.table["model"])
    assert set(result.generators) == set(result.table["model"])
    assert result.table["mae"].notna().all()


def test_compare_without_neural(series: pd.Series) -> None:
    result = compare_models(series, origins=3, include_neural=False)

    assert set(result.table["kind"]) == {"baseline"}
    assert not any(name.startswith("neural_") for name in result.table["model"])


def test_compare_survives_a_failing_model(
    series: pd.Series, monkeypatch: pytest.MonkeyPatch
) -> None:
    class ExplodingForecaster:
        name = "boom"

        def fit(self, series: pd.Series) -> ExplodingForecaster:
            raise RuntimeError("training blew up")

        def predict(self, horizon: int) -> np.ndarray:  # pragma: no cover - never reached
            raise AssertionError("predict must not be called after a failed fit")

    monkeypatch.setattr(
        compare_module,
        "default_baselines",
        lambda: [NaiveForecaster(), ExplodingForecaster()],
    )

    result = compare_models(series, origins=3, include_neural=False)
    row = result.table.set_index("model").loc["boom"]

    assert np.isnan(row["mae"])
    assert result.table.set_index("model").loc["naive", "mae"] >= 0.0
    assert "training blew up" in result.generators["boom"]
    assert result.winner == "naive"
