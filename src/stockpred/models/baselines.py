"""Reference baselines that a neural model must beat before it earns a place.

These are dependency-free on purpose: a baseline you cannot run on a fresh
machine is not a baseline. ``statsforecast`` is wired in only if installed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
import pandas as pd


class Forecaster(Protocol):
    name: str

    def fit(self, series: pd.Series) -> Forecaster: ...

    def predict(self, horizon: int) -> np.ndarray: ...


@dataclass
class NaiveForecaster:
    """Tomorrow equals today (random walk). The honest default."""

    name: str = "naive"

    def __post_init__(self) -> None:
        self._last: float | None = None

    def fit(self, series: pd.Series) -> NaiveForecaster:
        if len(series) == 0:
            raise ValueError("cannot fit on an empty series")
        self._last = float(series.iloc[-1])
        return self

    def predict(self, horizon: int) -> np.ndarray:
        if self._last is None:
            raise RuntimeError("fit before predict")
        return np.full(horizon, self._last, dtype=float)


@dataclass
class DriftForecaster:
    """Random walk with drift: last value plus the mean past per-step change."""

    name: str = "drift"

    def __post_init__(self) -> None:
        self._last: float | None = None
        self._drift: float = 0.0

    def fit(self, series: pd.Series) -> DriftForecaster:
        if len(series) < 2:
            raise ValueError("drift needs at least two observations")
        values = series.astype(float)
        self._last = float(values.iloc[-1])
        self._drift = float((values.iloc[-1] - values.iloc[0]) / (len(values) - 1))
        return self

    def predict(self, horizon: int) -> np.ndarray:
        if self._last is None:
            raise RuntimeError("fit before predict")
        steps = np.arange(1, horizon + 1, dtype=float)
        return self._last + self._drift * steps


@dataclass
class SeasonalNaiveForecaster:
    """Value from ``season`` steps ago, repeated."""

    season: int = 5
    name: str = "seasonal_naive"

    def __post_init__(self) -> None:
        self._tail: np.ndarray | None = None

    def fit(self, series: pd.Series) -> SeasonalNaiveForecaster:
        if len(series) < self.season:
            raise ValueError(f"need at least {self.season} observations")
        self._tail = series.astype(float).to_numpy()[-self.season :]
        return self

    def predict(self, horizon: int) -> np.ndarray:
        if self._tail is None:
            raise RuntimeError("fit before predict")
        repeats = int(np.ceil(horizon / self.season))
        return np.tile(self._tail, repeats)[:horizon]


@dataclass
class MovingAverageForecaster:
    window: int = 5
    name: str = "moving_average"

    def __post_init__(self) -> None:
        self._mean: float | None = None

    def fit(self, series: pd.Series) -> MovingAverageForecaster:
        if len(series) < self.window:
            raise ValueError(f"need at least {self.window} observations")
        self._mean = float(series.astype(float).iloc[-self.window :].mean())
        return self

    def predict(self, horizon: int) -> np.ndarray:
        if self._mean is None:
            raise RuntimeError("fit before predict")
        return np.full(horizon, self._mean, dtype=float)


def default_baselines() -> list[Forecaster]:
    return [
        NaiveForecaster(),
        DriftForecaster(),
        MovingAverageForecaster(),
        SeasonalNaiveForecaster(),
    ]


def evaluate_forecasters(
    series: pd.Series,
    forecasters: list[Forecaster],
    *,
    horizon: int = 1,
    origins: int = 20,
) -> pd.DataFrame:
    """Rolling-origin evaluation: fit on history only, score the next ``horizon``."""
    values = series.astype(float).reset_index(drop=True)
    needed = horizon + 1
    if len(values) < needed + 2:
        raise ValueError("series too short for rolling-origin evaluation")
    origins = min(origins, len(values) - needed)
    rows: list[dict[str, float | str]] = []
    for forecaster in forecasters:
        errors: list[float] = []
        for offset in range(origins):
            train_end = len(values) - needed - offset + 1
            train = values.iloc[:train_end]
            actual = values.iloc[train_end : train_end + horizon].to_numpy()
            predicted = forecaster.fit(train).predict(horizon)
            errors.append(float(np.mean(np.abs(predicted - actual))))
        rows.append(
            {
                "model": forecaster.name,
                "mae": float(np.mean(errors)),
                "horizon": horizon,
                "origins": origins,
            }
        )
    return pd.DataFrame(rows).sort_values("mae").reset_index(drop=True)
