"""Forecast models. Baselines first: a model that cannot beat them is noise."""

from __future__ import annotations

from stockpred.models.baselines import (
    DriftForecaster,
    Forecaster,
    MovingAverageForecaster,
    NaiveForecaster,
    SeasonalNaiveForecaster,
    default_baselines,
    evaluate_forecasters,
)

__all__ = [
    "DriftForecaster",
    "Forecaster",
    "MovingAverageForecaster",
    "NaiveForecaster",
    "SeasonalNaiveForecaster",
    "default_baselines",
    "evaluate_forecasters",
]
