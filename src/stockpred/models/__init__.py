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
from stockpred.models.neural import SUPPORTED_MODELS, NeuralForecastAdapter

__all__ = [
    "DriftForecaster",
    "Forecaster",
    "MovingAverageForecaster",
    "NaiveForecaster",
    "NeuralForecastAdapter",
    "SUPPORTED_MODELS",
    "SeasonalNaiveForecaster",
    "default_baselines",
    "evaluate_forecasters",
]
