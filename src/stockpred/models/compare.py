"""Head-to-head comparison: baselines and neural models under one scoring harness.

A neural model earns its place only by beating the dependency-free baselines on
the *same* rolling-origin windows, so both families are scored through a single
private path (:func:`_rolling_mae`) rather than through model-specific code.

Per-model failures are recorded as ``mae=nan`` with the reason attached to the
model's description: an auditable bad row beats an aborted comparison.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from stockpred.models.baselines import Forecaster, default_baselines, evaluate_forecasters
from stockpred.models.neural import SUPPORTED_MODELS, NeuralForecastAdapter

KIND_BASELINE = "baseline"
KIND_NEURAL = "neural"

NEURAL_MODELS: tuple[str, ...] = SUPPORTED_MODELS

_BASELINE_NOTES: dict[str, str] = {
    "naive": "随机游走基线：明日价格等于今日价值。",
    "drift": "带漂移随机游走基线：最后价值加历史平均每步变动。",
    "moving_average": "移动平均基线：最近 5 个观测的均值。",
    "seasonal_naive": "季节朴素基线：取 5 步之前的数值并按周期重复。",
}

_NEURAL_NOTES: dict[str, str] = {
    "nhits": "NHITS：多尺度分层插值的 MLP 神经预测模型（neuralforecast 实现）。",
    "lstm": "LSTM：循环神经网络编码历史窗口后输出多步预测。",
    "patchtst": "PatchTST：把序列切成 patch 后用 Transformer 编码的预测模型。",
}


@dataclass(frozen=True)
class ComparisonResult:
    table: pd.DataFrame
    winner: str
    generators: dict[str, str]


class _HorizonChecked:
    """Re-assert the Forecaster output contract before anything is scored.

    A short/overlong return would otherwise broadcast silently inside the MAE.
    """

    def __init__(self, forecaster: Forecaster) -> None:
        self._forecaster = forecaster
        self.name = forecaster.name

    def fit(self, series: pd.Series) -> _HorizonChecked:
        self._forecaster.fit(series)
        return self

    def predict(self, horizon: int) -> np.ndarray:
        output = np.asarray(self._forecaster.predict(horizon), dtype=float)
        if output.shape != (horizon,):
            raise RuntimeError(
                f"{self.name} returned shape {output.shape}, expected ({horizon},)"
            )
        return output


def _rolling_mae(forecaster: Forecaster, series: pd.Series, horizon: int, origins: int) -> float:
    """Mean MAE over the rolling origins, using the shared evaluation harness.

    ``evaluate_forecasters`` already owns the origin arithmetic (and the "fit on
    history only" rule); calling it with a single forecaster keeps one source of
    truth while still letting each model fail in isolation.
    """
    table = evaluate_forecasters(
        series, [_HorizonChecked(forecaster)], horizon=horizon, origins=origins
    )
    return float(table["mae"].iloc[0])


def _describe(model: str) -> str:
    if model.startswith("neural_"):
        return _NEURAL_NOTES.get(model.removeprefix("neural_"), "神经预测模型（neuralforecast）。")
    return _BASELINE_NOTES.get(model, "基线模型。")


def compare_models(
    series: pd.Series,
    *,
    horizon: int = 1,
    origins: int = 5,
    neural_max_steps: int = 30,
    seed: int = 7,
    include_neural: bool = True,
) -> ComparisonResult:
    """Score every forecaster on identical rolling origins and rank by MAE."""
    if horizon <= 0:
        raise ValueError(f"horizon must be positive, got {horizon}")
    if origins <= 0:
        raise ValueError(f"origins must be positive, got {origins}")

    values = series.astype(float)
    candidates: list[tuple[Forecaster, str]] = [
        (forecaster, KIND_BASELINE) for forecaster in default_baselines()
    ]
    if include_neural:
        candidates.extend(
            (
                NeuralForecastAdapter(
                    model, horizon=horizon, max_steps=neural_max_steps, seed=seed
                ),
                KIND_NEURAL,
            )
            for model in NEURAL_MODELS
        )

    rows: list[dict[str, float | int | str]] = []
    generators: dict[str, str] = {}
    for forecaster, kind in candidates:
        notes = _describe(forecaster.name)
        try:
            mae = _rolling_mae(forecaster, values, horizon, origins)
        except Exception as exc:  # a broken model must not take the comparison down
            mae = float("nan")
            notes = f"{notes}（本次评估失败：{type(exc).__name__}: {exc}）"
        generators[forecaster.name] = notes
        rows.append(
            {
                "model": forecaster.name,
                "mae": mae,
                "horizon": horizon,
                "origins": origins,
                "kind": kind,
            }
        )

    table = (
        pd.DataFrame(rows)[["model", "mae", "horizon", "origins", "kind"]]
        .sort_values("mae", ascending=True, kind="stable", na_position="last")
        .reset_index(drop=True)
    )
    return ComparisonResult(
        table=table,
        winner=str(table["model"].iloc[0]),
        generators=generators,
    )
