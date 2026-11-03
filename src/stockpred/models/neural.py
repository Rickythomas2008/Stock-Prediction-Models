"""Deep-learning forecasters behind the same ``Forecaster`` protocol as the baselines.

The point of this module is substitutability: a neural model is only useful if it
can be dropped into the identical rolling-origin harness that scores the
baselines. Everything version-specific about ``neuralforecast`` / ``torch`` lives
in this adapter and nowhere else.

The heavy imports are deferred to :meth:`NeuralForecastAdapter.fit` so that the
core package (and its test suite) stays importable without the optional
``models`` extra installed.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from stockpred.errors import ProviderUnavailable

# name used on the command line / in tables -> attribute name in neuralforecast.models
_MODEL_CLASSES: dict[str, str] = {
    "nhits": "NHITS",
    "lstm": "LSTM",
    "patchtst": "PatchTST",
}

SUPPORTED_MODELS: tuple[str, ...] = tuple(_MODEL_CLASSES)

_SERIES_ID = "series"
# Fixed alias so the prediction column name does not drift with the library's defaults.
_DEFAULT_ALIAS = "yhat"
_INSTALL_HINT = "pip install -e '.[models]'"


def _load_backend() -> tuple[Any, Any, Any]:
    """Import torch + neuralforecast on demand; translate absence into a typed error."""
    try:
        import torch
        from neuralforecast import NeuralForecast
        from neuralforecast import models as nf_models
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise ProviderUnavailable(
            f"neural forecast models need the optional 'models' extra: {_INSTALL_HINT}"
        ) from exc
    return torch, NeuralForecast, nf_models


def _to_long_frame(values: pd.Series) -> tuple[pd.DataFrame, str]:
    """Reshape a one-column series into neuralforecast's (unique_id, ds, y) long format."""
    index = values.index
    stamps: pd.DatetimeIndex
    if isinstance(index, pd.DatetimeIndex) or pd.api.types.is_datetime64_any_dtype(index):
        stamps = pd.DatetimeIndex(pd.to_datetime(index))
    elif pd.api.types.is_numeric_dtype(index):
        # Integers are technically parseable as nanosecond epochs, which is never
        # what a caller means; a synthetic daily axis is the honest fallback.
        stamps = pd.date_range("2000-01-01", periods=len(values), freq="D")
    else:
        try:
            stamps = pd.DatetimeIndex(pd.to_datetime(index))
        except (TypeError, ValueError, pd.errors.OutOfBoundsDatetime):
            stamps = pd.date_range("2000-01-01", periods=len(values), freq="D")
    if stamps.isna().any():
        stamps = pd.date_range("2000-01-01", periods=len(values), freq="D")

    frame = pd.DataFrame(
        {
            "unique_id": _SERIES_ID,
            "ds": stamps,
            "y": values.to_numpy(dtype=float),
        }
    )
    frame = frame.dropna(subset=["y"]).sort_values("ds")
    # neuralforecast rejects duplicated timestamps outright.
    frame = frame.drop_duplicates(subset=["ds"], keep="last").reset_index(drop=True)

    try:
        freq = pd.infer_freq(frame["ds"])
    except (TypeError, ValueError):
        freq = None
    return frame, freq or "D"


class NeuralForecastAdapter:
    """Wraps a neuralforecast model behind the Forecaster protocol."""

    def __init__(
        self,
        model: str = "nhits",
        *,
        horizon: int = 1,
        max_steps: int = 50,
        seed: int = 7,
        input_size: int | None = None,
        **model_kwargs: Any,
    ) -> None:
        key = model.strip().lower()
        if key not in _MODEL_CLASSES:
            supported = ", ".join(SUPPORTED_MODELS)
            raise ValueError(f"unknown neural model {model!r}; supported: {supported}")
        if horizon <= 0:
            raise ValueError(f"horizon must be positive, got {horizon}")
        if max_steps <= 0:
            raise ValueError(f"max_steps must be positive, got {max_steps}")
        if input_size is not None and input_size <= 0:
            raise ValueError(f"input_size must be positive, got {input_size}")

        self._model = key
        self._horizon = int(horizon)
        self._max_steps = int(max_steps)
        self._seed = int(seed)
        self._input_size = input_size
        self._model_kwargs: dict[str, Any] = dict(model_kwargs)
        self._nf: Any | None = None
        self._column = _DEFAULT_ALIAS

    @property
    def name(self) -> str:
        return f"neural_{self._model}"

    def _resolve_input_size(self, n_obs: int) -> int:
        """Longest sensible lookback: half the history, capped per horizon, never < 2*h."""
        if self._input_size is not None:
            return int(self._input_size)
        cap = 2 * self._horizon if self._horizon > 1 else 20
        return max(min(n_obs // 2, cap), 2 * self._horizon)

    def _min_observations(self, n_obs: int) -> int:
        return max(self._resolve_input_size(n_obs), 3 * self._horizon, 10)

    def fit(self, series: pd.Series) -> NeuralForecastAdapter:
        values = series.astype(float)
        if len(values) == 0:
            raise ValueError("cannot fit on an empty series")
        required = self._min_observations(len(values))
        if len(values) < required:
            raise ValueError(
                f"{self.name} needs at least {required} observations "
                f"(input_size/windows), got {len(values)}"
            )

        torch, NeuralForecast, nf_models = _load_backend()
        frame, freq = _to_long_frame(values)
        if len(frame) < required:
            raise ValueError(
                f"{self.name} needs at least {required} non-missing observations, got {len(frame)}"
            )

        input_size = self._resolve_input_size(len(frame))
        # Trainer options (accelerator/devices/logger/enable_progress_bar) are model
        # constructor kwargs in neuralforecast 3.3; NeuralForecast itself only takes
        # models + freq, so the training flags travel with the network.
        options: dict[str, Any] = {
            "h": self._horizon,
            "input_size": input_size,
            "max_steps": self._max_steps,
            "random_seed": self._seed,
            "alias": _DEFAULT_ALIAS,
            "accelerator": "cpu",
            "devices": 1,
            "logger": False,
            "enable_progress_bar": False,
            "enable_model_summary": False,
        }
        options.update(self._model_kwargs)

        torch.manual_seed(self._seed)
        np.random.seed(self._seed)

        network = getattr(nf_models, _MODEL_CLASSES[self._model])(**options)
        nf = NeuralForecast(models=[network], freq=freq)
        nf.fit(df=frame)

        self._nf = nf
        self._column = str(options["alias"])
        return self

    def predict(self, horizon: int) -> np.ndarray:
        if self._nf is None:
            raise RuntimeError("fit before predict")
        if horizon <= 0:
            raise ValueError(f"horizon must be positive, got {horizon}")

        steps = max(horizon, self._horizon)
        forecast = self._nf.predict() if steps == self._horizon else self._nf.predict(h=steps)
        forecast = forecast.sort_values("ds")
        values = np.asarray(forecast[self._column].to_numpy(dtype=float))
        if values.shape != (steps,):
            raise RuntimeError(
                f"{self.name} produced {values.shape} values, expected ({steps},) "
                f"for horizon {horizon}"
            )
        return values[:horizon]
