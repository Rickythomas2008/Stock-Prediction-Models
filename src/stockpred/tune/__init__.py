"""Hyper-parameter search with persisted studies.

The legacy repo hand-rolls an evolution strategy for this. Optuna does it with a
storage backend, so a search survives a restart and can be audited afterwards --
which is the property that matters when a live pipeline depends on the winner.

The objective is deliberately *pluggable*: tests pass a cheap synthetic objective
instead of running a full backtest per trial.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import optuna
from optuna.samplers import TPESampler

from stockpred.errors import StockpredError


class TuningError(StockpredError):
    """Search configuration or objective misbehaved."""


@dataclass(frozen=True)
class SearchSpace:
    """A flat parameter grid: name -> (low, high) inclusive bounds."""

    ints: Mapping[str, tuple[int, int]] = field(default_factory=dict)
    floats: Mapping[str, tuple[float, float]] = field(default_factory=dict)

    def is_empty(self) -> bool:
        return not self.ints and not self.floats

    def validate(self) -> None:
        """Fail before the search starts; a bad range found mid-study is a wasted study."""
        for name, (low, high) in self.ints.items():
            if low >= high:
                raise TuningError(f"int range for {name} must be increasing")
        for name, (low, high) in self.floats.items():
            if low >= high:
                raise TuningError(f"float range for {name} must be increasing")

    def suggest(self, trial: optuna.Trial) -> dict[str, Any]:
        params: dict[str, Any] = {}
        for name, (low, high) in self.ints.items():
            params[name] = trial.suggest_int(name, low, high)
        for name, (low, high) in self.floats.items():
            params[name] = trial.suggest_float(name, low, high)
        return params


@dataclass(frozen=True)
class TuningResult:
    study_name: str
    best_params: dict[str, Any]
    best_value: float
    n_trials: int
    storage: str | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "study_name": self.study_name,
            "best_params": self.best_params,
            "best_value": self.best_value,
            "n_trials": self.n_trials,
            "storage": self.storage,
        }


def run_search(
    objective: Callable[[dict[str, Any]], float],
    space: SearchSpace,
    *,
    study_name: str,
    n_trials: int = 20,
    seed: int = 7,
    storage: str | Path | None = None,
    direction: str = "maximize",
) -> TuningResult:
    """Maximise (or minimise) ``objective`` over ``space``.

    ``storage`` accepts a path; the study is loaded when it already exists, so a
    re-run continues instead of discarding earlier trials.
    """
    if space.is_empty():
        raise TuningError("search space is empty")
    space.validate()
    if n_trials < 1:
        raise TuningError("n_trials must be >= 1")
    if direction not in {"maximize", "minimize"}:
        raise TuningError(f"direction must be maximize or minimize, got {direction!r}")

    storage_url = None
    if storage is not None:
        path = Path(storage)
        path.parent.mkdir(parents=True, exist_ok=True)
        storage_url = f"sqlite:///{path.as_posix()}"

    study = optuna.create_study(
        study_name=study_name,
        direction=direction,
        sampler=TPESampler(seed=seed),
        storage=storage_url,
        load_if_exists=True,
    )

    def _wrapped(trial: optuna.Trial) -> float:
        params = space.suggest(trial)
        trial.set_user_attr("params", params)
        value = float(objective(params))
        if value != value:  # NaN means the trial produced no usable score
            raise optuna.TrialPruned("objective returned NaN")
        return value

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study.optimize(_wrapped, n_trials=n_trials, catch=(Exception,), show_progress_bar=False)
    if not study.best_trials:
        raise TuningError(f"no trial completed for study {study_name!r}")

    return TuningResult(
        study_name=study_name,
        best_params=dict(study.best_params),
        best_value=float(study.best_value),
        n_trials=len(study.trials),
        storage=str(storage) if storage is not None else None,
    )


__all__ = ["SearchSpace", "TuningError", "TuningResult", "run_search"]
