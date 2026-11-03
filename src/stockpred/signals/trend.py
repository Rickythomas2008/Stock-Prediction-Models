"""Trend signals: Donchian breakout (turtle) and moving-average cross.

Both are migrations of the rule-based agents in ``agent/1.turtle-agent.ipynb`` and
``agent/2.moving-average-agent.ipynb``. The notebooks emit a per-bar signal series
for one symbol; here the same rule is evaluated once, on the bar visible up to
``as_of``, so the result can be dropped into :class:`stockpred.pipeline.Pipeline`
as an ``AlphaModel``. ``docs/SIGNAL-MIGRATION.md`` records the mapping and the
deliberate differences.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date

import pandas as pd

from stockpred.pipeline import _history_until


@dataclass(frozen=True)
class TurtleSignal:
    """Donchian breakout on the close: new high -> ``+1``, new low -> ``-1``.

    The reference range is shifted by one bar, like the original
    ``trend.shift(1).rolling(count).max()``: a bar must not sit inside its own
    range, otherwise every bar would be its own high and the rule would never fire.
    """

    count: int = 20
    name: str = "turtle"

    def __post_init__(self) -> None:
        if self.count < 1:
            raise ValueError("count must be at least 1")

    def score(self, bars: Mapping[str, pd.DataFrame], *, as_of: date) -> pd.Series:
        scores: dict[str, float] = {}
        for symbol, frame in bars.items():
            history = _history_until(frame, as_of)
            if history is None:
                continue
            value = self._score_series(history["close"].astype(float))
            if pd.isna(value):
                continue
            scores[symbol] = float(value)
        return pd.Series(scores, dtype=float).sort_values(ascending=False)

    def _score_series(self, close: pd.Series) -> float:
        """Score the last bar of ``close``; ``NaN`` means "not enough history".

        Returning ``NaN`` rather than scoring a short window keeps a partial
        window from silently producing a weaker signal -- the caller skips the
        symbol instead.
        """
        if len(close) < self.count + 1:
            return float("nan")
        values = close.to_numpy(dtype=float)
        reference = values[-self.count - 1 : -1]
        last = values[-1]
        if last > reference.max():
            return 1.0
        if last < reference.min():
            return -1.0
        return 0.0


@dataclass(frozen=True)
class MovingAverageSignal:
    """Fast/slow MA cross: fast above slow -> ``+1``, fast below -> ``-1``.

    The original guards the comparison with ``min_periods=1`` and only ever emits
    ``0``/``1``; here the windows must be full, because a half-warm MA compared
    against another half-warm MA is not a cross.
    """

    short_window: int = 10
    long_window: int = 30
    name: str = "moving_average"

    def __post_init__(self) -> None:
        if self.short_window < 1:
            raise ValueError("short_window must be at least 1")
        if self.short_window >= self.long_window:
            raise ValueError(
                f"short_window ({self.short_window}) must be smaller than "
                f"long_window ({self.long_window})"
            )

    def score(self, bars: Mapping[str, pd.DataFrame], *, as_of: date) -> pd.Series:
        scores: dict[str, float] = {}
        for symbol, frame in bars.items():
            history = _history_until(frame, as_of)
            if history is None:
                continue
            value = self._score_series(history["close"].astype(float))
            if pd.isna(value):
                continue
            scores[symbol] = float(value)
        return pd.Series(scores, dtype=float).sort_values(ascending=False)

    def _score_series(self, close: pd.Series) -> float:
        """Score the last bar of ``close``; ``NaN`` means "not enough history"."""
        if len(close) < self.long_window:
            return float("nan")
        values = close.to_numpy(dtype=float)
        short = values[-self.short_window :].mean()
        long = values[-self.long_window :].mean()
        if short > long:
            return 1.0
        if short < long:
            return -1.0
        return 0.0
