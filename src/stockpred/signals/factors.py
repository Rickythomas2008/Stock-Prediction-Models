"""Cross-sectional factor alphas: short-horizon reversal and low volatility.

Both implement :class:`stockpred.pipeline.AlphaModel`, so they slot into the same
``Pipeline`` as ``MomentumAlpha`` and are compared on identical walk-forward folds
rather than argued about.

Why these two: 20-day momentum on a 20-name book behaved as downside protection,
not as an edge (docs/CONFIG-WALKFORWARD.md). Reversal is the other side of that
same short-horizon effect, and low volatility is the classic defensive tilt that
does not depend on a directional call. All three are price-only, so the
comparison is not confounded by a different data source.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date

import pandas as pd

from stockpred.pipeline import PipelineError, _history_until


@dataclass(frozen=True)
class ReversalAlpha:
    """Score = minus the trailing return: the worst recent performer ranks first."""

    lookback: int = 5
    name: str = "reversal"

    def __post_init__(self) -> None:
        if self.lookback < 1:
            raise ValueError(f"lookback must be at least 1, got {self.lookback!r}")

    def score(self, bars: Mapping[str, pd.DataFrame], *, as_of: date) -> pd.Series:
        scores: dict[str, float] = {}
        for symbol, frame in bars.items():
            history = _history_until(frame, as_of)
            if history is None or len(history) < self.lookback + 1:
                continue
            close = history["close"].astype(float).to_numpy()
            past = close[-self.lookback - 1]
            if past <= 0:
                continue
            scores[symbol] = float(-(close[-1] / past - 1.0))
        if not scores:
            raise PipelineError(f"no symbol has {self.lookback + 1}+ bars up to {as_of}")
        return pd.Series(scores, dtype=float).sort_values(ascending=False)


@dataclass(frozen=True)
class LowVolatilityAlpha:
    """Score = minus realised volatility: the calmest name ranks first."""

    lookback: int = 60
    name: str = "low_vol"

    def __post_init__(self) -> None:
        if self.lookback < 2:
            raise ValueError(f"lookback must be at least 2, got {self.lookback!r}")

    def score(self, bars: Mapping[str, pd.DataFrame], *, as_of: date) -> pd.Series:
        scores: dict[str, float] = {}
        for symbol, frame in bars.items():
            history = _history_until(frame, as_of)
            if history is None or len(history) < self.lookback + 1:
                continue
            close = history["close"].astype(float)
            vol = close.pct_change().tail(self.lookback).std()
            if vol is None or not pd.notna(vol):
                continue
            scores[symbol] = float(-vol)
        if not scores:
            raise PipelineError(f"no symbol has {self.lookback + 1}+ bars up to {as_of}")
        return pd.Series(scores, dtype=float).sort_values(ascending=False)
