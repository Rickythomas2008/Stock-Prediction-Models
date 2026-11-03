"""A-B-C-D swing pattern, migrated from ``agent/23.abcd-strategy-agent.ipynb``.

The original ``abcd(trend, skip_loop=4, ma=7)`` walks four nested loops over the
whole MA history -- O(n^4) candidate quadruples -- and then marks every index that
appears as an A/C point but not as a B/D point as a buy. On a few thousand bars
that is billions of comparisons, which is why the notebook needs ``%%time`` and
why it cannot run inside a decision loop.

This module keeps the *shape* being detected (A < B > C < D, with C > A and
D > B on the moving average) and replaces the search with a single-pass swing
detector plus a confirmation check against the most recent bar. Differences from
the notebook are listed in the class docstring and in ``docs/SIGNAL-MIGRATION.md``.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd

from stockpred.pipeline import _history_until

# An A-B-C triangle needs three swing points, i.e. at least five MA observations
# (low, high, low, plus the two bars that establish the turns).
_MIN_WINDOW_POINTS = 5

_HIGH = "high"
_LOW = "low"


def _swing_pivots(values: np.ndarray) -> list[tuple[int, str]]:
    """Alternating swing highs/lows, in one pass over ``values``.

    A pivot is emitted only when the direction of travel reverses, so the result
    alternates high/low and no two pivots share a bar. This is the linear
    stand-in for the notebook's quadruple search: it keeps the turns that could
    plausibly play the role of A, B, C or D and discards everything between them.
    """
    pivots: list[tuple[int, str]] = []
    if values.size < 2:
        return pivots
    direction = 0  # 0 = undecided, +1 = rising, -1 = falling
    extreme = 0
    for i in range(1, values.size):
        if values[i] < values[extreme]:
            if direction > 0:
                pivots.append((extreme, _HIGH))
            if direction >= 0:
                direction = -1
            extreme = i
        elif values[i] > values[extreme]:
            if direction < 0:
                pivots.append((extreme, _LOW))
            if direction <= 0:
                direction = 1
            extreme = i
    return pivots


def _find_abc(
    values: np.ndarray, pivots: list[tuple[int, str]], min_span: int
) -> tuple[int, int, int] | None:
    """Most recent A-B-C triple, newest first; ``None`` when the shape is absent.

    ``min_span`` is the original ``skip_loop`` stride expressed as a rule: legs
    shorter than that were never visited by the notebook's ``range(..., skip)``
    walk, so they must not count here either.
    """
    for j in range(len(pivots) - 1, 1, -1):
        a_index, a_kind = pivots[j - 2]
        b_index, b_kind = pivots[j - 1]
        c_index, c_kind = pivots[j]
        if c_kind != _LOW or b_kind != _HIGH or a_kind != _LOW:
            continue
        if b_index - a_index < min_span or c_index - b_index < min_span:
            continue
        if values[b_index] > values[a_index] and values[c_index] < values[b_index]:
            if values[c_index] > values[a_index]:
                return a_index, b_index, c_index
    return None


@dataclass(frozen=True)
class ABCDSignal:
    """Score the last bar against the newest A-B-C swing on the MA.

    Verdicts: the D leg has taken out B -> ``+1.0``; the shape is intact but D is
    still inside the A-B-C range -> ``0.0``; price has lost C -> ``-1.0``.

    What differs from the notebook's ``abcd()``:

    * *Complexity*: the notebook enumerates every ``(a, b, c, d)`` quadruple with a
      stride of ``skip_loop`` over the full history (O(n^4), and it materialises
      every hit in a Python list). Here swing points are found in one pass and
      only the most recent A-B-C triple of the last ``lookback`` bars is used:
      O(lookback).
    * *What the output means*: the notebook returns a per-bar array that marks
      A/C-but-not-B/D bars as ``+1`` and B/D-but-not-A/C bars as ``-1`` -- a
      retroactive marker of where the shape happened, which cannot be traded
      (it needs the future to be complete). Here the output is a stance at
      ``as_of``: has the pattern been confirmed by D, or has it failed.
    * *Lookback and stride*: the notebook saw the whole history and used
      ``skip_loop`` as a search stride. Here ``lookback`` bounds the window so the
      cost per symbol is constant, and ``skip_loop`` becomes the minimum leg
      length, because short legs were never sampled by the stride anyway.
    * *Missing pieces*: no usable A-B-C -> ``0.0`` (the notebook would emit all
      zeros too, since no quadruples means empty buy/sell sets).
    """

    skip_loop: int = 4
    ma_window: int = 7
    lookback: int = 60
    name: str = "abcd"

    def __post_init__(self) -> None:
        if self.skip_loop < 1:
            raise ValueError("skip_loop must be at least 1")
        if self.ma_window < 1:
            raise ValueError("ma_window must be at least 1")
        if self.lookback < _MIN_WINDOW_POINTS:
            raise ValueError(f"lookback must be at least {_MIN_WINDOW_POINTS}")

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

        Only the last ``lookback`` MA values are inspected, so a long history
        costs nothing extra and the result cannot depend on bars the decision
        could not have seen.
        """
        if len(close) < self.ma_window + _MIN_WINDOW_POINTS:
            return float("nan")
        ma = close.rolling(self.ma_window).mean().dropna()
        if ma.size < _MIN_WINDOW_POINTS:
            return float("nan")
        values = ma.to_numpy(dtype=float)[-self.lookback :]

        triple = _find_abc(values, _swing_pivots(values), self.skip_loop)
        if triple is None:
            return 0.0
        _, b_index, c_index = triple

        # Failure is checked first: once price closes below C the "up-down-up"
        # reading is dead, no matter what D did before that.
        if values[-1] < values[c_index]:
            return -1.0
        if values[c_index + 1 :].size and values[c_index + 1 :].max() > values[b_index]:
            return 1.0
        return 0.0
