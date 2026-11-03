"""Threshold signal with a hysteresis delay, migrated from ``agent/3.signal-rolling-agent.ipynb``.

The notebook's ``buy_stock`` fuses two things that do not belong together: the
decision rule and a toy broker that prints trades. Only the decision rule is a
signal, so the state machine is factored out into a pure function
(:func:`signal_rolling_states`) and the broker stays behind in the notebook.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd

from stockpred.pipeline import _history_until


def signal_rolling_states(
    close: pd.Series,
    *,
    window: int = 5,
    threshold: float = 0.02,
    delay: int = 2,
) -> pd.Series:
    """Long-only 0/1 state machine over ``close``, with a minimum dwell time.

    Entry needs ``close / mean(close[-window:]) - 1 > threshold``; exit needs the
    same deviation below ``-threshold``. After a flip, the next flip is refused
    for ``delay`` bars.

    Pure and prefix-stable on purpose: the state at bar ``i`` depends only on
    bars ``0..i``, which is what makes the rule auditable in a backtest and lets
    a test drive it with a hand-written price path. Rows before the window is
    full stay ``0`` -- that is warm-up, not a decision.
    """
    if window < 1:
        raise ValueError("window must be at least 1")
    if delay < 0:
        raise ValueError("delay must not be negative")

    index = close.index if isinstance(close, pd.Series) else None
    values = np.asarray(pd.Series(close, dtype=float).to_numpy(), dtype=float)
    states = np.zeros(values.shape[0], dtype=np.int64)
    if values.size == 0:
        return pd.Series(states, index=index, dtype="int64")

    trailing_mean = pd.Series(values).rolling(window, min_periods=window).mean().to_numpy()

    state = 0
    last_change: int | None = None
    for i in range(values.shape[0]):
        mean = trailing_mean[i]
        if not np.isfinite(mean) or mean <= 0.0:
            states[i] = state
            continue
        deviation = values[i] / mean - 1.0
        if state == 0 and deviation > threshold:
            wanted = 1
        elif state == 1 and deviation < -threshold:
            wanted = 0
        else:
            wanted = state
        if wanted != state and (last_change is None or i - last_change >= delay):
            state = wanted
            last_change = i
        states[i] = state

    return pd.Series(states, index=index, dtype="int64")


@dataclass(frozen=True)
class SignalRollingSignal:
    """Deviation-from-mean entry with hysteresis; long-only, so scores are ``0`` or ``+1``.

    The original agent is long-only too (``state`` starts at 0 and a "sell" only
    returns it to 0). Emitting ``-1`` here would invent a short leg the strategy
    never took, so the score range is deliberately one-sided.
    """

    window: int = 5
    threshold: float = 0.02
    delay: int = 2
    name: str = "signal_rolling"

    def __post_init__(self) -> None:
        if self.window < 1:
            raise ValueError("window must be at least 1")
        if self.threshold < 0:
            raise ValueError("threshold must not be negative")
        if self.delay < 0:
            raise ValueError("delay must not be negative")

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

        One bar more than ``window`` is required: at exactly ``window`` rows the
        state machine has only just become definable, and the notebook's own loop
        also ran one bar behind the first full window.
        """
        if len(close) < self.window + 1:
            return float("nan")
        states = signal_rolling_states(
            close, window=self.window, threshold=self.threshold, delay=self.delay
        )
        return float(states.iloc[-1])
