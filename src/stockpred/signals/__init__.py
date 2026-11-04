"""Rule-based signal generators migrated from the legacy notebooks' agents.

Every class implements :class:`stockpred.pipeline.AlphaModel` (``name`` plus
``score(bars, as_of=...)``), so a signal can be handed straight to
:class:`stockpred.pipeline.Pipeline` in place of ``MomentumAlpha``.

Scores are a stance in ``-1..+1``: negative is bearish. They are *not* expected
returns -- the constructors only rank them.
"""

from __future__ import annotations

from stockpred.signals.factors import LowVolatilityAlpha, ReversalAlpha
from stockpred.signals.pattern import ABCDSignal
from stockpred.signals.rolling import SignalRollingSignal, signal_rolling_states
from stockpred.signals.trend import MovingAverageSignal, TurtleSignal

__all__ = [
    "ABCDSignal",
    "LowVolatilityAlpha",
    "MovingAverageSignal",
    "ReversalAlpha",
    "SignalRollingSignal",
    "TurtleSignal",
    "signal_rolling_states",
]
