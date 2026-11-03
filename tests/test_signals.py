"""Contract tests for the migrated rule-based signals.

The four classes under test replace the notebooks' per-bar signal arrays with one
score per symbol at ``as_of``, so every test here is about the seam the
:class:`~stockpred.pipeline.AlphaModel` protocol exposes: direction, missing
history, causality, and whether the pipeline can actually consume them.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from datetime import date

import numpy as np
import pandas as pd
import pytest

from stockpred.errors import LookaheadError
from stockpred.eval.leakage import assert_causal
from stockpred.pipeline import AlphaModel, Pipeline, TopNConstructor
from stockpred.risk import RiskState
from stockpred.signals import (
    ABCDSignal,
    MovingAverageSignal,
    SignalRollingSignal,
    TurtleSignal,
)
from stockpred.signals.rolling import signal_rolling_states

ALL_SIGNALS = [
    pytest.param(TurtleSignal(), id="turtle"),
    pytest.param(MovingAverageSignal(), id="moving-average"),
    pytest.param(SignalRollingSignal(), id="signal-rolling"),
    pytest.param(ABCDSignal(), id="abcd"),
]


def _frame(closes: Sequence[float], start: str = "2024-01-01") -> pd.DataFrame:
    values = [float(value) for value in closes]
    days = pd.date_range(start, periods=len(values), freq="B")
    return pd.DataFrame(
        {
            "date": days,
            "open": values,
            "high": [value * 1.01 for value in values],
            "low": [value * 0.99 for value in values],
            "close": values,
            "volume": 1_000_000.0,
        }
    )


def _segment(start: float, end: float, count: int) -> list[float]:
    return list(np.linspace(start, end, count, endpoint=False))


def _zigzag(seed: int = 3, legs: int = 12, leg: int = 12) -> list[float]:
    """Alternating up/down legs, so the ABCD swing detector has work to do."""
    rng = np.random.default_rng(seed)
    price = 100.0
    out: list[float] = []
    for k in range(legs):
        target = price + (25 if k % 2 == 0 else -20) * (0.7 + 0.6 * rng.random())
        out += list(np.linspace(price, target, leg, endpoint=False))
        price = target
    return out


def _uptrend(count: int = 120) -> list[float]:
    return [100.0 * 1.014**i for i in range(count)]


def _downtrend(count: int = 120) -> list[float]:
    return [100.0 * 0.986**i for i in range(count)]


def _abcd_with_confirmed_d() -> list[float]:
    """A(90) -> B(118) -> C(100) -> D runs through B."""
    return (
        _segment(100, 90, 13)
        + _segment(90, 118, 13)
        + _segment(118, 100, 13)
        + _segment(100, 135, 13)
    )


def _abcd_with_unconfirmed_d() -> list[float]:
    """Same A-B-C, then a weak D leg that stalls below B."""
    return (
        _segment(100, 90, 13)
        + _segment(90, 118, 13)
        + _segment(118, 100, 13)
        + _segment(100, 104, 6)
    )


def _abcd_that_breaks_c() -> list[float]:
    """Same A-B-C, a weak D, then a close below C."""
    return (
        _segment(100, 90, 13)
        + _segment(90, 118, 13)
        + _segment(118, 100, 13)
        + _segment(100, 108, 6)
        + _segment(108, 88, 12)
    )


def _prefix_feature(signal):
    """Turn a one-shot scorer into a per-bar feature so ``assert_causal`` can run.

    A prefix-only scorer is causal by construction; the point of the check is
    that it stays causal after the migration, i.e. that no scorer reads past the
    series it is handed.
    """

    def feature(frame: pd.DataFrame) -> pd.Series:
        close = frame["close"].astype(float)
        values = [signal._score_series(close.iloc[: i + 1]) for i in range(len(close))]
        return pd.Series(values, index=frame.index, dtype=float)

    return feature


def test_turtle_scores_breakout_up_and_breakdown_down() -> None:
    turtle = TurtleSignal(count=20)
    assert turtle._score_series(pd.Series(_uptrend())) == 1.0
    assert turtle._score_series(pd.Series(_downtrend())) == -1.0


def test_moving_average_follows_the_cross() -> None:
    ma = MovingAverageSignal(short_window=10, long_window=30)
    assert ma._score_series(pd.Series(_uptrend())) == 1.0
    assert ma._score_series(pd.Series(_downtrend())) == -1.0


def test_signal_rolling_stays_long_only() -> None:
    """A crash must score 0, not -1: the original agent never shorted."""

    rolling = SignalRollingSignal(window=5, threshold=0.02, delay=2)
    assert rolling._score_series(pd.Series(_uptrend())) == 1.0
    assert rolling._score_series(pd.Series(_downtrend())) == 0.0


def test_abcd_confirmed_d_scores_long() -> None:
    assert ABCDSignal()._score_series(pd.Series(_abcd_with_confirmed_d())) == 1.0


def test_abcd_break_below_c_scores_short() -> None:
    assert ABCDSignal()._score_series(pd.Series(_abcd_that_breaks_c())) == -1.0


def test_abcd_structure_without_d_confirmation_is_flat() -> None:
    """A-B-C intact but D has not taken out B: no position, either way."""

    assert ABCDSignal()._score_series(pd.Series(_abcd_with_unconfirmed_d())) == 0.0


def test_abcd_without_structure_is_flat() -> None:
    """No A-B-C turns means nothing to trade -- a monotone ramp is not a pattern."""

    assert ABCDSignal()._score_series(pd.Series(_uptrend())) == 0.0


@pytest.mark.parametrize("signal", ALL_SIGNALS)
def test_symbol_with_short_history_is_skipped(signal) -> None:
    bars = {
        "LONG": _frame(_uptrend()),
        "SHORT": _frame([100.0, 101.0, 102.0, 101.5, 103.0]),
    }
    as_of = date(2024, 12, 31)

    scores = signal.score(bars, as_of=as_of)

    assert "SHORT" not in scores.index
    assert "LONG" in scores.index
    assert scores.dtype == float


@pytest.mark.parametrize("signal", ALL_SIGNALS)
def test_no_usable_symbol_returns_empty_not_an_exception(signal) -> None:
    """Skipping is per symbol; if nothing qualifies the caller decides, not us."""

    bars = {"SHORT": _frame([100.0, 101.0, 102.0])}
    scores = signal.score(bars, as_of=date(2024, 12, 31))
    assert scores.empty
    assert scores.dtype == float


def test_moving_average_rejects_inverted_windows() -> None:
    with pytest.raises(ValueError, match="short_window"):
        MovingAverageSignal(short_window=30, long_window=30)
    with pytest.raises(ValueError, match="short_window"):
        MovingAverageSignal(short_window=40, long_window=30)


def test_signal_rolling_states_respect_the_delay() -> None:
    # 12 flat bars, one spike that triggers entry, then a crash: the exit
    # threshold is met on the first crash bar but must wait ``delay`` bars.
    path = [100.0] * 12 + [106.0] + [90.0] * 8
    close = pd.Series(path)

    states = signal_rolling_states(close, window=5, threshold=0.02, delay=2)

    assert states.iloc[:12].eq(0).all(), "warm-up must not be long"
    assert states.iloc[12] == 1, "spike above the trailing mean must enter"
    assert states.iloc[13] == 1, "flip inside the delay window must be refused"
    assert states.iloc[14] == 0, "flip must happen once the delay has elapsed"


def test_signal_rolling_delay_is_what_blocks_the_flip() -> None:
    path = [100.0] * 12 + [106.0] + [90.0] * 8
    close = pd.Series(path)

    blocked = signal_rolling_states(close, window=5, threshold=0.02, delay=2)
    immediate = signal_rolling_states(close, window=5, threshold=0.02, delay=0)

    assert blocked.iloc[13] == 1
    assert immediate.iloc[13] == 0, "without the delay the same bar would flip"


def test_signal_rolling_states_are_prefix_stable() -> None:
    """The state at a bar must not change when later bars arrive."""

    close = pd.Series(_uptrend(40) + _downtrend(20))
    full = signal_rolling_states(close, window=5, threshold=0.02, delay=2)
    cut = signal_rolling_states(close.iloc[:30], window=5, threshold=0.02, delay=2)
    pd.testing.assert_series_equal(full.iloc[:30], cut)


def test_abcd_single_symbol_is_fast() -> None:
    rng = np.random.default_rng(7)
    prices = 100.0 + np.cumsum(rng.normal(0.0, 1.0, 500))
    close = pd.Series(prices)
    signal = ABCDSignal(lookback=60)

    timings = []
    for _ in range(5):
        started = time.perf_counter()
        signal._score_series(close)
        timings.append(time.perf_counter() - started)

    median_ms = sorted(timings)[len(timings) // 2] * 1000.0
    assert median_ms < 5.0, f"ABCD scored in {median_ms:.2f} ms, budget is 5 ms"


@pytest.mark.parametrize("signal", ALL_SIGNALS)
def test_signal_satisfies_the_alpha_protocol(signal) -> None:
    assert isinstance(signal, AlphaModel)
    assert isinstance(signal.name, str) and signal.name


@pytest.mark.parametrize("signal", ALL_SIGNALS)
def test_signal_drives_the_pipeline(signal) -> None:
    """The whole point of the migration: drop-in for ``MomentumAlpha``."""

    rng = np.random.default_rng(11)
    bars = {
        "UP": _frame(_uptrend()),
        "DOWN": _frame(_downtrend()),
        "ABCD": _frame(_abcd_with_confirmed_d() * 3),
        "CHOP": _frame(list(100.0 + np.cumsum(rng.normal(0.0, 1.2, 120)))),
    }
    prices = {symbol: float(frame["close"].iloc[-1]) for symbol, frame in bars.items()}
    as_of = date(2024, 12, 31)

    pipeline = Pipeline(
        signal,
        TopNConstructor(top_n=2, max_position=0.25),
        capital=100_000.0,
    )
    package, plan = pipeline.decide(
        bars,
        as_of=as_of,
        bar_date=as_of,
        prices=prices,
        state=RiskState(cash=100_000.0),
    )

    assert package.signal
    assert package.rationale.startswith(f"alpha={signal.name}")
    assert all(shares % 100 == 0 for shares in package.target_shares.values())
    assert isinstance(plan.proposals, tuple)


@pytest.mark.parametrize(
    ("signal", "closes"),
    [
        pytest.param(
            TurtleSignal(),
            _uptrend(40) + _downtrend(20) + _uptrend(20),
            id="turtle",
        ),
        pytest.param(
            MovingAverageSignal(),
            _uptrend(40) + _downtrend(20) + _uptrend(20),
            id="moving-average",
        ),
        pytest.param(
            SignalRollingSignal(),
            _uptrend(40) + _downtrend(20) + _uptrend(20),
            id="signal-rolling",
        ),
        pytest.param(ABCDSignal(), _zigzag(), id="abcd"),
    ],
)
def test_scores_are_causal(signal, closes: list[float]) -> None:
    frame = _frame(closes)
    feature = _prefix_feature(signal)
    values = feature(frame).dropna()
    # A constant feature would satisfy the causality check vacuously.
    assert values.nunique() > 1, f"{signal.name} scored the same value on every bar"
    assert_causal(feature, frame, label=signal.name)


def test_causality_guard_would_catch_a_leak() -> None:
    """Negative control: the check above is only evidence if it can fail."""

    frame = _frame(_uptrend(40) + _downtrend(20))

    def leaking(frame_: pd.DataFrame) -> pd.Series:
        close = frame_["close"].astype(float)
        return pd.Series(
            [close.iloc[: i + 1].max() / close.max() for i in range(len(close))],
            index=frame_.index,
            dtype=float,
        )

    with pytest.raises(LookaheadError, match="depends on future data"):
        assert_causal(leaking, frame, label="leak")


@pytest.mark.parametrize("signal", ALL_SIGNALS)
def test_rows_after_as_of_are_ignored(signal) -> None:
    history = _uptrend(80)
    future = _downtrend(40)
    trimmed = {"X": _frame(history)}
    extended = {"X": _frame(history + [value for value in future if value > 0])}
    as_of = date(2024, 3, 1)

    cut = signal.score(trimmed, as_of=as_of)
    full = signal.score(extended, as_of=as_of)

    pd.testing.assert_series_equal(cut, full)
