from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from stockpred.pipeline import PipelineError
from stockpred.signals.factors import LowVolatilityAlpha, ReversalAlpha

AS_OF = date(2024, 3, 20)


def _frame(closes: list[float]) -> pd.DataFrame:
    days = pd.bdate_range(end="2024-03-20", periods=len(closes))
    return pd.DataFrame(
        {
            "date": days,
            "open": closes,
            "high": [c * 1.01 for c in closes],
            "low": [c * 0.99 for c in closes],
            "close": closes,
            "volume": [1_000_000.0] * len(closes),
        }
    )


def _bars(**series: list[float]) -> dict[str, pd.DataFrame]:
    return {symbol: _frame(closes) for symbol, closes in series.items()}


def test_reversal_ranks_the_worst_recent_performer_first() -> None:
    bars = _bars(
        UP=[10.0, 10.2, 10.4, 10.6, 10.8, 11.0],
        FLAT=[10.0, 10.0, 10.0, 10.0, 10.0, 10.0],
        DOWN=[10.0, 9.8, 9.6, 9.4, 9.2, 9.0],
    )

    scores = ReversalAlpha(lookback=5).score(bars, as_of=AS_OF)

    assert list(scores.index) == ["DOWN", "FLAT", "UP"]
    assert scores["DOWN"] == pytest.approx(0.10, abs=1e-9)


def test_low_volatility_ranks_the_calmest_name_first() -> None:
    bars = _bars(
        CALM=[10.0, 10.01, 10.0, 10.01, 10.0, 10.01, 10.0],
        WILD=[10.0, 10.9, 10.0, 10.9, 10.0, 10.9, 10.0],
    )

    scores = LowVolatilityAlpha(lookback=6).score(bars, as_of=AS_OF)

    assert list(scores.index) == ["CALM", "WILD"]
    assert scores["CALM"] > scores["WILD"]


def test_reversal_needs_enough_history() -> None:
    bars = _bars(UP=[10.0, 10.1, 10.2])
    with pytest.raises(PipelineError, match="no symbol has"):
        ReversalAlpha(lookback=30).score(bars, as_of=AS_OF)


def test_low_volatility_needs_enough_history() -> None:
    bars = _bars(UP=[10.0, 10.1, 10.2])
    with pytest.raises(PipelineError, match="no symbol has"):
        LowVolatilityAlpha(lookback=30).score(bars, as_of=AS_OF)


def test_alphas_reject_degenerate_parameters() -> None:
    with pytest.raises(ValueError):
        ReversalAlpha(lookback=0)
    with pytest.raises(ValueError):
        LowVolatilityAlpha(lookback=1)
