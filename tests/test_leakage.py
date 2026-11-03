from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stockpred.errors import LookaheadError
from stockpred.eval.leakage import assert_causal, find_lookahead


@pytest.fixture
def frame() -> pd.DataFrame:
    n = 30
    return pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=n),
            "close": 10.0 + np.arange(n, dtype=float) * 0.1,
        }
    )


def test_trailing_mean_is_causal(frame: pd.DataFrame) -> None:
    assert_causal(lambda df: df["close"].rolling(3).mean(), frame)
    assert find_lookahead(lambda df: df["close"].rolling(3).mean(), frame) is None


def test_shift_negative_is_detected(frame: pd.DataFrame) -> None:
    with pytest.raises(LookaheadError, match="depends on future data"):
        assert_causal(lambda df: df["close"].shift(-1), frame)


def test_centred_rolling_mean_is_detected(frame: pd.DataFrame) -> None:
    with pytest.raises(LookaheadError):
        assert_causal(lambda df: df["close"].rolling(3, center=True).mean(), frame)


def test_full_series_statistic_is_detected(frame: pd.DataFrame) -> None:
    leaky = lambda df: pd.Series(np.full(len(df), df["close"].max()))  # noqa: E731

    with pytest.raises(LookaheadError, match="depends on future data"):
        assert_causal(leaky, frame, label="max_feature")


def test_future_shift_positive_is_causal(frame: pd.DataFrame) -> None:
    assert_causal(lambda df: df["close"].shift(1), frame)


def test_find_lookahead_reports_a_cut_point(frame: pd.DataFrame) -> None:
    cut = find_lookahead(lambda df: df["close"].shift(-1), frame)
    assert cut is not None
    assert 2 <= cut < len(frame)


def test_short_frame_is_rejected() -> None:
    with pytest.raises(ValueError, match="at least 4 rows"):
        assert_causal(lambda df: df["close"], pd.DataFrame({"close": [1.0, 2.0]}))
