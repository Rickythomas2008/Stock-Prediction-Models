from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from stockpred.eval.walkforward import (
    InvalidSplitError,
    assert_no_overlap,
    fold_boundaries,
    slice_frame,
    walk_forward_folds,
)


@pytest.fixture
def dates() -> pd.DatetimeIndex:
    start = pd.Timestamp("2024-01-01")
    return pd.DatetimeIndex([start + pd.Timedelta(days=i) for i in range(40)])


def test_folds_do_not_overlap(dates: pd.DatetimeIndex) -> None:
    folds = walk_forward_folds(dates, train_size=20, test_size=5)
    assert_no_overlap(folds)
    assert len(folds) == 4
    assert folds[0].train_start == dates[0]


def test_purge_shrinks_training_window(dates: pd.DatetimeIndex) -> None:
    plain = walk_forward_folds(dates, train_size=20, test_size=5)
    purged = walk_forward_folds(dates, train_size=20, test_size=5, purge=3)

    assert purged[0].train_end < plain[0].train_end
    gap = dates.get_loc(plain[0].test_start) - dates.get_loc(purged[0].train_end)
    assert gap == 4  # 3 purged rows + the boundary row


def test_embargo_skips_rows_after_test(dates: pd.DatetimeIndex) -> None:
    plain = walk_forward_folds(dates, train_size=20, test_size=5)
    embargoed = walk_forward_folds(dates, train_size=20, test_size=5, embargo=2)

    first_next = dates.get_loc(embargoed[1].test_start)
    plain_next = dates.get_loc(plain[1].test_start)
    assert first_next == plain_next + 2


def test_rolling_window_caps_training_size(dates: pd.DatetimeIndex) -> None:
    folds = walk_forward_folds(dates, train_size=10, test_size=5, max_train_size=10)
    span = dates.get_loc(folds[-1].train_end) - dates.get_loc(folds[-1].train_start)
    assert span == 9


def test_step_controls_stride(dates: pd.DatetimeIndex) -> None:
    steady = walk_forward_folds(dates, train_size=20, test_size=5)
    strided = walk_forward_folds(dates, train_size=20, test_size=5, step=10)
    assert len(strided) < len(steady)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"train_size": 0, "test_size": 5},
        {"train_size": 20, "test_size": 0},
        {"train_size": 20, "test_size": 5, "purge": -1},
        {"train_size": 20, "test_size": 5, "step": 0},
        {"train_size": 20, "test_size": 5, "max_train_size": 5},
    ],
)
def test_invalid_configs_raise(dates: pd.DatetimeIndex, kwargs: dict[str, int]) -> None:
    with pytest.raises(InvalidSplitError):
        walk_forward_folds(dates, **kwargs)


def test_too_short_series_raises() -> None:
    with pytest.raises(InvalidSplitError, match="no fold fits"):
        walk_forward_folds(pd.date_range("2024-01-01", periods=10), train_size=20, test_size=5)


def test_purge_eating_window_raises(dates: pd.DatetimeIndex) -> None:
    with pytest.raises(InvalidSplitError, match="purge consumed"):
        walk_forward_folds(dates, train_size=5, test_size=5, purge=5)


def test_leaky_fold_is_detected() -> None:
    from stockpred.eval.walkforward import Fold

    leaky = [
        Fold(
            index=0,
            train_start=pd.Timestamp("2024-01-01"),
            train_end=pd.Timestamp("2024-01-10"),
            test_start=pd.Timestamp("2024-01-08"),
            test_end=pd.Timestamp("2024-01-12"),
        )
    ]
    with pytest.raises(InvalidSplitError, match="leaks"):
        assert_no_overlap(leaky)


def test_slice_frame_train_and_test(dates: pd.DatetimeIndex) -> None:
    frame = pd.DataFrame({"date": dates[:12], "close": np.arange(12, dtype=float)})
    fold = walk_forward_folds(dates[:12], train_size=8, test_size=4)[0]

    train = slice_frame(frame, fold, part="train")
    test = slice_frame(frame, fold, part="test")

    assert len(train) == 8
    assert len(test) == 4
    assert train["date"].max() < test["date"].min()


def test_fold_boundaries_table(dates: pd.DatetimeIndex) -> None:
    folds = walk_forward_folds(dates, train_size=20, test_size=5)
    table = fold_boundaries(folds)
    assert list(table.columns) == [
        "fold",
        "train_start",
        "train_end",
        "test_start",
        "test_end",
    ]
    assert table["fold"].tolist() == list(range(len(folds)))
    assert isinstance(table["test_start"].iloc[0], date)
    assert table["test_end"].iloc[0] - table["test_start"].iloc[0] == timedelta(days=4)
