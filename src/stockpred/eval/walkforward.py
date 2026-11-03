"""Walk-forward splits with purge and embargo.

Random or single train/test splits leak future information into the training
set. This module produces expanding (or rolling) folds where the test window
always follows the train window, with a gap that removes label overlap.

Terminology (Lopez de Prado):
* **purge** - drop training samples whose label horizon reaches into the test window
* **embargo** - drop training samples immediately after the test window
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import pandas as pd

from stockpred.errors import StockpredError


class InvalidSplitError(StockpredError):
    """A split configuration that cannot produce a leak-free fold."""


@dataclass(frozen=True)
class Fold:
    index: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp

    def describe(self) -> str:
        return (
            f"fold {self.index}: train [{self.train_start.date()}..{self.train_end.date()}] "
            f"test [{self.test_start.date()}..{self.test_end.date()}]"
        )


def _as_index(dates) -> pd.DatetimeIndex:
    index = pd.DatetimeIndex(pd.to_datetime(pd.Series(list(dates)))).normalize().unique()
    return index.sort_values()


def walk_forward_folds(
    dates,
    *,
    train_size: int,
    test_size: int,
    step: int | None = None,
    purge: int = 0,
    embargo: int = 0,
    max_train_size: int | None = None,
) -> list[Fold]:
    """Generate leak-free expanding (or rolling) folds over ``dates``.

    ``train_size`` / ``test_size`` / ``purge`` / ``embargo`` are counted in
    **observations** (rows), not calendar days, because label horizons are
    defined in rows.
    """
    index = _as_index(dates)
    if train_size <= 0 or test_size <= 0:
        raise InvalidSplitError("train_size and test_size must be positive")
    if purge < 0 or embargo < 0:
        raise InvalidSplitError("purge and embargo must be non-negative")
    step = test_size if step is None else step
    if step <= 0:
        raise InvalidSplitError("step must be positive")
    if max_train_size is not None and max_train_size < train_size:
        raise InvalidSplitError("max_train_size must be >= train_size")

    folds: list[Fold] = []
    fold_index = 0
    cursor = train_size
    while cursor + test_size <= len(index):
        train_stop = cursor - purge
        train_start_pos = 0 if max_train_size is None else max(0, train_stop - max_train_size)
        if train_stop - train_start_pos <= 0:
            raise InvalidSplitError("purge consumed the whole training window")

        test_slice = index[cursor : cursor + test_size]
        folds.append(
            Fold(
                index=fold_index,
                train_start=index[train_start_pos],
                train_end=index[train_stop - 1],
                test_start=test_slice[0],
                test_end=test_slice[-1],
            )
        )
        fold_index += 1
        cursor += step + embargo

    if not folds:
        raise InvalidSplitError(
            f"no fold fits: {len(index)} observations, train {train_size}, test {test_size}"
        )
    return folds


def assert_no_overlap(folds: list[Fold]) -> None:
    """Raise if any fold's training window touches or passes its test window."""
    for fold in folds:
        if fold.train_end >= fold.test_start:
            raise InvalidSplitError(
                f"fold {fold.index} leaks: train ends {fold.train_end.date()} "
                f">= test starts {fold.test_start.date()}"
            )
    for earlier, later in zip(folds, folds[1:], strict=False):
        if earlier.test_end >= later.test_start:
            raise InvalidSplitError(
                f"folds {earlier.index} and {later.index} overlap in the test window"
            )


def slice_frame(frame: pd.DataFrame, fold: Fold, *, part: str = "train") -> pd.DataFrame:
    """Return rows of ``frame`` (indexed or columned by ``date``) for a fold part."""
    column = "date" if "date" in frame.columns else frame.index.name
    if column is None or column not in getattr(frame, "columns", []):
        series = pd.to_datetime(frame.index)
        mask = (series >= (fold.train_start if part == "train" else fold.test_start)) & (
            series <= (fold.train_end if part == "train" else fold.test_end)
        )
        return frame.loc[mask]

    values = pd.to_datetime(frame[column])
    lo = fold.train_start if part == "train" else fold.test_start
    hi = fold.train_end if part == "train" else fold.test_end
    return frame.loc[(values >= lo) & (values <= hi)]


def fold_boundaries(folds: list[Fold]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "fold": f.index,
                "train_start": f.train_start.date(),
                "train_end": f.train_end.date(),
                "test_start": f.test_start.date(),
                "test_end": f.test_end.date(),
            }
            for f in folds
        ]
    )


def to_date_range(fold: Fold) -> tuple[date, date]:
    return fold.test_start.date(), fold.test_end.date()
