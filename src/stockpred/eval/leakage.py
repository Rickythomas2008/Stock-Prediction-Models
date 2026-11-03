"""Lookahead detection by construction.

Rationale: a leaking feature changes when you append future rows. So the test
is mechanical -- recompute the feature on a truncated frame and compare the
rows that both versions contain. If they differ, the feature saw the future.

This is stronger than eyeballing ``shift`` signs, and it runs in CI.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import pandas as pd

from stockpred.errors import LookaheadError

FeatureFn = Callable[[pd.DataFrame], pd.Series]


def assert_causal(
    feature_fn: FeatureFn,
    frame: pd.DataFrame,
    *,
    cut_points: Sequence[int] | None = None,
    atol: float = 1e-9,
    label: str = "feature",
) -> None:
    """Raise ``LookaheadError`` if ``feature_fn`` depends on rows after each cut."""
    if len(frame) < 4:
        raise ValueError("need at least 4 rows to test causality")
    points = list(cut_points) if cut_points is not None else _default_cut_points(len(frame))
    base = feature_fn(frame)
    for cut in points:
        truncated = frame.iloc[:cut].reset_index(drop=True)
        recomputed = feature_fn(truncated)
        if not recomputed.index.equals(base.index[:cut]):
            recomputed = pd.Series(recomputed.to_numpy(), index=base.index[:cut])
        left = base.iloc[:cut].to_numpy(dtype=float)
        right = recomputed.to_numpy(dtype=float)
        mismatched = ~_close(left, right, atol)
        mismatched &= ~(_isnan(left) & _isnan(right))
        if mismatched.any():
            row = int(mismatched.argmax())
            raise LookaheadError(
                f"{label} depends on future data: at cut={cut} row {row} "
                f"computed {right[row]!r} with history vs {left[row]!r} with future rows"
            )


def find_lookahead(
    feature_fn: FeatureFn,
    frame: pd.DataFrame,
    *,
    cut_points: Sequence[int] | None = None,
    atol: float = 1e-9,
) -> int | None:
    """First cut point that leaks, or ``None`` when the feature is causal."""
    try:
        assert_causal(feature_fn, frame, cut_points=cut_points, atol=atol)
    except LookaheadError as exc:
        text = str(exc)
        marker = "cut="
        start = text.find(marker)
        if start == -1:
            return 0
        digits = text[start + len(marker) :].split()[0]
        return int(digits)
    return None


def _default_cut_points(length: int) -> list[int]:
    candidates = {length // 3, length // 2, (2 * length) // 3, length - 1}
    return sorted(p for p in candidates if 2 <= p < length)


def _close(left, right, atol: float):
    try:
        import numpy as np

        return np.isclose(left, right, atol=atol, rtol=0.0, equal_nan=True)
    except TypeError:  # pragma: no cover - very old numpy
        return left == right


def _isnan(values):
    import numpy as np

    return np.isnan(values)
