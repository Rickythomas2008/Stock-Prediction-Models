"""Self-computed price adjustment factors from raw prices plus corporate actions.

Why not use the vendor's 复权 series: akshare and baostock disagree on adjusted
returns by up to 0.50% on the same symbol and window (docs/SOURCE-PARITY.md),
because each computes its own factors. Unadjusted bars, by contrast, matched
20/20 day-for-day. So the rule here is: take raw prices (which both vendors agree
on) and derive the factors ourselves from the dividend/split record — one
adjustment convention, no vendor opinion in the middle.

Factor definition (ex-date ``d``, previous raw close ``P``, cash dividend ``div``
per share, 送转 ``r`` shares per share held):

    k = P * (1 + r) / (P - div)

``k >= 1`` and is applied to every bar **before** ``d``:

* 后复权 (``hfq``): ``adj(t) = raw(t) * prod(k for actions after t)``
* 前复权 (``qfq``): ``adj(t) = hfq(t) / hfq(last)``

Actions with no previous close, or a dividend that swallows the price, are data
errors and raise instead of being skipped — a silently dropped dividend shifts
every return after it.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import pandas as pd

from stockpred.errors import DataQualityError


@dataclass(frozen=True)
class CorporateAction:
    """One ex-date event. ``share_ratio`` is shares per share (0.5 = 10送5)."""

    ex_date: date
    cash_per_share: float = 0.0
    share_ratio: float = 0.0

    def __post_init__(self) -> None:
        if self.cash_per_share < 0:
            raise ValueError(f"cash_per_share must be >= 0, got {self.cash_per_share!r}")
        if self.share_ratio < 0:
            raise ValueError(f"share_ratio must be >= 0, got {self.share_ratio!r}")
        if self.cash_per_share == 0 and self.share_ratio == 0:
            raise ValueError("an action must pay cash or shares")


def _previous_close(bars: pd.DataFrame, ex_date: date) -> float:
    dates = pd.to_datetime(bars["date"])
    before = bars.loc[dates < pd.Timestamp(ex_date)]
    if before.empty:
        raise DataQualityError(f"no bar before ex-date {ex_date}; cannot price the action")
    return float(before["close"].iloc[-1])


def action_factor(bars: pd.DataFrame, action: CorporateAction) -> float:
    """``k`` for one action, using our own raw close as ``P``."""
    price = _previous_close(bars, action.ex_date)
    residual = price - action.cash_per_share
    if residual <= 0:
        raise DataQualityError(
            f"dividend {action.cash_per_share} swallows close {price} on {action.ex_date}"
        )
    return price * (1.0 + action.share_ratio) / residual


def relevant_actions(bars: pd.DataFrame, actions: list[CorporateAction]) -> list[CorporateAction]:
    """Keep only actions that actually fall inside the bar window.

    A dividend before the first bar would only rescale the whole series (level, not
    return), and it has no observable previous close here — applying it would raise
    for every long history fetched with a short window.
    """
    if bars.empty:
        return []
    dates = pd.to_datetime(bars["date"])
    first, last = dates.iloc[0], dates.iloc[-1]
    return [
        action
        for action in sorted(actions, key=lambda a: a.ex_date)
        if first < pd.Timestamp(action.ex_date) <= last
    ]


def backward_adjusted(bars: pd.DataFrame, actions: list[CorporateAction]) -> pd.DataFrame:
    """后复权 bars: each action's factor scales the bars from its ex-date onward.

    The ex-date bar is already post-dividend, so scaling it up by ``k`` restores the
    pre-dividend price level and the series stays continuous across the event.
    """
    inside = relevant_actions(bars, actions)
    if not inside:
        return bars.copy()
    dates = pd.to_datetime(bars["date"])
    factors = pd.Series(1.0, index=bars.index, dtype=float)
    for action in inside:
        k = action_factor(bars, action)
        factors.loc[dates >= pd.Timestamp(action.ex_date)] *= k
    frame = bars.copy()
    for column in ("open", "high", "low", "close"):
        frame[column] = frame[column].astype(float) * factors
    return frame


def forward_adjusted(bars: pd.DataFrame, actions: list[CorporateAction]) -> pd.DataFrame:
    """前复权 bars: 后复权 rescaled so the last close matches the raw last close."""
    frame = backward_adjusted(bars, actions)
    if frame.empty:
        return frame
    raw_last = float(bars["close"].astype(float).iloc[-1])
    adj_last = float(frame["close"].iloc[-1])
    if adj_last <= 0:
        raise DataQualityError("backward-adjusted close is not positive")
    scale = raw_last / adj_last
    for column in ("open", "high", "low", "close"):
        frame[column] = frame[column] * scale
    return frame


def adjustment_drift(ours: pd.DataFrame, theirs: pd.DataFrame, *, column: str = "close") -> float:
    """Max relative difference on overlapping dates, for vendor comparison."""
    merged = ours.merge(theirs, on="date", suffixes=("_ours", "_theirs"))
    if merged.empty:
        return 0.0
    left = merged[f"{column}_ours"].astype(float)
    right = merged[f"{column}_theirs"].astype(float)
    return float(((left - right).abs() / right.abs()).max())
