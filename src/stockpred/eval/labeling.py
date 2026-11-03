"""Triple-barrier labeling and the purge window it implies.

A label that looks N bars ahead contaminates the N bars before the test window.
That is precisely why ``walkforward.walk_forward_folds`` exposes ``purge``: this
module computes the labels and hands back the horizon that must be purged, so the
two numbers cannot drift apart.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

import numpy as np
import pandas as pd


class Label(IntEnum):
    """Lopez de Prado's triple barrier: which barrier was touched first."""

    LOWER = -1
    VERTICAL = 0
    UPPER = 1


@dataclass(frozen=True)
class LabelResult:
    labels: pd.Series
    touch_bar: pd.Series
    horizon: int

    @property
    def purge_bars(self) -> int:
        """Bars that must be dropped before a test window starting at ``t``."""
        return self.horizon

    def distribution(self) -> dict[str, int]:
        counts = self.labels.value_counts().to_dict()
        return {
            "upper": int(counts.get(int(Label.UPPER), 0)),
            "lower": int(counts.get(int(Label.LOWER), 0)),
            "vertical": int(counts.get(int(Label.VERTICAL), 0)),
        }


def triple_barrier_labels(
    close: pd.Series,
    *,
    upper_pct: float = 0.02,
    lower_pct: float = 0.02,
    horizon: int = 5,
    volatility_scaled: bool = False,
    vol_window: int = 20,
) -> LabelResult:
    """Label each bar by the first barrier it touches within ``horizon`` bars.

    Barriers are symmetric thresholds on the forward return; ``volatility_scaled``
    replaces the fixed percentage with ``k`` rolling standard deviations, which is
    what makes the labels comparable across calm and violent regimes.
    """
    if horizon < 1:
        raise ValueError("horizon must be >= 1")
    if upper_pct <= 0 or lower_pct <= 0:
        raise ValueError("barrier widths must be positive")

    prices = pd.Series(close, dtype=float).reset_index(drop=True)
    if len(prices) < horizon + 2:
        raise ValueError(f"need at least {horizon + 2} observations, got {len(prices)}")

    if volatility_scaled:
        vol = prices.pct_change().rolling(vol_window).std()
        upper = vol.fillna(vol.dropna().iloc[0] if vol.notna().any() else 0.01) * upper_pct * 100
        lower = vol.fillna(vol.dropna().iloc[0] if vol.notna().any() else 0.01) * lower_pct * 100
    else:
        upper = pd.Series(upper_pct, index=prices.index)
        lower = pd.Series(lower_pct, index=prices.index)

    labels: list[int] = []
    touches: list[int] = []
    for index in range(len(prices)):
        if index + horizon >= len(prices):
            labels.append(int(Label.VERTICAL))
            touches.append(-1)
            continue
        entry = prices.iloc[index]
        up_level = entry * (1.0 + float(upper.iloc[index]))
        down_level = entry * (1.0 - float(lower.iloc[index]))
        outcome, touch = int(Label.VERTICAL), horizon
        for step in range(1, horizon + 1):
            price = float(prices.iloc[index + step])
            if price >= up_level:
                outcome, touch = int(Label.UPPER), step
                break
            if price <= down_level:
                outcome, touch = int(Label.LOWER), step
                break
        labels.append(outcome)
        touches.append(touch)

    return LabelResult(
        labels=pd.Series(labels, dtype=int),
        touch_bar=pd.Series(touches, dtype=int),
        horizon=horizon,
    )


def assert_labels_honour_purge(
    result: LabelResult,
    *,
    train_end: int,
    test_start: int,
) -> None:
    """Raise if a training label reaches into the test window."""
    if train_end >= test_start:
        raise ValueError("train_end must be before test_start")
    last_train_label_bar = train_end - 1 + result.horizon
    if last_train_label_bar >= test_start:
        raise ValueError(
            f"label horizon {result.horizon} leaks into the test window: "
            f"last training label resolves at bar {last_train_label_bar} "
            f">= test start {test_start}; purge at least {result.horizon} bars"
        )


def meta_labels(
    primary: pd.Series,
    *,
    returns: pd.Series,
    threshold: float = 0.0,
) -> pd.Series:
    """Meta-labeling: 1 when the primary signal was right, 0 when it was wrong.

    The primary model decides direction; the meta model only decides size.
    """
    if len(primary) != len(returns):
        raise ValueError("primary and returns must have the same length")
    signal = pd.Series(primary, dtype=float).reset_index(drop=True)
    realised = pd.Series(returns, dtype=float).reset_index(drop=True)
    edge = signal * realised
    out = (edge > threshold).astype(int)
    return out


def sample_uniqueness(result: LabelResult) -> pd.Series:
    """Average uniqueness of each label's span (1.0 = no overlap with any other label).

    Overlapping labels inflate the effective sample size. Lopez de Prado's measure:
    count how many labels cover each bar (concurrency), then average ``1/concurrency``
    over the bars a label spans. Reporting a metric without this is how a 5-bar
    horizon on 30 bars gets mistaken for 30 independent observations.
    """
    span = int(result.horizon)
    if span <= 0:
        raise ValueError("horizon must be positive")

    count = len(result.labels)
    concurrency = np.zeros(count + span, dtype=float)
    for index in range(count):
        concurrency[index : index + span] += 1.0
    concurrency = np.maximum(concurrency, 1.0)

    scores = np.empty(count, dtype=float)
    for index in range(count):
        window = concurrency[index : index + span]
        scores[index] = float(np.mean(1.0 / window))
    return pd.Series(scores, name="uniqueness")
