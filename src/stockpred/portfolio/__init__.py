"""Long-only portfolio construction, sized for A-share lots.

A 100k account cannot buy fractional shares, so a weight vector is only a
proposal: ``to_lots`` turns it into whole 100-share lots and reports the cash
that the rounding leaves behind. Ignoring that rounding is how a backtest ends
up holding 27.3 shares of a 170 CNY stock.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from stockpred.backtest.rules import LOT_SIZE


@dataclass(frozen=True)
class Allocation:
    shares: dict[str, int]
    invested: float
    leftover_cash: float
    dropped: tuple[str, ...]

    @property
    def invested_pct(self) -> float:
        total = self.invested + self.leftover_cash
        return self.invested / total if total else 0.0


def equal_weight(symbols: list[str]) -> pd.Series:
    if not symbols:
        raise ValueError("need at least one symbol")
    return pd.Series(1.0 / len(symbols), index=list(symbols))


def inverse_volatility(returns: pd.DataFrame, *, floor: float = 1e-6) -> pd.Series:
    """Risk parity by inverse volatility; a zero-variance asset cannot dominate."""
    if returns.empty:
        raise ValueError("returns frame is empty")
    vol = returns.std(ddof=1).clip(lower=floor)
    weights = (1.0 / vol) / (1.0 / vol).sum()
    return weights


def min_variance(returns: pd.DataFrame, *, shrinkage: float = 0.1, iters: int = 500) -> pd.Series:
    """Long-only minimum-variance weights via projected gradient descent.

    Shrinkage keeps the covariance invertible when the sample is short, which is
    the normal case for a few hundred daily bars.
    """
    if returns.empty or returns.shape[1] < 1:
        raise ValueError("returns frame is empty")
    cov = returns.cov().to_numpy(dtype=float)
    n = cov.shape[0]
    target = np.eye(n) * np.trace(cov) / n
    cov = (1.0 - shrinkage) * cov + shrinkage * target
    cov += np.eye(n) * 1e-10

    weights = np.full(n, 1.0 / n)
    step = 1.0 / (np.linalg.eigvalsh(cov).max() * 2.0)
    for _ in range(iters):
        gradient = 2.0 * cov @ weights
        weights = weights - step * gradient
        weights = np.clip(weights, 0.0, None)
        total = weights.sum()
        if total <= 0:
            weights = np.full(n, 1.0 / n)
            break
        weights = weights / total
    return pd.Series(weights, index=list(returns.columns))


def apply_caps(weights: pd.Series, *, max_position: float = 0.25) -> pd.Series:
    """Cap any single name, redistributing the excess to uncapped names."""
    if not 0.0 < max_position <= 1.0:
        raise ValueError("max_position must be in (0, 1]")
    result = weights.astype(float).clip(lower=0.0)
    for _ in range(20):
        total = result.sum()
        if total <= 0:
            raise ValueError("all weights are zero")
        result = result / total
        over = result > max_position
        if not over.any():
            return result
        excess = float((result[over] - max_position).sum())
        result[over] = max_position
        under = ~over
        if not under.any():
            return result
        room = float((max_position - result[under]).sum())
        if room <= 0:
            return result
        result[under] = result[under] + excess * (max_position - result[under]) / room
    return result


def to_lots(
    weights: pd.Series,
    *,
    capital: float,
    prices: dict[str, float],
    lot: int = LOT_SIZE,
    min_position_value: float | None = None,
) -> Allocation:
    """Convert target weights into whole lots, largest weight first.

    ``min_position_value`` (default: one lot at the current price) drops names
    whose allocation cannot even buy a single lot, instead of silently sizing
    them to zero inside the dict.
    """
    if capital <= 0:
        raise ValueError("capital must be positive")
    ordered = weights[weights > 0].sort_values(ascending=False)
    shares: dict[str, int] = {}
    invested = 0.0
    dropped: list[str] = []
    cash = float(capital)

    for symbol, weight in ordered.items():
        price = prices.get(symbol)
        if price is None or price <= 0:
            dropped.append(symbol)
            continue
        target_value = weight * capital
        one_lot = lot * price
        floor_value = one_lot if min_position_value is None else min_position_value
        if target_value < max(floor_value, one_lot):
            dropped.append(symbol)
            continue
        lots = int(target_value // one_lot)
        while lots > 0 and lots * one_lot > cash:
            lots -= 1
        if lots <= 0:
            dropped.append(symbol)
            continue
        cost = lots * one_lot
        shares[symbol] = lots * lot
        invested += cost
        cash -= cost

    return Allocation(
        shares=shares,
        invested=invested,
        leftover_cash=cash,
        dropped=tuple(dropped),
    )


def weights_from_allocation(allocation: Allocation, prices: dict[str, float]) -> pd.Series:
    values = {
        symbol: shares * prices[symbol] for symbol, shares in allocation.shares.items()
    }
    return pd.Series(values, dtype=float)
