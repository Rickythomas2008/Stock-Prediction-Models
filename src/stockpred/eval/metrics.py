"""Performance metrics with explicit, auditable definitions.

Every metric states its convention in the docstring because "Sharpe" means at
least three different things in practice, and a live-decision system must not
answer with an ambiguous number.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import pandas as pd

TRADING_DAYS_PER_YEAR = 252


@dataclass(frozen=True)
class PerformanceMetrics:
    total_return: float
    cagr: float
    annual_volatility: float
    sharpe: float
    max_drawdown: float
    calmar: float
    win_rate: float
    turnover: float
    n_days: int


def daily_returns(equity: pd.Series) -> pd.Series:
    """Simple returns from an equity curve; first observation yields NaN."""
    return equity.astype(float).pct_change()


def total_return(equity: pd.Series) -> float:
    if len(equity) < 2:
        return 0.0
    start, end = float(equity.iloc[0]), float(equity.iloc[-1])
    if start <= 0:
        raise ValueError("equity curve must start above zero")
    return end / start - 1.0


def cagr(equity: pd.Series, *, periods_per_year: int = TRADING_DAYS_PER_YEAR) -> float:
    if len(equity) < 2:
        return 0.0
    start, end = float(equity.iloc[0]), float(equity.iloc[-1])
    years = (len(equity) - 1) / periods_per_year
    if years <= 0 or start <= 0:
        return 0.0
    return (end / start) ** (1.0 / years) - 1.0


def annual_volatility(returns: pd.Series, *, periods_per_year: int = TRADING_DAYS_PER_YEAR) -> float:
    clean = returns.dropna()
    if len(clean) < 2:
        return 0.0
    return float(clean.std(ddof=1) * math.sqrt(periods_per_year))


def sharpe(
    returns: pd.Series,
    *,
    risk_free_rate: float = 0.0,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> float:
    """Annualised Sharpe using sample std (ddof=1) and a per-annum risk-free rate."""
    clean = returns.dropna()
    if len(clean) < 2:
        return 0.0
    excess = clean - risk_free_rate / periods_per_year
    std = float(excess.std(ddof=1))
    if std == 0.0:
        return 0.0
    return float(excess.mean() / std * math.sqrt(periods_per_year))


def max_drawdown(equity: pd.Series) -> float:
    """Deepest peak-to-trough decline, returned as a negative fraction."""
    if len(equity) < 2:
        return 0.0
    curve = equity.astype(float)
    peak = curve.cummax()
    drawdown = curve / peak - 1.0
    return float(drawdown.min())


def drawdown_series(equity: pd.Series) -> pd.Series:
    curve = equity.astype(float)
    return curve / curve.cummax() - 1.0


def calmar(equity: pd.Series, *, periods_per_year: int = TRADING_DAYS_PER_YEAR) -> float:
    mdd = max_drawdown(equity)
    if mdd == 0.0:
        return 0.0
    return cagr(equity, periods_per_year=periods_per_year) / abs(mdd)


def win_rate(trade_pnls: pd.Series) -> float:
    """Share of closed trades with positive P&L; 0.0 when there are none."""
    if len(trade_pnls) == 0:
        return 0.0
    return float((trade_pnls > 0).mean())


def turnover(traded_notional: float, average_equity: float) -> float:
    """Traded notional over average equity (a cost proxy, not an annualised figure)."""
    if average_equity <= 0:
        return 0.0
    return traded_notional / average_equity


def compute_metrics(
    equity: pd.Series,
    *,
    traded_notional: float = 0.0,
    trade_pnls: pd.Series | None = None,
    risk_free_rate: float = 0.0,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> PerformanceMetrics:
    returns = daily_returns(equity)
    return PerformanceMetrics(
        total_return=total_return(equity),
        cagr=cagr(equity, periods_per_year=periods_per_year),
        annual_volatility=annual_volatility(returns, periods_per_year=periods_per_year),
        sharpe=sharpe(returns, risk_free_rate=risk_free_rate, periods_per_year=periods_per_year),
        max_drawdown=max_drawdown(equity),
        calmar=calmar(equity, periods_per_year=periods_per_year),
        win_rate=win_rate(
            pd.Series(dtype=float) if trade_pnls is None else trade_pnls.astype(float)
        ),
        turnover=turnover(traded_notional, float(equity.astype(float).mean())),
        n_days=int(len(equity)),
    )


def metrics_to_dict(metrics: PerformanceMetrics) -> dict[str, float | int]:
    return asdict(metrics)


def compare_against_benchmarks(
    strategy: pd.Series,
    benchmarks: dict[str, pd.Series],
) -> pd.DataFrame:
    """Side-by-side metric table; a strategy without benchmarks is not a result."""
    rows = []
    for label, curve in {"strategy": strategy, **benchmarks}.items():
        metrics = compute_metrics(curve.astype(float))
        row = metrics_to_dict(metrics)
        row["label"] = label
        rows.append(row)
    frame = pd.DataFrame(rows).set_index("label")
    return frame[["total_return", "cagr", "annual_volatility", "sharpe", "max_drawdown", "n_days"]]
