from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stockpred.eval import BenchmarkSuite
from stockpred.eval.metrics import (
    annual_volatility,
    cagr,
    compute_metrics,
    daily_returns,
    max_drawdown,
    sharpe,
    total_return,
    win_rate,
)


def _constant_growth(rate: float, n: int = 253, start: float = 100.0) -> pd.Series:
    return pd.Series(start * (1.0 + rate) ** np.arange(n))


def test_total_return_on_known_series() -> None:
    equity = pd.Series([100.0, 110.0, 121.0])
    assert total_return(equity) == pytest.approx(0.21)


def test_cagr_of_one_year_doubling() -> None:
    equity = pd.Series([100.0] + [0.0] * 251 + [200.0])
    equity = pd.Series(np.linspace(100.0, 200.0, 253))
    assert cagr(equity) == pytest.approx((2.0) ** (252 / 252) - 1.0, rel=1e-6)


def test_max_drawdown_known_answer() -> None:
    equity = pd.Series([100.0, 120.0, 90.0, 130.0])
    assert max_drawdown(equity) == pytest.approx(90.0 / 120.0 - 1.0)


def test_sharpe_zero_for_constant_returns() -> None:
    assert annual_volatility(daily_returns(_constant_growth(0.0))) == pytest.approx(0.0)
    assert sharpe(daily_returns(_constant_growth(0.0))) == pytest.approx(0.0)


def test_sharpe_matches_manual_formula() -> None:
    returns = pd.Series([0.01, -0.005, 0.02, 0.0, -0.01, 0.015])
    manual = returns.mean() / returns.std(ddof=1) * np.sqrt(252)
    assert sharpe(returns) == pytest.approx(manual)


def test_volatility_scales_with_sqrt_of_year() -> None:
    returns = pd.Series([0.01, -0.01] * 126)
    assert annual_volatility(returns) == pytest.approx(returns.std(ddof=1) * np.sqrt(252))


def test_win_rate_counts_positive_trades() -> None:
    assert win_rate(pd.Series([1.0, -1.0, 2.0, 3.0])) == pytest.approx(0.75)
    assert win_rate(pd.Series(dtype=float)) == pytest.approx(0.0)


def test_compute_metrics_is_complete() -> None:
    equity = _constant_growth(0.001)
    metrics = compute_metrics(equity, traded_notional=50_000.0)

    assert metrics.n_days == len(equity)
    assert metrics.total_return == pytest.approx(equity.iloc[-1] / equity.iloc[0] - 1.0)
    assert metrics.turnover == pytest.approx(50_000.0 / equity.mean())
    assert metrics.max_drawdown <= 0.0


def test_buy_and_hold_equals_price_growth() -> None:
    close = pd.Series([10.0, 11.0, 12.0])
    curve = BenchmarkSuite(close, initial_cash=100_000.0).buy_and_hold()
    assert curve.iloc[0] == pytest.approx(100_000.0)
    assert curve.iloc[-1] == pytest.approx(120_000.0)


def test_cash_benchmark_is_flat() -> None:
    curve = BenchmarkSuite(pd.Series([10.0, 11.0]), initial_cash=100_000.0).cash()
    assert curve.nunique() == 1


def test_random_benchmark_is_reproducible() -> None:
    close = pd.Series(np.linspace(10.0, 12.0, 50))
    suite = BenchmarkSuite(close)
    first = suite.random_signal(seed=42)
    second = suite.random_signal(seed=42)
    pd.testing.assert_series_equal(first, second)


def test_table_contains_all_benchmarks() -> None:
    close = pd.Series(np.linspace(10.0, 12.0, 50))
    suite = BenchmarkSuite(close)
    table = suite.table(suite.buy_and_hold())

    assert set(table.index) == {"strategy", "buy_and_hold", "cash", "random_signal"}
    assert "sharpe" in table.columns


def test_short_input_is_rejected() -> None:
    with pytest.raises(ValueError, match="at least two observations"):
        BenchmarkSuite(pd.Series([10.0]))
