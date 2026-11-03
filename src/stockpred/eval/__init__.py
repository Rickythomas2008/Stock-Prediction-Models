"""Evaluation protocol: leak-free splits, metrics, and mandatory benchmarks.

A reported result must always carry its benchmarks. ``BenchmarkSuite`` builds
the three references this project requires (buy-and-hold, equal-weight,
random-signal) so a strategy number is never presented alone.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from stockpred.eval.leakage import assert_causal, find_lookahead
from stockpred.eval.metrics import (
    TRADING_DAYS_PER_YEAR,
    PerformanceMetrics,
    compare_against_benchmarks,
    compute_metrics,
    drawdown_series,
    metrics_to_dict,
)
from stockpred.eval.report import (
    DISCLOSURES,
    CostImpact,
    benchmark_warnings,
    cost_impact,
    failing_gates,
    render_report,
)
from stockpred.eval.walkforward import (
    Fold,
    InvalidSplitError,
    assert_no_overlap,
    fold_boundaries,
    slice_frame,
    walk_forward_folds,
)

__all__ = [
    "DISCLOSURES",
    "TRADING_DAYS_PER_YEAR",
    "BenchmarkSuite",
    "CostImpact",
    "Fold",
    "InvalidSplitError",
    "PerformanceMetrics",
    "assert_causal",
    "assert_no_overlap",
    "benchmark_warnings",
    "compare_against_benchmarks",
    "compute_metrics",
    "cost_impact",
    "drawdown_series",
    "failing_gates",
    "find_lookahead",
    "fold_boundaries",
    "metrics_to_dict",
    "render_report",
    "slice_frame",
    "walk_forward_folds",
]


class BenchmarkSuite:
    """Reference equity curves that every strategy is measured against."""

    def __init__(self, close: pd.Series, *, initial_cash: float = 100_000.0) -> None:
        if len(close) < 2:
            raise ValueError("benchmarks need at least two observations")
        self.close = close.astype(float).reset_index(drop=True)
        self.initial_cash = float(initial_cash)

    def buy_and_hold(self) -> pd.Series:
        """All cash into the asset at the first close, no costs."""
        units = self.initial_cash / float(self.close.iloc[0])
        return (self.close * units).rename("buy_and_hold")

    def cash(self) -> pd.Series:
        """Flat cash: the benchmark that answers "was trading worth it at all?"."""
        return pd.Series(
            np.full(len(self.close), self.initial_cash, dtype=float), name="cash"
        )

    def random_signal(self, *, seed: int = 7, cost_per_turn: float = 0.0) -> pd.Series:
        """Deterministic coin-flip long/flat strategy with optional per-switch cost."""
        rng = np.random.default_rng(seed)
        returns = self.close.pct_change().fillna(0.0)
        positions = rng.integers(0, 2, size=len(self.close)).astype(float)
        switches = np.abs(np.diff(positions, prepend=positions[0]))
        strategy_returns = positions * returns - switches * cost_per_turn
        return (self.initial_cash * (1.0 + strategy_returns).cumprod()).rename("random_signal")

    def all(self, *, seed: int = 7) -> dict[str, pd.Series]:
        return {
            "buy_and_hold": self.buy_and_hold(),
            "cash": self.cash(),
            "random_signal": self.random_signal(seed=seed),
        }

    def table(self, strategy: pd.Series, *, seed: int = 7) -> pd.DataFrame:
        return compare_against_benchmarks(strategy, self.all(seed=seed))
