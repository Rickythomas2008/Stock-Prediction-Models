from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from stockpred.eval import BenchmarkSuite
from stockpred.eval.report import (
    DISCLOSURES,
    benchmark_warnings,
    cost_impact,
    failing_gates,
    render_report,
    summarize_dates,
)
from stockpred.eval.walkforward import walk_forward_folds


@pytest.fixture
def curves() -> tuple[pd.Series, pd.Series]:
    n = 200
    gross = pd.Series(100_000.0 * (1.0009) ** np.arange(n))
    net = pd.Series(100_000.0 * (1.0006) ** np.arange(n))
    return gross, net


def test_cost_impact_quantifies_consumption(curves) -> None:
    gross, net = curves
    impact = cost_impact(gross, net, traded_notional=80_000.0)

    assert impact.return_consumed > 0
    assert 0 < impact.consumed_share < 1
    assert impact.gross.total_return > impact.net.total_return


def test_cost_impact_zero_gross_is_safe() -> None:
    flat = pd.Series(np.full(10, 100.0))
    impact = cost_impact(flat, flat)
    assert impact.consumed_share == pytest.approx(0.0)


def test_length_mismatch_raises(curves) -> None:
    gross, net = curves
    with pytest.raises(ValueError, match="same length"):
        cost_impact(gross, net.iloc[:-5])


def test_report_contains_both_cost_versions_and_benchmarks(curves) -> None:
    gross, net = curves
    impact = cost_impact(gross, net, traded_notional=80_000.0)
    suite = BenchmarkSuite(gross, initial_cash=100_000.0)
    table = suite.table(impact.net.n_days and net)

    text = render_report(
        strategy_name="unit-test",
        impact=impact,
        benchmark_table=table,
        generated_at=datetime(2026, 11, 3, 9, 0),
    )

    assert "含成本（净）" in text
    assert "不含成本（毛）" in text
    assert "buy_and_hold" in text
    assert "random_signal" in text
    assert "2026-11-03 09:00" in text
    for disclosure in DISCLOSURES:
        assert disclosure in text


def test_report_lists_folds_when_given(curves) -> None:
    gross, net = curves
    impact = cost_impact(gross, net)
    suite = BenchmarkSuite(gross)
    dates = pd.date_range("2024-01-01", periods=len(gross), freq="D")

    text = render_report(
        strategy_name="folds",
        impact=impact,
        benchmark_table=suite.table(net),
        folds=walk_forward_folds(dates, train_size=100, test_size=20),
    )

    assert "折区间" in text
    assert "fold" in text


def test_gates_reject_losing_strategy() -> None:
    declining = pd.Series([100.0, 95.0, 90.0, 92.0])
    impact = cost_impact(declining, declining)
    reasons = failing_gates(impact)

    assert any("not positive" in reason for reason in reasons)
    assert len(reasons) >= 1


def test_gates_reject_excessive_cost_drag() -> None:
    gross = pd.Series(100_000.0 * (1.001) ** np.arange(100))
    net = pd.Series(100_000.0 * (0.9995) ** np.arange(100))
    impact = cost_impact(gross, net, traded_notional=500_000.0)
    reasons = failing_gates(impact)

    assert any("costs consume" in reason or "not positive" in reason for reason in reasons)


def test_gates_pass_a_healthy_strategy() -> None:
    healthy = pd.Series(100_000.0 * (1.0012) ** np.arange(250))
    impact = cost_impact(healthy, healthy)
    assert failing_gates(impact) == []


def test_summarize_dates(curves) -> None:
    dates = pd.date_range("2024-01-01", periods=200, freq="D")
    folds = walk_forward_folds(dates, train_size=100, test_size=20)
    first, last = summarize_dates(folds)
    assert first < last


def test_benchmark_warnings_flag_underperformance() -> None:
    from stockpred.eval.metrics import compute_metrics

    strategy = compute_metrics(pd.Series(100_000.0 * (1.0005) ** np.arange(100)))
    hold = compute_metrics(pd.Series(100_000.0 * (1.002) ** np.arange(100)))
    cash = compute_metrics(pd.Series(np.full(100, 100_000.0)))

    warnings = benchmark_warnings(strategy, {"buy_and_hold": hold, "cash": cash})

    assert any("跑输买入持有" in item for item in warnings)


def test_benchmark_warnings_quiet_when_strategy_wins() -> None:
    from stockpred.eval.metrics import compute_metrics

    winners = compute_metrics(pd.Series(100_000.0 * (1.002) ** np.arange(100)))
    losers = compute_metrics(pd.Series(100_000.0 * (1.0002) ** np.arange(100)))

    warnings = benchmark_warnings(winners, {"buy_and_hold": losers})

    assert not any("跑输买入持有" in item for item in warnings)


def test_benchmark_warnings_flag_cash_defeat() -> None:
    from stockpred.eval.metrics import compute_metrics

    declining = compute_metrics(pd.Series(100_000.0 * (0.999) ** np.arange(50)))
    cash = compute_metrics(pd.Series(np.full(50, 100_000.0)))

    warnings = benchmark_warnings(declining, {"cash": cash})

    assert any("未跑赢现金基准" in item for item in warnings)
