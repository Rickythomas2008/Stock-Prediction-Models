from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from stockpred.data.adjust import (
    CorporateAction,
    action_factor,
    backward_adjusted,
    forward_adjusted,
    relevant_actions,
)
from stockpred.data.dividends import (
    parse_akshare_detail,
    parse_baostock_rows,
    read_cached,
    write_cache,
)
from stockpred.errors import DataQualityError


def _bars(closes: list[float], *, start: str = "2024-01-02") -> pd.DataFrame:
    days = pd.bdate_range(start, periods=len(closes))
    return pd.DataFrame(
        {
            "date": days,
            "open": closes,
            "high": [c * 1.01 for c in closes],
            "low": [c * 0.99 for c in closes],
            "close": closes,
            "volume": [1_000_000.0] * len(closes),
        }
    )


def test_cash_dividend_factor_uses_our_own_close() -> None:
    bars = _bars([10.0, 10.0, 9.5])
    action = CorporateAction(ex_date=date(2024, 1, 4), cash_per_share=0.5)

    assert action_factor(bars, action) == pytest.approx(10.0 / 9.5)


def test_share_ratio_factor_scales_shares() -> None:
    bars = _bars([10.0, 10.0, 6.67])
    action = CorporateAction(ex_date=date(2024, 1, 4), share_ratio=0.5)  # 10送5

    assert action_factor(bars, action) == pytest.approx(1.5)


def test_combined_action_multiplies_both_effects() -> None:
    bars = _bars([10.0, 10.0, 6.33])
    action = CorporateAction(ex_date=date(2024, 1, 4), cash_per_share=0.5, share_ratio=0.5)

    assert action_factor(bars, action) == pytest.approx(10.0 * 1.5 / 9.5)


def test_backward_adjustment_keeps_the_series_continuous() -> None:
    """A cash dividend with no market move must not print as a price drop."""
    bars = _bars([10.0, 10.0, 9.5])
    action = CorporateAction(ex_date=date(2024, 1, 4), cash_per_share=0.5)

    frame = backward_adjusted(bars, [action])

    k = 10.0 / 9.5
    assert frame["close"].iloc[0] == pytest.approx(10.0)
    assert frame["close"].iloc[1] == pytest.approx(10.0)
    assert frame["close"].iloc[2] == pytest.approx(9.5 * k)  # ex-day bar scaled up
    assert frame["close"].iloc[2] == pytest.approx(frame["close"].iloc[1])


def test_forward_adjustment_holds_the_last_close_and_shifts_history() -> None:
    bars = _bars([10.0, 10.0, 9.5])
    action = CorporateAction(ex_date=date(2024, 1, 4), cash_per_share=0.5)

    frame = forward_adjusted(bars, [action])

    assert frame["close"].iloc[-1] == pytest.approx(9.5)  # 前复权 pins the latest price
    assert frame["close"].iloc[0] == pytest.approx(9.5, rel=1e-6)
    assert frame["close"].iloc[1] == pytest.approx(frame["close"].iloc[2], rel=1e-6)


def test_actions_before_the_window_are_ignored() -> None:
    """A 2002 dividend cannot be priced from a 2024 window, and does not matter."""
    bars = _bars([10.0, 10.0, 10.0])
    old = CorporateAction(ex_date=date(2002, 7, 25), cash_per_share=0.5)
    inside = CorporateAction(ex_date=date(2024, 1, 4), cash_per_share=0.5)

    assert relevant_actions(bars, [old, inside]) == [inside]

    frame = backward_adjusted(bars, [old, inside])
    assert frame["close"].iloc[0] == pytest.approx(10.0)
    assert frame["close"].iloc[-1] > 10.0


def test_action_without_previous_bar_is_an_error() -> None:
    """Guards the factor maths directly; the window filter keeps such actions out."""
    bars = _bars([10.0, 10.0])
    action = CorporateAction(ex_date=date(2023, 12, 1), cash_per_share=0.5)

    with pytest.raises(DataQualityError, match="no bar before ex-date"):
        action_factor(bars, action)
    assert relevant_actions(bars, [action]) == []


def test_dividend_larger_than_price_is_an_error() -> None:
    bars = _bars([10.0, 10.0])
    action = CorporateAction(ex_date=date(2024, 1, 4), cash_per_share=20.0)

    with pytest.raises(DataQualityError, match="swallows close"):
        action_factor(bars, action)


def test_action_needs_a_payout() -> None:
    with pytest.raises(ValueError):
        CorporateAction(ex_date=date(2024, 1, 4))


def test_parse_akshare_detail_converts_per_ten_shares() -> None:
    frame = pd.DataFrame(
        {
            "公告日期": ["2024-06-20", "2024-01-10"],
            "送股(股)": [5.0, 0.0],
            "转增(股)": [3.0, 0.0],
            "派息(税前)(元)": [23.882, 10.0],
            "除权除息日": ["2024-06-27", "2024-01-17"],
            "进度": ["实施", "实施"],
        }
    )

    actions = parse_akshare_detail(frame)

    assert [a.ex_date.isoformat() for a in actions] == ["2024-01-17", "2024-06-27"]
    assert actions[0].cash_per_share == pytest.approx(1.0)
    assert actions[1].cash_per_share == pytest.approx(2.3882)
    assert actions[1].share_ratio == pytest.approx(0.8)  # 5送3 -> 0.5 + 0.3


def test_parse_baostock_rows_keeps_per_share_units() -> None:
    rows = [
        {
            "dividOperateDate": "2024-06-27",
            "dividCashPsBeforeTax": "0.5",
            "dividStocksPs": "0.2",
            "dividReserveToStockPs": "0.3",
        },
        {"dividOperateDate": "", "dividCashPsBeforeTax": "1.0"},
    ]

    actions = parse_baostock_rows(rows)

    assert len(actions) == 1
    assert actions[0].cash_per_share == pytest.approx(0.5)
    assert actions[0].share_ratio == pytest.approx(0.5)


def test_dividend_cache_round_trips(tmp_path: Path) -> None:
    actions = [CorporateAction(ex_date=date(2024, 6, 27), cash_per_share=2.3882, share_ratio=0.8)]
    write_cache(tmp_path, "600519", "akshare", actions)

    restored = read_cached(tmp_path, "600519", "akshare")

    assert restored == actions
    assert read_cached(tmp_path, "000001", "akshare") is None
