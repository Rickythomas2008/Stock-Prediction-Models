from __future__ import annotations

import pytest

from stockpred.data.providers.baostock_provider import to_baostock_code


@pytest.mark.parametrize(
    ("symbol", "expected"),
    [
        ("600519", "sh.600519"),
        ("000001", "sz.000001"),
        ("300750", "sz.300750"),
        ("600519.SH", "sh.600519"),
        ("000001.SZ", "sz.000001"),
        ("sh.600519", "sh.600519"),
        ("SZ.000001", "sz.000001"),
    ],
)
def test_symbol_mapping(symbol: str, expected: str) -> None:
    assert to_baostock_code(symbol) == expected


@pytest.mark.parametrize("symbol", ["830799", "688981.HK", "AAPL"])
def test_unsupported_symbols_are_rejected(symbol: str) -> None:
    with pytest.raises(ValueError):
        to_baostock_code(symbol)
