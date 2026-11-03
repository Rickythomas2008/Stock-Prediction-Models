"""Provider contract.

A provider returns bars in the canonical schema (``date, open, high, low,
close, volume[, amount, adj_factor]``) and never mutates the caller's data.

Units are part of the contract, not the vendor's choice: ``volume`` is in
**shares** (股) and ``amount`` in CNY. Vendors that quote volume in lots (手)
must convert inside their adapter — a 100x unit slip is invisible to the schema
and silently corrupts every volume-based feature.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date
from typing import Protocol, runtime_checkable

import pandas as pd


@runtime_checkable
class MarketDataProvider(Protocol):
    #: Short identifier recorded in cache manifests and run records.
    name: str

    def fetch_daily(
        self,
        symbol: str,
        start: date,
        end: date,
        *,
        adjust: str = "",
    ) -> pd.DataFrame:
        """Daily bars for ``symbol`` between ``start`` and ``end`` (inclusive)."""
        ...

    def trading_calendar(self) -> Sequence[date]:
        """All known trading days, ascending."""
        ...
