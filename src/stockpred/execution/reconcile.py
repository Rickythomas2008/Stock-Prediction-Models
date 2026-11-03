"""Reconcile what we expected against what the broker reported.

Paper trading is only worth anything if the expected result is checked against a
second source. Here that second source is either the paper broker's own fills or,
in the live path, the broker's daily statement.

Mismatches are classified rather than counted, because the four classes need
different responses: a missing fill is an execution bug, an unexpected fill is a
control failure, a quantity mismatch is a lot-sizing bug, and a price mismatch is
a data or slippage problem.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from datetime import date
from typing import Any

DEFAULT_PRICE_TOLERANCE = 0.01
DEFAULT_SHARE_TOLERANCE = 0


@dataclass(frozen=True)
class TradeRecord:
    date: date
    symbol: str
    side: str
    shares: int
    price: float

    @classmethod
    def from_mapping(cls, row: Mapping[str, Any]) -> TradeRecord:
        return cls(
            date=_as_date(row["date"]),
            symbol=str(row["symbol"]),
            side=str(row["side"]),
            shares=int(row["shares"]),
            price=float(row["price"]),
        )

    @property
    def key(self) -> tuple[date, str, str]:
        return (self.date, self.symbol, self.side)


@dataclass(frozen=True)
class Discrepancy:
    kind: str
    key: tuple[date, str, str]
    expected: tuple[int, float] | None
    actual: tuple[int, float] | None
    detail: str


@dataclass(frozen=True)
class ReconciliationReport:
    expected_count: int
    actual_count: int
    discrepancies: tuple[Discrepancy, ...] = ()
    matched: int = 0
    price_tolerance: float = DEFAULT_PRICE_TOLERANCE
    share_tolerance: int = DEFAULT_SHARE_TOLERANCE

    @property
    def ok(self) -> bool:
        return not self.discrepancies

    def by_kind(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for item in self.discrepancies:
            counts[item.kind] = counts.get(item.kind, 0) + 1
        return counts

    def summary(self) -> str:
        if self.ok:
            return f"reconciled: {self.matched} trades, no discrepancy"
        parts = ", ".join(f"{kind}={count}" for kind, count in sorted(self.by_kind().items()))
        return f"reconciliation failed: {parts} (matched {self.matched}/{self.expected_count})"

    def as_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "expected_count": self.expected_count,
            "actual_count": self.actual_count,
            "matched": self.matched,
            "discrepancies": [
                {
                    "kind": item.kind,
                    "key": [item.key[0].isoformat(), item.key[1], item.key[2]],
                    "expected": item.expected,
                    "actual": item.actual,
                    "detail": item.detail,
                }
                for item in self.discrepancies
            ],
        }


@dataclass
class ReconcileAccumulator:
    """Collect several days of reconciliation and keep the worst outcome."""

    reports: list[ReconciliationReport] = field(default_factory=list)

    def add(self, report: ReconciliationReport) -> None:
        self.reports.append(report)

    @property
    def ok(self) -> bool:
        return all(report.ok for report in self.reports)

    @property
    def days(self) -> int:
        return len(self.reports)

    def failing_kinds(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for report in self.reports:
            for kind, count in report.by_kind().items():
                counts[kind] = counts.get(kind, 0) + count
        return counts


def reconcile(
    expected: Iterable[TradeRecord | Mapping[str, Any]],
    actual: Iterable[TradeRecord | Mapping[str, Any]],
    *,
    price_tolerance: float = DEFAULT_PRICE_TOLERANCE,
    share_tolerance: int = DEFAULT_SHARE_TOLERANCE,
) -> ReconciliationReport:
    expected_records = [_as_record(item) for item in expected]
    actual_records = [_as_record(item) for item in actual]

    expected_map = _index(expected_records, "expected")
    actual_map = _index(actual_records, "actual")

    discrepancies: list[Discrepancy] = []
    matched = 0

    for key, expected_record in expected_map.items():
        actual_record = actual_map.get(key)
        if actual_record is None:
            discrepancies.append(
                Discrepancy("missing_fill", key, (expected_record.shares, expected_record.price), None, "no broker record")
            )
            continue
        share_gap = abs(expected_record.shares - actual_record.shares)
        price_gap = abs(expected_record.price - actual_record.price)
        if share_gap > share_tolerance:
            discrepancies.append(
                Discrepancy(
                    "quantity_mismatch",
                    key,
                    (expected_record.shares, expected_record.price),
                    (actual_record.shares, actual_record.price),
                    f"shares differ by {share_gap}",
                )
            )
            continue
        if price_gap > price_tolerance:
            discrepancies.append(
                Discrepancy(
                    "price_mismatch",
                    key,
                    (expected_record.shares, expected_record.price),
                    (actual_record.shares, actual_record.price),
                    f"price differs by {price_gap:.4f} > {price_tolerance}",
                )
            )
            continue
        matched += 1

    for key, actual_record in actual_map.items():
        if key not in expected_map:
            discrepancies.append(
                Discrepancy(
                    "unexpected_fill",
                    key,
                    None,
                    (actual_record.shares, actual_record.price),
                    "broker filled something we did not order",
                )
            )

    return ReconciliationReport(
        expected_count=len(expected_records),
        actual_count=len(actual_records),
        discrepancies=tuple(discrepancies),
        matched=matched,
        price_tolerance=price_tolerance,
        share_tolerance=share_tolerance,
    )


def _index(records: list[TradeRecord], label: str) -> dict[tuple[date, str, str], TradeRecord]:
    out: dict[tuple[date, str, str], TradeRecord] = {}
    for record in records:
        if record.key in out:
            raise ValueError(f"duplicate {label} trade for {record.key}; reconciliation needs unique keys")
        out[record.key] = record
    return out


def _as_record(item: TradeRecord | Mapping[str, Any]) -> TradeRecord:
    return item if isinstance(item, TradeRecord) else TradeRecord.from_mapping(item)


def _as_date(value: Any) -> date:
    if isinstance(value, date) and not hasattr(value, "hour"):
        return value
    import pandas as pd

    return pd.Timestamp(value).date()
