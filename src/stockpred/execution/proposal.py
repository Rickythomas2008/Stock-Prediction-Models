"""Turn a decision package into an explicit, human-executable order sheet.

This is the R1/R2 route from the compliance ladder: no computer program places
the order, so nothing here is 程序化交易 under CSRC 2024 No.8 art.2. The sheet is
the artifact a person reads at the broker app, so it must be unambiguous --
price, quantities in whole lots, validity, and the risk verdict that allowed it.
"""

from __future__ import annotations

import csv
import io
from dataclasses import dataclass
from datetime import date

from stockpred.errors import BrokerNotConfirmed
from stockpred.pipeline import DecisionPackage, ExecutionPlan

SHEET_COLUMNS = (
    "as_of",
    "symbol",
    "side",
    "shares",
    "limit_price",
    "notional",
    "valid_until",
    "risk_status",
    "note",
)


@dataclass(frozen=True)
class SheetRow:
    as_of: date
    symbol: str
    side: str
    shares: int
    limit_price: float
    valid_until: date
    risk_status: str
    note: str

    @property
    def notional(self) -> float:
        return self.shares * self.limit_price

    def as_dict(self) -> dict[str, object]:
        return {
            "as_of": self.as_of.isoformat(),
            "symbol": self.symbol,
            "side": self.side,
            "shares": self.shares,
            "limit_price": round(self.limit_price, 2),
            "notional": round(self.notional, 2),
            "valid_until": self.valid_until.isoformat(),
            "risk_status": self.risk_status,
            "note": self.note,
        }


@dataclass(frozen=True)
class OrderSheet:
    rows: tuple[SheetRow, ...]
    as_of: date

    def approved_rows(self) -> tuple[SheetRow, ...]:
        return tuple(row for row in self.rows if row.risk_status == "approved")

    def refused_rows(self) -> tuple[SheetRow, ...]:
        return tuple(row for row in self.rows if row.risk_status != "approved")

    def to_csv(self) -> str:
        buffer = io.StringIO()
        writer = csv.DictWriter(buffer, fieldnames=SHEET_COLUMNS, lineterminator="\n")
        writer.writeheader()
        for row in self.rows:
            writer.writerow(row.as_dict())
        return buffer.getvalue()

    def to_text(self) -> str:
        """Human-readable sheet; the format used when typing into a broker app."""
        lines = [f"委托单 {self.as_of.isoformat()}（共 {len(self.approved_rows())} 笔可执行）"]
        for row in self.approved_rows():
            lines.append(
                f"  {row.side.upper():4} {row.symbol}  {row.shares:>6} 股  "
                f"限价 {row.limit_price:8.2f}  有效期至 {row.valid_until.isoformat()}"
            )
        refused = self.refused_rows()
        if refused:
            lines.append("被风控拒绝（不执行）：")
            for row in refused:
                lines.append(f"  {row.side.upper():4} {row.symbol}  {row.note}")
        return "\n".join(lines)


def build_order_sheet(
    package: DecisionPackage,
    plan: ExecutionPlan,
    *,
    valid_days: int = 1,
) -> OrderSheet:
    """Convert the approved part of an execution plan into a sheet of limit orders."""
    rows: list[SheetRow] = []
    for proposal in plan.proposals:
        verdict = plan.verdicts[proposal.symbol]
        status = "approved" if verdict.approved else "refused"
        note = (
            "ok"
            if verdict.approved
            else "refused: " + ",".join(verdict.violations)
        )
        rows.append(
            SheetRow(
                as_of=package.as_of,
                symbol=proposal.symbol,
                side=proposal.side,
                shares=proposal.shares,
                limit_price=proposal.price,
                valid_until=_add_days(package.as_of, valid_days),
                risk_status=status,
                note=note,
            )
        )
    return OrderSheet(rows=tuple(rows), as_of=package.as_of)


def require_broker_confirmation(confirmed_at: str | None) -> None:
    """Guard for anything that would send an order programmatically.

    Kept here (not only in config) so a caller cannot reach the order path
    without passing the filing check first.
    """
    if not confirmed_at:
        raise BrokerNotConfirmed(
            "order submission blocked: broker has not confirmed the program-trading "
            "filing (report first, trade afterwards - CSRC 2024 No.8 art.8)"
        )


def _add_days(start: date, days: int) -> date:
    from datetime import timedelta

    return start + timedelta(days=max(days, 0))
