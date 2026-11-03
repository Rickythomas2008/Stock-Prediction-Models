"""Execution layer.

Order of escalation, matching the compliance ladder in the plan:

* ``proposal`` -- R1/R2: an order sheet a person executes. No program sends it.
* ``paper``    -- R3: the same order lifecycle against real bars, no money.
* ``reconcile``-- the check that makes either of the above trustworthy.

Nothing in this package can place a live order; that requires a broker adapter
plus a confirmed program-trading filing (see ``proposal.require_broker_confirmation``).
"""

from __future__ import annotations

from stockpred.execution.paper import Fill, PaperBroker, Rejection, replay_sheet
from stockpred.execution.proposal import (
    SHEET_COLUMNS,
    OrderSheet,
    SheetRow,
    build_order_sheet,
    require_broker_confirmation,
)
from stockpred.execution.reconcile import (
    Discrepancy,
    ReconcileAccumulator,
    ReconciliationReport,
    TradeRecord,
    reconcile,
)

__all__ = [
    "SHEET_COLUMNS",
    "Discrepancy",
    "Fill",
    "OrderSheet",
    "PaperBroker",
    "ReconcileAccumulator",
    "ReconciliationReport",
    "Rejection",
    "SheetRow",
    "TradeRecord",
    "build_order_sheet",
    "reconcile",
    "replay_sheet",
    "require_broker_confirmation",
]
