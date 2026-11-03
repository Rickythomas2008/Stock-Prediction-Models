"""Monitoring: drift alerts and a tamper-evident decision log.

A live decision system needs two guarantees that a notebook never had: the model
is still valid, and the record of what it decided cannot be quietly rewritten.
"""

from __future__ import annotations

from stockpred.monitor.audit import (
    GENESIS,
    ChainStatus,
    append_record,
    last_payload,
    read_records,
    verify_chain,
)
from stockpred.monitor.drift import (
    PSI_MODERATE,
    PSI_STABLE,
    DriftAlert,
    DriftReport,
    detect_drift,
    hit_rate,
    population_stability_index,
)

__all__ = [
    "GENESIS",
    "PSI_MODERATE",
    "PSI_STABLE",
    "ChainStatus",
    "DriftAlert",
    "DriftReport",
    "append_record",
    "detect_drift",
    "hit_rate",
    "last_payload",
    "population_stability_index",
    "read_records",
    "verify_chain",
]
