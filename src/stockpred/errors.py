"""Typed errors. Every failure path that guards live money has its own type."""

from __future__ import annotations


class StockpredError(Exception):
    """Base class for all package errors."""


class DataQualityError(StockpredError):
    """Bars failed schema or integrity validation."""


class StaleDataError(StockpredError):
    """The data is older than the freshest bar that should be available.

    Raised instead of silently producing a decision on stale prices.
    """


class LookaheadError(StockpredError):
    """A bar dated in the future relative to the decision timestamp."""


class ProviderUnavailable(StockpredError):
    """A market-data provider (or its optional dependency) is not available."""


class BrokerNotConfirmed(StockpredError):
    """Auto-order attempted before the broker confirmed program-trading filing.

    CSRC 《证券市场程序化交易管理规定（试行）》art.8 (effective 2024-10-08):
    reporting first, trading afterwards.
    """


class MissingRunConfig(StockpredError):
    """A run directory has no (or a tampered) config.json."""
