"""Bar schema and validation for A-share daily data.

Validation is deliberately strict: a bad bar that slips through becomes a bad
decision. Every violation raises :class:`stockpred.errors.DataQualityError`
rather than being silently dropped or imputed.
"""

from __future__ import annotations

import pandas as pd

try:  # pandera >= 0.24 splits the pandas backend into its own module
    import pandera.pandas as pa
except ImportError:  # pragma: no cover - older pandera
    import pandera as pa  # type: ignore[no-redef]

from stockpred.errors import DataQualityError

REQUIRED_COLUMNS: tuple[str, ...] = ("date", "open", "high", "low", "close", "volume")
OPTIONAL_COLUMNS: tuple[str, ...] = ("amount", "adj_factor")

#: Canonical unit of ``volume``: shares (股), never lots (手). The check below
#: cannot detect a unit slip, so providers normalise and each adapter has a test.
VOLUME_UNIT = "shares"
#: Canonical unit of ``amount``: CNY, in yuan (not 万元).
AMOUNT_UNIT = "CNY"


def _ohlc_consistent(df: pd.DataFrame) -> pd.Series:
    body_high = df[["open", "close"]].max(axis=1)
    body_low = df[["open", "close"]].min(axis=1)
    return (df["high"] >= body_high) & (df["low"] <= body_low)


BAR_SCHEMA = pa.DataFrameSchema(
    columns={
        "date": pa.Column(pa.DateTime, nullable=False),
        "open": pa.Column(float, pa.Check.gt(0), nullable=False),
        "high": pa.Column(float, pa.Check.gt(0), nullable=False),
        "low": pa.Column(float, pa.Check.gt(0), nullable=False),
        "close": pa.Column(float, pa.Check.gt(0), nullable=False),
        "volume": pa.Column(float, pa.Check.ge(0), nullable=False),
        "amount": pa.Column(float, pa.Check.ge(0), nullable=True, required=False),
        "adj_factor": pa.Column(float, pa.Check.gt(0), nullable=True, required=False),
    },
    checks=[
        pa.Check(lambda df: df["date"].is_monotonic_increasing, name="dates_ascending"),
        pa.Check(lambda df: df["date"].is_unique, name="dates_unique"),
        pa.Check(_ohlc_consistent, name="high_low_consistent"),
    ],
    coerce=True,
    strict=False,
)


def validate_bars(df: pd.DataFrame, *, symbol: str | None = None) -> pd.DataFrame:
    """Validate and return bars, raising ``DataQualityError`` on any violation."""
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise DataQualityError(f"{_label(symbol)}missing required columns: {missing}")
    if df.empty:
        raise DataQualityError(f"{_label(symbol)}no rows")
    try:
        return BAR_SCHEMA.validate(df, lazy=False)
    except pa.errors.SchemaError as exc:
        raise DataQualityError(f"{_label(symbol)}schema violation: {exc}") from exc


def _label(symbol: str | None) -> str:
    return f"[{symbol}] " if symbol else ""
