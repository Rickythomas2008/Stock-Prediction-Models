"""Drift detection: is the world still the one the model was fitted on?

Three signals, because they fail at different times and mean different things:

* **feature PSI** -- the inputs moved (the market regime changed)
* **prediction shift** -- the outputs moved even if inputs look stable (model or
  scaling problem)
* **hit rate** -- the signal stopped working (the only one that measures money)

Thresholds follow the usual PSI reading: <0.1 stable, 0.1-0.25 moderate,
>0.25 major.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

PSI_STABLE = 0.10
PSI_MODERATE = 0.25
MIN_BIN_SHARE = 1e-6


@dataclass(frozen=True)
class DriftAlert:
    kind: str
    subject: str
    value: float
    threshold: float
    severity: str
    message: str


@dataclass(frozen=True)
class DriftReport:
    psi: dict[str, float] = field(default_factory=dict)
    prediction_shift: float = 0.0
    hit_rate_recent: float | None = None
    hit_rate_baseline: float | None = None
    alerts: tuple[DriftAlert, ...] = ()

    @property
    def ok(self) -> bool:
        return not self.alerts

    def by_severity(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for alert in self.alerts:
            counts[alert.severity] = counts.get(alert.severity, 0) + 1
        return counts

    def summary(self) -> str:
        if self.ok:
            return "no drift detected"
        parts = [f"{alert.kind}:{alert.subject}({alert.severity})" for alert in self.alerts]
        return "drift: " + ", ".join(parts)

    def as_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "psi": {key: round(value, 6) for key, value in self.psi.items()},
            "prediction_shift": round(self.prediction_shift, 6),
            "hit_rate_recent": self.hit_rate_recent,
            "hit_rate_baseline": self.hit_rate_baseline,
            "alerts": [
                {
                    "kind": alert.kind,
                    "subject": alert.subject,
                    "value": round(alert.value, 6),
                    "threshold": alert.threshold,
                    "severity": alert.severity,
                    "message": alert.message,
                }
                for alert in self.alerts
            ],
        }


def population_stability_index(
    baseline: Sequence[float] | pd.Series | np.ndarray,
    current: Sequence[float] | pd.Series | np.ndarray,
    *,
    bins: int = 10,
) -> float:
    """PSI with quantile edges taken from the baseline (so bins are fixed)."""
    base = np.asarray(baseline, dtype=float)
    cur = np.asarray(current, dtype=float)
    base = base[~np.isnan(base)]
    cur = cur[~np.isnan(cur)]
    if len(base) < 10 or len(cur) < 5:
        raise ValueError("PSI needs at least 10 baseline and 5 current observations")

    edges = np.unique(np.quantile(base, np.linspace(0.0, 1.0, bins + 1)))
    if len(edges) < 3:
        raise ValueError("baseline is too flat to bin")

    base_counts, _ = np.histogram(base, bins=edges)
    cur_counts, _ = np.histogram(cur, bins=edges)
    base_share = np.maximum(base_counts / max(len(base), 1), MIN_BIN_SHARE)
    cur_share = np.maximum(cur_counts / max(len(cur), 1), MIN_BIN_SHARE)
    return float(np.sum((cur_share - base_share) * np.log(cur_share / base_share)))


def detect_drift(
    baseline_features: pd.DataFrame,
    current_features: pd.DataFrame,
    *,
    baseline_predictions: pd.Series | None = None,
    current_predictions: pd.Series | None = None,
    hit_rate_baseline: float | None = None,
    hit_rate_recent: float | None = None,
    psi_threshold: float = PSI_MODERATE,
    shift_sigma_threshold: float = 1.0,
    hit_rate_floor: float = 0.45,
) -> DriftReport:
    """Compare the serving window against the fitting window."""
    alerts: list[DriftAlert] = []
    psi_values: dict[str, float] = {}

    shared = [column for column in baseline_features.columns if column in current_features.columns]
    if not shared:
        raise ValueError("baseline and current frames share no feature columns")

    for column in shared:
        try:
            value = population_stability_index(
                baseline_features[column].to_numpy(), current_features[column].to_numpy()
            )
        except ValueError:
            continue
        psi_values[str(column)] = value
        if value > psi_threshold:
            severity = "major" if value > PSI_MODERATE else "moderate"
            alerts.append(
                DriftAlert(
                    kind="feature_psi",
                    subject=str(column),
                    value=value,
                    threshold=psi_threshold,
                    severity=severity,
                    message=f"feature {column} distribution moved (PSI {value:.3f})",
                )
            )

    shift = 0.0
    if baseline_predictions is not None and current_predictions is not None:
        base = pd.Series(baseline_predictions).astype(float).dropna()
        cur = pd.Series(current_predictions).astype(float).dropna()
        if len(base) > 1 and len(cur) > 0:
            sigma = float(base.std(ddof=1)) or 1.0
            shift = float(abs(cur.mean() - base.mean()) / sigma)
            if shift > shift_sigma_threshold:
                alerts.append(
                    DriftAlert(
                        kind="prediction_shift",
                        subject="mean",
                        value=shift,
                        threshold=shift_sigma_threshold,
                        severity="major" if shift > 2 * shift_sigma_threshold else "moderate",
                        message=f"prediction mean moved {shift:.2f} sigma",
                    )
                )

    if hit_rate_recent is not None and hit_rate_baseline is not None:
        if hit_rate_recent < hit_rate_floor:
            alerts.append(
                DriftAlert(
                    kind="hit_rate",
                    subject="recent",
                    value=float(hit_rate_recent),
                    threshold=hit_rate_floor,
                    severity="major",
                    message=(
                        f"recent hit rate {hit_rate_recent:.1%} below floor {hit_rate_floor:.0%} "
                        f"(baseline {hit_rate_baseline:.1%})"
                    ),
                )
            )
        elif hit_rate_recent < hit_rate_baseline - 0.10:
            alerts.append(
                DriftAlert(
                    kind="hit_rate",
                    subject="recent",
                    value=float(hit_rate_recent),
                    threshold=float(hit_rate_baseline - 0.10),
                    severity="moderate",
                    message=(
                        f"hit rate degraded {hit_rate_baseline:.1%} -> {hit_rate_recent:.1%}"
                    ),
                )
            )

    return DriftReport(
        psi=psi_values,
        prediction_shift=shift,
        hit_rate_recent=hit_rate_recent,
        hit_rate_baseline=hit_rate_baseline,
        alerts=tuple(alerts),
    )


def hit_rate(predictions: Sequence[float] | pd.Series, outcomes: Sequence[float] | pd.Series) -> float:
    """Share of predictions whose sign matched the realised outcome."""
    pred = np.sign(np.asarray(predictions, dtype=float))
    real = np.sign(np.asarray(outcomes, dtype=float))
    mask = (pred != 0) & (real != 0) & ~np.isnan(pred) & ~np.isnan(real)
    if not mask.any():
        return 0.0
    return float((pred[mask] == real[mask]).mean())
