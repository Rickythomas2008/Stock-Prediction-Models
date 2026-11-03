"""Cross-provider agreement check for bar frames.

Two providers covering the same symbol must agree on prices and — now that
``volume`` is canonicalised to shares — on volume too. A mismatch is a data
integrity finding, not formatting noise, so this module returns numeric deltas
plus a pass flag instead of silently picking one side.

Tolerances exist because the vendors round differently (``amount`` to the yuan,
prices to the 分), not because disagreement is acceptable.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

PRICE_TOLERANCE = 0.011
AMOUNT_TOLERANCE = 1.0
VOLUME_REL_TOLERANCE = 0.01


@dataclass(frozen=True)
class CrossCheck:
    sessions: int
    findings: list[str] = field(default_factory=list)
    ok: bool = True


def compare_bars(
    left: pd.DataFrame,
    right: pd.DataFrame,
    *,
    name_left: str,
    name_right: str,
) -> CrossCheck:
    """Compare two providers on their overlapping dates."""
    merged = left.merge(right, on="date", suffixes=(f"_{name_left}", f"_{name_right}"))
    if merged.empty:
        return CrossCheck(0, [f"no overlapping dates between {name_left} and {name_right}"], False)

    findings = [f"overlapping sessions: {len(merged)}"]
    ok = True
    for column, tolerance in (
        ("open", PRICE_TOLERANCE),
        ("high", PRICE_TOLERANCE),
        ("low", PRICE_TOLERANCE),
        ("close", PRICE_TOLERANCE),
        ("amount", AMOUNT_TOLERANCE),
    ):
        here, there = f"{column}_{name_left}", f"{column}_{name_right}"
        if here not in merged or there not in merged:
            continue
        delta = float((merged[here] - merged[there]).abs().max())
        good = delta <= tolerance
        ok = ok and good
        findings.append(
            f"{column:<6} max|diff|={delta:.4f} tol={tolerance} {'ok' if good else 'FAIL'}"
        )

    here, there = f"volume_{name_left}", f"volume_{name_right}"
    if here in merged and there in merged:
        usable = merged[(merged[here] > 0) & (merged[there] > 0)]
        if usable.empty:
            findings.append("volume ratio n/a (no positive volume on both sides)")
        else:
            ratio = float((usable[there] / usable[here]).median())
            good = abs(ratio - 1.0) <= VOLUME_REL_TOLERANCE
            ok = ok and good
            findings.append(
                f"volume median ratio {name_right}/{name_left}={ratio:.4f} "
                f"tol={VOLUME_REL_TOLERANCE} {'ok' if good else 'FAIL (unit mismatch?)'}"
            )
    return CrossCheck(len(merged), findings, ok)
