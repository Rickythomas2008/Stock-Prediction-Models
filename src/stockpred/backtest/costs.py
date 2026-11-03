"""A-share transaction costs as they hit a real account.

Rates follow the 2023-08-28 stamp-duty cut (0.1% -> 0.05%, sell side only) plus
the two-way 0.001% transfer fee. The brokerage commission carries the statutory
CNY 5 minimum, which is why a small trade is priced by the floor and not by the
rate: 万 2.5 on 1,000 CNY is 0.25, so the actual charge is 5.

All cost functions are monotone in notional and never negative; the engine relies
on both properties when it solves for an affordable lot size.
"""

from __future__ import annotations

from dataclasses import dataclass

BPS = 10_000.0

_RATE_FIELDS = (
    "commission_rate",
    "min_commission",
    "stamp_duty_rate",
    "transfer_fee_rate",
    "slippage_bps",
)


@dataclass(frozen=True)
class CostModel:
    commission_rate: float = 0.00025  # 佣金 万 2.5
    min_commission: float = 5.0  # 单笔最低 5 元
    stamp_duty_rate: float = 0.0005  # 印花税，仅卖出
    transfer_fee_rate: float = 0.00001  # 过户费 0.001%，双向
    slippage_bps: float = 0.0  # 滑点，基点，买入加价 / 卖出减价

    def __post_init__(self) -> None:
        for name in _RATE_FIELDS:
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be non-negative, got {getattr(self, name)!r}")

    def _commission(self, notional: float) -> float:
        return max(notional * self.commission_rate, self.min_commission)

    def _transfer_fee(self, notional: float) -> float:
        return notional * self.transfer_fee_rate

    def buy_cost(self, notional: float) -> float:
        """Total buy-side fees: commission (floored) plus transfer fee."""
        _require_non_negative(notional)
        if notional == 0.0:
            return 0.0
        return self._commission(notional) + self._transfer_fee(notional)

    def sell_cost(self, notional: float) -> float:
        """Total sell-side fees: commission, transfer fee and stamp duty."""
        _require_non_negative(notional)
        if notional == 0.0:
            return 0.0
        return self._commission(notional) + self._transfer_fee(notional) + notional * self.stamp_duty_rate

    def buy_fill_price(self, ref_price: float) -> float:
        """Price actually paid per share: the reference price marked up by slippage."""
        _require_positive_price(ref_price)
        return ref_price * (1.0 + self.slippage_bps / BPS)

    def sell_fill_price(self, ref_price: float) -> float:
        """Price actually received per share: reference price marked down by slippage."""
        _require_positive_price(ref_price)
        return ref_price * (1.0 - self.slippage_bps / BPS)


def _require_non_negative(notional: float) -> None:
    # A negative notional is an upstream bug; clamping it here would hide it
    # behind a plausible-looking minimum commission.
    if notional < 0:
        raise ValueError(f"notional must be non-negative, got {notional!r}")


def _require_positive_price(price: float) -> None:
    if price <= 0:
        raise ValueError(f"price must be positive, got {price!r}")
