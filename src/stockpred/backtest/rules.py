"""A-share trading rules that make a naive backtest lie.

T+1 settlement, 100-share lots, board-specific price limits and sealed
limit-up / limit-down sessions are the four frictions that turn an "ideal" fill
into a rejected one. Each is a separate gate so that a rejection can name its
own cause instead of collapsing into a generic "cannot trade".
"""

from __future__ import annotations

from datetime import date

LOT_SIZE = 100

ST_LIMIT = 0.05
MAIN_BOARD_LIMIT = 0.10
GROWTH_BOARD_LIMIT = 0.20

_GROWTH_BOARD_PREFIXES = ("300", "301")  # 创业板
_STAR_BOARD_PREFIXES = ("688", "689")  # 科创板
_BSE_PREFIXES = ("8", "4")  # 北交所


def round_lot(shares: float) -> int:
    """Floor to a whole 100-share lot; anything below one lot is zero.

    Non-positive input returns 0 rather than a negative size: a caller that asks
    for a short position must go through the sell path, not through the lot math.
    """
    if shares <= 0:
        return 0
    return int(shares // LOT_SIZE) * LOT_SIZE


def price_limit_ratio(symbol: str, *, is_st: bool = False) -> float:
    """Daily price-limit ratio for ``symbol``.

    ST clearance takes precedence over the board rule (the risk-warning regime is
    the tighter constraint in the spec this engine implements).
    """
    if is_st:
        return ST_LIMIT
    code = _board_code(symbol)
    if code.startswith(_GROWTH_BOARD_PREFIXES) or code.startswith(_STAR_BOARD_PREFIXES):
        return GROWTH_BOARD_LIMIT
    if code.startswith(_BSE_PREFIXES):
        return GROWTH_BOARD_LIMIT
    return MAIN_BOARD_LIMIT


def limit_prices(
    prev_close: float, symbol: str, *, is_st: bool = False
) -> tuple[float, float]:
    """(limit-down, limit-up) for the next session, rounded to the 0.01 tick.

    The ``+ 1e-9`` before rounding lifts float noise (e.g. 10.65 * 0.9 landing at
    9.584999999) onto the correct side of the half-cent boundary.
    """
    if prev_close <= 0:
        raise ValueError(f"prev_close must be positive, got {prev_close!r}")
    ratio = price_limit_ratio(symbol, is_st=is_st)
    down = round(prev_close * (1.0 - ratio) + 1e-9, 2)
    up = round(prev_close * (1.0 + ratio) + 1e-9, 2)
    return down, up


def can_buy(
    *,
    prev_close: float,
    open_price: float,
    high: float,
    low: float,
    symbol: str,
    is_st: bool = False,
    suspended: bool = False,
) -> bool:
    """Whether a buy order can be filled on this bar.

    A gap-locked limit-up session prints ``low == limit_up``: every trade of the
    day settled at the ceiling, so there is nothing left to buy. ``open_price``
    is accepted for call-site symmetry with :func:`can_sell` (and for tick-level
    checks later); the daily seal test uses the day's low.
    """
    if suspended:
        return False
    _, limit_up = limit_prices(prev_close, symbol, is_st=is_st)
    return low < limit_up


def can_sell(
    *,
    prev_close: float,
    open_price: float,
    high: float,
    low: float,
    symbol: str,
    is_st: bool = False,
    suspended: bool = False,
) -> bool:
    """Whether a sell order can be filled on this bar (mirror of :func:`can_buy`).

    A gap-locked limit-down session prints ``high == limit_down``: no buyer, so
    the position cannot be exited that day.
    """
    if suspended:
        return False
    limit_down, _ = limit_prices(prev_close, symbol, is_st=is_st)
    return high > limit_down


class T1Ledger:
    """Per-symbol share ledger enforcing T+1 sellability with FIFO consumption.

    Shares bought on day ``d`` become sellable on any day strictly after ``d``;
    sells consume the oldest sellable buy lots first, which matches how a
    broker's available position is actually released.
    """

    __slots__ = ("_lots",)

    def __init__(self) -> None:
        self._lots: dict[date, int] = {}

    def buy(self, day: date, shares: int) -> None:
        if shares <= 0:
            raise ValueError(f"buy shares must be positive, got {shares}")
        self._lots[day] = self._lots.get(day, 0) + shares

    def sellable(self, day: date) -> int:
        """Shares held before ``day`` (today's purchases excluded)."""
        return sum(held for buy_day, held in self._lots.items() if buy_day < day)

    @property
    def total_shares(self) -> int:
        return sum(self._lots.values())

    def sell(self, day: date, shares: int) -> None:
        if shares <= 0:
            raise ValueError(f"sell shares must be positive, got {shares}")
        available = self.sellable(day)
        if shares > available:
            raise ValueError(
                f"cannot sell {shares} shares on {day}: only {available} are T+1 sellable"
            )
        remaining = shares
        for buy_day in sorted(self._lots):
            if buy_day >= day:
                break
            held = self._lots[buy_day]
            take = min(held, remaining)
            held -= take
            remaining -= take
            if held:
                self._lots[buy_day] = held
            else:
                del self._lots[buy_day]
            if remaining == 0:
                break

    def __repr__(self) -> str:  # pragma: no cover - diagnostics only
        return f"T1Ledger(lots={dict(sorted(self._lots.items()))})"


def _board_code(symbol: str) -> str:
    """Digits of ``symbol``; tolerates ``600519.SH``, ``sh.600519`` and plain codes."""
    return "".join(ch for ch in symbol if ch.isdigit())
