"""A Gymnasium trading environment for A-share daily bars.

This is the piece the legacy `agent/` notebooks lacked: they hard-code a training
loop against one CSV, so nothing is reusable, testable, or comparable. Here the
market is an environment with an explicit contract, which makes the 23 agent
notebooks reducible to "which policy do you plug in".

Frictions modelled (the ones that change behaviour):
* T+1 -- bought today cannot be sold today
* 100-share lots -- the action is a lot count, not a target weight
* commission floor -- small orders are priced by the 5 CNY minimum, not the rate
* optional limit-up/limit-down block

Deliberately **not** modelled here: fill probability, queue position, intraday
paths. Those belong to a simulator with tick data; pretending a daily bar env has
them would be false precision.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

try:  # gymnasium is an optional extra
    import gymnasium as gym
    from gymnasium import spaces
except ImportError as exc:  # pragma: no cover - only without the extra
    raise ImportError(
        "gymnasium is required for stockpred.envs; install with: pip install -e '.[rl]'"
    ) from exc

from stockpred.backtest.costs import CostModel
from stockpred.backtest.rules import LOT_SIZE, can_buy, can_sell


class AshareTradingEnv(gym.Env):
    """Single-symbol, long-only, daily bars.

    Observation: ``[close/close[0], position_lots/max_lots, cash/capital, step/n]``
    Action: ``Discrete(2*n_lots + 1)`` -- 0..n_lots-1 buy that many lots,
    n_lots is hold, n_lots+1.. is sell that many lots.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        bars: pd.DataFrame,
        *,
        initial_cash: float = 100_000.0,
        max_lots: int = 5,
        cost_model: CostModel | None = None,
        symbol: str = "600519",
        block_limits: bool = True,
    ) -> None:
        super().__init__()
        required = {"date", "open", "high", "low", "close"}
        missing = required - set(bars.columns)
        if missing:
            raise ValueError(f"bars missing columns: {sorted(missing)}")
        if len(bars) < 3:
            raise ValueError("need at least 3 bars")
        if max_lots < 1:
            raise ValueError("max_lots must be >= 1")

        self.bars = bars.reset_index(drop=True)
        self.initial_cash = float(initial_cash)
        self.max_lots = int(max_lots)
        self.cost_model = cost_model or CostModel()
        self.symbol = symbol
        self.block_limits = block_limits

        self._n = len(self.bars)
        self._n_actions = 2 * self.max_lots + 1

        self.action_space = spaces.Discrete(self._n_actions)
        # Finite bounds keep gymnasium's checker quiet and catch a scaled-input
        # mistake early (an observation outside these ranges means a bug, not a market).
        self.observation_space = spaces.Box(
            low=np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([1e4, 1.0, 1e4, 1.0], dtype=np.float32),
            shape=(4,),
            dtype=np.float32,
        )

        self._reset_state()

    # ---------------------------------------------------------------- gym API
    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        self._reset_state()
        return self._observation(), self._info()

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        action = int(action)
        if not 0 <= action < self._n_actions:
            raise ValueError(f"action must be in [0, {self._n_actions - 1}], got {action}")

        row = self.bars.iloc[self._step + 1]
        prev_close = float(self.bars.iloc[self._step]["close"])
        price = float(row["open"])

        # A new session: everything bought before today is now T+1 sellable.
        self._bought_today = 0

        equity_before = self._equity(prev_close)
        side, lots = self._decode(action)
        executed = 0
        if lots:
            executed = self._trade(side, lots, price, prev_close, row)
        self._step += 1

        equity_after = self._equity(float(row["close"]))
        reward = (equity_after - equity_before) / max(self.initial_cash, 1.0)

        terminated = self._step >= self._n - 1
        info = self._info(executed_lots=executed, price=price)
        return self._observation(), float(reward), terminated, False, info

    # -------------------------------------------------------------- internals
    def _reset_state(self) -> None:
        self._step = 0
        self.cash = self.initial_cash
        self.shares = 0
        self._bought_today = 0
        self._base_close = float(self.bars.iloc[0]["close"])

    def _decode(self, action: int) -> tuple[str, int]:
        if action < self.max_lots:
            return "buy", action + 1
        if action == self.max_lots:
            return "hold", 0
        return "sell", action - self.max_lots

    def _trade(self, side: str, lots: int, price: float, prev_close: float, row: pd.Series) -> int:
        if self.block_limits:
            allowed = can_buy if side == "buy" else can_sell
            if not allowed(
                prev_close=prev_close,
                open_price=price,
                high=float(row["high"]),
                low=float(row["low"]),
                symbol=self.symbol,
            ):
                return 0

        if side == "buy":
            if self.cash <= 0:
                return 0
            affordable_lots = min(lots, int(self.cash // (price * LOT_SIZE)))
            while affordable_lots > 0:
                notional = affordable_lots * LOT_SIZE * price
                if notional + self.cost_model.buy_cost(notional) <= self.cash + 1e-9:
                    break
                affordable_lots -= 1
            if affordable_lots <= 0:
                return 0
            notional = affordable_lots * LOT_SIZE * price
            self.cash -= notional + self.cost_model.buy_cost(notional)
            self.shares += affordable_lots * LOT_SIZE
            self._bought_today += affordable_lots * LOT_SIZE
            return affordable_lots

        sellable = self.shares - self._bought_today  # T+1
        sellable_lots = min(lots, sellable // LOT_SIZE)
        if sellable_lots <= 0:
            return 0
        notional = sellable_lots * LOT_SIZE * price
        self.cash += notional - self.cost_model.sell_cost(notional)
        self.shares -= sellable_lots * LOT_SIZE
        return sellable_lots

    def _equity(self, price: float) -> float:
        return self.cash + self.shares * price

    def _observation(self) -> np.ndarray:
        price = float(self.bars.iloc[self._step]["close"])
        max_shares = self.max_lots * LOT_SIZE
        return np.array(
            [
                price / max(self._base_close, 1e-9),
                self.shares / max(max_shares, 1),
                self.cash / max(self.initial_cash, 1e-9),
                self._step / max(self._n - 1, 1),
            ],
            dtype=np.float32,
        )

    def _info(self, *, executed_lots: int = 0, price: float = 0.0) -> dict[str, Any]:
        return {
            "step": self._step,
            "cash": float(self.cash),
            "shares": int(self.shares),
            "equity": float(self._equity(float(self.bars.iloc[self._step]["close"]))),
            "executed_lots": int(executed_lots),
            "price": float(price),
            "symbol": self.symbol,
        }

    @property
    def n_actions(self) -> int:
        return self._n_actions
