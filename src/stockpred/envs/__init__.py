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
        observation_mode: str = "compact",
        action_mode: str = "lots",
        window_size: int = 30,
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
        if observation_mode not in {"compact", "price_diffs"}:
            raise ValueError(
                f"observation_mode must be 'compact' or 'price_diffs', got {observation_mode!r}"
            )
        if action_mode not in {"lots", "unit"}:
            raise ValueError(f"action_mode must be 'lots' or 'unit', got {action_mode!r}")
        if observation_mode == "price_diffs" and window_size < 2:
            raise ValueError("window_size must be >= 2 for price_diffs observations")

        self.bars = bars.reset_index(drop=True)
        self.initial_cash = float(initial_cash)
        self.max_lots = int(max_lots)
        self.cost_model = cost_model or CostModel()
        self.symbol = symbol
        self.block_limits = block_limits
        self.observation_mode = observation_mode
        self.action_mode = action_mode
        self.window_size = int(window_size)

        self._n = len(self.bars)
        # 'unit' reproduces the legacy notebooks' three-way action (buy one unit /
        # hold / sell one unit), so an original agent can be attached without
        # rewriting its output layer.
        self._n_actions = 3 if action_mode == "unit" else 2 * self.max_lots + 1

        self.action_space = spaces.Discrete(self._n_actions)
        # Finite bounds keep gymnasium's checker quiet and catch a scaled-input
        # mistake early (an observation outside these ranges means a bug, not a market).
        if observation_mode == "price_diffs":
            self.observation_space = spaces.Box(
                low=np.array([-1e4] * (self.window_size - 1), dtype=np.float32),
                high=np.array([1e4] * (self.window_size - 1), dtype=np.float32),
                shape=(self.window_size - 1,),
                dtype=np.float32,
            )
        else:
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
        if self.action_mode == "unit":
            # 0 = buy one lot, 1 = hold, 2 = sell one lot
            return {0: ("buy", 1), 1: ("hold", 0), 2: ("sell", 1)}[action]
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
        if self.observation_mode == "price_diffs":
            # Legacy state: consecutive close differences over the window, exactly
            # what `Agent.get_state()` produced in agent/5..22 notebooks.
            start = max(self._step - self.window_size + 1, 0)
            window = self.bars["close"].iloc[start : self._step + 1].to_numpy(dtype=float)
            if len(window) < self.window_size:
                pad = np.full(self.window_size - len(window), window[0], dtype=float)
                window = np.concatenate([pad, window])
            diffs = np.diff(window)
            return diffs.astype(np.float32)

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
