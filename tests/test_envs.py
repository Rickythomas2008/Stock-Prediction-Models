from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stockpred.backtest import CostModel
from stockpred.envs import AshareTradingEnv

gym = pytest.importorskip("gymnasium")


@pytest.fixture
def bars() -> pd.DataFrame:
    n = 40
    days = pd.bdate_range("2024-01-01", periods=n)
    closes = [10.0 * (1.002**i) for i in range(n)]
    return pd.DataFrame(
        {
            "date": days,
            "open": closes,
            "high": [c * 1.01 for c in closes],
            "low": [c * 0.99 for c in closes],
            "close": closes,
            "volume": 1_000_000.0,
        }
    )


def _env(bars: pd.DataFrame, **kwargs) -> AshareTradingEnv:
    return AshareTradingEnv(bars, max_lots=3, **kwargs)


def test_env_passes_gymnasium_checker(bars: pd.DataFrame) -> None:
    from gymnasium.utils.env_checker import check_env

    check_env(_env(bars), skip_render_check=True)


def test_action_space_covers_buy_hold_sell(bars: pd.DataFrame) -> None:
    env = _env(bars)
    assert env.n_actions == 7  # 3 buys + hold + 3 sells
    assert env.action_space.n == 7
    assert env.observation_space.shape == (4,)


def test_buy_fills_whole_lots_and_charges_fees(bars: pd.DataFrame) -> None:
    env = _env(bars)
    env.reset(seed=0)
    _, reward, _, _, info = env.step(2)  # buy 3 lots

    assert info["executed_lots"] == 3
    assert info["shares"] == 300
    price = info["price"]
    notional = 300 * price
    expected_cash = 100_000.0 - notional - CostModel().buy_cost(notional)
    assert info["cash"] == pytest.approx(expected_cash)
    assert isinstance(reward, float)


def test_t_plus_one_is_structural_at_daily_granularity(bars: pd.DataFrame) -> None:
    """Buy and sell cannot happen in one action, so a same-day round trip is impossible.

    At daily-bar resolution a single ``step`` is one session: the action is either
    a buy, a hold, or a sell. That is what enforces T+1 here -- plus a defensive
    guard on shares bought in the current step.
    """
    env = _env(bars)
    env.reset(seed=0)
    _, _, _, _, info = env.step(2)  # buy 3 lots

    assert info["executed_lots"] == 3
    assert env._bought_today == 300  # guard is armed for this session


def test_sell_allowed_on_the_next_session(bars: pd.DataFrame) -> None:
    env = _env(bars)
    env.reset(seed=0)
    env.step(2)  # buy 3 lots on session 1
    _, _, _, _, info = env.step(5)  # session 2: sell 2 lots

    assert info["executed_lots"] == 2
    assert info["shares"] == 100


def test_no_action_can_both_buy_and_sell(bars: pd.DataFrame) -> None:
    env = _env(bars)
    env.reset(seed=0)
    env.step(2)
    shares_before = env.shares
    cash_before = env.cash
    env.step(2)  # another buy
    assert env.shares > shares_before
    assert env.cash < cash_before


def test_hold_action_does_nothing(bars: pd.DataFrame) -> None:
    env = _env(bars)
    env.reset(seed=0)
    _, _, _, _, info = env.step(3)
    assert info["executed_lots"] == 0
    assert info["shares"] == 0
    assert info["cash"] == pytest.approx(100_000.0)


def test_cannot_buy_more_than_cash_allows(bars: pd.DataFrame) -> None:
    env = _env(bars, initial_cash=250.0)
    env.reset(seed=0)
    _, _, _, _, info = env.step(2)  # wants 3 lots, cash covers 2 at most
    assert info["executed_lots"] <= 2
    assert info["cash"] >= 0.0


def test_episode_terminates_at_last_bar(bars: pd.DataFrame) -> None:
    env = _env(bars)
    env.reset(seed=0)
    terminated = False
    steps = 0
    while not terminated:
        _, _, terminated, truncated, _ = env.step(3)
        steps += 1
        assert not truncated
    assert steps == len(bars) - 1


def test_reset_restores_initial_state(bars: pd.DataFrame) -> None:
    env = _env(bars)
    env.reset(seed=0)
    env.step(2)
    obs, info = env.reset(seed=0)

    assert info["shares"] == 0
    assert info["cash"] == pytest.approx(100_000.0)
    assert obs.shape == (4,)


def test_invalid_action_rejected(bars: pd.DataFrame) -> None:
    env = _env(bars)
    env.reset(seed=0)
    with pytest.raises(ValueError, match="action must be"):
        env.step(99)


def test_sealed_limit_up_blocks_buy() -> None:
    days = pd.bdate_range("2024-01-01", periods=4)
    frame = pd.DataFrame(
        {
            "date": days,
            "open": [10.0, 10.0, 11.0, 11.0],
            "high": [10.1, 10.1, 11.0, 11.0],
            "low": [9.9, 9.9, 11.0, 11.0],
            "close": [10.0, 10.0, 11.0, 11.0],
            "volume": 1_000.0,
        }
    )
    env = AshareTradingEnv(frame, max_lots=2)
    env.reset(seed=0)
    env.step(1)  # session 2: buy 2 lots at 10.0
    _, _, _, _, sealed = env.step(1)  # session 3: open == high == low == limit-up
    assert sealed["executed_lots"] == 0


def test_bad_frames_rejected() -> None:
    with pytest.raises(ValueError, match="missing columns"):
        AshareTradingEnv(pd.DataFrame({"close": [1.0, 2.0, 3.0]}))
    short = pd.DataFrame(
        {
            "date": pd.bdate_range("2024-01-01", periods=2),
            "open": [1.0, 1.0],
            "high": [1.0, 1.0],
            "low": [1.0, 1.0],
            "close": [1.0, 1.0],
        }
    )
    with pytest.raises(ValueError, match="at least 3 bars"):
        AshareTradingEnv(short)
    frame = pd.DataFrame(
        {
            "date": pd.bdate_range("2024-01-01", periods=5),
            "open": [1.0] * 5,
            "high": [1.0] * 5,
            "low": [1.0] * 5,
            "close": [1.0] * 5,
        }
    )
    with pytest.raises(ValueError, match="max_lots"):
        AshareTradingEnv(frame, max_lots=0)


def test_ppo_trains_without_crashing(bars: pd.DataFrame) -> None:
    """The env must be usable by a real RL library, not just by the checker."""
    from stable_baselines3 import PPO
    from stable_baselines3.common.env_checker import check_env as sb3_check_env

    env = _env(bars)
    sb3_check_env(env, warn=False)
    model = PPO(
        "MlpPolicy",
        env,
        n_steps=32,
        batch_size=32,
        n_epochs=1,
        seed=0,
        verbose=0,
        device="cpu",
    )
    model.learn(total_timesteps=64)
    obs, _ = env.reset(seed=0)
    action, _ = model.predict(obs, deterministic=True)
    assert 0 <= int(action) < env.n_actions


def test_reward_is_equity_change(bars: pd.DataFrame) -> None:
    env = _env(bars)
    env.reset(seed=0)
    _, reward, _, _, info = env.step(3)  # hold through one session
    start_equity = 100_000.0
    assert reward == pytest.approx((info["equity"] - start_equity) / start_equity)
    assert np.isfinite(reward)
