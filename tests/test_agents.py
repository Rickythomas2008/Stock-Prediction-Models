from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stockpred.agents import (
    MIGRATED_AGENTS,
    ACConfig,
    ACDuelRecurrentAgent,
    ActorCriticAgent,
    AgentError,
    AgentPolicy,
    DoubleDuelRecurrentDQNAgent,
    DQNAgent,
    DQNConfig,
    DuelingDQNAgent,
    ESConfig,
    EvolutionStrategyAgent,
    NEConfig,
    NeuroEvolutionAgent,
    NoveltySearchESAgent,
    episode_returns,
)
from stockpred.envs import AshareTradingEnv

WINDOW = 12


@pytest.fixture
def bars() -> pd.DataFrame:
    n = 26
    days = pd.bdate_range("2024-01-01", periods=n)
    rng = np.random.default_rng(5)
    returns = np.clip(rng.normal(0.001, 0.010, n), -0.09, 0.09)
    closes = list(12.0 * np.cumprod(1 + returns))
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


@pytest.fixture
def env(bars: pd.DataFrame) -> AshareTradingEnv:
    return AshareTradingEnv(
        bars,
        max_lots=2,
        observation_mode="price_diffs",
        action_mode="unit",
        window_size=WINDOW,
    )


def _small_config(cls):
    if cls in {DQNAgent, DuelingDQNAgent, DoubleDuelRecurrentDQNAgent}:
        return {"config": DQNConfig(hidden=16, batch_size=8, buffer_size=200, target_sync=8)}
    if cls in {ActorCriticAgent, ACDuelRecurrentAgent}:
        return {"config": ACConfig(hidden=16)}
    if cls is EvolutionStrategyAgent:
        return {"config": ESConfig(population_size=4, hidden=8)}
    if cls is NeuroEvolutionAgent:
        return {"config": NEConfig(population_size=4, hidden=8)}
    if cls is NoveltySearchESAgent:
        return {"config": ESConfig(population_size=4, hidden=8)}
    return {}


def test_eight_agents_are_registered() -> None:
    assert len(MIGRATED_AGENTS) == 8
    names = {cls.__name__ for cls in MIGRATED_AGENTS}
    assert names == {
        "DQNAgent",
        "DuelingDQNAgent",
        "DoubleDuelRecurrentDQNAgent",
        "ActorCriticAgent",
        "ACDuelRecurrentAgent",
        "EvolutionStrategyAgent",
        "NeuroEvolutionAgent",
        "NoveltySearchESAgent",
    }


@pytest.mark.parametrize("agent_cls", MIGRATED_AGENTS, ids=lambda cls: cls.__name__)
def test_agent_satisfies_policy_protocol(agent_cls, env: AshareTradingEnv) -> None:
    obs, _ = env.reset(seed=0)
    agent = agent_cls(obs.shape[0], env.n_actions, **_small_config(agent_cls))

    assert isinstance(agent, AgentPolicy)
    assert isinstance(agent.name, str) and agent.name
    action = agent.act(obs, deterministic=True)
    assert 0 <= action < env.n_actions


@pytest.mark.parametrize("agent_cls", MIGRATED_AGENTS, ids=lambda cls: cls.__name__)
def test_agent_completes_short_learning_run(agent_cls, env: AshareTradingEnv) -> None:
    obs, _ = env.reset(seed=0)
    agent = agent_cls(obs.shape[0], env.n_actions, **_small_config(agent_cls))
    stats = agent.learn(env, total_steps=8)

    assert isinstance(stats, dict) and stats
    assert all(isinstance(value, float) for value in stats.values())
    assert np.isfinite(list(stats.values())).all()


@pytest.mark.parametrize("agent_cls", MIGRATED_AGENTS, ids=lambda cls: cls.__name__)
def test_wrong_observation_dimension_is_rejected(agent_cls, env: AshareTradingEnv) -> None:
    obs, _ = env.reset(seed=0)
    agent = agent_cls(obs.shape[0], env.n_actions, **_small_config(agent_cls))
    with pytest.raises(AgentError, match="obs_dim"):
        agent.act(np.zeros(3, dtype=np.float32))


@pytest.mark.parametrize("agent_cls", MIGRATED_AGENTS, ids=lambda cls: cls.__name__)
def test_zero_steps_rejected(agent_cls, env: AshareTradingEnv) -> None:
    obs, _ = env.reset(seed=0)
    agent = agent_cls(obs.shape[0], env.n_actions, **_small_config(agent_cls))
    with pytest.raises(AgentError, match="total_steps"):
        agent.learn(env, total_steps=0)


def test_invalid_sizes_rejected() -> None:
    with pytest.raises(AgentError, match="positive"):
        DQNAgent(0, 3)
    with pytest.raises(AgentError, match="positive"):
        ActorCriticAgent(5, 0)


def test_dqn_variants_carry_the_right_switches() -> None:
    plain = DQNAgent(12, 3, config=DQNConfig(hidden=8))
    dueling = DuelingDQNAgent(12, 3, config=DQNConfig(hidden=8))
    full = DoubleDuelRecurrentDQNAgent(12, 3, config=DQNConfig(hidden=8))

    assert (plain.config.double, plain.config.dueling, plain.config.recurrent) == (False, False, False)
    assert (dueling.config.double, dueling.config.dueling, dueling.config.recurrent) == (False, True, False)
    assert (full.config.double, full.config.dueling, full.config.recurrent) == (True, True, True)
    assert len({plain.name, dueling.name, full.name}) == 3


def test_dqn_buffer_is_bounded(env: AshareTradingEnv) -> None:
    obs, _ = env.reset(seed=0)
    agent = DQNAgent(
        obs.shape[0], env.n_actions, config=DQNConfig(hidden=8, batch_size=4, buffer_size=10, target_sync=4)
    )
    agent.learn(env, total_steps=15)
    assert len(agent.buffer) <= 10


def test_dqn_epsilon_decays(env: AshareTradingEnv) -> None:
    obs, _ = env.reset(seed=0)
    agent = DQNAgent(
        obs.shape[0],
        env.n_actions,
        config=DQNConfig(hidden=8, epsilon_start=1.0, epsilon_end=0.05, epsilon_decay_steps=10),
    )
    start = agent._epsilon()
    agent.learn(env, total_steps=12)
    assert agent._epsilon() < start


def test_actor_critic_reports_finite_losses(env: AshareTradingEnv) -> None:
    obs, _ = env.reset(seed=0)
    agent = ActorCriticAgent(obs.shape[0], env.n_actions, config=ACConfig(hidden=8))
    stats = agent.learn(env, total_steps=8)

    assert stats["policy_loss"] == pytest.approx(stats["policy_loss"])  # not NaN
    assert stats["value_loss"] >= 0.0
    assert stats["entropy"] > 0.0


def test_ac_variant_has_dueling_and_recurrent(env: AshareTradingEnv) -> None:
    obs, _ = env.reset(seed=0)
    agent = ACDuelRecurrentAgent(obs.shape[0], env.n_actions, config=ACConfig(hidden=8))
    assert agent.config.dueling is True
    assert agent.config.recurrent is True
    assert agent.name == "ac_duel_recurrent"


def test_evolution_strategy_moves_weights(env: AshareTradingEnv) -> None:
    obs, _ = env.reset(seed=0)
    agent = EvolutionStrategyAgent(obs.shape[0], env.n_actions, config=ESConfig(population_size=4, hidden=8))
    before = agent.policy.get_params().copy()
    stats = agent.learn(env, total_steps=8)

    assert not np.allclose(before, agent.policy.get_params())
    assert stats["generations"] >= 1


def test_neuro_evolution_keeps_population_size(env: AshareTradingEnv) -> None:
    obs, _ = env.reset(seed=0)
    agent = NeuroEvolutionAgent(obs.shape[0], env.n_actions, config=NEConfig(population_size=4, hidden=8))
    agent.learn(env, total_steps=8)

    assert len(agent.population) == 4
    assert agent.champion is not None


def test_novelty_search_grows_archive(env: AshareTradingEnv) -> None:
    obs, _ = env.reset(seed=0)
    agent = NoveltySearchESAgent(obs.shape[0], env.n_actions, config=ESConfig(population_size=4, hidden=8))
    stats = agent.learn(env, total_steps=8)

    assert stats["archive_size"] > 0
    assert len(agent.archive.signatures) == int(stats["archive_size"])


def test_same_seed_gives_same_greedy_actions(bars: pd.DataFrame) -> None:
    def run() -> list[int]:
        env = AshareTradingEnv(
            bars, max_lots=2, observation_mode="price_diffs", action_mode="unit", window_size=WINDOW
        )
        obs, _ = env.reset(seed=0)
        agent = DQNAgent(obs.shape[0], env.n_actions, config=DQNConfig(hidden=8, seed=11))
        agent.learn(env, total_steps=6)
        obs, _ = env.reset(seed=0)
        actions = []
        for _ in range(5):
            action = agent.act(obs, deterministic=True)
            actions.append(action)
            obs, _, done, _, _ = env.step(action)
            if done:
                break
        return actions

    assert run() == run()


def test_agent_works_with_compact_observations(bars: pd.DataFrame) -> None:
    compact = AshareTradingEnv(bars, max_lots=2, observation_mode="compact", action_mode="lots")
    obs, _ = compact.reset(seed=0)
    assert obs.shape == (4,)

    agent = DQNAgent(obs.shape[0], compact.n_actions, config=DQNConfig(hidden=8))
    agent.learn(compact, total_steps=6)
    assert 0 <= agent.act(obs, deterministic=True) < compact.n_actions


def test_episode_returns_helper(env: AshareTradingEnv) -> None:
    obs, _ = env.reset(seed=0)
    agent = EvolutionStrategyAgent(obs.shape[0], env.n_actions, config=ESConfig(population_size=2, hidden=8))
    value = episode_returns(env, agent, max_steps=5)
    assert np.isfinite(value)
