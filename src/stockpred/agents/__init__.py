"""Eight migrated agents, each mapped to the notebook it came from.

The remaining notebooks in `agent/` are kept as reference material and are
deliberately not implemented here -- see `docs/AGENT-MIGRATION.md` for the split
and the reason for each decision.

+------------------------------+--------------------------------------+
| agent                        | legacy notebook                      |
+==============================+======================================+
| ``DQNAgent``                 | `agent/5.q-learning-agent.ipynb`     |
| ``DuelingDQNAgent``          | `agent/10.duel-q-learning-agent.ipynb`|
| ``DoubleDuelRecurrentDQNAgent`` | `agent/13.double-duel-recurrent-q-learning-agent.ipynb` |
| ``ActorCriticAgent``         | `agent/14.actor-critic-agent.ipynb`  |
| ``ACDuelRecurrentAgent``     | `agent/17.actor-critic-duel-recurrent-agent.ipynb` |
| ``EvolutionStrategyAgent``   | `agent/6.evolution-strategy-agent.ipynb` + `updated-NES-google.ipynb` |
| ``NeuroEvolutionAgent``      | `agent/21.neuro-evolution-agent.ipynb` |
| ``NoveltySearchESAgent``     | `agent/22.neuro-evolution-novelty-search-agent.ipynb` |
+------------------------------+--------------------------------------+
"""

from __future__ import annotations

from stockpred.agents.actor_critic import (
    ACConfig,
    ACDuelRecurrentAgent,
    ActorCriticAgent,
)
from stockpred.agents.base import AgentError, AgentPolicy, episode_returns
from stockpred.agents.dqn import (
    DoubleDuelRecurrentDQNAgent,
    DQNAgent,
    DQNConfig,
    DuelingDQNAgent,
)
from stockpred.agents.evolution import (
    ESConfig,
    EvolutionStrategyAgent,
    NEConfig,
    NeuroEvolutionAgent,
    NoveltySearchESAgent,
)

MIGRATED_AGENTS = (
    DQNAgent,
    DuelingDQNAgent,
    DoubleDuelRecurrentDQNAgent,
    ActorCriticAgent,
    ACDuelRecurrentAgent,
    EvolutionStrategyAgent,
    NeuroEvolutionAgent,
    NoveltySearchESAgent,
)

__all__ = [
    "MIGRATED_AGENTS",
    "ACConfig",
    "ACDuelRecurrentAgent",
    "ActorCriticAgent",
    "AgentError",
    "AgentPolicy",
    "DQNAgent",
    "DQNConfig",
    "DoubleDuelRecurrentDQNAgent",
    "DuelingDQNAgent",
    "ESConfig",
    "EvolutionStrategyAgent",
    "NEConfig",
    "NeuroEvolutionAgent",
    "NoveltySearchESAgent",
    "episode_returns",
]
