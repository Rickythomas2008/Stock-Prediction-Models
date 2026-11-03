"""Shared contract for the migrated agents.

Every agent here plays against :class:`stockpred.envs.AshareTradingEnv` and
exposes the same three things, so the comparison harness never needs to know
which algorithm it is holding:

* ``name``  -- stable identifier used in reports
* ``act(obs)`` -- choose an action for one observation
* ``learn(env, total_steps)`` -- improve against the environment

The legacy notebooks had no such contract: each one re-implemented its own
training loop against a hard-coded CSV, which is why 23 of them could not be
compared to each other.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class AgentPolicy(Protocol):
    name: str

    def act(self, obs: np.ndarray, *, deterministic: bool = False) -> int: ...

    def learn(self, env: Any, *, total_steps: int) -> dict[str, float]: ...


class AgentError(RuntimeError):
    """An agent was configured or driven in a way its algorithm cannot support."""


def flatten(obs: np.ndarray) -> np.ndarray:
    array = np.asarray(obs, dtype=np.float32).reshape(-1)
    if array.size == 0:
        raise AgentError("empty observation")
    return array


def episode_returns(env: Any, policy: Any, *, max_steps: int | None = None) -> float:
    """Run one greedy episode and return the summed reward."""
    obs, _ = env.reset()
    total = 0.0
    steps = 0
    done = False
    while not done:
        action = policy.act(obs, deterministic=True)
        obs, reward, done, truncated, _ = env.step(action)
        total += float(reward)
        steps += 1
        if max_steps is not None and steps >= max_steps:
            break
        if truncated:
            break
    return total


def as_action_list(actions: int | Sequence[int]) -> list[int]:
    if isinstance(actions, int):
        return [actions]
    return list(actions)
