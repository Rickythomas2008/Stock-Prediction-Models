"""The DQN family: `agent/5`, `agent/10`, `agent/13` migrated to PyTorch.

The three originals were three copies of the same file differing only in two
switches -- double-Q, dueling heads, recurrent encoder. Keeping them as three
classes here would copy that mistake, so one implementation carries the switches
and three named agents expose the three historical configurations:

===============  ==========  =======  =========
legacy notebook  double      dueling  recurrent
===============  ==========  =======  =========
`5`              no          no       no
`10`             no          yes      no
`13`             yes         yes      yes
===============  ==========  =======  =========
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any

import numpy as np

try:  # torch is an optional extra
    import torch
    from torch import nn
except ImportError as exc:  # pragma: no cover - only without the extra
    raise ImportError(
        "torch is required for stockpred.agents; install with: pip install -e '.[models]'"
    ) from exc

from stockpred.agents.base import AgentError, flatten


class QNetwork(nn.Module):
    """Q-values from a price-difference window, optionally dueling and recurrent."""

    def __init__(
        self,
        obs_dim: int,
        n_actions: int,
        *,
        hidden: int = 64,
        dueling: bool = False,
        recurrent: bool = False,
    ) -> None:
        super().__init__()
        self.dueling = dueling
        self.recurrent = recurrent
        if recurrent:
            # The window already encodes recent history, so the LSTM runs over the
            # window as a sequence; hidden state is not carried across env steps.
            self.encoder = nn.LSTM(input_size=1, hidden_size=hidden, batch_first=True)
            feature_dim = hidden
        else:
            self.encoder = nn.Sequential(
                nn.Linear(obs_dim, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU()
            )
            feature_dim = hidden

        self.value = nn.Linear(feature_dim, 1)
        if dueling:
            self.advantage = nn.Linear(feature_dim, n_actions)
        else:
            self.head = nn.Linear(feature_dim, n_actions)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.recurrent:
            sequence = x.unsqueeze(-1)
            _, (hidden, _) = self.encoder(sequence)
            features = hidden[-1]
        else:
            features = self.encoder(x)

        if not self.dueling:
            return self.head(features)

        value = self.value(features)
        advantage = self.advantage(features)
        return value + advantage - advantage.mean(dim=1, keepdim=True)


@dataclass
class DQNConfig:
    hidden: int = 64
    double: bool = False
    dueling: bool = False
    recurrent: bool = False
    lr: float = 1e-3
    gamma: float = 0.99
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 400
    batch_size: int = 32
    buffer_size: int = 5_000
    target_sync: int = 50
    seed: int = 7


class DQNAgent:
    """Deep Q-learning with a replay buffer; the flags select the legacy variant."""

    def __init__(
        self,
        obs_dim: int,
        n_actions: int,
        *,
        name: str = "dqn",
        config: DQNConfig | None = None,
    ) -> None:
        if obs_dim <= 0 or n_actions <= 0:
            raise AgentError("obs_dim and n_actions must be positive")
        self.name = name
        self.obs_dim = int(obs_dim)
        self.n_actions = int(n_actions)
        self.config = config or DQNConfig()

        torch.manual_seed(self.config.seed)
        np.random.seed(self.config.seed)
        self._rng = np.random.default_rng(self.config.seed)

        self.q = QNetwork(
            self.obs_dim,
            self.n_actions,
            hidden=self.config.hidden,
            dueling=self.config.dueling,
            recurrent=self.config.recurrent,
        )
        self.target = QNetwork(
            self.obs_dim,
            self.n_actions,
            hidden=self.config.hidden,
            dueling=self.config.dueling,
            recurrent=self.config.recurrent,
        )
        self.target.load_state_dict(self.q.state_dict())
        self.optimizer = torch.optim.Adam(self.q.parameters(), lr=self.config.lr)
        self.buffer: deque[tuple[np.ndarray, int, float, np.ndarray, bool]] = deque(
            maxlen=self.config.buffer_size
        )
        self.steps_done = 0

    # ------------------------------------------------------------------ acting
    def _epsilon(self) -> float:
        span = max(self.config.epsilon_decay_steps, 1)
        progress = min(self.steps_done / span, 1.0)
        return self.config.epsilon_start + progress * (
            self.config.epsilon_end - self.config.epsilon_start
        )

    def act(self, obs: np.ndarray, *, deterministic: bool = False) -> int:
        vector = flatten(obs)
        if vector.size != self.obs_dim:
            raise AgentError(f"expected obs_dim {self.obs_dim}, got {vector.size}")
        if not deterministic and self._rng.random() < self._epsilon():
            return int(self._rng.integers(self.n_actions))
        with torch.no_grad():
            q_values = self.q(torch.from_numpy(vector).unsqueeze(0))
        return int(torch.argmax(q_values, dim=1).item())

    # ---------------------------------------------------------------- learning
    def learn(self, env: Any, *, total_steps: int) -> dict[str, float]:
        if total_steps < 1:
            raise AgentError("total_steps must be >= 1")
        obs, _ = env.reset(seed=self.config.seed)
        episode_reward = 0.0
        rewards: list[float] = []
        losses: list[float] = []

        for step in range(total_steps):
            action = self.act(obs)
            next_obs, reward, done, truncated, _ = env.step(action)
            self.buffer.append((flatten(obs), action, float(reward), flatten(next_obs), bool(done)))
            episode_reward += float(reward)
            obs = next_obs
            self.steps_done += 1

            loss = self._train_step()
            if loss is not None:
                losses.append(loss)

            if step % self.config.target_sync == 0:
                self.target.load_state_dict(self.q.state_dict())

            if done or truncated:
                rewards.append(episode_reward)
                episode_reward = 0.0
                obs, _ = env.reset(seed=self.config.seed + step + 1)

        return {
            "steps": float(total_steps),
            "episodes": float(len(rewards)),
            "total_reward": float(np.sum(rewards)),
            "mean_episode_reward": float(np.mean(rewards)) if rewards else 0.0,
            "mean_loss": float(np.mean(losses)) if losses else 0.0,
            "epsilon": self._epsilon(),
        }

    def _train_step(self) -> float | None:
        if len(self.buffer) < self.config.batch_size:
            return None
        indices = self._rng.integers(0, len(self.buffer), self.config.batch_size)
        batch = [self.buffer[int(index)] for index in indices]
        obs = torch.from_numpy(np.stack([item[0] for item in batch]))
        actions = torch.tensor([item[1] for item in batch], dtype=torch.int64).unsqueeze(1)
        rewards = torch.tensor([item[2] for item in batch], dtype=torch.float32).unsqueeze(1)
        next_obs = torch.from_numpy(np.stack([item[3] for item in batch]))
        dones = torch.tensor([float(item[4]) for item in batch], dtype=torch.float32).unsqueeze(1)

        q_values = self.q(obs).gather(1, actions)
        with torch.no_grad():
            if self.config.double:
                # Double DQN: online net picks the action, target net scores it.
                next_actions = torch.argmax(self.q(next_obs), dim=1, keepdim=True)
                next_q = self.target(next_obs).gather(1, next_actions)
            else:
                next_q = self.target(next_obs).max(dim=1, keepdim=True).values
            target = rewards + self.config.gamma * next_q * (1.0 - dones)

        loss = nn.functional.smooth_l1_loss(q_values, target)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q.parameters(), 1.0)
        self.optimizer.step()
        return float(loss.item())


class DuelingDQNAgent(DQNAgent):
    """`agent/10`: dueling heads, no double-Q, no recurrence."""

    def __init__(self, obs_dim: int, n_actions: int, *, config: DQNConfig | None = None) -> None:
        base = config or DQNConfig()
        super().__init__(
            obs_dim,
            n_actions,
            name="dueling_dqn",
            config=DQNConfig(**{**base.__dict__, "dueling": True, "double": False, "recurrent": False}),
        )


class DoubleDuelRecurrentDQNAgent(DQNAgent):
    """`agent/13`: all three switches on at once."""

    def __init__(self, obs_dim: int, n_actions: int, *, config: DQNConfig | None = None) -> None:
        base = config or DQNConfig()
        super().__init__(
            obs_dim,
            n_actions,
            name="double_duel_recurrent_dqn",
            config=DQNConfig(**{**base.__dict__, "dueling": True, "double": True, "recurrent": True}),
        )
