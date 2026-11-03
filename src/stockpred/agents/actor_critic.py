"""Actor-critic: `agent/14` and `agent/17` migrated to PyTorch.

`14` is actor + critic with a softmax policy and a learned baseline. `17` adds the
dueling decomposition and a recurrent encoder, which is why the two share one
implementation with switches -- same reason as the DQN family.
"""

from __future__ import annotations

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


class ActorCriticNet(nn.Module):
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
            self.encoder = nn.LSTM(input_size=1, hidden_size=hidden, batch_first=True)
            feature_dim = hidden
        else:
            self.encoder = nn.Sequential(
                nn.Linear(obs_dim, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh()
            )
            feature_dim = hidden

        self.critic = nn.Linear(feature_dim, 1)
        if dueling:
            self.advantage = nn.Linear(feature_dim, n_actions)
        else:
            self.actor = nn.Linear(feature_dim, n_actions)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.recurrent:
            _, (hidden, _) = self.encoder(x.unsqueeze(-1))
            features = hidden[-1]
        else:
            features = self.encoder(x)

        value = self.critic(features)
        if self.dueling:
            advantage = self.advantage(features)
            logits = advantage - advantage.mean(dim=1, keepdim=True)
        else:
            logits = self.actor(features)
        return logits, value


@dataclass
class ACConfig:
    hidden: int = 64
    dueling: bool = False
    recurrent: bool = False
    lr: float = 2e-3
    gamma: float = 0.99
    entropy_bonus: float = 0.01
    seed: int = 7


class ActorCriticAgent:
    def __init__(
        self,
        obs_dim: int,
        n_actions: int,
        *,
        name: str = "actor_critic",
        config: ACConfig | None = None,
    ) -> None:
        if obs_dim <= 0 or n_actions <= 0:
            raise AgentError("obs_dim and n_actions must be positive")
        self.name = name
        self.obs_dim = int(obs_dim)
        self.n_actions = int(n_actions)
        self.config = config or ACConfig()

        torch.manual_seed(self.config.seed)
        self._rng = np.random.default_rng(self.config.seed)
        self.net = ActorCriticNet(
            self.obs_dim,
            self.n_actions,
            hidden=self.config.hidden,
            dueling=self.config.dueling,
            recurrent=self.config.recurrent,
        )
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.config.lr)

    def _policy(self, obs: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
        vector = flatten(obs)
        if vector.size != self.obs_dim:
            raise AgentError(f"expected obs_dim {self.obs_dim}, got {vector.size}")
        return self.net(torch.from_numpy(vector).unsqueeze(0))

    def act(self, obs: np.ndarray, *, deterministic: bool = False) -> int:
        with torch.no_grad():
            logits, _ = self._policy(obs)
            probabilities = torch.softmax(logits, dim=1).squeeze(0).numpy()
        if deterministic:
            return int(np.argmax(probabilities))
        return int(self._rng.choice(self.n_actions, p=probabilities / probabilities.sum()))

    def learn(self, env: Any, *, total_steps: int) -> dict[str, float]:
        if total_steps < 1:
            raise AgentError("total_steps must be >= 1")

        obs, _ = env.reset(seed=self.config.seed)
        log_probs: list[torch.Tensor] = []
        values: list[torch.Tensor] = []
        entropies: list[torch.Tensor] = []
        rewards: list[float] = []

        for _ in range(total_steps):
            logits, value = self._policy(obs)
            distribution = torch.distributions.Categorical(logits=logits)
            action = distribution.sample()
            next_obs, reward, done, truncated, _ = env.step(int(action.item()))

            log_probs.append(distribution.log_prob(action))
            values.append(value.reshape(()))
            entropies.append(distribution.entropy())
            rewards.append(float(reward))
            obs = next_obs
            if done or truncated:
                obs, _ = env.reset(seed=self.config.seed + len(rewards))

        # Monte-Carlo returns with the critic as baseline: one batch, one update.
        returns = _discounted(rewards, self.config.gamma)
        returns_t = torch.tensor(returns, dtype=torch.float32)
        values_t = torch.stack(values)
        advantage = (returns_t - values_t).detach()

        policy_loss = -(torch.stack(log_probs) * advantage).mean()
        value_loss = nn.functional.mse_loss(values_t, returns_t)
        entropy = torch.stack(entropies).mean()
        loss = policy_loss + 0.5 * value_loss - self.config.entropy_bonus * entropy

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
        self.optimizer.step()

        return {
            "steps": float(total_steps),
            "mean_reward": float(np.mean(rewards)) if rewards else 0.0,
            "total_reward": float(np.sum(rewards)),
            "policy_loss": float(policy_loss.item()),
            "value_loss": float(value_loss.item()),
            "entropy": float(entropy.item()),
        }


def _discounted(rewards: list[float], gamma: float) -> list[float]:
    out = [0.0] * len(rewards)
    running = 0.0
    for index in range(len(rewards) - 1, -1, -1):
        running = rewards[index] + gamma * running
        out[index] = running
    return out


class ACDuelRecurrentAgent(ActorCriticAgent):
    """`agent/17`: actor-critic with dueling logits and a recurrent encoder."""

    def __init__(self, obs_dim: int, n_actions: int, *, config: ACConfig | None = None) -> None:
        base = config or ACConfig()
        super().__init__(
            obs_dim,
            n_actions,
            name="ac_duel_recurrent",
            config=ACConfig(**{**base.__dict__, "dueling": True, "recurrent": True}),
        )
