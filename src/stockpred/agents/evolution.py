"""Evolution strategies: `agent/6`, `agent/21`, `agent/22`, `updated-NES-google`.

These three did not differ by architecture but by *search*:

* `6` + `updated-NES-google` -- Gaussian perturbation ES: sample a population
  around the current weights, move along the reward-weighted average.
* `21` -- neuro-evolution: a population of small networks, tournament selection,
  mutation and crossover.
* `22` -- the same population search plus a novelty archive, so a policy that
  behaves differently gets credit before it earns reward.

None of them needs gradients, which is why they attach to the environment's
reward signal directly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from stockpred.agents.base import AgentError, flatten


def _relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0.0)


def _softmax(x: np.ndarray) -> np.ndarray:
    shifted = x - np.max(x, axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=-1, keepdims=True)


class PolicyNetwork:
    """Two-layer numpy policy: observation -> hidden -> action logits."""

    def __init__(self, obs_dim: int, n_actions: int, hidden: int = 16, seed: int = 7) -> None:
        rng = np.random.default_rng(seed)
        scale = np.sqrt(1.0 / (obs_dim + hidden))
        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.hidden = hidden
        self.w1 = rng.normal(0.0, scale, size=(obs_dim, hidden))
        self.b1 = np.zeros(hidden)
        self.w2 = rng.normal(0.0, scale, size=(hidden, n_actions))
        self.b2 = np.zeros(n_actions)

    # flat parameter view, so a search can perturb weights without knowing shapes
    def get_params(self) -> np.ndarray:
        return np.concatenate(
            [self.w1.ravel(), self.b1.ravel(), self.w2.ravel(), self.b2.ravel()]
        )

    def set_params(self, params: np.ndarray) -> None:
        index = 0
        for array in (self.w1, self.b1, self.w2, self.b2):
            size = array.size
            array[...] = params[index : index + size].reshape(array.shape)
            index += size

    @property
    def n_params(self) -> int:
        return self.w1.size + self.b1.size + self.w2.size + self.b2.size

    def logits(self, obs: np.ndarray) -> np.ndarray:
        features = _relu(obs @ self.w1 + self.b1)
        return features @ self.w2 + self.b2

    def probabilities(self, obs: np.ndarray) -> np.ndarray:
        return _softmax(self.logits(obs))


def _evaluate(env: Any, policy: PolicyNetwork, *, seed: int | None = None) -> float:
    obs, _ = env.reset(seed=seed)
    total = 0.0
    done = truncated = False
    while not done and not truncated:
        probabilities = policy.probabilities(flatten(obs))
        action = int(np.argmax(probabilities))
        obs, reward, done, truncated, _ = env.step(action)
        total += float(reward)
    return total


@dataclass
class ESConfig:
    population_size: int = 12
    sigma: float = 0.05
    learning_rate: float = 0.05
    hidden: int = 16
    seed: int = 7


class EvolutionStrategyAgent:
    """`agent/6` / `updated-NES-google`: rank-weighted Gaussian perturbation ES."""

    def __init__(
        self,
        obs_dim: int,
        n_actions: int,
        *,
        name: str = "evolution_strategy",
        config: ESConfig | None = None,
    ) -> None:
        if obs_dim <= 0 or n_actions <= 0:
            raise AgentError("obs_dim and n_actions must be positive")
        self.name = name
        self.obs_dim = int(obs_dim)
        self.n_actions = int(n_actions)
        self.config = config or ESConfig()
        self._rng = np.random.default_rng(self.config.seed)
        self.policy = PolicyNetwork(
            self.obs_dim, self.n_actions, hidden=self.config.hidden, seed=self.config.seed
        )

    def act(self, obs: np.ndarray, *, deterministic: bool = False) -> int:
        vector = flatten(obs)
        if vector.size != self.obs_dim:
            raise AgentError(f"expected obs_dim {self.obs_dim}, got {vector.size}")
        probabilities = self.policy.probabilities(vector)
        if deterministic:
            return int(np.argmax(probabilities))
        return int(self._rng.choice(self.n_actions, p=probabilities / probabilities.sum()))

    def learn(self, env: Any, *, total_steps: int) -> dict[str, float]:
        if total_steps < 1:
            raise AgentError("total_steps must be >= 1")

        theta = self.policy.get_params()
        rewards_history: list[float] = []
        generations = 0
        # One generation costs `population_size` episodes, so the step budget maps
        # to generations rather than to timesteps.
        max_generations = max(int(total_steps // max(self.config.population_size, 1)), 1)

        for generation in range(max_generations):
            noise = [
                self._rng.normal(0.0, 1.0, size=theta.size)
                for _ in range(self.config.population_size)
            ]
            scores = np.empty(self.config.population_size, dtype=float)
            for index, direction in enumerate(noise):
                self.policy.set_params(theta + self.config.sigma * direction)
                scores[index] = _evaluate(env, self.policy, seed=self.config.seed + generation)

            # Rank-normalise: scale-free, which keeps the step size meaningful even
            # when reward magnitudes drift between generations.
            order = np.argsort(scores)
            ranks = np.empty_like(order, dtype=float)
            ranks[order] = np.arange(len(scores), dtype=float)
            centred = (ranks - ranks.mean()) / (ranks.std() + 1e-8)

            step = np.zeros_like(theta)
            for index, direction in enumerate(noise):
                step += centred[index] * direction
            theta = theta + self.config.learning_rate / (len(noise) * self.config.sigma) * step
            self.policy.set_params(theta)
            rewards_history.append(float(scores.mean()))
            generations += 1

        return {
            "generations": float(generations),
            "best_generation_mean": float(np.max(rewards_history)) if rewards_history else 0.0,
            "last_generation_mean": rewards_history[-1] if rewards_history else 0.0,
            "final_episode_reward": _evaluate(env, self.policy, seed=self.config.seed),
        }


@dataclass
class NEConfig:
    population_size: int = 12
    hidden: int = 16
    mutation_sigma: float = 0.1
    crossover_rate: float = 0.5
    tournament_size: int = 3
    novelty_weight: float = 0.0
    seed: int = 7


class NeuroEvolutionAgent:
    """`agent/21`: population of networks, tournament selection, mutation, crossover."""

    def __init__(
        self,
        obs_dim: int,
        n_actions: int,
        *,
        name: str = "neuro_evolution",
        config: NEConfig | None = None,
    ) -> None:
        if obs_dim <= 0 or n_actions <= 0:
            raise AgentError("obs_dim and n_actions must be positive")
        self.name = name
        self.obs_dim = int(obs_dim)
        self.n_actions = int(n_actions)
        self.config = config or NEConfig()
        self._rng = np.random.default_rng(self.config.seed)
        self.population: list[PolicyNetwork] = [
            PolicyNetwork(self.obs_dim, self.n_actions, self.config.hidden, seed=self.config.seed + i)
            for i in range(self.config.population_size)
        ]
        self.champion = self.population[0]

    def act(self, obs: np.ndarray, *, deterministic: bool = False) -> int:
        vector = flatten(obs)
        if vector.size != self.obs_dim:
            raise AgentError(f"expected obs_dim {self.obs_dim}, got {vector.size}")
        probabilities = self.champion.probabilities(vector)
        if deterministic:
            return int(np.argmax(probabilities))
        return int(self._rng.choice(self.n_actions, p=probabilities / probabilities.sum()))

    def learn(self, env: Any, *, total_steps: int) -> dict[str, float]:
        if total_steps < 1:
            raise AgentError("total_steps must be >= 1")
        generations = max(int(total_steps // max(self.config.population_size, 1)), 1)
        best_scores: list[float] = []

        for generation in range(generations):
            scores = np.array(
                [
                    self._score(env, policy, seed=self.config.seed + generation)
                    for policy in self.population
                ],
                dtype=float,
            )
            best_scores.append(float(scores.max()))
            champion_index = int(np.argmax(scores))
            self.champion = self.population[champion_index]

            parents = [
                self.population[self._tournament(scores)] for _ in range(self.config.population_size)
            ]
            children: list[PolicyNetwork] = []
            while len(children) < self.config.population_size:
                left, right = parents[len(children) % len(parents)], parents[(len(children) + 1) % len(parents)]
                child = PolicyNetwork(self.obs_dim, self.n_actions, self.config.hidden)
                child.set_params(self._crossover(left.get_params(), right.get_params()))
                child.set_params(child.get_params() + self._rng.normal(0.0, self.config.mutation_sigma, child.n_params))
                children.append(child)
            children[0] = self.population[champion_index]  # elitism
            self.population = children

        return {
            "generations": float(generations),
            "best_ever": float(max(best_scores)) if best_scores else 0.0,
            "last_generation_best": best_scores[-1] if best_scores else 0.0,
            "final_episode_reward": _evaluate(env, self.champion, seed=self.config.seed),
        }

    # -------------------------------------------------------------- internals
    def _score(self, env: Any, policy: PolicyNetwork, *, seed: int) -> float:
        reward = _evaluate(env, policy, seed=seed)
        if self.config.novelty_weight <= 0.0:
            return reward
        return reward + self.config.novelty_weight * _novelty(policy, self._behaviour_signature(env, policy))

    def _tournament(self, scores: np.ndarray) -> int:
        contenders = self._rng.integers(0, len(scores), self.config.tournament_size)
        return int(contenders[int(np.argmax(scores[contenders]))])

    def _crossover(self, left: np.ndarray, right: np.ndarray) -> np.ndarray:
        mask = self._rng.random(left.size) < self.config.crossover_rate
        return np.where(mask, left, right)

    @staticmethod
    def _behaviour_signature(env: Any, policy: PolicyNetwork) -> np.ndarray:
        """Behavioural descriptor: the greedy action taken at each step."""
        obs, _ = env.reset(seed=0)
        actions: list[int] = []
        done = truncated = False
        while not done and not truncated:
            actions.append(int(np.argmax(policy.probabilities(flatten(obs)))))
            obs, _, done, truncated, _ = env.step(actions[-1])
        return np.array(actions, dtype=float)


class _NoveltyArchive:
    def __init__(self, k: int = 5) -> None:
        self.k = k
        self.signatures: list[np.ndarray] = []

    def score(self, signature: np.ndarray) -> float:
        if not self.signatures:
            self.signatures.append(signature)
            return 0.0
        distances = sorted(float(np.mean(np.abs(signature - other))) for other in self.signatures)
        value = float(np.mean(distances[: self.k]))
        self.signatures.append(signature)
        return value


def _novelty(policy: PolicyNetwork, signature: np.ndarray) -> float:
    # Module-level archive keeps the ES and NE variants comparable; a per-agent
    # archive would make the novelty bonus depend on call order.
    global _ARCHIVE
    try:
        archive = _ARCHIVE
    except NameError:
        archive = _NoveltyArchive()
        _ARCHIVE = archive
    return archive.score(signature)


class NoveltySearchESAgent(EvolutionStrategyAgent):
    """`agent/22`: ES with a novelty bonus on top of reward (behaviour-space archive)."""

    def __init__(self, obs_dim: int, n_actions: int, *, config: ESConfig | None = None) -> None:
        super().__init__(obs_dim, n_actions, name="novelty_es", config=config)
        self.archive = _NoveltyArchive()

    def learn(self, env: Any, *, total_steps: int) -> dict[str, float]:
        if total_steps < 1:
            raise AgentError("total_steps must be >= 1")

        theta = self.policy.get_params()
        history: list[float] = []
        generations = max(int(total_steps // max(self.config.population_size, 1)), 1)

        for generation in range(generations):
            noise = [
                self._rng.normal(0.0, 1.0, size=theta.size)
                for _ in range(self.config.population_size)
            ]
            scores = np.empty(self.config.population_size, dtype=float)
            for index, direction in enumerate(noise):
                self.policy.set_params(theta + self.config.sigma * direction)
                reward = _evaluate(env, self.policy, seed=self.config.seed + generation)
                signature = NeuroEvolutionAgent._behaviour_signature(env, self.policy)
                scores[index] = reward + self.archive.score(signature)

            order = np.argsort(scores)
            ranks = np.empty_like(order, dtype=float)
            ranks[order] = np.arange(len(scores), dtype=float)
            centred = (ranks - ranks.mean()) / (ranks.std() + 1e-8)

            step = np.zeros_like(theta)
            for index, direction in enumerate(noise):
                step += centred[index] * direction
            theta = theta + self.config.learning_rate / (len(noise) * self.config.sigma) * step
            self.policy.set_params(theta)
            history.append(float(scores.mean()))

        return {
            "generations": float(generations),
            "archive_size": float(len(self.archive.signatures)),
            "last_generation_mean": history[-1] if history else 0.0,
            "final_episode_reward": _evaluate(env, self.policy, seed=self.config.seed),
        }
