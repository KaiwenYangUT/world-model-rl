"""Harness adapter for the official DreamerV3 agent (.venv-dreamer; T12).

Wraps an `embodied.jax.Agent` (built and loaded by scripts/eval_dreamer.py) in the protocol of
rlproj.adapters: `reset(n, seed)` / `act(obs, first)`. Uses `agent.policy(..., mode='eval')`,
which samples its action (PLAN.md §3), and passes `first` as `is_first`, so the RSSM state of
a slot restarts with each new episode. Sampling seeds come from the agent's own action counter
(seeded by config.seed); `reset` rewinds that counter, so an eval of a freshly loaded agent is
reproducible. The `seed` argument is recorded but cannot reseed the jitted policy.
"""

import numpy as np

from rlproj.adapters import Adapter


class DreamerAdapter(Adapter):

    def __init__(self, agent):
        self.agent = agent

    def reset(self, n, seed):
        self._n = n
        self.seed = seed
        with self.agent.n_actions.lock:
            self.agent.n_actions.value = 0
        self._carry = self.agent.init_policy(n)

    def act(self, obs, first):
        batch = {
            "image": np.asarray(obs, np.uint8),
            "reward": np.zeros(self._n, np.float32),
            "is_first": np.asarray(first, bool),
            "is_last": np.zeros(self._n, bool),
            "is_terminal": np.zeros(self._n, bool),
        }
        self._carry, acts, _ = self.agent.policy(self._carry, batch, mode="eval")
        return np.asarray(acts["action"]).reshape(self._n).astype(np.int64)
