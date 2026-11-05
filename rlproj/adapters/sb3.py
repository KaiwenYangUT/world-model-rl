"""Harness adapter for SB3 / sb3-contrib models (PPO, RecurrentPPO, DQN). Needs torch.

Eval policies (PLAN.md §3): "sample" (PPO, PPO-LSTM), "greedy", or "epsilon" (DQN, with
`epsilon`=0.05: greedy action replaced by a uniform one with that probability).
"""

import numpy as np
import torch

from rlproj.adapters import N_ACTIONS, Adapter

POLICIES = ("sample", "greedy", "epsilon")


class SB3Adapter(Adapter):

    def __init__(self, model, policy="sample", epsilon=0.05, recurrent=False):
        if policy not in POLICIES:
            raise ValueError(f"policy must be one of {POLICIES}, got {policy!r}")
        self.model = model
        self.policy = policy
        self.epsilon = epsilon
        self.recurrent = recurrent

    def reset(self, n, seed):
        self._n = n
        torch.manual_seed(seed)  # SB3 samples actions with torch
        self._rng = np.random.default_rng(seed)
        self._state = None

    def act(self, obs, first):
        deterministic = self.policy != "sample"
        if self.recurrent:
            actions, self._state = self.model.predict(
                obs, state=self._state, episode_start=first, deterministic=deterministic)
        else:
            actions, _ = self.model.predict(obs, deterministic=deterministic)
        actions = np.asarray(actions).reshape(self._n)
        if self.policy == "epsilon":
            explore = self._rng.random(self._n) < self.epsilon
            actions = np.where(explore, self._rng.integers(N_ACTIONS, size=self._n), actions)
        return actions
