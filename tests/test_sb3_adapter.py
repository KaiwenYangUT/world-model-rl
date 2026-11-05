"""Tests for rlproj.adapters.sb3 (needs torch + sb3; skipped in .venv-dreamer).

    .venv-sb3/bin/python -m unittest tests.test_sb3_adapter -v
"""

import unittest

import numpy as np

try:
    from sb3_contrib import RecurrentPPO
    from stable_baselines3 import DQN

    from rlproj.adapters.sb3 import SB3Adapter
except ImportError:  # .venv-dreamer has no torch
    RecurrentPPO = None

from rlproj.envs.crafter_task import CrafterTask


def frames(seed, n=2):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=(n, 64, 64, 3), dtype=np.uint8)


@unittest.skipIf(RecurrentPPO is None, "torch / sb3 not installed in this venv")
class TestRecurrentAdapter(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.model = RecurrentPPO("CnnLstmPolicy", CrafterTask(), n_steps=16, batch_size=16,
                                 device="cpu", seed=0,
                                 policy_kwargs={"lstm_hidden_size": 32,
                                                "enable_critic_lstm": False})

    def hidden(self, adapter):
        return np.array(adapter._state[0])  # (n_layers, n_envs, hidden)

    def test_state_carries_over_and_resets_per_slot(self):
        a = SB3Adapter(self.model, policy="sample", recurrent=True)
        a.reset(2, seed=0)
        a.act(frames(1), np.array([True, True]))
        h1 = self.hidden(a)
        a.act(frames(2), np.array([False, False]))
        h2 = self.hidden(a)
        self.assertFalse(np.allclose(h1, h2), "state should change from step to step")
        a.act(frames(3), np.array([True, False]))  # slot 0 starts a new episode
        h3 = self.hidden(a)

        fresh = SB3Adapter(self.model, policy="sample", recurrent=True)
        fresh.reset(2, seed=0)
        fresh.act(frames(3), np.array([True, True]))
        h_fresh = self.hidden(fresh)
        np.testing.assert_allclose(h3[:, 0], h_fresh[:, 0], rtol=1e-5, atol=1e-6)
        self.assertFalse(np.allclose(h3[:, 1], h_fresh[:, 1]),
                         "slot 1 continued its episode, so its state must keep history")

    def test_reset_clears_state(self):
        a = SB3Adapter(self.model, recurrent=True)
        a.reset(2, seed=0)
        a.act(frames(1), np.array([True, True]))
        a.reset(2, seed=0)
        self.assertIsNone(a._state)

    def test_same_seed_same_sampled_actions(self):
        out = []
        for _ in range(2):
            a = SB3Adapter(self.model, policy="sample", recurrent=True)
            a.reset(2, seed=7)
            first = np.array([True, True])
            acts = []
            for t in range(5):
                acts.append(a.act(frames(t), first))
                first = np.array([False, False])
            out.append(np.stack(acts))
        np.testing.assert_array_equal(out[0], out[1])


@unittest.skipIf(RecurrentPPO is None, "torch / sb3 not installed in this venv")
class TestEpsilonAdapter(unittest.TestCase):

    def test_epsilon_rate(self):
        model = DQN("CnnPolicy", CrafterTask(), buffer_size=100, learning_starts=0,
                    device="cpu", seed=0)
        greedy = SB3Adapter(model, policy="greedy")
        eps = SB3Adapter(model, policy="epsilon", epsilon=0.5)
        obs = frames(0, n=8)
        greedy.reset(8, seed=0)
        eps.reset(8, seed=0)
        g = greedy.act(obs, np.ones(8, bool))
        diffs = np.mean([np.mean(eps.act(obs, np.zeros(8, bool)) != g) for _ in range(50)])
        # A random action differs from greedy with prob 0.5 * 16/17 ~= 0.47.
        self.assertGreater(diffs, 0.35)
        self.assertLess(diffs, 0.6)
        self.assertTrue(((eps.act(obs, np.zeros(8, bool)) >= 0) &
                         (eps.act(obs, np.zeros(8, bool)) < 17)).all())


if __name__ == "__main__":
    unittest.main()
