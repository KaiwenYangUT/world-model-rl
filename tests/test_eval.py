"""Tests for rlproj.eval (harness, metrics) and rlproj.utils.csvlog. Stdlib unittest.

    .venv-sb3/bin/python -m unittest discover -s tests -t . -v
"""

import csv
import os
import tempfile
import unittest

import numpy as np

from rlproj.adapters import Adapter, RandomAdapter
from rlproj.envs.crafter_task import HELDOUT_SEEDS, NOT_UNLOCKED, TRAIN_SEEDS
from rlproj.eval.harness import (
    ACHIEVEMENTS,
    eval_world_seeds,
    evaluate,
    success_rates,
    summarize,
)
from rlproj.eval.metrics import area_under_curve, crafter_score, steps_to_fraction, wilson_ci
from rlproj.utils.csvlog import EVAL_COLUMNS, write_rows

CFG = {"target": "place_table", "reward_mode": "crafter", "length": 10000}


class FakeEnv:
    """Episode length = world_seed % 5 + 2; place_table unlocks at step 2 on even seeds."""

    def __init__(self, cfg, split):
        self.world_seed = None

    def reset(self, options=None):
        self.world_seed = options["world_seed"]
        self.t = 0
        return np.full((64, 64, 3), self.world_seed % 256, np.uint8), {}

    def step(self, action):
        self.t += 1
        ach = dict.fromkeys(ACHIEVEMENTS, 0)
        if self.world_seed % 2 == 0 and self.t >= 2:
            ach["place_table"] = 1
        done = self.t >= self.world_seed % 5 + 2
        obs = np.full((64, 64, 3), self.world_seed % 256, np.uint8)
        info = {"achievements": ach, "crafter_reward": 1.0 if self.t == 2 else 0.0}
        return obs, 0.5, done, False, info


class RecordingAdapter(Adapter):
    """Checks the protocol: obs shape, and that first=True marks exactly each episode start."""

    def reset(self, n, seed):
        self.n = n
        self.calls = []

    def act(self, obs, first):
        assert obs.shape == (self.n, 64, 64, 3) and obs.dtype == np.uint8
        assert first.shape == (self.n,) and first.dtype == bool
        self.calls.append((obs[:, 0, 0, 0].copy(), first.copy()))
        return np.zeros(self.n, dtype=np.int64)


class TestMetrics(unittest.TestCase):

    def test_wilson(self):
        self.assertEqual(wilson_ci(0, 0), (0.0, 1.0))
        lo, hi = wilson_ci(1, 20)
        self.assertAlmostEqual(lo, 0.0089, places=4)
        self.assertAlmostEqual(hi, 0.2361, places=4)
        lo, hi = wilson_ci(10, 20)
        self.assertAlmostEqual(lo, 0.2993, places=4)
        self.assertAlmostEqual(hi, 0.7007, places=4)
        self.assertEqual(wilson_ci(0, 20)[0], 0.0)
        self.assertAlmostEqual(wilson_ci(0, 20)[1], 0.1611, places=4)

    def test_crafter_score(self):
        self.assertAlmostEqual(crafter_score([0.0] * 22), 0.0)
        self.assertAlmostEqual(crafter_score([1.0] * 22), 100.0)
        # exp(mean(log(1 + [100, 0]))) - 1 = sqrt(101) - 1
        self.assertAlmostEqual(crafter_score([1.0, 0.0]), 101 ** 0.5 - 1)
        with self.assertRaises(ValueError):
            crafter_score([1.5])

    def test_auc_and_steps_to_fraction(self):
        steps = [0, 10, 20, 30]
        self.assertAlmostEqual(area_under_curve(steps, [0, 0.2, 0.4, 0.4]), 8 / 30)  # (1 + 3 + 4) / 30
        self.assertEqual(steps_to_fraction(steps, [0, 0.1, 0.3, 0.4]), 20)
        self.assertIsNone(steps_to_fraction(steps, [0, 0, 0, 0]))
        with self.assertRaises(ValueError):
            area_under_curve([0, 0], [1, 1])


class TestHarness(unittest.TestCase):

    def test_world_seeds(self):
        a = eval_world_seeds("train", 20, 0)
        self.assertEqual(a, eval_world_seeds("train", 20, 0))
        self.assertEqual(len(set(a)), 20)
        self.assertTrue(set(a) <= set(TRAIN_SEEDS))
        self.assertNotEqual(a, eval_world_seeds("train", 20, 1))
        self.assertTrue(set(eval_world_seeds("heldout", 30, 0)) <= set(HELDOUT_SEEDS))
        long = eval_world_seeds("train", 120, 0)  # more than the pool: every seed used
        self.assertEqual(len(long), 120)
        self.assertEqual(set(long), set(TRAIN_SEEDS))

    def test_fake_env_episodes_and_first_flags(self):
        adapter = RecordingAdapter()
        eps = evaluate(adapter, CFG, "train", 7, rng_seed=0, n_envs=3, env_fn=FakeEnv)
        seeds = eval_world_seeds("train", 7, 0)
        self.assertEqual([e["episode"] for e in eps], list(range(7)))
        self.assertEqual([e["world_seed"] for e in eps], seeds)
        for e in eps:
            s = e["world_seed"]
            self.assertEqual(e["length"], s % 5 + 2)
            self.assertAlmostEqual(e["return"], 0.5 * e["length"])
            self.assertEqual(e["crafter_return"], 1.0)
            self.assertEqual(e["first_place_table"], 2 if s % 2 == 0 else NOT_UNLOCKED)
            self.assertEqual(e["first_collect_wood"], NOT_UNLOCKED)
        # first=True exactly once per episode, and the obs that comes with it is the new
        # episode's first frame (FakeEnv obs encodes world_seed % 256).
        starts = [(obs[k], call) for call, (obs, first) in enumerate(adapter.calls)
                  for k in range(3) if first[k]]
        self.assertEqual(len(starts), 7)
        self.assertEqual(sorted(o for o, _ in starts), sorted(s % 256 for s in seeds))
        self.assertTrue(adapter.calls[0][1].all())

    def test_summarize(self):
        eps = evaluate(RecordingAdapter(), CFG, "train", 7, rng_seed=0, n_envs=3,
                       env_fn=FakeEnv)
        rows = summarize(eps, CFG, exp="t", method="fake", seed=0, ckpt_env_steps=100,
                         wall_clock_s=1.5)
        self.assertEqual([r["target"] for r in rows],
                         ["collect_wood", "place_table", "make_wood_pickaxe"])
        for r in rows:
            self.assertEqual(set(r), set(EVAL_COLUMNS))
        pt = rows[1]
        k = sum(e["world_seed"] % 2 == 0 for e in eps)
        self.assertAlmostEqual(pt["success_rate"], round(k / 7, 4))
        self.assertEqual((pt["success_ci_lo"], pt["success_ci_hi"]),
                         tuple(round(x, 4) for x in wilson_ci(k, 7)))
        self.assertEqual(pt["mean_steps_to_success"], 2.0 if k else None)
        self.assertIsNone(rows[0]["mean_steps_to_success"])
        self.assertAlmostEqual(success_rates(eps)["place_table"], k / 7)

    def test_real_env_random_policy_is_reproducible(self):
        a = evaluate(RandomAdapter(), CFG, "heldout", 4, rng_seed=3, n_envs=2)
        b = evaluate(RandomAdapter(), CFG, "heldout", 4, rng_seed=3, n_envs=2)
        self.assertEqual(a, b)
        c = evaluate(RandomAdapter(), CFG, "heldout", 4, rng_seed=4, n_envs=2)
        self.assertNotEqual([e["world_seed"] for e in a], [e["world_seed"] for e in c])
        for e in a:
            self.assertFalse(e["truncated"] and e["terminated"])
            self.assertGreater(e["length"], 0)


class TestCsvLog(unittest.TestCase):

    def test_write_append_and_schema(self):
        row = dict.fromkeys(EVAL_COLUMNS, 1)
        row["mean_steps_to_success"] = None
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "eval.csv")
            write_rows(path, [row], "eval", append=True)
            write_rows(path, [row], "eval", append=True)
            with open(path) as f:
                lines = list(csv.reader(f))
            self.assertEqual(lines[0], list(EVAL_COLUMNS))
            self.assertEqual(len(lines), 3)
            self.assertEqual(lines[1][EVAL_COLUMNS.index("mean_steps_to_success")], "")
            with self.assertRaises(ValueError):
                write_rows(path, [{"exp": 1}], "eval", append=True)
            with open(path, "w") as f:
                f.write("a,b\n")
            with self.assertRaises(ValueError):
                write_rows(path, [row], "eval", append=True)


if __name__ == "__main__":
    unittest.main()
