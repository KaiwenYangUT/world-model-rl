"""Tests for rlproj.envs.crafter_task. Stdlib unittest, so no pytest is needed.

Run from the repo root, in either venv:
    .venv-sb3/bin/python -m unittest discover -s tests -v
"""

import hashlib
import os
import subprocess
import sys
import unittest

import numpy as np
from gymnasium.utils.env_checker import check_env

from rlproj.envs.crafter_task import (
    HELDOUT_SEEDS,
    TRAIN_SEEDS,
    CrafterTask,
    make_env,
)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ACTIONS = np.random.default_rng(123).integers(17, size=100)


def fingerprint(env, world_seed, actions=ACTIONS):
    """sha256 over the first frame and a fixed-action trajectory (obs, reward, flags, inventory)."""
    h = hashlib.sha256()
    obs, _ = env.reset(options={"world_seed": world_seed})
    h.update(obs.tobytes())
    for a in actions:
        obs, reward, terminated, truncated, info = env.step(a)
        h.update(obs.tobytes())
        h.update(repr((round(reward, 6), terminated, truncated,
                       sorted(info["inventory"].items()))).encode())
        if terminated or truncated:
            break
    return h.hexdigest()


def first_frame(env, world_seed):
    obs, _ = env.reset(options={"world_seed": world_seed})
    return obs


def world_map(env, world_seed):
    env.reset(options={"world_seed": world_seed})
    return env._env._world._mat_map.copy()


def unlock_target_at(env, step_index, name):
    """Patch the raw Crafter step so `name` reads as unlocked from step `step_index` on."""
    real_step = env._env.step
    count = {"n": 0}

    def fake_step(action):
        obs, reward, done, info = real_step(action)
        count["n"] += 1
        if count["n"] >= step_index:
            info["achievements"][name] = max(1, info["achievements"][name])
        return obs, reward, done, info

    env._env.step = fake_step


class TestDeterminism(unittest.TestCase):

    def test_same_seed_same_first_frame_and_trajectory(self):
        a, b = CrafterTask(), CrafterTask()
        for s in (0, 7, 49, 10000):
            np.testing.assert_array_equal(first_frame(a, s), first_frame(b, s))
            self.assertEqual(fingerprint(a, s), fingerprint(b, s))

    def test_long_trajectory_with_other_envs_alive(self):
        # Unpatched crafter.Env diverged here within 50-150 steps (chunk-set order, see
        # crafter_task.py), so use a longer run and disturb the memory layout in between.
        long_actions = np.random.default_rng(7).integers(17, size=500)
        for s in (0, 3):
            a = CrafterTask()
            other = CrafterTask()
            fingerprint(other, 99, long_actions[:200])
            b = CrafterTask()
            self.assertEqual(fingerprint(a, s, long_actions), fingerprint(b, s, long_actions))

    def test_episode_history_does_not_matter(self):
        fresh = CrafterTask()
        used = CrafterTask()
        for s in (5, 11, 10003):  # play other episodes first
            obs, _ = used.reset(options={"world_seed": s})
            for a in ACTIONS[:30]:
                used.step(a)
        self.assertEqual(fingerprint(fresh, 3), fingerprint(used, 3))

    def test_same_across_processes_and_hash_seeds(self):
        expected = fingerprint(CrafterTask(), 7)
        code = ("from tests.test_crafter_task import fingerprint; "
                "from rlproj.envs.crafter_task import CrafterTask; "
                "print(fingerprint(CrafterTask(), 7))")
        for hash_seed in ("0", "12345"):
            env = dict(os.environ, PYTHONHASHSEED=hash_seed)
            out = subprocess.run([sys.executable, "-c", code], cwd=REPO_ROOT, env=env,
                                 capture_output=True, text=True, check=True)
            self.assertEqual(out.stdout.strip(), expected, msg=f"PYTHONHASHSEED={hash_seed}")

    def test_different_seeds_give_different_maps(self):
        env = CrafterTask()
        seeds = TRAIN_SEEDS[:10] + HELDOUT_SEEDS[:10]
        maps = {hashlib.sha256(world_map(env, s).tobytes()).hexdigest() for s in seeds}
        self.assertEqual(len(maps), len(seeds))


class TestSeedPools(unittest.TestCase):

    def test_pools(self):
        self.assertEqual(len(TRAIN_SEEDS), 50)
        self.assertEqual(len(HELDOUT_SEEDS), 100)
        self.assertFalse(set(TRAIN_SEEDS) & set(HELDOUT_SEEDS))

    def test_reset_samples_from_pool_reproducibly(self):
        def sampled(seed, n=30, split="train"):
            env = make_env({}, split=split, seed=seed)
            env._env.reset = lambda: None  # skip world generation (~0.5 s); only seeds matter
            return [env.reset()[1]["world_seed"] for _ in range(n)]

        train = sampled(0)
        self.assertEqual(train, sampled(0))
        self.assertNotEqual(train, sampled(1))
        self.assertTrue(set(train) <= set(TRAIN_SEEDS))
        self.assertTrue(set(sampled(0, split="heldout")) <= set(HELDOUT_SEEDS))

    def test_forced_world_seed(self):
        env = make_env({}, split="train", seed=0)
        _, info = env.reset(options={"world_seed": 10042})
        self.assertEqual(info["world_seed"], 10042)


class TestRewardModes(unittest.TestCase):

    def test_crafter_mode_passes_reward_through_and_continues_on_success(self):
        env = CrafterTask(target="place_table", reward_mode="crafter")
        env.reset(options={"world_seed": 0})
        unlock_target_at(env, 5, "place_table")
        for t, a in enumerate(ACTIONS[:10], start=1):
            _, reward, terminated, truncated, info = env.step(a)
            self.assertEqual(reward, info["crafter_reward"])
            self.assertEqual(info["success"], t >= 5)
            # The fake unlock happens after Crafter computed its reward, so only check that
            # success is logged and does not end the episode.
            self.assertFalse(terminated or truncated)
        self.assertEqual(info["first_unlock"]["place_table"], 5)

    def test_sparse_mode_rewards_only_target_and_terminates(self):
        env = CrafterTask(target="place_table", reward_mode="sparse")
        env.reset(options={"world_seed": 0})
        unlock_target_at(env, 5, "place_table")
        rewards = []
        for a in ACTIONS:
            _, reward, terminated, truncated, info = env.step(a)
            rewards.append(reward)
            if terminated or truncated:
                break
        self.assertEqual(rewards, [0.0] * 4 + [1.0])
        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertTrue(info["success"])
        self.assertEqual(info["first_unlock"]["place_table"], 5)

    def test_tracked_achievements_logged_for_any_target(self):
        env = CrafterTask(target="collect_stone")
        _, info = env.reset(options={"world_seed": 0})
        self.assertEqual(set(info["first_unlock"]),
                         {"collect_wood", "place_table", "make_wood_pickaxe", "collect_stone"})

    def test_length_limit_truncates(self):
        env = CrafterTask(length=20)
        env.reset(options={"world_seed": 0})
        for t in range(1, 21):
            _, _, terminated, truncated, _ = env.step(0)  # noop
            if t < 20:
                self.assertFalse(terminated or truncated)
        self.assertFalse(terminated)
        self.assertTrue(truncated)

    def test_bad_arguments(self):
        with self.assertRaises(ValueError):
            CrafterTask(target="not_an_achievement")
        with self.assertRaises(ValueError):
            CrafterTask(reward_mode="dense")
        with self.assertRaises(ValueError):
            make_env({}, split="test")


class TestGymnasiumApi(unittest.TestCase):

    def test_check_env(self):
        env = CrafterTask(length=200)
        check_env(env, skip_render_check=True)

    def test_spaces(self):
        env = CrafterTask()
        obs, _ = env.reset(seed=0)
        self.assertEqual(obs.shape, (64, 64, 3))
        self.assertEqual(obs.dtype, np.uint8)
        self.assertTrue(env.observation_space.contains(obs))
        self.assertEqual(env.action_space.n, 17)


if __name__ == "__main__":
    unittest.main()
