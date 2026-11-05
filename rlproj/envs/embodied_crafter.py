"""CrafterTask as a DreamerV3 `embodied.Env` (.venv-dreamer only: imports embodied/elements).

Mirrors the observation/action spaces of upstream third_party/dreamerv3/embodied/envs/crafter.py
(image, reward, is_first, is_last, is_terminal; action + reset), but the episodes come from
our CrafterTask, so seed pools, reward mode, and success are the same as for every other method.

Every finished episode is appended as one JSON line to `episode_log` with wall time, the env's
own step count, split, world seed, length, return, success, and first-unlock steps.
"""

import json
import time

import elements
import embodied
import numpy as np

from rlproj.envs.crafter_task import TRACKED_ACHIEVEMENTS, make_env


class EmbodiedCrafterTask(embodied.Env):

    def __init__(self, env_cfg, split, seed, episode_log=None):
        self._task = make_env(env_cfg, split, seed=seed)
        self._split = split
        self._log = episode_log
        self._done = True
        self._env_steps = 0
        self._episode = None

    @property
    def obs_space(self):
        return {
            "image": elements.Space(np.uint8, self._task.observation_space.shape),
            "reward": elements.Space(np.float32),
            "is_first": elements.Space(bool),
            "is_last": elements.Space(bool),
            "is_terminal": elements.Space(bool),
            "log/success": elements.Space(np.float32),
            "log/crafter_reward": elements.Space(np.float32),
        }

    @property
    def act_space(self):
        return {
            "action": elements.Space(np.int32, (), 0, self._task.action_space.n),
            "reset": elements.Space(bool),
        }

    def step(self, action):
        if action["reset"] or self._done:
            image, info = self._task.reset()
            self._done = False
            self._episode = {"split": self._split, "world_seed": info["world_seed"],
                             "length": 0, "return": 0.0, "crafter_return": 0.0}
            return self._obs(image, 0.0, False, False, True, 0.0, 0.0)
        image, reward, terminated, truncated, info = self._task.step(int(action["action"]))
        self._env_steps += 1
        self._done = terminated or truncated
        ep = self._episode
        ep["length"] += 1
        ep["return"] += float(reward)
        ep["crafter_return"] += float(info["crafter_reward"])
        if self._done and self._log:
            ep.update(time=time.time(), env_steps=self._env_steps, success=bool(info["success"]),
                      terminated=bool(terminated), truncated=bool(truncated),
                      **{f"first_{a}": info["first_unlock"][a] for a in TRACKED_ACHIEVEMENTS})
            with open(self._log, "a") as f:
                f.write(json.dumps(ep) + "\n")
        return self._obs(image, reward, self._done, terminated, False,
                         float(info["success"]), info["crafter_reward"])

    @staticmethod
    def _obs(image, reward, is_last, is_terminal, is_first, success, crafter_reward):
        return {"image": image, "reward": np.float32(reward), "is_first": is_first,
                "is_last": bool(is_last), "is_terminal": bool(is_terminal),
                "log/success": np.float32(success),
                "log/crafter_reward": np.float32(crafter_reward)}
