"""Train an SB3 / sb3-contrib agent (PPO, RecurrentPPO, DQN) on CrafterTask (PLAN.md §4).

    .venv-sb3/bin/python scripts/train_sb3.py --config configs/ppo_cnn_250k.yaml --seed 0

Writes:
    runs/<exp>/<method>/seed<k>/ckpt/step_<n>.zip (+ .json with training wall-clock)
                                     every cfg.eval.every env steps, including step 0
    results/<exp>/<method>/seed<k>/train_episodes.csv   one row per finished training episode
    results/<exp>/<method>/seed<k>/meta.json            git, config, hardware, timings
Then runs scripts/eval.py on all checkpoints unless --no-eval.
"""

import argparse
import csv
import functools
import json
import os
import sys
import time

import torch
from sb3_contrib import RecurrentPPO
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
from rlproj.envs.crafter_task import SPLITS, TRACKED_ACHIEVEMENTS, make_env  # noqa: E402
from rlproj.utils.config import load_config  # noqa: E402
from rlproj.utils.meta import RunMeta  # noqa: E402

ALGOS = {"PPO": PPO, "RecurrentPPO": RecurrentPPO, "DQN": DQN}
VEC_ENVS = {"dummy": DummyVecEnv, "subproc": SubprocVecEnv}
EPISODE_COLUMNS = (["env_steps", "wall_clock_s", "env_index", "world_seed", "return", "length",
                    "success"] + [f"first_{a}" for a in TRACKED_ACHIEVEMENTS])


def run_dirs(cfg, seed):
    rel = os.path.join(cfg["exp"], cfg["method"], f"seed{seed}")
    return os.path.join(REPO_ROOT, "runs", rel), os.path.join(REPO_ROOT, "results", rel)


class CheckpointAndLog(BaseCallback):
    """Saves a checkpoint every `every` env steps (and at step 0) with the training wall-clock
    so far, and logs every finished episode from the Monitor `episode` info."""

    def __init__(self, every, ckpt_dir, clock):
        super().__init__()
        self.every = every
        self.ckpt_dir = ckpt_dir
        self.clock = clock  # callable -> training seconds so far (excludes eval and warm-up)
        self.next_save = 0
        self.episodes = []

    def _save(self, steps):
        path = os.path.join(self.ckpt_dir, f"step_{steps}")
        self.model.save(path)
        with open(path + ".json", "w") as f:
            json.dump({"env_steps": steps, "wall_clock_s": round(self.clock(), 3)}, f)

    def _on_training_start(self):
        self._save(0)
        self.next_save = self.every

    def _on_step(self):
        for i, info in enumerate(self.locals["infos"]):
            if "episode" in info:
                row = {"env_steps": self.num_timesteps, "wall_clock_s": round(self.clock(), 3),
                       "env_index": i, "world_seed": info["world_seed"],
                       "return": round(info["episode"]["r"], 4), "length": info["episode"]["l"],
                       "success": info["success"]}
                row.update({f"first_{a}": info["first_unlock"][a] for a in TRACKED_ACHIEVEMENTS})
                self.episodes.append(row)
        if self.num_timesteps >= self.next_save:
            self._save(self.next_save)
            self.next_save += self.every
        return True


def train(cfg, seed, config_path):
    run_dir, results_dir = run_dirs(cfg, seed)
    ckpt_dir = os.path.join(run_dir, "ckpt")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    meta = RunMeta(cfg, config_path=config_path, device=cfg["device"])
    meta.data["seed"] = seed

    venv = make_vec_env(functools.partial(make_env, cfg["env"], "train"), n_envs=cfg["n_envs"],
                        seed=seed, vec_env_cls=VEC_ENVS[cfg["vec_env"]])
    t0 = time.perf_counter()
    if cfg.get("warm_cache", False):
        for world_seed in SPLITS["train"]:
            venv.env_method("reset", options={"world_seed": world_seed})
    warmup_s = time.perf_counter() - t0

    model = ALGOS[cfg["algo"]](cfg["policy"], venv, device=cfg["device"], seed=seed, verbose=0,
                               **cfg.get("algo_kwargs", {}))
    start = time.perf_counter()
    callback = CheckpointAndLog(cfg["eval"]["every"], ckpt_dir,
                                clock=lambda: time.perf_counter() - start)
    model.learn(cfg["total_steps"], callback=callback)
    train_s = time.perf_counter() - start
    final_steps = model.num_timesteps  # may overshoot total_steps by < one rollout
    venv.close()

    with open(os.path.join(results_dir, "train_episodes.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=EPISODE_COLUMNS)
        writer.writeheader()
        writer.writerows(callback.episodes)
    meta.finish(os.path.join(results_dir, "meta.json"), env_steps=final_steps,
                train_wall_clock_s=round(train_s, 2), warmup_s=round(warmup_s, 2),
                wall_clock_per_10k_steps_s=round(train_s / final_steps * 1e4, 2),
                train_sps=round(final_steps / train_s, 1), torch_version=torch.__version__,
                n_train_episodes=len(callback.episodes))
    print(f"[train] {cfg['method']} seed {seed}: {final_steps} steps in {train_s:.0f} s "
          f"({final_steps / train_s:.0f} sps), {len(callback.episodes)} episodes", flush=True)
    return run_dir, results_dir


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--no-eval", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config)
    train(cfg, args.seed, args.config)
    if not args.no_eval:
        sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))
        import eval as eval_script
        eval_script.eval_run(cfg, args.seed, args.config)


if __name__ == "__main__":
    main()
