"""T05: SB3 PPO CnnPolicy steps/sec on cpu vs mps (PLAN.md §4 Stage 0).

    .venv-sb3/bin/python scripts/bench_device.py --config configs/stage0_bench_device.yaml

Writes to cfg["out_dir"]:
    bench_device.csv  one row per run (device x vec_env x repeat)
    meta.json         git hash, config, hardware, wall-clock, load averages, summary

Per run: build the vec env, warm the world cache (optional, timed separately), build PPO with
a fixed seed, and time `learn(steps)`. A callback splits each iteration into rollout time
(env stepping + policy forward) and update time (`PPO.train`). "steady" numbers skip the first
iteration, which includes device warm-up.
"""

import argparse
import csv
import functools
import os
import sys
import time

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rlproj.envs.crafter_task import SPLITS, make_env  # noqa: E402
from rlproj.utils.config import load_config  # noqa: E402
from rlproj.utils.meta import RunMeta  # noqa: E402

VEC_ENVS = {"dummy": DummyVecEnv, "subproc": SubprocVecEnv}
COLUMNS = ["device", "vec_env", "repeat", "n_envs", "env_steps", "wall_s", "sps", "sps_steady",
           "rollout_s", "update_s", "first_iter_s", "warmup_s", "train_loss", "approx_kl",
           "torch_threads", "load_avg_1m_before"]


class IterationTimer(BaseCallback):
    """Records (rollout start, rollout end, num_timesteps) for every PPO iteration.
    MPS kernels run asynchronously, so it synchronizes before each timestamp; otherwise
    update work queued by PPO.train() would be billed to the next rollout."""

    def __init__(self, device):
        super().__init__()
        self.iterations = []
        self._start = None
        self._sync = torch.mps.synchronize if device == "mps" else (lambda: None)

    def _now(self):
        self._sync()
        return time.perf_counter()

    def _on_rollout_start(self):
        self._start = self._now()

    def _on_rollout_end(self):
        self.iterations.append((self._start, self._now(), self.num_timesteps))

    def _on_step(self):
        return True


def run_once(cfg, device, vec_env, repeat):
    ppo_cfg = dict(cfg["ppo"])
    n_envs = ppo_cfg.pop("n_envs")
    load = os.getloadavg()[0]
    venv = make_vec_env(functools.partial(make_env, cfg["env"], "train"), n_envs=n_envs,
                        seed=cfg["seed"], vec_env_cls=VEC_ENVS[vec_env])
    t0 = time.perf_counter()
    if cfg.get("warm_cache", False):
        for world_seed in SPLITS["train"]:
            venv.env_method("reset", options={"world_seed": world_seed})
    warmup = time.perf_counter() - t0

    model = PPO("CnnPolicy", venv, device=device, seed=cfg["seed"], verbose=0, **ppo_cfg)
    timer = IterationTimer(device)
    start = timer._now()
    model.learn(cfg["steps"], callback=timer)
    end = timer._now()
    values = dict(model.logger.name_to_value)  # last PPO.train() call
    venv.close()

    its = timer.iterations
    # Update time of iteration i = next rollout start (or learn end) - this rollout end.
    next_starts = [s for s, _, _ in its[1:]] + [end]
    rollout = sum(e - s for s, e, _ in its)
    update = sum(n - e for (_, e, _), n in zip(its, next_starts))
    steps = its[-1][2]
    first_steps = its[0][2]
    steady_wall = end - next_starts[0]
    return {
        "device": device, "vec_env": vec_env, "repeat": repeat, "n_envs": n_envs,
        "env_steps": steps, "wall_s": round(end - start, 3),
        "sps": round(steps / (end - start), 1),
        "sps_steady": round((steps - first_steps) / steady_wall, 1),
        "rollout_s": round(rollout, 3), "update_s": round(update, 3),
        "first_iter_s": round(next_starts[0] - its[0][0], 3), "warmup_s": round(warmup, 2),
        "train_loss": values.get("train/loss"), "approx_kl": values.get("train/approx_kl"),
        "torch_threads": torch.get_num_threads(), "load_avg_1m_before": round(load, 2),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    if "mps" in cfg["devices"] and not torch.backends.mps.is_available():
        raise SystemExit("mps requested but torch.backends.mps.is_available() is False")
    meta = RunMeta(cfg, config_path=args.config, device=",".join(cfg["devices"]))
    os.makedirs(cfg["out_dir"], exist_ok=True)

    rows = []
    for repeat in range(cfg["repeats"]):
        for vec_env in cfg["vec_envs"]:
            for device in cfg["devices"]:
                row = run_once(cfg, device, vec_env, repeat)
                rows.append(row)
                print(f"[{device:>3} {vec_env:>7} r{repeat}] {row['sps']:7.1f} sps "
                      f"(steady {row['sps_steady']:7.1f}; rollout {row['rollout_s']:.1f} s, "
                      f"update {row['update_s']:.1f} s; loss {row['train_loss']}; "
                      f"load {row['load_avg_1m_before']})", flush=True)

    with open(os.path.join(cfg["out_dir"], "bench_device.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    summary = {}
    for vec_env in cfg["vec_envs"]:
        for device in cfg["devices"]:
            sel = [r for r in rows if r["device"] == device and r["vec_env"] == vec_env]
            summary[f"{device}/{vec_env}"] = {
                "sps_steady_mean": round(sum(r["sps_steady"] for r in sel) / len(sel), 1),
                "update_s_mean": round(sum(r["update_s"] for r in sel) / len(sel), 2),
                "rollout_s_mean": round(sum(r["rollout_s"] for r in sel) / len(sel), 2)}
    best = max(summary, key=lambda k: summary[k]["sps_steady_mean"])
    meta.finish(os.path.join(cfg["out_dir"], "meta.json"), torch_version=torch.__version__,
                mps_available=torch.backends.mps.is_available(), summary=summary,
                fastest=best)
    print("\nsteady-state sps (mean over repeats):")
    for key, s in summary.items():
        print(f"  {key:15s} {s['sps_steady_mean']:7.1f}  (rollout {s['rollout_s_mean']:.1f} s, "
              f"update {s['update_s_mean']:.1f} s per run)")
    print(f"fastest: {best}")


if __name__ == "__main__":
    main()
