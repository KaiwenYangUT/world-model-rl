"""T04: random-policy baseline and raw Crafter env throughput (PLAN.md §4 Stage 0).

    .venv-sb3/bin/python scripts/random_baseline.py --config configs/stage0_random.yaml

Writes to cfg["out_dir"]:
    env_sps.csv          raw env steps/sec for each process count (measured first)
    random_episodes.csv  one row per episode: world seed, length, return, first-unlock step of
                         every achievement (empty = never unlocked)
    random.csv           eval.csv schema (PLAN.md §7), one row per (split, achievement as target)
    meta.json            git hash, config, hardware, wall-clock, load averages, summary

Episodes are deterministic and independent of the worker count: episode i of a split uses
world seed pool[i % len(pool)] and action RNG default_rng([seed, split_index, i]).
"""

import argparse
import csv
import functools
import multiprocessing
import os
import sys
import time

import crafter
import gymnasium as gym
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rlproj.envs.crafter_task import SPLITS, make_env  # noqa: E402
from rlproj.eval.metrics import wilson_ci  # noqa: E402
from rlproj.utils.config import load_config  # noqa: E402
from rlproj.utils.meta import RunMeta  # noqa: E402

ACHIEVEMENTS = tuple(crafter.constants.achievements)
N_ACTIONS = len(crafter.constants.actions)
EVAL_COLUMNS = ("exp, method, seed, ckpt_env_steps, wall_clock_s, split, n_episodes, success_rate, "
                "success_ci_lo, success_ci_hi, mean_return, mean_len, mean_steps_to_success, "
                "target, reward_mode").split(", ")


def run_episode(env_cfg, seed, split, index):
    pool = SPLITS[split]
    world_seed = pool[index % len(pool)]
    rng = np.random.default_rng([seed, list(SPLITS).index(split), index])
    env = make_env(env_cfg, split)
    env.reset(options={"world_seed": world_seed})
    first = dict.fromkeys(ACHIEVEMENTS)
    total, length = 0.0, 0
    while True:
        _, reward, terminated, truncated, info = env.step(rng.integers(N_ACTIONS))
        length += 1
        total += reward
        for name, count in info["achievements"].items():
            if count > 0 and first[name] is None:
                first[name] = length
        if terminated or truncated:
            break
    row = {"split": split, "episode": index, "world_seed": world_seed, "length": length,
           "return": round(total, 4), "terminated": terminated, "truncated": truncated}
    row.update({f"first_{name}": first[name] for name in ACHIEVEMENTS})
    return row


def _run_episode_args(args):
    return run_episode(*args)


def measure_single(env_cfg, steps, seed, warm_cache):
    """Plain loop in this process; separates step() time from reset() time."""
    env = make_env(env_cfg, "train", seed=seed)
    t0 = time.perf_counter()
    if warm_cache:
        for world_seed in SPLITS["train"]:
            env.reset(options={"world_seed": world_seed})
    warmup = time.perf_counter() - t0
    rng = np.random.default_rng(seed)
    t0 = time.perf_counter()
    env.reset()
    reset_times = [time.perf_counter() - t0]
    step_time, lengths, length = 0.0, [], 0
    start = time.perf_counter()
    for _ in range(steps):
        t = time.perf_counter()
        _, _, terminated, truncated, _ = env.step(rng.integers(N_ACTIONS))
        step_time += time.perf_counter() - t
        length += 1
        if terminated or truncated:
            lengths.append(length)
            length = 0
            t = time.perf_counter()
            env.reset()
            reset_times.append(time.perf_counter() - t)
    wall = time.perf_counter() - start
    return {"sps": steps / wall, "sps_step_only": steps / step_time, "wall_s": wall,
            "n_resets": len(reset_times), "reset_mean_s": float(np.mean(reset_times)),
            "mean_episode_len": float(np.mean(lengths)) if lengths else float("nan"),
            "startup_s": "", "warmup_s": warmup}


def measure_async(env_cfg, steps, procs, seed, warm_cache):
    """gymnasium AsyncVectorEnv, SAME_STEP autoreset: every step() is one transition per env.
    Startup (process spawn + first reset) and the world-cache warm-up (every worker resets once
    on each train seed) are timed separately and excluded from sps."""
    fns = [functools.partial(make_env, env_cfg, "train", seed + i) for i in range(procs)]
    t0 = time.perf_counter()
    venv = gym.vector.AsyncVectorEnv(fns, autoreset_mode=gym.vector.AutoresetMode.SAME_STEP)
    venv.reset(seed=seed)
    startup = time.perf_counter() - t0
    t0 = time.perf_counter()
    if warm_cache:
        for world_seed in SPLITS["train"]:
            venv.reset(options={"world_seed": world_seed})  # passed to every worker
        venv.reset(seed=seed)
    warmup = time.perf_counter() - t0
    rng = np.random.default_rng(seed)
    calls = -(-steps // procs)
    done = 0
    start = time.perf_counter()
    for _ in range(calls):
        _, _, terminated, truncated, _ = venv.step(rng.integers(N_ACTIONS, size=procs))
        done += int(np.sum(terminated | truncated))
    wall = time.perf_counter() - start
    venv.close()
    transitions = calls * procs
    return {"sps": transitions / wall, "sps_step_only": "", "wall_s": wall,
            "n_resets": done, "reset_mean_s": "", "startup_s": startup,
            "warmup_s": warmup,
            "mean_episode_len": transitions / done if done else float("nan"),
            "env_steps": transitions}


def summarize(episodes, cfg):
    rows = []
    for split in cfg["splits"]:
        eps = [e for e in episodes if e["split"] == split]
        n = len(eps)
        mean_return = float(np.mean([e["return"] for e in eps]))
        mean_len = float(np.mean([e["length"] for e in eps]))
        for name in ACHIEVEMENTS:
            steps = [e[f"first_{name}"] for e in eps if e[f"first_{name}"] is not None]
            lo, hi = wilson_ci(len(steps), n)
            rows.append({
                "exp": cfg["exp"], "method": cfg["method"], "seed": cfg["seed"],
                "ckpt_env_steps": 0, "wall_clock_s": 0, "split": split, "n_episodes": n,
                "success_rate": round(len(steps) / n, 4), "success_ci_lo": round(lo, 4),
                "success_ci_hi": round(hi, 4), "mean_return": round(mean_return, 4),
                "mean_len": round(mean_len, 2),
                "mean_steps_to_success": round(float(np.mean(steps)), 2) if steps else "",
                "target": name, "reward_mode": cfg["env"]["reward_mode"]})
    return rows


def suggest_target(rows):
    """PLAN.md §4 gate on the train split: place_table if 0 < success <= 10%;
    make_wood_pickaxe if place_table >= 20%; otherwise leave it to a human."""
    rate = {r["target"]: r["success_rate"] for r in rows if r["split"] == "train"}
    pt = rate["place_table"]
    if 0 < pt <= 0.10:
        return "place_table"
    if pt >= 0.20:
        return "make_wood_pickaxe"
    return f"undecided (place_table random success = {pt})"


def write_csv(path, rows, columns=None):
    columns = columns or list(rows[0])
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: ("" if row.get(k) is None else row.get(k)) for k in columns})


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    meta = RunMeta(cfg, config_path=args.config, device="cpu")
    out_dir = cfg["out_dir"]
    os.makedirs(out_dir, exist_ok=True)

    # 1) Throughput first, before the parallel episode workers load the machine.
    sps_rows = []
    warm = bool(cfg["sps"].get("warm_cache", False))
    for procs in cfg["sps"]["procs"]:
        load = os.getloadavg()[0]
        if procs == 1:
            r = measure_single(cfg["env"], cfg["sps"]["steps"], cfg["seed"], warm)
            r.update(env_steps=cfg["sps"]["steps"])
        else:
            r = measure_async(cfg["env"], cfg["sps"]["steps"], procs, cfg["seed"], warm)
        r.update(procs=procs, warm_cache=warm, load_avg_1m_before=round(load, 2))
        sps_rows.append(r)
        print(f"[sps] procs={procs}: {r['sps']:.0f} sps over {r['env_steps']} steps "
              f"({r['wall_s']:.1f} s, {r['n_resets']} resets, load {load:.2f})", flush=True)
    for r in sps_rows:
        for k, v in r.items():
            if isinstance(v, float):
                r[k] = round(v, 4)
    write_csv(os.path.join(out_dir, "env_sps.csv"), sps_rows,
              ["procs", "env_steps", "wall_s", "sps", "sps_step_only", "n_resets",
               "reset_mean_s", "mean_episode_len", "startup_s", "warmup_s", "warm_cache",
               "load_avg_1m_before"])

    # 2) Random-policy episodes on each split.
    jobs = [(cfg["env"], cfg["seed"], split, i)
            for split in cfg["splits"] for i in range(cfg["episodes_per_split"])]
    t0 = time.perf_counter()
    with multiprocessing.get_context("spawn").Pool(cfg["eval_procs"]) as pool:
        episodes = []
        for k, row in enumerate(pool.imap(_run_episode_args, jobs), start=1):
            episodes.append(row)
            if k % 50 == 0 or k == len(jobs):
                print(f"[episodes] {k}/{len(jobs)} ({time.perf_counter() - t0:.0f} s)", flush=True)
    episode_wall = time.perf_counter() - t0
    write_csv(os.path.join(out_dir, "random_episodes.csv"), episodes)
    rows = summarize(episodes, cfg)
    write_csv(os.path.join(out_dir, "random.csv"), rows, EVAL_COLUMNS)

    lengths = {s: np.array([e["length"] for e in episodes if e["split"] == s])
               for s in cfg["splits"]}
    length_stats = {s: {"min": int(v.min()), "p25": float(np.percentile(v, 25)),
                        "median": float(np.median(v)), "p75": float(np.percentile(v, 75)),
                        "max": int(v.max()), "mean": float(v.mean()),
                        "frac_truncated": float(np.mean([e["truncated"] for e in episodes
                                                         if e["split"] == s]))}
                    for s, v in lengths.items()}
    tracked = {f"{r['split']}/{r['target']}": [r["success_rate"], r["success_ci_lo"],
                                               r["success_ci_hi"]]
               for r in rows if r["target"] in ("collect_wood", "place_table",
                                                "make_wood_pickaxe", "collect_stone")}
    target = suggest_target(rows)
    meta.finish(os.path.join(out_dir, "meta.json"), episode_phase_wall_s=round(episode_wall, 2),
                episode_length=length_stats, tracked_success=tracked, suggested_target=target)

    print("\nsuccess rate [95% Wilson CI]:")
    for key, (p, lo, hi) in tracked.items():
        print(f"  {key:32s} {p:6.1%}  [{lo:.1%}, {hi:.1%}]")
    for s, st in length_stats.items():
        print(f"episode length {s}: median {st['median']:.0f}, p25 {st['p25']:.0f}, "
              f"p75 {st['p75']:.0f}, min {st['min']}, max {st['max']}, "
              f"truncated {st['frac_truncated']:.1%}")
    print(f"suggested target (PLAN.md §4 gate): {target}")


if __name__ == "__main__":
    main()
