"""Shared evaluation harness: every method's numbers come from here (CONTRIBUTING.md rule 2).

    episodes = evaluate(adapter, env_cfg, split, n_episodes, rng_seed, n_envs=8)
    rows = summarize(episodes, env_cfg, exp=..., method=..., seed=..., ckpt_env_steps=...,
                     wall_clock_s=...)
    write_rows(".../eval.csv", rows, "eval", append=True)

Determinism: the evaluated worlds depend only on (split, n_episodes, rng_seed), so every
checkpoint and every method is scored on the same worlds. Envs run in this process in a fixed
order and the adapter is reset with rng_seed, so a re-run of the same checkpoint with the same
(rng_seed, n_envs) gives identical numbers.

Imports only numpy, crafter, and gymnasium (via rlproj.envs).
"""

import crafter
import numpy as np

from rlproj.envs.crafter_task import NOT_UNLOCKED, SPLITS, TRACKED_ACHIEVEMENTS, make_env
from rlproj.eval.metrics import wilson_ci

ACHIEVEMENTS = tuple(crafter.constants.achievements)


def eval_world_seeds(split, n_episodes, rng_seed):
    """World seeds for an eval: a seeded permutation of the split's pool, repeated if
    n_episodes exceeds the pool size. Independent of the policy being evaluated."""
    pool = np.array(SPLITS[split])
    rng = np.random.default_rng([rng_seed, list(SPLITS).index(split)])
    reps = -(-n_episodes // len(pool))
    seeds = np.concatenate([rng.permutation(pool) for _ in range(reps)])[:n_episodes]
    return [int(s) for s in seeds]


def evaluate(adapter, env_cfg, split, n_episodes, rng_seed, n_envs=8, env_fn=None):
    """Play n_episodes with `adapter` on `split`, n_envs episodes at a time.

    Returns one dict per episode, in episode order: split, episode, world_seed, length,
    return (env reward, per env_cfg reward_mode), crafter_return, terminated, truncated, and
    first_<achievement> = episode step of its first unlock (-1 if never) for all 22.
    `env_fn(env_cfg, split)` replaces make_env (tests use it).
    """
    env_fn = env_fn or make_env
    world_seeds = eval_world_seeds(split, n_episodes, rng_seed)
    n = min(n_envs, n_episodes)
    envs = [env_fn(env_cfg, split) for _ in range(n)]
    adapter.reset(n, seed=rng_seed)
    obs = None
    first = np.ones(n, dtype=bool)
    current = [None] * n  # per slot: running record of the episode being played, or None
    episodes = [None] * n_episodes
    next_episode = 0

    def start(slot):
        nonlocal next_episode
        index = next_episode
        next_episode += 1
        ob, _ = envs[slot].reset(options={"world_seed": world_seeds[index]})
        current[slot] = {"split": split, "episode": index, "world_seed": world_seeds[index],
                         "length": 0, "return": 0.0, "crafter_return": 0.0,
                         "first": dict.fromkeys(ACHIEVEMENTS, NOT_UNLOCKED)}
        first[slot] = True
        return ob

    first_obs = [start(slot) for slot in range(n)]
    obs = np.stack(first_obs)
    while any(rec is not None for rec in current):
        actions = np.asarray(adapter.act(obs, first.copy()))
        first[:] = False
        for slot in range(n):
            rec = current[slot]
            if rec is None:
                continue
            ob, reward, terminated, truncated, info = envs[slot].step(int(actions[slot]))
            rec["length"] += 1
            rec["return"] += float(reward)
            rec["crafter_return"] += float(info["crafter_reward"])
            for name, count in info["achievements"].items():
                if count > 0 and rec["first"][name] == NOT_UNLOCKED:
                    rec["first"][name] = rec["length"]
            if terminated or truncated:
                firsts = rec.pop("first")
                rec.update(terminated=bool(terminated), truncated=bool(truncated),
                           **{f"first_{name}": firsts[name] for name in ACHIEVEMENTS})
                rec["return"] = round(rec["return"], 6)
                rec["crafter_return"] = round(rec["crafter_return"], 6)
                episodes[rec["episode"]] = rec
                current[slot] = None
                if next_episode < n_episodes:
                    ob = start(slot)
            obs[slot] = ob
    return episodes


def summarize(episodes, env_cfg, exp, method, seed, ckpt_env_steps, wall_clock_s,
              achievements=None):
    """eval.csv rows (PLAN.md §7): one per (split, achievement).

    `target` in a row is the achievement whose success that row reports. By default the
    rows cover TRACKED_ACHIEVEMENTS plus env_cfg's target. mean_return uses the env reward
    (env_cfg reward_mode); mean_steps_to_success is over successful episodes only (empty if
    none succeeded).
    """
    target = env_cfg.get("target", "place_table")
    names = achievements or tuple(dict.fromkeys(TRACKED_ACHIEVEMENTS + (target,)))
    rows = []
    for split in dict.fromkeys(e["split"] for e in episodes):
        eps = [e for e in episodes if e["split"] == split]
        n = len(eps)
        mean_return = float(np.mean([e["return"] for e in eps]))
        mean_len = float(np.mean([e["length"] for e in eps]))
        for name in names:
            steps = [e[f"first_{name}"] for e in eps if e[f"first_{name}"] != NOT_UNLOCKED]
            lo, hi = wilson_ci(len(steps), n)
            rows.append({
                "exp": exp, "method": method, "seed": seed, "ckpt_env_steps": ckpt_env_steps,
                "wall_clock_s": wall_clock_s, "split": split, "n_episodes": n,
                "success_rate": round(len(steps) / n, 4), "success_ci_lo": round(lo, 4),
                "success_ci_hi": round(hi, 4), "mean_return": round(mean_return, 4),
                "mean_len": round(mean_len, 2),
                "mean_steps_to_success": round(float(np.mean(steps)), 2) if steps else None,
                "target": name, "reward_mode": env_cfg.get("reward_mode", "crafter")})
    return rows


def success_rates(episodes, split=None):
    """{achievement: success rate} over all 22 achievements, e.g. for crafter_score."""
    eps = [e for e in episodes if split is None or e["split"] == split]
    return {name: float(np.mean([e[f"first_{name}"] != NOT_UNLOCKED for e in eps]))
            for name in ACHIEVEMENTS}
