"""RQ3 follow-up: does DreamerV3's critic anticipate the table reward its reward head misses?

    .venv-dreamer/bin/python scripts/value_check.py --config configs/analysis_value_check_sparse_r128.yaml

Background: in the sparse-reward ratio-128 runs the reward head almost never predicts the +1 for
placing a table (scripts/wm_eval.py), yet the policy learns to place tables. Our guess is that the
reward reaches the policy through the critic instead, since the critic is also trained on real
replay rewards (`repval` loss, third_party/dreamerv3/dreamerv3/agent.py). If so, the critic's
value should rise in the steps before a table is placed. This script measures that.

For each seed and step checkpoint in the config, it plays the harness's eval worlds (same world
seeds and order as scripts/eval_dreamer.py) with the eval-mode policy and records, at every step,
the critic's value of the current posterior state (`val` head, de-normalised with `valnorm`, i.e.
in reward units: with this reward it estimates the discounted chance of placing a table) and the
wood in the inventory. No training, nothing in third_party is edited: the agent's policy function
is replaced here, before the agent is built, by a copy of upstream `Agent.policy` that also
returns the value (and the critic parameters are added to the policy's parameter set).

Writes results/<exp>/<method>/value_check.csv (long format: seed, ckpt_env_steps, group, key, n,
mean_value) and value_check_meta.json. Groups:
  before_table  key = k: value k steps before the step whose action placed the table
                (k = 0: the state in which the agent chose place_table); successful episodes only.
  no_table      every step of episodes that never placed a table.
  wood          key = wood in the inventory at that step, all steps of all episodes.
"""

import argparse
import csv
import os
import sys
import time
from collections import defaultdict
from functools import partial

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))
from wm_eval import build_config  # noqa: E402  (sets sys.path for rlproj and dreamerv3)

import numpy as np  # noqa: E402

from rlproj.envs.crafter_task import make_env  # noqa: E402
from rlproj.eval.harness import eval_world_seeds  # noqa: E402
from rlproj.utils.config import load_config  # noqa: E402
from rlproj.utils.meta import RunMeta  # noqa: E402
from train_dreamer import make_crafter_env  # noqa: E402

COLUMNS = ["seed", "ckpt_env_steps", "group", "key", "n", "mean_value"]
WOOD_BINS = 5  # wood 0, 1, 2, 3, 4+


def patch_policy_with_value():
    """Make Agent.policy also return the critic's value. Same as upstream Agent.policy (pinned
    commit) except the extra output and no replay-context entries (not needed here)."""
    from dreamerv3 import agent as dv3

    def policy(self, carry, obs, mode="train"):
        (enc_carry, dyn_carry, dec_carry, prevact) = carry
        kw = dict(training=False, single=True)
        reset = obs["is_first"]
        enc_carry, _, tokens = self.enc(enc_carry, obs, reset, **kw)
        dyn_carry, _, feat = self.dyn.observe(dyn_carry, tokens, prevact, reset, **kw)
        if dec_carry:
            dec_carry, _, _ = self.dec(dec_carry, feat, reset, **kw)
        inp = self.feat2tensor(feat)
        act = dv3.sample(self.pol(inp, bdims=1))
        voffset, vscale = self.valnorm.stats()
        out = {"value": self.val(inp, bdims=1).pred() * vscale + voffset}
        return (enc_carry, dyn_carry, dec_carry, act), act, out

    dv3.Agent.policy = policy
    # The jitted policy only receives parameters whose names match policy_keys.
    dv3.Agent.policy_keys = property(lambda self: "^(enc|dyn|dec|pol|val)/")


def play(agent, env_cfg, split, n_episodes, rng_seed, n_envs):
    """Like rlproj.eval.harness.evaluate, but keeps per-step values and wood counts."""
    world_seeds = eval_world_seeds(split, n_episodes, rng_seed)
    n = min(n_envs, n_episodes)
    envs = [make_env(env_cfg, split) for _ in range(n)]
    with agent.n_actions.lock:
        agent.n_actions.value = 0
    carry = agent.init_policy(n)
    first = np.ones(n, dtype=bool)
    current, episodes, next_episode = [None] * n, [None] * n_episodes, 0

    def start(slot):
        nonlocal next_episode
        index = next_episode
        next_episode += 1
        ob, _ = envs[slot].reset(options={"world_seed": world_seeds[index]})
        current[slot] = {"episode": index, "values": [], "wood": [], "wood_now": 0,
                         "table_index": -1}
        first[slot] = True
        return ob

    obs = np.stack([start(slot) for slot in range(n)])
    while any(rec is not None for rec in current):
        batch = {"image": obs.astype(np.uint8), "reward": np.zeros(n, np.float32),
                 "is_first": first.copy(), "is_last": np.zeros(n, bool),
                 "is_terminal": np.zeros(n, bool)}
        carry, acts, outs = agent.policy(carry, batch, mode="eval")
        actions = np.asarray(acts["action"]).reshape(n)
        values = np.asarray(outs["value"], np.float64).reshape(n)
        first[:] = False
        for slot in range(n):
            rec = current[slot]
            if rec is None:
                continue
            rec["values"].append(float(values[slot]))
            rec["wood"].append(rec["wood_now"])
            ob, reward, terminated, truncated, info = envs[slot].step(int(actions[slot]))
            rec["wood_now"] = int(info["inventory"]["wood"])
            if reward > 0:  # sparse reward: +1 only on the first place_table
                rec["table_index"] = len(rec["values"]) - 1
            if terminated or truncated:
                del rec["wood_now"]
                episodes[rec["episode"]] = rec
                current[slot] = None
                if next_episode < n_episodes:
                    ob = start(slot)
            obs[slot] = ob
    return episodes


def aggregate(episodes, seed, steps, max_k):
    rows = []
    add = lambda group, key, vals: rows.append({
        "seed": seed, "ckpt_env_steps": steps, "group": group, "key": key, "n": len(vals),
        "mean_value": round(float(np.mean(vals)), 6) if vals else None})
    before = defaultdict(list)
    no_table, wood = [], defaultdict(list)
    for ep in episodes:
        t = ep["table_index"]
        if t >= 0:
            for k in range(max_k + 1):
                if t - k >= 0:
                    before[k].append(ep["values"][t - k])
        else:
            no_table += ep["values"]
        for v, w in zip(ep["values"], ep["wood"]):
            wood[min(w, WOOD_BINS - 1)].append(v)
    for k in range(max_k + 1):
        add("before_table", k, before[k])
    add("no_table", "", no_table)
    for w in range(WOOD_BINS):
        add("wood", f"{w}+" if w == WOOD_BINS - 1 else w, wood[w])
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    vc = load_config(args.config)
    cfg = load_config(os.path.join(REPO_ROOT, vc["run_config"]))
    out_dir = os.path.join(REPO_ROOT, "results", cfg["exp"], cfg["method"])
    meta = RunMeta(vc, config_path=args.config, device="cpu (jax)")
    patch_policy_with_value()
    import elements

    rows, summary, t0 = [], [], time.perf_counter()
    for seed in vc["seeds"]:
        rel = os.path.join(cfg["exp"], cfg["method"], f"seed{seed}")
        dreamer_main, config = build_config(cfg, seed, os.path.join(REPO_ROOT, "runs", rel, "logdir"))
        dreamer_main.make_env = partial(make_crafter_env, cfg["env"], None, "train")
        agent = dreamer_main.make_agent(config)
        for steps in vc["checkpoints"]:
            ckpt = os.path.join(REPO_ROOT, "runs", rel, "ckpt_steps", f"step_{steps}")
            elements.checkpoint.load(ckpt, dict(agent=agent.load))
            eps = play(agent, cfg["env"], vc["split"], vc["n_episodes"], vc["rng_seed"],
                       vc["n_envs"])
            r = aggregate(eps, seed, steps, vc["max_k"])
            rows += r
            n_table = sum(ep["table_index"] >= 0 for ep in eps)
            get = lambda g, k: next(x["mean_value"] for x in r if x["group"] == g and x["key"] == k)
            summary.append({"seed": seed, "ckpt_env_steps": steps, "episodes": len(eps),
                            "tables": n_table, "steps": sum(len(ep["values"]) for ep in eps)})
            print(f"[value_check] seed {seed} @ {steps}: tables {n_table}/{len(eps)}, "
                  f"value k=0 {get('before_table', 0)}, k=20 {get('before_table', 20)}, "
                  f"no-table eps {get('no_table', '')}", flush=True)
    path = os.path.join(out_dir, "value_check.csv")
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    meta.finish(os.path.join(out_dir, "value_check_meta.json"), run_config=cfg,
                episodes=summary, analysis_wall_clock_s=round(time.perf_counter() - t0, 1))
    print(f"[value_check] wrote {os.path.relpath(path, REPO_ROOT)}", flush=True)


if __name__ == "__main__":
    main()
