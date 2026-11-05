"""Evaluate every checkpoint of a run with the shared harness (PLAN.md §3, CONTRIBUTING.md rule 2).

    .venv-sb3/bin/python scripts/eval.py --config configs/ppo_cnn_250k.yaml --seed 0

Reads runs/<exp>/<method>/seed<k>/ckpt/step_<n>.zip (+ .json) and writes, overwriting:
    results/<exp>/<method>/seed<k>/eval.csv           PLAN.md §7 schema, sorted by checkpoint
    results/<exp>/<method>/seed<k>/eval_episodes.csv  every eval episode (ckpt_env_steps added)
    results/<exp>/<method>/seed<k>/eval_meta.json     git, config, hardware, eval wall-clock
Checkpoints are evaluated in parallel worker processes; each result depends only on the
checkpoint and cfg.eval, so the files are identical across re-runs and worker counts.
"""

import argparse
import glob
import json
import multiprocessing
import os
import re
import sys
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
from rlproj.eval.harness import evaluate, summarize  # noqa: E402
from rlproj.utils.config import load_config  # noqa: E402
from rlproj.utils.csvlog import write_rows  # noqa: E402
from rlproj.utils.meta import RunMeta  # noqa: E402


def _eval_checkpoint(args):
    cfg, seed, path = args
    import torch
    from sb3_contrib import RecurrentPPO
    from stable_baselines3 import DQN, PPO

    from rlproj.adapters.sb3 import SB3Adapter

    torch.set_num_threads(1)
    algos = {"PPO": PPO, "RecurrentPPO": RecurrentPPO, "DQN": DQN}
    ev = cfg["eval"]
    model = algos[cfg["algo"]].load(path, device=ev.get("device", "cpu"))
    adapter = SB3Adapter(model, policy=ev["policy"], epsilon=ev.get("epsilon", 0.05),
                         recurrent=cfg["algo"] == "RecurrentPPO")
    with open(path[: -len(".zip")] + ".json") as f:
        info = json.load(f)
    rows, episodes = [], []
    for split in ev["splits"]:
        eps = evaluate(adapter, cfg["env"], split, ev["n_episodes"], ev["rng_seed"],
                       n_envs=ev["n_envs"])
        rows += summarize(eps, cfg["env"], exp=cfg["exp"], method=cfg["method"], seed=seed,
                          ckpt_env_steps=info["env_steps"], wall_clock_s=info["wall_clock_s"])
        episodes += [dict(e, ckpt_env_steps=info["env_steps"]) for e in eps]
    return info["env_steps"], rows, episodes


def eval_run(cfg, seed, config_path=None):
    rel = os.path.join(cfg["exp"], cfg["method"], f"seed{seed}")
    ckpts = glob.glob(os.path.join(REPO_ROOT, "runs", rel, "ckpt", "step_*.zip"))
    ckpts.sort(key=lambda p: int(re.search(r"step_(\d+)\.zip$", p).group(1)))
    if not ckpts:
        raise SystemExit(f"no checkpoints under runs/{rel}/ckpt")
    results_dir = os.path.join(REPO_ROOT, "results", rel)
    os.makedirs(results_dir, exist_ok=True)
    meta = RunMeta(cfg, config_path=config_path, device=cfg["eval"].get("device", "cpu"))
    meta.data["seed"] = seed
    t0 = time.perf_counter()
    jobs = [(cfg, seed, p) for p in ckpts]
    with multiprocessing.get_context("spawn").Pool(cfg["eval"].get("workers", 1)) as pool:
        out = []
        for steps, rows, episodes in pool.imap_unordered(_eval_checkpoint, jobs):
            out.append((steps, rows, episodes))
            pt = [r for r in rows if r["target"] == cfg["env"].get("target", "place_table")]
            print(f"[eval] {cfg['method']} seed {seed} step {steps}: " + ", ".join(
                f"{r['split']} {r['target']} {r['success_rate']:.2f}" for r in pt), flush=True)
    out.sort(key=lambda x: x[0])
    write_rows(os.path.join(results_dir, "eval.csv"), [r for _, rows, _ in out for r in rows],
               "eval")
    episodes = [e for _, _, eps in out for e in eps]
    write_rows(os.path.join(results_dir, "eval_episodes.csv"), episodes,
               ["ckpt_env_steps"] + [k for k in episodes[0] if k != "ckpt_env_steps"])
    meta.finish(os.path.join(results_dir, "eval_meta.json"), n_checkpoints=len(ckpts),
                eval_wall_clock_s=round(time.perf_counter() - t0, 2))
    print(f"[eval] wrote {len(ckpts)} checkpoints to results/{rel}/eval.csv", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    eval_run(load_config(args.config), args.seed, args.config)


if __name__ == "__main__":
    main()
