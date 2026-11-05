"""Evaluate the latest DreamerV3 checkpoint of a run with the shared harness (T12).

    .venv-dreamer/bin/python scripts/eval_dreamer.py --config configs/dreamer_size1m_r32_100k.yaml --seed 0

Upstream keeps only the latest checkpoint (elements.Checkpoint keep=1), so this scores one
checkpoint per run. Its env-step count is interpolated from the run's train_episodes.csv
(scripts/dreamer_results.py) at the checkpoint's timestamp. Same eval settings as the SB3
runs (cfg["eval"] if present, else 50 episodes per split, 8 envs, rng_seed 1234). Writes
results/<exp>/<method>/seed<k>/{eval_harness.csv, eval_harness_episodes.csv,
eval_harness_meta.json}.
"""

import argparse
import csv
import datetime
import json
import os
import sys
from functools import partial

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))
from wm_eval import build_config  # noqa: E402  (sets sys.path for rlproj and dreamerv3)

import numpy as np  # noqa: E402

from rlproj.adapters.dreamer import DreamerAdapter  # noqa: E402
from rlproj.eval.harness import evaluate, summarize  # noqa: E402
from rlproj.utils.config import load_config  # noqa: E402
from rlproj.utils.csvlog import write_rows  # noqa: E402
from rlproj.utils.meta import RunMeta  # noqa: E402
from train_dreamer import make_crafter_env  # noqa: E402

EVAL_DEFAULTS = {"n_episodes": 50, "n_envs": 8, "rng_seed": 1234, "splits": ["train", "heldout"]}


def ckpt_env_steps(results_dir, ckpt_name):
    """Interpolate global env steps at the checkpoint time from train_episodes.csv."""
    stamp = datetime.datetime.strptime(ckpt_name[:15], "%Y%m%dT%H%M%S").astimezone()
    with open(os.path.join(results_dir, "meta.json")) as f:
        t0 = datetime.datetime.fromisoformat(json.load(f)["started"]).timestamp()
    with open(os.path.join(results_dir, "train_episodes.csv"), newline="") as f:
        rows = list(csv.DictReader(f))
    wall = np.array([float(r["wall_clock_s"]) for r in rows])
    steps = np.array([int(r["env_steps"]) for r in rows])
    order = np.argsort(wall)
    return int(round(float(np.interp(stamp.timestamp() - t0, wall[order], steps[order])))), \
        round(stamp.timestamp() - t0, 1)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    ev = dict(EVAL_DEFAULTS, **cfg.get("eval", {}))
    rel = os.path.join(cfg["exp"], cfg["method"], f"seed{args.seed}")
    logdir = os.path.join(REPO_ROOT, "runs", rel, "logdir")
    results_dir = os.path.join(REPO_ROOT, "results", rel)
    meta = RunMeta(cfg, config_path=args.config, device="cpu (jax)")
    meta.data["seed"] = args.seed

    dreamer_main, config = build_config(cfg, args.seed, logdir)
    dreamer_main.make_env = partial(make_crafter_env, cfg["env"], None, "train")
    import elements
    agent = dreamer_main.make_agent(config)
    ckpt_dir = elements.Path(logdir) / "ckpt"
    ckpt_name = (ckpt_dir / "latest").read_text().strip()
    elements.checkpoint.load(ckpt_dir / ckpt_name, dict(agent=agent.load))
    steps, wall = ckpt_env_steps(results_dir, ckpt_name)

    adapter = DreamerAdapter(agent)
    rows, episodes = [], []
    for split in ev["splits"]:
        eps = evaluate(adapter, cfg["env"], split, ev["n_episodes"], ev["rng_seed"],
                       n_envs=ev["n_envs"])
        rows += summarize(eps, cfg["env"], exp=cfg["exp"], method=cfg["method"], seed=args.seed,
                          ckpt_env_steps=steps, wall_clock_s=wall)
        episodes += [dict(e, ckpt_env_steps=steps) for e in eps]
        pt = next(r for r in rows if r["split"] == split and r["target"] == "place_table")
        print(f"[eval_dreamer] {cfg['method']} seed {args.seed} @ {steps} steps, {split}: "
              f"place_table {pt['success_rate']:.2f}, return {pt['mean_return']:.2f}",
              flush=True)
    write_rows(os.path.join(results_dir, "eval_harness.csv"), rows, "eval")
    write_rows(os.path.join(results_dir, "eval_harness_episodes.csv"), episodes,
               ["ckpt_env_steps"] + [k for k in episodes[0] if k != "ckpt_env_steps"])
    meta.finish(os.path.join(results_dir, "eval_harness_meta.json"), checkpoint=ckpt_name,
                ckpt_env_steps=steps, eval=ev)


if __name__ == "__main__":
    main()
