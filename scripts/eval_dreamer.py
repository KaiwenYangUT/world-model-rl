"""Evaluate DreamerV3 checkpoints of a run with the shared harness (T12).

    .venv-dreamer/bin/python scripts/eval_dreamer.py --config configs/dreamer_size1m_r32_100k.yaml --seed 0

Same eval settings as the SB3 runs (cfg["eval"] if present, else 50 episodes per split,
8 envs, rng_seed 1234). Two cases:
  * Step checkpoints (runs with cfg["ckpt_every_steps"]; scripts/train_dreamer.py writes
    runs/<exp>/<method>/seed<k>/ckpt_steps/step_<n>/): every one is scored, and the output has
    the same names and schema as scripts/eval.py for SB3: results/<exp>/<method>/seed<k>/
    {eval.csv, eval_episodes.csv, eval_meta.json}. This is the DreamerV3 learning curve.
  * Otherwise (older runs): upstream keeps only its latest checkpoint (elements.Checkpoint
    keep=1), so only that one is scored. Its env-step count is interpolated from the run's
    train_episodes.csv (scripts/dreamer_results.py) at the checkpoint's timestamp. Writes
    {eval_harness.csv, eval_harness_episodes.csv, eval_harness_meta.json}.
"""

import argparse
import csv
import datetime
import glob
import json
import os
import re
import sys
import time
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


def score(agent, cfg, ev, seed, steps, wall):
    adapter = DreamerAdapter(agent)
    rows, episodes = [], []
    for split in ev["splits"]:
        eps = evaluate(adapter, cfg["env"], split, ev["n_episodes"], ev["rng_seed"],
                       n_envs=ev["n_envs"])
        rows += summarize(eps, cfg["env"], exp=cfg["exp"], method=cfg["method"], seed=seed,
                          ckpt_env_steps=steps, wall_clock_s=wall)
        episodes += [dict(e, ckpt_env_steps=steps) for e in eps]
        pt = next(r for r in rows if r["split"] == split and r["target"] == "place_table")
        print(f"[eval_dreamer] {cfg['method']} seed {seed} @ {steps} steps, {split}: "
              f"place_table {pt['success_rate']:.2f}, return {pt['mean_return']:.2f}",
              flush=True)
    return rows, episodes


def write(results_dir, prefix, rows, episodes):
    write_rows(os.path.join(results_dir, f"{prefix}.csv"), rows, "eval")
    write_rows(os.path.join(results_dir, f"{prefix}_episodes.csv"), episodes,
               ["ckpt_env_steps"] + [k for k in episodes[0] if k != "ckpt_env_steps"])


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
    os.makedirs(results_dir, exist_ok=True)

    step_dirs = glob.glob(os.path.join(REPO_ROOT, "runs", rel, "ckpt_steps", "step_*"))
    step_dirs = [d for d in step_dirs if os.path.exists(os.path.join(d, "done"))]
    step_dirs.sort(key=lambda d: int(re.search(r"step_(\d+)$", d).group(1)))
    if step_dirs:
        t0 = time.perf_counter()
        rows, episodes = [], []
        for d in step_dirs:
            with open(os.path.join(d, "info.json")) as f:
                info = json.load(f)
            elements.checkpoint.load(d, dict(agent=agent.load))
            r, e = score(agent, cfg, ev, args.seed, info["env_steps"], info["wall_clock_s"])
            rows += r
            episodes += e
        write(results_dir, "eval", rows, episodes)
        meta.finish(os.path.join(results_dir, "eval_meta.json"), n_checkpoints=len(step_dirs),
                    eval_wall_clock_s=round(time.perf_counter() - t0, 2), eval=ev)
        print(f"[eval_dreamer] wrote {len(step_dirs)} checkpoints to results/{rel}/eval.csv",
              flush=True)
        return

    ckpt_dir = elements.Path(logdir) / "ckpt"
    ckpt_name = (ckpt_dir / "latest").read_text().strip()
    elements.checkpoint.load(ckpt_dir / ckpt_name, dict(agent=agent.load))
    steps, wall = ckpt_env_steps(results_dir, ckpt_name)
    rows, episodes = score(agent, cfg, ev, args.seed, steps, wall)
    write(results_dir, "eval_harness", rows, episodes)
    meta.finish(os.path.join(results_dir, "eval_harness_meta.json"), checkpoint=ckpt_name,
                ckpt_env_steps=steps, eval=ev)


if __name__ == "__main__":
    main()
