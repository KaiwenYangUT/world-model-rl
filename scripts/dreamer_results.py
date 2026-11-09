"""Turn DreamerV3 episode logs (scripts/train_dreamer.py) into results/ CSVs (PLAN.md §7 schema).

    .venv-sb3/bin/python scripts/dreamer_results.py --config configs/dreamer_size1m_r32_100k.yaml --seed 0

PRELIMINARY / NOT THE SHARED HARNESS: these learning-curve rows come from upstream's in-loop
eval driver (`train_eval`, sampling policy, mode='eval') on our CrafterTask envs: eval env 0
on the train pool, env 1 on the held-out pool, world seeds sampled by the env (not the
harness's fixed list). scripts/eval_dreamer.py (T12) scores the final checkpoint with the
shared harness. Upstream triggers eval rounds on wall time,
so each round's env-step count is interpolated from the train envs' own step counters
(the driver steps all train envs in lockstep, so global steps ~= n_train_envs x env steps).

Writes results/<exp>/dreamerv3/seed<k>/{eval.csv, eval_episodes.csv, train_episodes.csv,
dreamer_results.json}. For runs with step checkpoints (cfg["ckpt_every_steps"]) the in-loop eval
goes to eval_inloop.csv / eval_inloop_episodes.csv, and eval.csv comes from the harness.
"""

import argparse
import datetime
import glob
import json
import os
import re
import sys

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
from rlproj.envs.crafter_task import TRACKED_ACHIEVEMENTS  # noqa: E402
from rlproj.eval.harness import summarize  # noqa: E402
from rlproj.utils.config import load_config  # noqa: E402
from rlproj.utils.csvlog import write_rows  # noqa: E402

TRAIN_COLUMNS = (["env_steps", "wall_clock_s", "env_index", "world_seed", "return", "length",
                  "success"] + [f"first_{a}" for a in TRACKED_ACHIEVEMENTS])


def read_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    rel = os.path.join(cfg["exp"], cfg["method"], f"seed{args.seed}")
    ep_dir = os.path.join(REPO_ROOT, "runs", rel, "episodes")
    out_dir = os.path.join(REPO_ROOT, "results", rel)
    os.makedirs(out_dir, exist_ok=True)

    train, evals = [], []
    for path in sorted(glob.glob(os.path.join(ep_dir, "*.jsonl"))):
        role, index = re.match(r"(train|eval)(\d+)_", os.path.basename(path)).groups()
        for ep in read_jsonl(path):
            ep["env_index"] = int(index)
            (train if role == "train" else evals).append(ep)
    if not train:
        raise SystemExit(f"no training episodes under {ep_dir}")
    n_train_envs = len({e["env_index"] for e in train})

    # Wall time -> global env steps, from the train envs' counters.
    train.sort(key=lambda e: e["time"])
    times = np.array([e["time"] for e in train])
    steps = np.maximum.accumulate(np.array([n_train_envs * e["env_steps"] for e in train]))
    meta_path = os.path.join(out_dir, "meta.json")
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            t0 = datetime.datetime.fromisoformat(json.load(f)["started"]).timestamp()
    else:  # run still going: upstream writes logdir/config.yaml at launch
        t0 = os.path.getmtime(os.path.join(REPO_ROOT, "runs", rel, "logdir", "config.yaml"))

    def global_steps(t):
        return int(round(float(np.interp(t, np.concatenate([[t0], times]),
                                         np.concatenate([[0], steps])))))

    train_rows = [{"env_steps": global_steps(e["time"]), "wall_clock_s": round(e["time"] - t0, 1),
                   "env_index": e["env_index"], "world_seed": e["world_seed"],
                   "return": round(e["return"], 4), "length": e["length"],
                   "success": e["success"],
                   **{f"first_{a}": e[f"first_{a}"] for a in TRACKED_ACHIEVEMENTS}}
                  for e in train]
    write_rows(os.path.join(out_dir, "train_episodes.csv"), train_rows, TRAIN_COLUMNS)

    # Eval rounds: episodes separated by more than half the report interval start a new round.
    gap = 0.5 * float(cfg["dreamer_flags"].get("run.report_every", 300))
    evals.sort(key=lambda e: e["time"])
    rounds, last = [], None
    for ep in evals:
        if last is None or ep["time"] - last > gap:
            rounds.append([])
        rounds[-1].append(ep)
        last = ep["time"]
    rows, episodes = [], []
    for eps in rounds:
        start = eps[0]["time"]
        ckpt = global_steps(start)
        for i, e in enumerate(eps):
            e.setdefault("episode", i)
        rows += summarize(eps, cfg["env"], exp=cfg["exp"], method=cfg["method"], seed=args.seed,
                          ckpt_env_steps=ckpt, wall_clock_s=round(start - t0, 1))
        episodes += [{"ckpt_env_steps": ckpt, **e} for e in eps]
    if rows:
        # Runs with step checkpoints get their eval.csv from the harness (scripts/eval_dreamer.py),
        # so the in-loop numbers go to eval_inloop*.csv instead.
        prefix = "eval_inloop" if cfg.get("ckpt_every_steps") else "eval"
        write_rows(os.path.join(out_dir, f"{prefix}.csv"), rows, "eval")
        write_rows(os.path.join(out_dir, f"{prefix}_episodes.csv"), episodes, list(episodes[0]))
    summary = {"eval_source": "dreamerv3 train_eval in-loop eval driver (not the shared "
                              "harness; scripts/eval_dreamer.py scores the final checkpoint)",
               "n_train_envs": n_train_envs, "n_train_episodes": len(train),
               "n_eval_rounds": len(rounds),
               "eval_episodes_per_round": [len(r) for r in rounds],
               "last_global_env_steps": int(steps[-1])}
    with open(os.path.join(out_dir, "dreamer_results.json"), "w") as f:
        json.dump(summary, f, indent=2)
        f.write("\n")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
