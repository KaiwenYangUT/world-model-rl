"""Headline numbers per method from the results CSVs (PLAN.md §3 metrics, §6 week-1 gate).

    .venv-sb3/bin/python plots/summarize_results.py --exp main

"Final" = the last two eval checkpoints of every seed, pooled (PLAN.md §6 uses the last 2 eval
points). Success CIs are Wilson intervals on the pooled episode counts; return CIs are normal
approximations on the pooled episodes. Writes results/<exp>/summary.csv and summary.md.
"""

import argparse
import csv
import glob
import math
import os
import sys
from collections import defaultdict

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
from rlproj.eval.harness import ACHIEVEMENTS  # noqa: E402
from rlproj.eval.metrics import (  # noqa: E402
    area_under_curve,
    crafter_score,
    steps_to_fraction,
    wilson_ci,
)

TARGET = "place_table"
NAMES = {"ppo_cnn": "PPO", "ppo_lstm": "PPO-LSTM", "dqn": "DQN", "dreamerv3": "DreamerV3",
         "dqn_nstep3": "DQN 3-step", "dreamerv3_r64": "DreamerV3 ratio 64",
         "dreamerv3_r128": "DreamerV3 ratio 128"}


def read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def mean_ci(values):
    v = np.asarray(values, float)
    half = 1.96 * v.std(ddof=1) / math.sqrt(len(v)) if len(v) > 1 else float("nan")
    return float(v.mean()), float(v.mean() - half), float(v.mean() + half)


def random_reference():
    eps = [e for e in read_csv(os.path.join(REPO_ROOT, "results", "stage0",
                                            "random_episodes.csv")) if e["split"] == "train"]
    succ = sum(e[f"first_{TARGET}"] != "" for e in eps)
    return {"success": (succ / len(eps),) + wilson_ci(succ, len(eps)),
            "return": mean_ci([float(e["return"]) for e in eps])}


def summarize_method(exp, method):
    seed_dirs = sorted(glob.glob(os.path.join(REPO_ROOT, "results", exp, method, "seed*")))
    episodes, curves, budget, sources = [], defaultdict(dict), 0, set()
    for d in seed_dirs:
        path = os.path.join(d, "eval_episodes.csv")
        if not os.path.exists(path):
            continue
        seed = os.path.basename(d)
        eps = read_csv(path)
        ckpts = sorted({int(e["ckpt_env_steps"]) for e in eps})
        harness = os.path.join(d, "eval_harness_episodes.csv")
        if os.path.exists(harness):  # DreamerV3: shared-harness eval of the final checkpoint
            final = read_csv(harness)
            budget = max(budget, int(final[0]["ckpt_env_steps"]))
            episodes += [dict(e, seed=seed) for e in final]
            sources.add("harness, final ckpt")
        else:
            budget = max(budget, ckpts[-1])
            last2 = set(ckpts[-2:])
            episodes += [dict(e, seed=seed) for e in eps if int(e["ckpt_env_steps"]) in last2]
            sources.add("harness, last 2 ckpts" if "dreamer" not in method else "in-loop eval")
        for c in ckpts:
            sel = [e for e in eps if int(e["ckpt_env_steps"]) == c and e["split"] == "train"]
            curves[seed][c] = np.mean([int(e[f"first_{TARGET}"]) >= 0 for e in sel])
    if not episodes:
        return None
    row = {"method": NAMES.get(method, method), "n_seeds": len(curves),
           "budget_env_steps": budget, "final_source": "; ".join(sorted(sources))}
    for split in ("train", "heldout"):
        eps = [e for e in episodes if e["split"] == split]
        k = sum(int(e[f"first_{TARGET}"]) >= 0 for e in eps)
        lo, hi = wilson_ci(k, len(eps))
        row[f"{split}_n"] = len(eps)
        row[f"{split}_success"], row[f"{split}_ci_lo"], row[f"{split}_ci_hi"] = (
            k / len(eps), lo, hi)
        row[f"{split}_return"], row[f"{split}_return_lo"], row[f"{split}_return_hi"] = mean_ci(
            [float(e["return"]) for e in eps])
        if all(f"first_{a}" in eps[0] for a in ACHIEVEMENTS):
            row[f"{split}_crafter_score"] = crafter_score(
                [np.mean([int(e[f"first_{a}"]) >= 0 for e in eps]) for a in ACHIEVEMENTS])
        else:
            row[f"{split}_crafter_score"] = None
    row["gap_pp"] = 100 * (row["train_success"] - row["heldout_success"])
    # Seed-mean learning curve on train seeds (common checkpoints only).
    common = sorted(set.intersection(*[set(c) for c in curves.values()]))
    mean_curve = [np.mean([curves[s][c] for s in curves]) for c in common]
    row["auc_train_success"] = area_under_curve(common, mean_curve) if len(common) > 1 else None
    row["steps_to_half_final"] = steps_to_fraction(common, mean_curve,
                                                   final=np.mean(mean_curve[-2:]))
    return row


def gate(row, ref):
    """PLAN.md §6 week-1 criterion on the train split."""
    r_succ, _, r_hi = ref["success"]
    r_ret, _, r_ret_hi = ref["return"]
    a = row["train_return"] >= r_ret + 1.0 and row["train_return_lo"] > r_ret_hi
    need = max(2 * r_succ, r_succ + 0.10)
    b = row["train_success"] >= need and row["train_ci_lo"] > r_hi
    return a, b, need


def fmt(x, pct=False):
    if x is None:
        return "n/a"
    return f"{100 * x:.1f}%" if pct else f"{x:.2f}"


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--exp", default="main")
    args = parser.parse_args()
    methods = sorted({p.split(os.sep)[-2] for p in glob.glob(
        os.path.join(REPO_ROOT, "results", args.exp, "*", "seed*"))},
        key=lambda m: list(NAMES).index(m) if m in NAMES else 99)
    ref = random_reference()
    rows = [r for r in (summarize_method(args.exp, m) for m in methods) if r]
    for r in rows:
        r["gate_a_return"], r["gate_b_success"], r["gate_b_needed"] = gate(r, ref)

    out = os.path.join(REPO_ROOT, "results", args.exp)
    with open(os.path.join(out, "summary.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    lines = [f"Final = last 2 eval checkpoints x all seeds, pooled. Target = {TARGET}. "
             f"Random policy (train, n=200): success {fmt(ref['success'][0], True)} "
             f"[{fmt(ref['success'][1], True)}, {fmt(ref['success'][2], True)}], "
             f"return {ref['return'][0]:.2f}.", "",
             "| Method | Seeds | Steps | Train success [95% CI] | Held-out success [95% CI] | "
             "Gap (pp) | Train return | Crafter score (train) | AUC | Steps to 50% | "
             "Gate (a) / (b) |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(
            f"| {r['method']} | {r['n_seeds']} | {r['budget_env_steps'] // 1000}k | "
            f"{fmt(r['train_success'], True)} [{fmt(r['train_ci_lo'], True)}, "
            f"{fmt(r['train_ci_hi'], True)}] (n={r['train_n']}) | "
            f"{fmt(r['heldout_success'], True)} [{fmt(r['heldout_ci_lo'], True)}, "
            f"{fmt(r['heldout_ci_hi'], True)}] (n={r['heldout_n']}) | {r['gap_pp']:+.1f} | "
            f"{r['train_return']:.2f} [{r['train_return_lo']:.2f}, {r['train_return_hi']:.2f}] | "
            f"{fmt(r['train_crafter_score'])} | {fmt(r['auc_train_success'], True)} | "
            f"{r['steps_to_half_final'] if r['steps_to_half_final'] is not None else 'n/a'} | "
            f"{'pass' if r['gate_a_return'] else 'fail'} / "
            f"{'pass' if r['gate_b_success'] else 'fail'} |")
    text = "\n".join(lines) + "\n"
    with open(os.path.join(out, "summary.md"), "w") as f:
        f.write(text)
    print(text)


if __name__ == "__main__":
    main()
