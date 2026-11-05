"""Learning curves from the shared-harness CSVs (PLAN.md §7): success vs env steps per method,
with the random-policy baseline as a reference line. --splits train (default; RQ1) or
train,heldout (held-out dashed; RQ2 detail for few methods).

    .venv-sb3/bin/python plots/plot_learning_curves.py --exp main [--splits train,heldout]

Reads results/<exp>/<method>/seed*/eval.csv and results/stage0/random.csv; writes
figures/<exp>_learning_curves.png. Lines are the mean over seeds; the band spans the
min..max over seeds (shown only with >= 2 seeds).
"""

import argparse
import csv
import glob
import os
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Fixed method -> categorical slot order (dataviz reference palette, light mode).
METHODS = {"ppo_cnn": ("PPO", "#2a78d6"), "ppo_lstm": ("PPO-LSTM", "#eb6834"),
           "dqn": ("DQN", "#1baf7a"), "dreamerv3": ("DreamerV3", "#eda100"),
           "dqn_nstep3": ("DQN 3-step", "#e87ba4"),
           "dreamerv3_r64": ("DreamerV3 ratio 64", "#008300")}
INK, INK_2, GRID, SURFACE, RANDOM = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb", "#8a8984"
PANELS = [("place_table", "place_table success (target)"),
          ("collect_wood", "collect_wood success"),
          ("make_wood_pickaxe", "make_wood_pickaxe success")]


def read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def load_eval(exp):
    """{method: {(target, split): {seed: [(steps, success, mean_return)]}}}"""
    data = defaultdict(lambda: defaultdict(dict))
    for path in sorted(glob.glob(os.path.join(REPO_ROOT, "results", exp, "*", "seed*",
                                              "eval.csv"))):
        method = path.split(os.sep)[-3]
        for r in read_csv(path):
            key = (r["target"], r["split"])
            data[method][key].setdefault(int(r["seed"]), []).append(
                (int(r["ckpt_env_steps"]), float(r["success_rate"]), float(r["mean_return"])))
    return data


def random_baseline():
    """Pooled train + held-out random success per achievement, and mean return."""
    rows = read_csv(os.path.join(REPO_ROOT, "results", "stage0", "random.csv"))
    succ, ret = {}, []
    for target in {r["target"] for r in rows}:
        rs = [r for r in rows if r["target"] == target]
        n = sum(int(r["n_episodes"]) for r in rs)
        succ[target] = sum(float(r["success_rate"]) * int(r["n_episodes"]) for r in rs) / n
    for split in {r["split"] for r in rows}:
        ret.append(float(next(r for r in rows if r["split"] == split)["mean_return"]))
    return succ, float(np.mean(ret))


def curve(per_seed, column):
    steps = sorted({s for runs in per_seed.values() for s, *_ in runs})
    table = np.full((len(per_seed), len(steps)), np.nan)
    for i, runs in enumerate(per_seed.values()):
        for s, succ, ret in runs:
            table[i, steps.index(s)] = (succ, ret)[column]
    return np.array(steps), table


def style(ax, title, ylabel):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", fontsize=11, color=INK)
    ax.set_xlabel("env steps", color=INK_2, fontsize=9)
    ax.set_ylabel(ylabel, color=INK_2, fontsize=9)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=8)
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(
        lambda x, _: f"{x / 1000:.0f}k" if x else "0"))


def draw(ax, steps, table, color, linestyle, label):
    mean = np.nanmean(table, axis=0)
    ax.plot(steps, mean, color=color, linestyle=linestyle, linewidth=2, label=label)
    if table.shape[0] >= 2:
        ax.fill_between(steps, np.nanmin(table, 0), np.nanmax(table, 0), color=color,
                        alpha=0.12 if linestyle == "-" else 0.07, linewidth=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--exp", default="main")
    parser.add_argument("--splits", default="train")
    args = parser.parse_args()
    splits = args.splits.split(",")
    data = load_eval(args.exp)
    if not data:
        raise SystemExit(f"no results/{args.exp}/*/seed*/eval.csv found")
    rand_succ, rand_ret = random_baseline()
    methods = [m for m in METHODS if m in data] + [m for m in data if m not in METHODS]

    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), facecolor=SURFACE)
    panels = PANELS + [("return", "mean episode return (standard Crafter reward)")]
    for ax, (target, title) in zip(axes.flat, panels):
        is_return = target == "return"
        style(ax, title, "return" if is_return else "success rate")
        for method in methods:
            name, color = METHODS.get(method, (method, INK_2))
            key_target = "place_table" if is_return else target
            for split, ls in (("train", "-"), ("heldout", "--")):
                if split not in splits:
                    continue
                per_seed = data[method].get((key_target, split))
                if not per_seed:
                    continue
                steps, table = curve(per_seed, 1 if is_return else 0)
                n = len(per_seed)
                tag = "" if len(splits) == 1 else (" train" if split == "train" else " held-out")
                draw(ax, steps, table, color, ls, f"{name}{tag} (n={n} seed{'s' * (n > 1)})")
        ref = rand_ret if is_return else rand_succ.get(target, 0.0)
        ax.axhline(ref, color=RANDOM, linestyle=":", linewidth=1.5,
                   label=f"random policy ({ref:.2f})" if is_return else
                   f"random policy ({ref:.1%})")
        if not is_return:
            ax.set_ylim(0, max(0.05, ax.get_ylim()[1]))
            ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
        ax.legend(fontsize=7.5, frameon=False, labelcolor=INK_2, loc="upper left")
    worlds = " + ".join({"train": "train worlds", "heldout": "held-out worlds (dashed)"}[s]
                        for s in splits)
    fig.suptitle(f"Crafter ({args.exp}) — eval on {worlds}, 50 episodes per checkpoint; line = "
                 "mean over seeds, band = min..max",
                 x=0.01, ha="left", fontsize=10, color=INK_2)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    os.makedirs(os.path.join(REPO_ROOT, "figures"), exist_ok=True)
    suffix = "" if splits == ["train"] else "_" + "_".join(splits)
    out = os.path.join(REPO_ROOT, "figures", f"{args.exp}_learning_curves{suffix}.png")
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    print(f"wrote {os.path.relpath(out, REPO_ROOT)}")


if __name__ == "__main__":
    main()
