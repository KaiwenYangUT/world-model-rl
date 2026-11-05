"""RQ3 figure: DreamerV3 world-model error vs open-loop horizon k (PLAN.md §8).

    .venv-sb3/bin/python plots/plot_wm_error.py --exp main

Reads results/<exp>/<method>/seed*/wm_error_{replay,eval_replay}.csv (scripts/wm_eval.py) and
writes figures/<exp>_wm_error[_<method>].png (--method, default dreamerv3). k = 0 is the posterior (observed) reference; k >= 1 is
open-loop imagination with the recorded actions. Mean over seeds when there are several.
"""

import argparse
import csv
import glob
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INK, INK_2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
# Two buffers: categorical slots 1 and 2 (dataviz reference palette, light mode).
SOURCES = {"replay": ("training episodes (train worlds)", "#2a78d6", "-"),
           "eval_replay": ("eval episodes (half held-out worlds)", "#eb6834", "--")}
PANELS = [("rew_mae_event", "reward MAE on reward steps (r ≠ 0)", "MAE"),
          ("rew_event_recall", "unlock recall (|r| > 0.5 caught by |r̂| > 0.5)", "recall"),
          ("cont_brier", "continuation Brier score", "Brier"),
          ("cum_rew_err", "|cumulative reward error| per step", "abs error")]


def load(exp, method, source):
    tables = []
    for path in sorted(glob.glob(os.path.join(REPO_ROOT, "results", exp, method, "seed*",
                                              f"wm_error_{source}.csv"))):
        with open(path, newline="") as f:
            tables.append(list(csv.DictReader(f)))
    return tables


def series(tables, column):
    ks = sorted({int(r["k"]) for t in tables for r in t})
    vals = np.full((len(tables), len(ks)), np.nan)
    for i, t in enumerate(tables):
        for r in t:
            if r[column] != "":
                vals[i, ks.index(int(r["k"]))] = float(r[column])
    with np.errstate(all="ignore"):
        mean = np.array([np.nan if np.isnan(c).all() else np.nanmean(c) for c in vals.T])
    return np.array(ks), mean


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--exp", default="main")
    parser.add_argument("--method", default="dreamerv3")
    args = parser.parse_args()
    data = {s: load(args.exp, args.method, s) for s in SOURCES}
    if not any(data.values()):
        raise SystemExit("no wm_error_*.csv found; run scripts/wm_eval.py first")
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), facecolor=SURFACE)
    for ax, (column, title, ylabel) in zip(axes.flat, PANELS):
        ax.set_facecolor(SURFACE)
        for source, (label, color, ls) in SOURCES.items():
            if not data[source]:
                continue
            ks, ys = series(data[source], column)
            ax.plot(ks, ys, color=color, linestyle=ls, linewidth=2, marker="o", markersize=4,
                    label=f"{label}, n={len(data[source])} seed(s)")
        ax.axvline(0.5, color=GRID, linewidth=1)
        ax.text(0.6, 0.97, "open loop →", transform=ax.get_xaxis_transform(), fontsize=8,
                color=INK_2, va="top")
        ax.set_title(title, loc="left", fontsize=11, color=INK)
        ax.set_xlabel("horizon k (0 = posterior)", color=INK_2, fontsize=9)
        ax.set_ylabel(ylabel, color=INK_2, fontsize=9)
        ax.grid(True, color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(GRID)
        ax.tick_params(colors=INK_2, labelsize=8)
        if column == "rew_event_recall":
            ax.set_ylim(-0.02, 1.02)
        ax.legend(fontsize=7.5, frameon=False, labelcolor=INK_2, loc="best")
    fig.suptitle(f"{args.method} (size1m) world model vs open-loop horizon ({args.exp}); context 16 "
                 "observed steps, then 16 imagined with recorded actions; latest checkpoint",
                 x=0.01, ha="left", fontsize=10, color=INK_2)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    tag = "" if args.method == "dreamerv3" else f"_{args.method}"
    out = os.path.join(REPO_ROOT, "figures", f"{args.exp}_wm_error{tag}.png")
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    print(f"wrote {os.path.relpath(out, REPO_ROOT)}")


if __name__ == "__main__":
    main()
