"""RQ3 follow-up figure: DreamerV3's critic value before a table is placed (scripts/value_check.py).

    .venv-sb3/bin/python plots/plot_value_check.py

Reads results/ablation_sparse/dreamerv3_r128/value_check.csv; writes
figures/ablation_sparse_value_check_dreamerv3_r128.png. Seeds are pooled (weighted by the number
of steps). Values differ ~3x between seeds, so each seed's value is first divided by its own
baseline at the same checkpoint, then the ratios are pooled (weighted by n). Left: value k steps
before the action that placed the table, relative to steps of episodes without a table (1 = no
difference). Right: value by wood in the inventory, relative to holding no wood. At 50k seed 0's
critic outputs ~0.0005 everywhere, so its ratios (1 table episode) are noise; drawn thin.
"""

import csv
import os
import sys
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot_learning_curves import INK_2, REPO_ROOT, SURFACE, style  # noqa: E402

SRC = os.path.join(REPO_ROOT, "results", "ablation_sparse", "dreamerv3_r128", "value_check.csv")
OUT = os.path.join(REPO_ROOT, "figures", "ablation_sparse_value_check_dreamerv3_r128.png")
# One hue, light -> dark with training progress (checkpoints are ordered).
COLORS = {50000: "#c9c8c3", 90000: "#f3a47c", 100000: "#c2410c"}


def pooled():
    """(ckpt, group, key) -> (n, n-weighted mean over seeds of value / seed baseline). Baseline:
    the seed's no-table episodes for before_table, its wood = 0 steps for wood."""
    with open(SRC, newline="") as f:
        rows = [r for r in csv.DictReader(f) if r["mean_value"]]
    base = {}
    for r in rows:
        if (r["group"], r["key"]) in (("no_table", ""), ("wood", "0")):
            base[(r["seed"], r["ckpt_env_steps"], r["group"])] = float(r["mean_value"])
    acc = defaultdict(lambda: [0, 0.0])
    for r in rows:
        ref = {"before_table": "no_table", "wood": "wood"}.get(r["group"])
        if ref is None:
            continue
        a = acc[(int(r["ckpt_env_steps"]), r["group"], r["key"])]
        a[0] += int(r["n"])
        a[1] += int(r["n"]) * float(r["mean_value"]) / base[(r["seed"], r["ckpt_env_steps"], ref)]
    return {k: (n, s / n) for k, (n, s) in acc.items() if n}


def main():
    data = pooled()
    ckpts = sorted({c for c, _, _ in data})
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.2), facecolor=SURFACE,
                                   gridspec_kw={"width_ratios": [1.6, 1]})
    style(ax1, "Critic value before placing a table (3 seeds)",
          "value / value in episodes without a table")
    ax1.set_xlabel("steps before the table is placed", color=INK_2, fontsize=9)
    ax1.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, _: f"{abs(x):.0f}"))
    for c in ckpts:
        pts = sorted((int(k), v) for (cc, g, k), (n, v) in data.items()
                     if cc == c and g == "before_table")
        n_tables = data.get((c, "before_table", "0"), (0, 0))[0]
        if pts:
            ax1.plot([-k for k, _ in pts], [v for _, v in pts], color=COLORS.get(c, INK_2),
                     linewidth=1 if n_tables < 10 else 2,
                     label=f"{c // 1000}k ckpt ({n_tables} tables)")
    ax1.axhline(1.0, color=INK_2, linestyle=":", linewidth=1.5)
    ax1.plot([], [], color=INK_2, linestyle=":", label="no difference")
    ax1.legend(frameon=False, fontsize=8, loc="upper left")

    style(ax2, "Critic value by wood held", "value / value with no wood")
    ax2.set_xlabel("wood in inventory", color=INK_2, fontsize=9)
    ax2.xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
    keys = ["0", "1", "2", "3", "4+"]
    width = 0.8 / max(1, len(ckpts))
    for i, c in enumerate(ckpts):
        xs = [j + (i - (len(ckpts) - 1) / 2) * width for j in range(len(keys))]
        ys = [data.get((c, "wood", k), (0, float("nan")))[1] for k in keys]
        ax2.bar(xs, ys, width=width, color=COLORS.get(c, INK_2), label=f"{c // 1000}k")
    ax2.axhline(1.0, color=INK_2, linestyle=":", linewidth=1.5)
    ax2.set_xticks(range(len(keys)), keys)
    ax2.legend(frameon=False, fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(OUT, dpi=150, facecolor=SURFACE)
    print(f"wrote {os.path.relpath(OUT, REPO_ROOT)}")


if __name__ == "__main__":
    main()
