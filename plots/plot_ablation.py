"""Reward-sparsity ablation figure: place_table success vs env steps for one method under the
standard Crafter reward (exp main) and the target-only sparse reward (exp ablation_sparse).

    .venv-sb3/bin/python plots/plot_ablation.py --method ppo_cnn

Reads results/{main,ablation_sparse}/<method>/seed*/eval.csv; writes
figures/ablation_sparse_<method>.png. Solid = train worlds, dashed = held-out worlds;
line = mean over seeds, band = min..max.
"""

import argparse
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot_learning_curves import (  # noqa: E402
    METHODS,
    REPO_ROOT,
    SURFACE,
    curve,
    draw,
    load_eval,
    random_baseline,
    style,
)

# Categorical slots 1 and 2 (dataviz reference palette, light mode).
EXPS = {"main": ("standard reward", "#2a78d6"), "ablation_sparse": ("sparse reward", "#eb6834")}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--method", default="ppo_cnn")
    args = parser.parse_args()
    name = METHODS.get(args.method, (args.method,))[0]
    fig, ax = plt.subplots(figsize=(7.5, 4.5), facecolor=SURFACE)
    style(ax, f"{name}: place_table success, standard vs target-only sparse reward",
          "success rate")
    for exp, (label, color) in EXPS.items():
        per_method = load_eval(exp).get(args.method, {})
        for split, ls in (("train", "-"), ("heldout", "--")):
            per_seed = per_method.get(("place_table", split))
            if not per_seed:
                continue
            steps, table = curve(per_seed, 0)
            n = len(per_seed)
            draw(ax, steps, table, color, ls,
                 f"{label}, {'train' if split == 'train' else 'held-out'} worlds (n={n})")
    rand = random_baseline()[0]["place_table"]
    ax.axhline(rand, color="#8a8984", linestyle=":", linewidth=1.5,
               label=f"random policy ({rand:.1%})")
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax.legend(fontsize=7.5, frameon=False, labelcolor="#52514e", loc="upper left")
    fig.tight_layout()
    out = os.path.join(REPO_ROOT, "figures", f"ablation_sparse_{args.method}.png")
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    print(f"wrote {os.path.relpath(out, REPO_ROOT)}")


if __name__ == "__main__":
    main()
