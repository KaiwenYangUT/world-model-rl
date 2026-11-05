"""RQ2 figure: final place_table success on train vs held-out worlds per method, 95% Wilson CIs.

    .venv-sb3/bin/python plots/plot_generalization.py --exp main

Reads results/<exp>/summary.csv (plots/summarize_results.py) and results/stage0/random.csv;
writes figures/<exp>_generalization.png. Final = last 2 eval checkpoints x all seeds, pooled.
"""

import argparse
import csv
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INK, INK_2, GRID, SURFACE, RANDOM = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb", "#8a8984"
SPLITS = {"train": ("train worlds (seeds 0-49)", "#2a78d6", "o"),
          "heldout": ("held-out worlds (10000-10099)", "#eb6834", "s")}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--exp", default="main")
    args = parser.parse_args()
    with open(os.path.join(REPO_ROOT, "results", args.exp, "summary.csv"), newline="") as f:
        rows = list(csv.DictReader(f))
    with open(os.path.join(REPO_ROOT, "results", "stage0", "random.csv"), newline="") as f:
        rand = [r for r in csv.DictReader(f) if r["target"] == "place_table"]
    rand_rate = sum(float(r["success_rate"]) * int(r["n_episodes"]) for r in rand) / sum(
        int(r["n_episodes"]) for r in rand)

    fig, ax = plt.subplots(figsize=(8, 0.9 * len(rows) + 1.6), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    for i, r in enumerate(rows):
        y = len(rows) - 1 - i
        for j, (split, (label, color, marker)) in enumerate(SPLITS.items()):
            p, lo, hi = (float(r[f"{split}_{k}"]) for k in ("success", "ci_lo", "ci_hi"))
            yy = y + (0.12 if split == "train" else -0.12)
            ax.plot([lo, hi], [yy, yy], color=color, linewidth=2, solid_capstyle="round")
            ax.plot(p, yy, marker=marker, color=color, markersize=8, linestyle="none",
                    markeredgecolor=SURFACE, markeredgewidth=2, label=label if i == 0 else None)
        ax.text(1.005, y, f"gap {float(r['gap_pp']):+.1f} pp", transform=ax.get_yaxis_transform(),
                va="center", fontsize=8.5, color=INK_2)
    ax.axvline(rand_rate, color=RANDOM, linestyle=":", linewidth=1.5,
               label=f"random policy ({rand_rate:.1%})")
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([f"{r['method']} ({r['n_seeds']} seed{'s' * (int(r['n_seeds']) > 1)}, "
                        f"{int(r['budget_env_steps']) // 1000}k)" for r in reversed(rows)],
                       color=INK, fontsize=9)
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax.set_xlabel("place_table success, final (last 2 eval checkpoints x seeds), 95% Wilson CI",
                  color=INK_2, fontsize=9)
    ax.grid(True, axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=8, left=False)
    ax.legend(fontsize=8, frameon=False, labelcolor=INK_2, loc="lower right")
    ax.set_title(f"Generalization to unseen worlds ({args.exp})", loc="left", fontsize=11,
                 color=INK)
    fig.tight_layout()
    out = os.path.join(REPO_ROOT, "figures", f"{args.exp}_generalization.png")
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    print(f"wrote {os.path.relpath(out, REPO_ROOT)}")


if __name__ == "__main__":
    main()
