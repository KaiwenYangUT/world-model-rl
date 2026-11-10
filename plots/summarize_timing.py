"""Timing summary for the idle-machine timing runs (configs/timing_*.yaml, CONTRIBUTING.md rule 5).

    .venv-sb3/bin/python plots/summarize_timing.py

SB3 runs: per-10k-step training wall-clock from the checkpoint .json files in
runs/timing/<method>/seed0/ckpt/ (training time only: no eval, no world-cache warm-up). "Steady"
is the median over the 10k segments after DQN's learning_starts (10k), i.e. segments 2-5 for every
method. DreamerV3: median fps/train and fps/policy over the steady log lines (after the prefill)
in runs/timing/<method>/seed0/logdir/metrics.jsonl. Writes results/timing/summary.{csv,md}.
"""

import csv
import glob
import json
import os
import re
import statistics

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(REPO_ROOT, "results", "timing")
SB3 = ["ppo_cnn", "ppo_lstm", "dqn", "dqn_nstep3"]
DREAMER = ["dreamerv3_size12m_r128"]
DREAMER_SIZE1M = {"fps_train": 764.0, "fps_policy": 5.97}  # sparse r128 seed 1 metrics.jsonl


def sb3_row(method):
    files = glob.glob(os.path.join(REPO_ROOT, "runs", "timing", method, "seed0", "ckpt", "step_*.json"))
    files.sort(key=lambda p: int(re.search(r"step_(\d+)\.json$", p).group(1)))
    pts = [(json.load(open(f))["env_steps"], json.load(open(f))["wall_clock_s"]) for f in files]
    segs = [(b[1] - a[1]) / (b[0] - a[0]) * 1e4 for a, b in zip(pts, pts[1:])]
    steady = statistics.median(segs[1:])
    meta = json.load(open(os.path.join(REPO_ROOT, "results", "timing", method, "seed0", "meta.json")))
    return {"method": method, "device": meta["device"], "steps": meta["env_steps"],
            "s_per_10k": round(steady, 1), "steps_per_s": round(1e4 / steady),
            "projected_1m_h": round(steady * 100 / 3600, 2),
            "projected_100k_h": round(steady * 10 / 3600, 3), "peak_mem_gb": "",
            "segments_s": " ".join(f"{s:.1f}" for s in segs)}


def dreamer_row(method):
    lines = [json.loads(l) for l in open(os.path.join(
        REPO_ROOT, "runs", "timing", method, "seed0", "logdir", "metrics.jsonl"))]
    fps = [(d["fps/train"], d["fps/policy"]) for d in lines
           if d.get("fps/train", 0) > 0 and d.get("fps/policy", 99) < 5]  # after prefill
    train = statistics.median(f for f, _ in fps)
    policy = statistics.median(p for _, p in fps)
    rss = None
    log = os.path.join(REPO_ROOT, "runs", "logs", "timing_dreamer_size12m.log")
    for line in open(log):
        if "maximum resident set size" in line:
            rss = int(line.split()[0]) / 1e9
    return {"method": method, "device": "cpu (jax)", "steps": lines[-1].get("step", ""),
            "s_per_10k": round(1e4 / policy, 1), "steps_per_s": round(policy, 2),
            "projected_1m_h": round(1e6 / policy / 3600, 1),
            "projected_100k_h": round(1e5 / policy / 3600, 1),
            "peak_mem_gb": round(rss, 1) if rss else "",
            "segments_s": f"fps/train {train:.0f} (size1m: {DREAMER_SIZE1M['fps_train']:.0f})"}


def main():
    rows = [sb3_row(m) for m in SB3] + [dreamer_row(m) for m in DREAMER]
    with open(os.path.join(OUT, "summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    head = ["Method", "Device", "Steps run", "Training s / 10k steps", "Steps / s",
            "Projected h / 100k", "Projected h / 1M", "Peak memory (GB)", "Detail"]
    md = ["Idle M2 Max, only job, on AC (runs/logs/timing_load.log). One run each; training "
          "time only (no eval). SB3: steady per-10k segments; DreamerV3: steady fps after "
          "prefill. 1M projections are extrapolations.", "",
          "| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for r in rows:
        md.append("| " + " | ".join(str(r[k]) for k in (
            "method", "device", "steps", "s_per_10k", "steps_per_s", "projected_100k_h",
            "projected_1m_h", "peak_mem_gb", "segments_s")) + " |")
    with open(os.path.join(OUT, "summary.md"), "w") as f:
        f.write("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
