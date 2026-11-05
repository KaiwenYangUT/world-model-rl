"""RQ3: DreamerV3 world-model reward / continuation error vs open-loop horizon (PLAN.md §8).

    .venv-dreamer/bin/python scripts/wm_eval.py --config configs/dreamer_size1m_r32_100k.yaml --seed 0

Loads the latest checkpoint of runs/<exp>/dreamerv3/seed<k>/logdir (agent weights) and the replay
chunks saved there, then runs the patched `Agent.report` (third_party/PATCHES.md) on `--batches`
report batches from each buffer. Each batch is report_length = 32 consecutive steps: the first
16 are observed (posterior, k = 0 reference), the last 16 are imagined open-loop with the
recorded actions (k = 1..16); steps after a new episode starts are masked.

PRELIMINARY: buffers instead of the T15 held-out trajectory files. `replay` = training episodes
(train pool, training policy); `eval_replay` = in-loop eval episodes, half on train-pool and half
on held-out worlds (the buffer does not record which). Writes results/<exp>/dreamerv3/seed<k>/
wm_error_{replay,eval_replay}.csv in the PLAN.md §7 wm_error schema, plus wm_eval_meta.json.
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from functools import partial

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DREAMER_ROOT = os.path.join(REPO_ROOT, "third_party", "dreamerv3")
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, DREAMER_ROOT)
os.environ["PYTHONPATH"] = os.pathsep.join([REPO_ROOT, DREAMER_ROOT])

import numpy as np  # noqa: E402

from rlproj.utils.config import load_config  # noqa: E402
from rlproj.utils.csvlog import write_rows  # noqa: E402
from rlproj.utils.meta import RunMeta  # noqa: E402

sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))
from train_dreamer import flags_to_argv, make_crafter_env  # noqa: E402


def build_config(cfg, seed, logdir):
    """Same parsing as upstream dreamerv3/main.py main()."""
    import elements
    import ruamel.yaml as yaml
    from dreamerv3 import main as dreamer_main
    configs = yaml.YAML(typ="safe").load(
        elements.Path(os.path.join(DREAMER_ROOT, "dreamerv3", "configs.yaml")).read())
    argv = flags_to_argv(cfg["dreamer_flags"]) + ["--logdir", logdir, "--seed", str(seed)]
    parsed, other = elements.Flags(configs=["defaults"]).parse_known(argv)
    config = elements.Config(configs["defaults"])
    for name in parsed.configs:
        config = config.update(configs[name])
    return dreamer_main, elements.Flags(config).parse(other)


def finalize(sums, exp, seed):
    rows = []
    ks = sorted({int(key.split("/")[1][1:]) for key in sums})
    div = lambda a, b: (a / b) if b > 0 else None
    for k in ks:
        g = lambda name: sums.get(f"wm/k{k:02d}/{name}", 0.0)
        tp, fn, fp = g("unlock_tp"), g("unlock_fn"), g("unlock_fp")
        rows.append({
            "exp": exp, "seed": seed, "k": k, "n": int(g("n")),
            "rew_mae": div(g("rew_abs"), g("n")),
            "rew_mae_event": div(g("rew_abs_event"), g("n_event")),
            "rew_event_recall": div(tp, tp + fn), "rew_event_precision": div(tp, tp + fp),
            "cont_brier": div(g("cont_brier"), g("n")), "cont_bce": div(g("cont_bce"), g("n")),
            "term_recall": div(g("term_hit"), g("n_term")),
            "cum_rew_err": div(g("cum_rew_abs"), g("n")) if k > 0 else None,
        })
    for r in rows:
        for key, value in r.items():
            if isinstance(value, float):
                r[key] = round(value, 6)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--batches", type=int, default=64)
    args = parser.parse_args()
    cfg = load_config(args.config)
    rel = os.path.join(cfg["exp"], cfg["method"], f"seed{args.seed}")
    logdir = os.path.join(REPO_ROOT, "runs", rel, "logdir")
    out_dir = os.path.join(REPO_ROOT, "results", rel)
    meta = RunMeta(cfg, config_path=args.config, device="cpu (jax)")
    meta.data["seed"] = args.seed

    dreamer_main, config = build_config(cfg, args.seed, logdir)
    # Same env (and so the same spaces) as training; see scripts/train_dreamer.py.
    dreamer_main.make_env = partial(make_crafter_env, cfg["env"], None, "train")
    import elements
    agent = dreamer_main.make_agent(config)
    ckpt_dir = elements.Path(logdir) / "ckpt"
    latest = ckpt_dir / (ckpt_dir / "latest").read_text().strip()
    elements.checkpoint.load(latest, dict(agent=agent.load))

    sources = {}
    for folder, mode in (("replay", "train"), ("eval_replay", "eval")):
        replay = dreamer_main.make_replay(config, folder, mode)
        replay.load()
        if len(replay) == 0:
            print(f"[wm_eval] {folder}: empty, skipped")
            continue
        stream = iter(agent.stream(dreamer_main.make_stream(config, replay, "report")))
        carry = agent.init_report(config.batch_size)
        sums = defaultdict(float)
        for _ in range(args.batches):
            carry, mets = agent.report(carry, next(stream))
            for key, value in mets.items():
                if key.startswith("wm/"):
                    sums[key] += float(np.asarray(value))
        rows = finalize(sums, cfg["exp"], args.seed)
        write_rows(os.path.join(out_dir, f"wm_error_{folder}.csv"), rows, "wm_error")
        sources[folder] = {"replay_items": len(replay), "batches": args.batches}
        print(f"[wm_eval] {folder}: " + ", ".join(
            f"k={r['k']} mae={r['rew_mae']} brier={r['cont_brier']}" for r in rows[:: 4]),
            flush=True)
    meta.finish(os.path.join(out_dir, "wm_eval_meta.json"), checkpoint=str(latest),
                checkpoint_name=os.path.basename(str(latest)), sources=sources,
                note="preliminary: replay buffers, not T15 held-out trajectories")


if __name__ == "__main__":
    main()
