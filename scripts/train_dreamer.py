"""DreamerV3 (official, third_party/dreamerv3) on CrafterTask (PLAN.md §4 Stage 2, T10).

    .venv-dreamer/bin/python scripts/train_dreamer.py --config configs/dreamer_size1m_r32_100k.yaml --seed 0

Runs upstream `dreamerv3/main.py` unmodified, with two substitutions made from here:
  * `embodied.run.train_eval` is wrapped so the train driver gets EmbodiedCrafterTask on the
    train pool, and the eval driver gets env 0 on the train pool and env 1 on the held-out
    pool (upstream passes the same make_env to both, so an env cannot tell them apart).
  * every env appends its finished episodes to runs/<exp>/dreamerv3/seed<k>/episodes/*.jsonl.
  * if cfg["ckpt_every_steps"] is set, the agent is also saved every that many env steps (and at
    step 0 and the end) to runs/<exp>/<method>/seed<k>/ckpt_steps/step_<n>/ (agent only, plus
    info.json with env_steps and wall_clock_s), for scripts/eval_dreamer.py. Upstream's own
    checkpoint (logdir/ckpt) is saved on wall time and keeps only the latest.
Upstream flags come from cfg["dreamer_flags"]; logdir and seed are set here. Afterwards,
scripts/dreamer_results.py turns the logs into results/ CSVs.
"""

import argparse
import json
import os
import sys
import time
from functools import partial

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DREAMER_ROOT = os.path.join(REPO_ROOT, "third_party", "dreamerv3")
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, DREAMER_ROOT)
# Env worker processes import rlproj by name, so they need the repo on their path too.
os.environ["PYTHONPATH"] = os.pathsep.join(
    [REPO_ROOT, DREAMER_ROOT] + [p for p in os.environ.get("PYTHONPATH", "").split(os.pathsep) if p])

from rlproj.utils.config import load_config  # noqa: E402
from rlproj.utils.meta import RunMeta  # noqa: E402

EVAL_SPLITS = ("train", "heldout")  # eval env index -> split


def make_crafter_env(env_cfg, log_dir, role, config, index, **overrides):
    """make_env replacement with upstream's signature (config, index) after binding."""
    from dreamerv3 import main as dreamer_main

    from rlproj.envs.embodied_crafter import EmbodiedCrafterTask

    split = "train" if role == "train" else EVAL_SPLITS[index % len(EVAL_SPLITS)]
    seed = hash((int(config.seed), 0 if role == "train" else 1, int(index))) % (2 ** 31 - 1)
    log = os.path.join(log_dir, f"{role}{index}_{split}.jsonl") if log_dir else None
    env = EmbodiedCrafterTask(env_cfg, split, seed=seed, episode_log=log)
    return dreamer_main.wrap_env(env, config)


def save_step_checkpoint(agent, root, env_steps, t0):
    """Write the agent's parameters to <root>/step_<env_steps>/ (same format as upstream)."""
    import elements
    path = os.path.join(root, f"step_{env_steps}")
    if os.path.exists(path):
        return
    elements.checkpoint.save(path, dict(agent=agent.save))
    with open(os.path.join(path, "info.json"), "w") as f:
        json.dump({"env_steps": env_steps, "wall_clock_s": round(time.time() - t0, 1)}, f)
    print(f"[train_dreamer] step checkpoint {path}", flush=True)


def with_step_checkpoints(make_agent, make_logger, every, root, t0):
    """Wrap upstream's make_agent / make_logger so the agent saves itself every `every` env
    steps. The env-step counter is the logger's step (upstream: `step = logger.step`, which the
    train driver increments once per env step); the check runs before each train-mode policy
    call, i.e. after the training updates for the steps so far."""
    state = {"next": 0, "last": None}

    def make_logger_():
        state["logger"] = make_logger()
        return state["logger"]

    def make_agent_():
        agent = make_agent()
        state["agent"] = agent
        policy = agent.policy

        def policy_(carry, obs, mode="train"):
            if mode == "train" and "logger" in state:
                steps = int(state["logger"].step)
                if steps >= state["next"]:
                    save_step_checkpoint(agent, root, steps, t0)
                    state["last"] = steps
                    state["next"] = (steps // every + 1) * every
            return policy(carry, obs, mode=mode)

        agent.policy = policy_
        return agent

    def finish():
        # Final save, unless a grid checkpoint was saved within half an interval of the end
        # (the train driver runs 10 steps per call, so a run of 100k steps ends at ~100,008
        # right after the step_100000 save; a near-duplicate would count twice in "last 2").
        if "agent" in state and "logger" in state:
            steps = int(state["logger"].step)
            if state["last"] is None or steps - state["last"] >= every // 2:
                save_step_checkpoint(state["agent"], root, steps, t0)

    return make_agent_, make_logger_, finish


def flags_to_argv(flags):
    argv = []
    for key, value in flags.items():
        if isinstance(value, (list, tuple)):
            argv += [f"--{key}"] + [str(v) for v in value]
        else:
            argv += [f"--{key}", str(value)]
    return argv


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    rel = os.path.join(cfg["exp"], cfg["method"], f"seed{args.seed}")
    run_dir = os.path.join(REPO_ROOT, "runs", rel)
    results_dir = os.path.join(REPO_ROOT, "results", rel)
    log_dir = os.path.join(run_dir, "episodes")
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    meta = RunMeta(cfg, config_path=args.config, device="cpu (jax)")
    meta.data["seed"] = args.seed
    t0 = time.time()

    import embodied
    from dreamerv3 import main as dreamer_main

    original = embodied.run.train_eval

    def train_eval(make_agent, make_replay_train, make_replay_eval, make_env_train,
                   make_env_eval, make_stream, make_logger, run_args):
        config = make_env_train.args[0]  # partial(make_env, config)
        finish = None
        if cfg.get("ckpt_every_steps"):
            make_agent, make_logger, finish = with_step_checkpoints(
                make_agent, make_logger, int(cfg["ckpt_every_steps"]),
                os.path.join(run_dir, "ckpt_steps"), t0)
        result = original(make_agent, make_replay_train, make_replay_eval,
                          partial(make_crafter_env, cfg["env"], log_dir, "train", config),
                          partial(make_crafter_env, cfg["env"], log_dir, "eval", config),
                          make_stream, make_logger, run_args)
        if finish:
            finish()
        return result

    embodied.run.train_eval = train_eval
    argv = flags_to_argv(cfg["dreamer_flags"]) + [
        "--logdir", os.path.join(run_dir, "logdir"), "--seed", str(args.seed),
        "--script", "train_eval"]
    print("dreamerv3 argv:", " ".join(argv), flush=True)
    dreamer_main.main(argv)
    import jax
    meta.finish(os.path.join(results_dir, "meta.json"), argv=argv, jax_version=jax.__version__)


if __name__ == "__main__":
    main()
