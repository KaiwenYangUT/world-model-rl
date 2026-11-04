# Contributing

**Read [PLAN.md](PLAN.md) first.** It holds the experimental design, the staged plan, the task
list (with statuses) and the progress log. Pick a `todo` task whose dependencies are `done`, set
it to `in progress`, and put your name in the Owner column.

## Environment
- Python **3.11** for everything (`/opt/homebrew/bin/python3.11` on macOS).
- Two virtual environments, never mixed:
  - `.venv-sb3`: torch, stable-baselines3, sb3-contrib, crafter, gymnasium. Used for PPO,
    RecurrentPPO, DQN, evaluation, and plots.
  - `.venv-dreamer`: CPU `jax==0.4.33`, `numpy<2`, DreamerV3 dependencies, crafter, gymnasium.
    Used for DreamerV3 training, its adapter, and the world-model evaluation.
- Core code in `rlproj/` (environments, evaluation harness, metrics) imports **only** numpy,
  crafter, and gymnasium, so it works in both environments.
- Primary machine: MacBook Pro M2 Max, 32 GB. Secondary: iMac M1, 8 GB (one job at a time).

## Rules
1. **Every experiment = one config file** in `configs/`. Never put hyperparameters only on a
   command line.
2. **Results come only from the shared harness** (`scripts/eval.py` →
   `results/<exp>/<method>/seed<k>/`), using the CSV schemas in PLAN.md §7. Every run writes
   `meta.json` (git hash, config, hardware, device, wall-clock).
3. **Third-party code** lives in `third_party/`, pinned as a submodule. Any local change needs an
   entry in `third_party/PATCHES.md`.
4. **Never commit** checkpoints, raw logdirs, or virtual environments (`runs/` is gitignored).
   Commit only small CSVs, JSON, and figures.
5. **Timing / compute-cost numbers** come from the M2 Max with nothing else running.
6. **Measure, don't assume.** Check library facts in docs or source and cite where you checked.
   Label guesses as guesses.
7. **Simplest thing first.** Use maintained libraries; don't reimplement algorithms.
8. **Git:** keep commits focused, with a message that says what changed and why. Tell the team
   before pushing to `main`.

## End of every work session
Update `PLAN.md`:
- Set the task statuses (`todo` / `in progress` / `done` / `blocked`) and the Owner column.
- Refresh the **Current status** block (stage, date, next action, blockers).
- Add a dated **Progress log** entry: what was done, key measurements (env steps/sec,
  wall-clock per 10k steps, whether learning was observed and the evidence), open issues,
  and the next step.
