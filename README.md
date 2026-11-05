# DQN / PPO / PPO-LSTM / DreamerV3 on Crafter (Apple Silicon)

ECE1508 RL course project. Plan, design decisions, and results: **[PLAN.md](PLAN.md)** (source of
truth). Preliminary results are at the top of PLAN.md; figures are in [figures/](figures/).

## Setup (macOS arm64, Python 3.11)
```bash
brew install python@3.11
git submodule update --init                     # third_party/dreamerv3, pinned
git -C third_party/dreamerv3 apply ../patches/dreamerv3-report-wm-horizon.patch   # RQ3 patch
/opt/homebrew/bin/python3.11 -m venv .venv-sb3
.venv-sb3/bin/pip install -r requirements-sb3.lock
/opt/homebrew/bin/python3.11 -m venv .venv-dreamer
.venv-dreamer/bin/pip install -r requirements-dreamer.lock
```
Always use the explicit interpreter path: on the M2 Max, `python3` is Anaconda.

## Tests
```bash
.venv-sb3/bin/python -m unittest discover -s tests -t .
.venv-dreamer/bin/python -m unittest discover -s tests -t .
```

## Reproduce
Every experiment is one config file in `configs/`. Seeds are given on the command line.

Stage 0: random baseline, env speed, device benchmark.
```bash
.venv-sb3/bin/python scripts/random_baseline.py --config configs/stage0_random.yaml
.venv-sb3/bin/python scripts/bench_device.py --config configs/stage0_bench_device.yaml
```

Model-free methods (training, then shared-harness eval of every checkpoint):
```bash
for s in 0 1 2; do .venv-sb3/bin/python scripts/train_sb3.py --config configs/ppo_cnn_250k.yaml --seed $s; done
```
The same command works for `ppo_lstm_250k.yaml`, `dqn_250k.yaml`, `dqn_nstep3_250k.yaml`, and
`ablation_sparse_ppo_cnn_250k.yaml`. To re-run eval only, use
`.venv-sb3/bin/python scripts/eval.py --config <cfg> --seed <k>`.

DreamerV3 (CPU JAX): training, log conversion, RQ3 world-model eval, then harness eval of the latest checkpoint.
```bash
C=configs/dreamer_size1m_r32_100k.yaml
.venv-dreamer/bin/python scripts/train_dreamer.py --config $C --seed 0
.venv-sb3/bin/python scripts/dreamer_results.py --config $C --seed 0
.venv-dreamer/bin/python scripts/wm_eval.py --config $C --seed 0
.venv-dreamer/bin/python scripts/eval_dreamer.py --config $C --seed 0
```

Tables and figures:
```bash
.venv-sb3/bin/python plots/summarize_results.py --exp main          # results/main/summary.md
.venv-sb3/bin/python plots/plot_learning_curves.py --exp main
.venv-sb3/bin/python plots/plot_learning_curves.py --exp main --splits train,heldout --methods ppo_cnn,ppo_lstm
.venv-sb3/bin/python plots/plot_generalization.py --exp main
.venv-sb3/bin/python plots/plot_wm_error.py --exp main
.venv-sb3/bin/python plots/plot_ablation.py --method ppo_cnn
```

## Layout
- `rlproj/`: shared code; its core imports only numpy, crafter, and gymnasium.
  - `envs/crafter_task.py`: `CrafterTask`, with seed pools, reward modes, a deterministic world cache, and a Crafter set-order fix.
  - `eval/`: the harness and metrics.
  - `adapters/`: SB3 and DreamerV3 policies behind one `reset` / `act` protocol.
  - `utils/`: config, `meta.json`, and CSV schemas.
- `scripts/`: entry points.
- `plots/`: figures and tables.
- `configs/`: one YAML per experiment.
- `results/`: small CSV and JSON outputs (committed).
- `runs/`: checkpoints and logs (gitignored).
- `third_party/`: the pinned DreamerV3 submodule. Every local change is listed in `third_party/PATCHES.md`.
