# PLAN — DQN / PPO / PPO-LSTM / DreamerV3 on Crafter (Apple Silicon, CPU-first)

> **Source of truth for the project.** Anyone who finishes a work session must:
> (1) update task statuses in §Task list, (2) update **Current status**, (3) add a dated entry to §Progress log.
> Statuses: `todo` · `in progress` · `done` · `blocked`.

## Current status
- **Date:** 2026-11-04
- **Stage:** **Stage 0** (setup and baselines) on the MacBook Pro M2 Max. T00–T03 done (both venvs built; DreamerV3 `debug` runs on CPU; `CrafterTask` + 15 passing tests).
- **Next action:** T04 (random baseline + env sps, idle machine) and T05 (CPU vs MPS). T06 (eval harness) is unblocked too.
- **Blockers:** none. (gymnasium 1.4.0 added to `.venv-dreamer`; the T03 tests pass in both venvs.)
- **Key dates:**
  - Week-1 kill/go gate: **Wed 2026-11-11**.
  - Results frozen: **Sun 2026-12-06**.
  - Presentation: 2nd week of December.
  - Final code and report: Dec 13–18.

---

## 0. Context
Three-person grad RL course project (~4–5 weeks).
- **Proposal:** compare DQN, PPO, PPO-LSTM, and DreamerV3 on a sparse-reward, partially observable visual task under a matched env-step budget.
- **RQ1 (learning efficiency):** success rate and mean return vs. env steps.
- **RQ2 (generalization):** performance drop on unseen map seeds.
- **RQ3 (world-model reliability):** DreamerV3 reward and continuation prediction error vs. open-loop horizon.
- **TA feedback:** watch compute and implementation cost; start with the simplest well-supported setup and get initial results early.

**Decisions (2026-11-04):**
- **Primary machine: MacBook Pro M2 Max, 32 GB.**
  - Secondary: iMac M1, 8 GB, for extra seeds of cheap methods.
  - Teammates' laptops: extra capacity for model-free seeds, plotting, and analysis.
  - No CUDA GPU is assumed.
- **Environment:** Crafter. **Algorithms:** SB3 / sb3-contrib (PPO, RecurrentPPO, DQN) and the official DreamerV3 (JAX, CPU).
- **Rewards:** main experiments use the **standard Crafter reward**. The ablation uses the **target-only sparse reward**.
- **Compute-cost numbers** in the report come **only from the M2 Max**, with timing benchmarks run alone. Runs on other machines are fine for learning curves. Every run logs its hardware.
- A teammate's earlier Craftium plan was unverified. Craftium was checked and rejected (see §2).

---

## 1. Machine check
### iMac M1 (secondary), measured 2026-11-04
| Item | Value | Note |
|---|---|---|
| Model | iMac21,1, Apple M1 | |
| CPU | 8 cores (4 performance + 4 efficiency) | |
| GPU | 8-core, Metal 4 | MPS available |
| RAM | **8 GB** | run one heavy job at a time |
| Disk free | 23 GB (89% used) | prune checkpoints |
| macOS | 26.1 (25B78) | |
| Python | Homebrew 3.14.3 (default), **3.11** at `/opt/homebrew/bin/python3.11`, system 3.9.6 | use **3.11** |
| PyTorch | 2.8.0 only in system py3.9 (MPS True) | py3.9 is too old for SB3; install fresh in the 3.11 venv |
| Tools | cmake, git; **no** ffmpeg, uv, or conda | GIF via imageio is enough for demos |

### MacBook Pro M2 Max (primary), measured 2026-11-04 (T00)
| Item | Value | Note |
|---|---|---|
| Model | MacBook Pro, Mac14,5, Apple M2 Max | arm64 |
| CPU | 12 cores (8 performance + 4 efficiency) | `sysctl hw.perflevel0/1.physicalcpu` |
| GPU | **30-core**, Metal 4 | `system_profiler SPDisplaysDataType` |
| RAM | **32 GB** (34359738368 B) | at check time: 5.6 GB of 7 GB swap in use, 72% memory free |
| Disk free | 533 GB (41% used) | plenty for checkpoints |
| macOS | 26.6.2 (25G83) | |
| Python | **3.11.18** at `/opt/homebrew/bin/python3.11` (`brew install python@3.11`, installed 2026-11-04; arm64, OpenSSL 3.6.3). Also present: `python3` on PATH = Anaconda 3.13.5 (`/opt/anaconda3`); Homebrew 3.14.6; python.org 3.12.2 (`/usr/local/bin/python3`); conda env `usc` 3.12.14; system 3.9.6 | use **3.11**. Always create venvs with the explicit path `/opt/homebrew/bin/python3.11`; Anaconda shadows `python3` |
| PyTorch | **2.15.0 in `.venv-sb3` (MPS True)** (T01). Outside the project: 2.9.1 in Anaconda base, 2.8.0 in system py3.9 | |
| JAX | **0.4.33 CPU in `.venv-dreamer`** (T02), device `TFRT_CPU_0` | |
| Tools | git (`/usr/bin/git`), Homebrew 6.0.15, conda 25.5.1, Xcode CLT (Apple clang 14.0.3); **no** cmake, ffmpeg, or uv | brew has `libomp`, `openblas`, `ollama` (not running) |
| Background load | load avg ≈ 2.3–2.5 at check time; WindowServer ≈ 44% CPU | close apps before timing benchmarks |

Commands used:
```bash
sw_vers; sysctl -n machdep.cpu.brand_string hw.ncpu hw.perflevel0.physicalcpu hw.perflevel1.physicalcpu hw.memsize; df -h ~ | tail -1
ls /opt/homebrew/bin | grep -E '^python3\.[0-9]+$'; which git cmake ffmpeg uv
system_profiler SPHardwareDataType SPDisplaysDataType; uptime; sysctl -n vm.swapusage
python3.11 -c "import torch;print(torch.__version__, torch.backends.mps.is_available())"   # after T01, inside .venv-sb3
```
- With 32 GB, 2–3 concurrent jobs fit in memory, but **timing benchmarks must run alone**.

**CPU vs MPS:**
- The SB3 PPO docs say PPO "is meant to be run primarily on the CPU, especially when you are not using a CNN".
- SB3 documents no MPS support, so we measure both devices (T05). CPU may beat MPS for a small CNN on 64×64 inputs.
- JAX runs CPU-only here.

---

## 2. Feasibility research (sources checked 2026-11-04)
### Environments
| Env | macOS arm64 | Facts checked | Verdict |
|---|---|---|---|
| **Crafter** v1.8.3 (`pip install crafter`) | Pure Python (numpy, pillow, imageio, opensimplex, ruamel.yaml) | 64×64×3 obs, 17 actions, local view (partially observable), 22 achievements. Reward +1 for the first unlock of each achievement per episode, ±0.1 for health changes. `Env(area=(64,64), view=(9,9), size=(64,64), reward=True, length=10000, seed=None)`. World seed per episode = `hash((seed, episode))`. Last commit Dec 2023 (stable, no longer maintained). **No published env FPS.** | **Primary.** The proposal names it as the fallback; built into every DreamerV3 implementation |
| MiniGrid v3.2.0 (2026-10-15) | "Python 3.11+ on Linux and macOS" | Pixels need `RGBImgPartialObsWrapper` → `ImgObsWrapper` (`ImgObsWrapper` alone is not pixels). Sparse reward `1-0.9*t/T`. No DreamerV3 implementation ships a MiniGrid config. rl-zoo MiniGrid hyperparameters use `FlatObsWrapper` + MLP | Fallback only |
| Craftium (`ChopTree-v0` exists) | **Effectively Linux-only.** Only a manylinux x86_64 wheel; a source build compiles Luanti. Issue #32 (open): Docker on an M2 Pro Mac fails | User report: 100–200 FPS capped, ~3k uncapped (Ryzen + 4090) | **Rejected** on Mac |

Crafter paper success rates (arXiv 2109.06780):

| Achievement | Random (no-reward table) | PPO, 1M steps | DreamerV2, 1M steps |
|---|---|---|---|
| collect_wood | 24.4% | 83.0% | 92.7% |
| place_table | 4.4% | 66.1% | 85.7% |
| make_wood_pickaxe | 0.3% | 21.1% | 59.6% |
| collect_stone | 0.0% | 3.0% | 42.7% |

### Algorithms
| Library | Python | Status | Fit |
|---|---|---|---|
| **Stable-Baselines3 2.9.2 (2026-10-26) + sb3-contrib** | ≥3.10, torch ≥2.8 | Active | **PPO (`CnnPolicy`), RecurrentPPO (`CnnLstmPolicy`), DQN (`CnnPolicy`)** — one API, one adapter |
| CleanRL | `>=3.8,<3.11`, torch 2.4.1, gym 0.23.1 | Docs-only commits in 2026 | Conflicts with py3.11; use only as a hyperparameter reference |
| **danijar/dreamerv3** (JAX) | 3.11+ | Active (last commit 2026-10-04). README: "tested on Linux and Mac"; supports `--jax.platform cpu` | **Primary DreamerV3.** Built-in `crafter` config, `size1m` preset, `--run.train_ratio`, open-loop `report()` to extend for RQ3 |
| SheepRL v0.8.5 (2026-10-27) | 3.10–3.13 | Active. README has an M-series Mac install section; default `fabric.accelerator=cpu`; no MPS config seen | **Fallback DreamerV3** (`dreamer_v3_XS`: 256 units, recurrent 256). Open issue #336 reports slow DreamerV3 |
| NM512/dreamerv3-torch | 3.11, torch 2.4.1 | **Archived 2026-07-05** | Rejected |
| NM512/r2dreamer | 3.11 only, torch 2.8 | Active; tested on Ubuntu only | Not chosen |

**Facts verified in the official DreamerV3 source** (`agent.py`, `rssm.py`, `configs.yaml`, `requirements.txt`, `embodied/envs/crafter.py`):
- **Requirements:** pins `jax[cuda12]==0.4.33`, `nvidia-cuda-nvcc-cu12`, `ale_py==0.9.0`, and `numpy<2`. On Mac, install CPU `jax==0.4.33` and skip the CUDA and ALE pins.
- **Configs:**
  - `size1m` = `rssm {deter 512, hidden 64, classes 4}`, depth 4, units 64.
  - Defaults: `batch_size 16`, `batch_length 64`, `run.envs 16`, `run.train_ratio 32`, `replay.size 5e6`.
  - The **crafter block overrides train_ratio to 512**, far too expensive on CPU; we use ≤ 32.
- **Script modes:** `train`, `train_eval`, `eval_only`, `parallel*`.
- **Logging and checkpoints:** logs to `metrics.jsonl` and `scores.jsonl`. `--run.from_checkpoint`; `save_every` is in **seconds** (900).
- **Crafter wrapper:** builds `crafter.Env(size, reward, seed)` and does **not** pass `length` or `area`, so we register our own env through a launcher.
- **Open-loop report:**
  - `report()` observes the first `T//2` steps, then calls `dyn.imagine(carry, secondhalf(prevact), length=T-T//2)` with the recorded actions.
  - `RSSM.imagine(carry, policy, length, training)` accepts an action sequence or a policy function.
- **Heads:** `self.rew(feat, 2).pred()` and `self.con(feat, 2).prob(1)`.
- **Unverified:** no published speed numbers exist for M1 or M2 CPUs. The `MLPHead` semantics in `embodied/jax` have not been read yet.

**Setup:** two venvs, both Python 3.11, so their pins never conflict:
- `.venv-sb3`: torch, sb3, sb3-contrib, crafter, gymnasium.
- `.venv-dreamer`: CPU jax 0.4.33, numpy<2, dreamerv3 deps, crafter.

Our shared harness depends only on numpy, crafter, and gymnasium, so it runs in both.

---

## 3. Experimental design
- **`CrafterTask` wrapper:**
  - Parameters: `target` (default `place_table`; `collect_wood` = sanity check, `make_wood_pickaxe` = stretch), `reward_mode ∈ {crafter, sparse}`, `length` (default 10000; lower only if eval cost requires it), and a seed pool.
  - **Main reward: `crafter`** (standard reward).
  - **Ablation reward: `sparse`**: +1 only on the first unlock of the target, no health reward, episode ends on success.
  - Always log success on collect_wood, place_table, and make_wood_pickaxe. **place_table is the headline metric.**
- **Metrics:**
  - **Success:** target unlocked in the episode.
  - **Steps-to-success:** in-episode step of the first unlock (successful episodes only).
  - **Sample efficiency:** env steps until eval success first reaches 50% of the final value, plus area under the learning curve.
  - The Crafter score (geometric mean over 22 achievements) is reported as secondary context.
- **Generalization:**
  - Train pool `N_train` = 50 world seeds (0–49). Held-out pool = 100 seeds (10000–10099).
  - The wrapper forces the world seed on reset, which touches Crafter internals `_seed` and `_episode`, so it needs a **determinism test**. (Done in T03. The test also exposed a Crafter set-order nondeterminism, now fixed in the wrapper; see `third_party/PATCHES.md`.)
  - Gap = success(train) − success(held-out).
- **Evaluation (single source of truth):**
  - Every method checkpoints every `eval_every` env steps (10k at a 100k budget).
  - `scripts/eval.py` loads each checkpoint through an adapter and runs the shared harness: 20 episodes on train seeds and 20 on held-out seeds, with a fixed eval RNG.
  - Eval policies: PPO and LSTM sample; DQN is ε-greedy with ε=0.05; DreamerV3 samples.
- **Compute logging:** each run writes `meta.json` with git hash, config, hardware, device, total wall-clock, and wall-clock per 10k steps.

---

## 4. Staged plan with gates
### Stage 0 — setup and baselines (Nov 4–6, M2 Max)
- **Deliverables:**
  - T00 machine check.
  - Both venvs with `requirements-*.txt` and `pip freeze` locks.
  - `CrafterTask` with a determinism test.
  - `random_baseline.py` → `results/stage0/random.csv`: success per target over 200 episodes, episode-length distribution, raw env steps/sec (1 process and 4 subprocesses).
  - `bench_device.py`: PPO-CNN steps/sec on cpu vs mps over 20k steps.
  - DreamerV3: the `debug` config, then `size1m` crafter for 2k steps, on `--jax.platform cpu`.
- **Verify:**
  - The same seed gives an identical first frame and an identical 100-step trajectory under fixed actions.
  - Random success is roughly consistent with the paper.
  - Dreamer prints `fps` and `train/loss/*`.
- **Gate:** choose the target whose random success is non-zero and ≤ ~10% (expected `place_table`). If `place_table` is ≥ 20%, use `make_wood_pickaxe`.
- **If slow or broken:**
  - JAX install fails → SheepRL XS.
  - Env < 300 sps → use 4 subprocess envs and fewer eval episodes.

### Stage 1 — PPO-CNN end-to-end + eval pipeline (Nov 6–9) → FIRST RESULT
- **Deliverables:**
  - `train_sb3.py --config configs/ppo_cnn_100k.yaml` (SB3 PPO, `CnnPolicy`, 8 envs, Atari-style hyperparameters).
  - `eval.py`; CSVs in `results/ppo_cnn/seed0/`.
  - `plot_learning_curves.py`: success vs env steps (train and held-out) with a random-baseline line.
  - Budget: 100k first; 250k if cheap.
- **Verify:** curves rise above random (§6); re-running eval on a checkpoint reproduces the same numbers; the CSV schema matches §7.
- **If slow:** 50k budget, 4 envs, fewer eval episodes, `collect_wood` to prove the pipeline.

### Stage 2 — DreamerV3 smallest model, one full run (Nov 6–10, in parallel with Stage 1)
- **Deliverables:**
  - `train_dreamer.py`: a launcher that registers `CrafterTask` and runs `--configs crafter size1m --run.train_ratio 32 --run.envs 4 --replay.size <budget> --jax.platform cpu`.
  - One 50k–100k-step run with a learning curve.
  - A throughput report: env steps/h, gradient updates/h, seconds per update, XLA compile time, peak RSS.
  - The Dreamer harness adapter.
- **Verify:**
  - `loss/rew`, `loss/con`, and image loss decrease.
  - `metrics.jsonl` and `scores.jsonl` are written.
  - A checkpoint plays 20 harness episodes.
- **Then:** fill §5 "Measured" and fix the Stage 4 budget B (100k or 250k).
- **If slow:** train_ratio 16, 50k budget, `run.envs 2`. If seconds per update > 3 → compare SheepRL XS.
- **Checkpointing:** `save_every` is time-based; find a step-indexed option, or read the step from the checkpoint.

### WEEK-1 GATE — Wed Nov 11 (see §6)

### Stage 3 — PPO-LSTM + DQN on the same harness (Nov 11–16)
- **Deliverables:**
  - `RecurrentPPO` (`CnnLstmPolicy`) and `DQN` (`CnnPolicy`, `buffer_size` ≤ 100k, `optimize_memory_usage=True`).
  - The memory settings matter: a 250k buffer storing both obs and next_obs is about 6 GB.
- **Verify:** each method gets a 50k smoke curve; a unit test confirms the LSTM adapter carries hidden state across steps and resets it on episode start.
- **If slow:** smaller LSTM budget, or n_envs=4. If DQN doesn't learn after 2–3 tries, report that as a finding.

### Stage 4 — final experiments (Nov 16–Dec 4; **results frozen Dec 6**)
- **Main runs:**
  - 4 methods × 3 seeds × B, with generalization eval on every checkpoint.
  - DreamerV3 runs on the M2 Max; model-free seeds can go to the iMac or teammates' laptops.
- **RQ3:** world-model error vs horizon (§8).
- **Ablation:** reward sparsity (`crafter` vs `sparse`) for PPO and DreamerV3 × 3 seeds. Alternative: DreamerV3 train_ratio {16, 32, 64}, only if the team prefers it and it fits.
- **Demos:** GIFs of the best seed per method on held-out seeds, plus an imagined-vs-real strip for Dreamer.
- **Run order:** PPO → Dreamer → DQN → LSTM per seed, so partial results stay balanced across methods.
- **Cut order if behind:**
  1. B=100k.
  2. LSTM at 2 seeds.
  3. Ablation on PPO only.
  4. DQN at 2 seeds.
  5. Never cut PPO, DreamerV3, or the held-out eval.
- **Dec 7–10:** presentation figures and slides.
- **Dec 11–16:** fixes from presentation feedback (minor reruns or plots), final README and report.

---

## 5. Compute budget (M2 Max, one job at a time) — **ALL GUESSES until "Measured" is filled**
- The guesses were reasoned for an M1. The M2 Max is likely 1.2–1.5× faster per core (also a guess), so treat these as upper bounds.
- **Dreamer wall-clock:** `B/sps_env + (B·train_ratio/1024)·t_update + compile`, where 1024 = batch 16 × length 64. At ratio 32, that is B/32 updates.

| Method | Guessed throughput | 50k | 100k | 250k | Measured |
|---|---|---|---|---|---|
| PPO-CNN (SB3, CPU) | 150–600 sps | 2–6 min | 3–11 min | 7–28 min | TBD (Stage 1) |
| PPO-LSTM | 0.3–0.5× PPO | 5–20 min | 10–40 min | 25–90 min | TBD (Stage 3) |
| DQN (train_freq 4) | 100–300 sps | 3–8 min | 6–17 min | 14–42 min | TBD (Stage 3) |
| DreamerV3 size1m, ratio 32 | t_update 0.3–1.5 s | 0.2–0.8 h | 0.3–1.5 h | 0.8–3.5 h | TBD (Stage 2) |
| DreamerV3 size1m, ratio 64 | 2× updates | 0.4–1.5 h | 0.7–3 h | 1.6–7 h | TBD |

**Totals (guess, including ~20% eval overhead):**
- B=100k: ≈ 6–10 machine-hours for the main runs (4 methods × 3 seeds) + 3–5 h for the ablation.
- B=250k: ≈ 15–25 h + 8–12 h.
- Either fits in Stage 4 overnight runs on the M2 Max.

**Memory:**
- Dreamer replay at 250k × 12 KB ≈ 3 GB, which is fine on 32 GB.
- On the 8 GB iMac, run one job at a time.

---

## 6. Week-1 kill criterion (Wed Nov 11)
Continue only if **both** of the following hold. Otherwise consider switching to the recommendation-system project.

**PPO is "learning".** Conditions: standard reward, train-seed pool, a run of ≤ 250k steps, measured over the last 2 eval points × 20 episodes.
- (a) Mean eval return is ≥ random + 1.0, with non-overlapping 95% CIs; **and**
- (b) Success on place_table (minimum: collect_wood) is ≥ max(2× random, random + 10 pp), with the Wilson CI lower bound above random's upper bound;
- and the curve trends upward.
- If only collect_wood passes, continue, but flag the risk.

**DreamerV3 is "completing".**
- A `size1m` run with train_ratio ≥ 16 and ≥ 50k env steps finishes with no crash, NaN, or OOM in ≤ 8 h.
- Final `loss/rew` and image loss are below their first-1k-step values.
- The checkpoint plays ≥ 20 harness episodes.
- The extrapolated Stage 4 main-run cost (B ≥ 50k × 3 seeds) is ≤ ~40 machine-hours.

**Rescue ladder (only if time remains in week 1):**
- PPO → `collect_wood`.
- DreamerV3 → SheepRL XS.
- Still failing → switch projects.

---

## 7. Repo structure
```
ECE1508_RL_Project/
├── PLAN.md  README.md  CONTRIBUTING.md
├── requirements-sb3.txt  requirements-dreamer.txt  (+ *.lock from pip freeze)
├── rlproj/                       # OUR code; core has no torch/jax imports
│   ├── envs/crafter_task.py      # CrafterTask, seed pools, make_env(cfg, split, seed)
│   ├── eval/harness.py           # evaluate(adapter, env_cfg, split, n_episodes, rng_seed) -> rows
│   ├── eval/metrics.py           # success, Wilson CI, steps-to-success, AUC
│   ├── adapters/sb3.py  adapters/dreamer.py   # Adapter protocol: reset(n) / act(obs, first) -> action
│   └── utils/{config.py, csvlog.py, meta.py}
├── scripts/  random_baseline.py bench_device.py train_sb3.py train_dreamer.py eval.py
│             collect_heldout.py wm_eval.py make_demo.py
├── configs/  stage0_random.yaml ppo_cnn_100k.yaml ppo_lstm_100k.yaml dqn_100k.yaml
│             dreamer_size1m_r32_100k.yaml ablation_*.yaml   # one file per experiment
├── third_party/dreamerv3/        # git submodule pinned to a commit; PATCHES.md lists every local change
├── results/<exp>/<method>/seed<k>/{eval.csv, train_episodes.csv, meta.json}   # small, committed
├── runs/                         # gitignored: checkpoints, raw logdirs
├── plots/  plot_learning_curves.py plot_generalization.py plot_wm_error.py plot_compute.py
├── figures/
└── tests/                        # seed determinism, harness, LSTM state reset
```
- **`eval.csv` columns:** `exp, method, seed, ckpt_env_steps, wall_clock_s, split, n_episodes, success_rate, success_ci_lo, success_ci_hi, mean_return, mean_len, mean_steps_to_success, target, reward_mode`
- **`wm_error.csv` columns:** `exp, seed, k, n, rew_mae, rew_mae_event, rew_event_recall, rew_event_precision, cont_brier, cont_bce, term_recall, cum_rew_err`

---

## 8. World-model reliability metric (RQ3)
**Held-out data.**
- `collect_heldout.py` runs the final DreamerV3 policy on the 100 held-out seeds. The PPO policy is an optional second distribution.
- It saves npz files (`image, action, reward, is_first, is_terminal`). None of this data goes into replay.

**Procedure, for each start index t0:**
1. Encode and run the posterior over context `[t0−C+1 … t0]` with `dyn.observe(carry, tokens, prevact, is_first)`, with C=16. This gives the posterior state at t0.
2. Imagine open-loop with `dyn.imagine(carry, actions[t0+1 … t0+H], length=H)` using the **recorded** actions. H=15 (= `imag_length`); go to 32 if stable.
3. Compute r̂_k = `rew(feat,2).pred()` and ĉ_k = `con(feat,2).prob(1)`. Compare them with the real r and with c = 1 − is_terminal.
4. Stop each rollout at the episode end.

**Metrics for each k:**
- Reward MAE, overall and on **event steps** (r≠0; the sparse reward makes overall MAE look small).
- Event precision and recall at |r̂| > 0.5.
- Continuation Brier score and BCE, plus terminal recall.
- Cumulative reward error.
- k=0 (posterior) as the reference point.
- Report mean ± 95% CI over episodes and seeds.
- Optional: plot errors for checkpoints at 25%, 50%, and 100% of training.

**Hook.**
- Add `wm_eval(self, data, context, horizon)` next to `report()` in the vendored `dreamerv3/agent.py`. It reuses `self.enc`, `self.dyn.observe/imagine`, `self.rew`, and `self.con`, and is jitted the same way as `report`.
- Record the change in `third_party/PATCHES.md`.
- `scripts/wm_eval.py` loads the checkpoint (`--run.from_checkpoint`).
- **Verify first:** `MLPHead` `.pred()/.prob()` semantics in `embodied/jax`.

---

## 9. Task list (work breakdown — no assignments)
★ = cleanly separable: a teammate can pick it up without blocking anyone. T00–T12 is the critical path to the first results. Put your name in **Owner** when you start a task.

| ID | Task | Produces | Depends | Effort | Parallel | Status | Owner |
|---|---|---|---|---|---|---|---|
| T00 | Machine check on M2 Max (§1) + log entry | hardware row | – | 15 min | yes | done | Kevin |
| T01 | SB3 venv, py3.11 (torch, sb3, sb3-contrib, crafter, gymnasium) | requirements-sb3 + lock | – | 1–2 h | yes | done | Kevin |
| T02 | Dreamer venv (CPU jax 0.4.33, numpy<2, deps minus CUDA/ALE) + pinned submodule | requirements-dreamer + lock, debug run log | – | 2–4 h | yes, with T01 | done | Kevin |
| T03 | `CrafterTask` wrapper + seed pools + determinism test | `rlproj/envs`, tests | T01 | 3–5 h | – | done | Kevin |
| T04 | Random baseline + env sps benchmark | `results/stage0/random.csv`, target choice | T03 | 2 h | – | todo | |
| T05 | CPU vs MPS PPO benchmark | device decision | T01 | 1 h | yes | todo | |
| T06 | Eval harness + adapter protocol + CSV schema + Wilson CI | `rlproj/eval`, tests | T03 | 4–6 h | – | todo | |
| T07 | `train_sb3.py` + PPO config + checkpointing | PPO run | T06 | 3–4 h | – | todo | |
| T08 | PPO 100k run + eval → **first result** | `results/ppo_cnn`, plot | T07, T09 | 0.5 day | – | todo | |
| T09 | ★ Plotting scripts against the §7 schema (start with fake CSVs) | `plots/*.py` | schema only | 4–6 h | **yes** | todo | |
| T10 | Dreamer launcher registering `CrafterTask`; size1m 2k-step smoke | `train_dreamer.py` | T02, T03 | 4–6 h | yes, with T06–T08 | todo | |
| T11 | Dreamer 50–100k run + throughput report | `results/dreamer`, §5 measured | T10 | 0.5 day + overnight | – | todo | |
| T12 | Dreamer checkpoint adapter for harness | `adapters/dreamer.py` | T06, T10 | 4–6 h | – | todo | |
| T13 | ★ PPO-LSTM config + adapter state handling + test | LSTM run | T06, T07 | 0.5 day | **yes** | todo | |
| T14 | ★ DQN config (memory-safe) + 50k smoke | DQN run | T06, T07 | 0.5 day | **yes** | todo | |
| T15 | ★ Held-out trajectory collector | `collect_heldout.py` + npz | T12 | 3–4 h | **yes** | todo | |
| T16 | ★ `wm_eval` patch + script + plot | `wm_error.csv`, figure | T10 (any checkpoint), T15 | 1 day | **yes** | todo | |
| T17 | ★ Demo GIF tool (incl. Dreamer imagined-vs-real) | `make_demo.py` | T06 | 4 h | **yes** | todo | |
| T18 | ★ TA update note (§10) | email text | T04 | 1 h | **yes** | todo | |
| T19 | ★ README with exact reproduction commands (kept current) | `README.md` | ongoing | ongoing | **yes** | todo | |
| T20 | Stage 4 main runs (4 methods × 3 seeds) | `results/main` | T08, T11–T14, gate | 1–2 weeks of nights | split by machine | todo | |
| T21 | Ablation runs (reward sparsity) | `results/ablation` | T20 setup | compute | split by machine | todo | |
| T22 | ★ Report skeleton + related work | report draft | – | ongoing | **yes** | todo | |
| T23 | (Optional) CUDA path if a GPU appears | timing comparison | T11 | 2–4 h | yes | todo | |
| T24 | ★ Multi-machine run sheet + results merge procedure | run sheet in this file | T07 | 1–2 h | **yes** | todo | |
| T25 | Presentation figures and slides (from frozen results) | slides | T20, T21, T16 | 1–2 days | partly | todo | |

**Suggested split:**
- **Critical path:** T00–T08 and T10–T11 on the M2 Max.
- **Separable from day 1:** T09, T18 (after T04), T19, T22.
- **Separable after T06/T07:** T13, T14, T17, T24.
- **Separable after a Dreamer checkpoint exists:** T15, T16.

---

## 10. Gaps vs. proposal — what to tell the TA (T18)
1. **Environment:** Crafter (2D top-down Minecraft-like), not 3D. Craftium is Linux-only and fails in Docker on Apple Silicon (craftium issue #32). The proposal pre-authorized this fallback.
2. **Scale:**
   - DreamerV3 `size1m` (≈1M params, vs up to 200M+ in the paper).
   - train_ratio ≤ 32 (vs Crafter's default 512).
   - Budgets 100k–250k (vs the 1M benchmark).
   - So this is a small-compute study, not comparable to published scores.
3. **Compute:** Apple Silicon only (M2 Max 32 GB primary), DreamerV3 on CPU via JAX. The cost table comes from the M2 Max; hardware is logged for every run.
4. **Generalization:** finite train-seed pool vs held-out seeds.
5. **Metric:** training on the standard Crafter reward. Headline = place_table success (also collect_wood and make_wood_pickaxe) and mean return. Crafter score is secondary.
6. **Ablation:** reward sparsity (standard vs target-only) for PPO and DreamerV3.
7. **Possible downgrades:** fewer LSTM or DQN seeds (cut order in §4).

---

## Progress log
Newest entry first. Template:
```
### YYYY-MM-DD — <name> — <machine>
- Completed: ...
- Measurements: env sps=..., wall-clock/10k steps=..., learning observed: yes/no (evidence)
- Issues / blockers: ...
- Next: ...
```

### 2026-11-04 — T03 — MacBook Pro M2 Max
- **Completed:** T03.
  - `rlproj/envs/crafter_task.py`: `CrafterTask` (gymnasium env), `TRAIN_SEEDS` (0–49), `HELDOUT_SEEDS` (10000–10099), and `make_env(cfg, split, seed)`.
  - `target` defaults to `place_table`. `reward_mode` is `crafter` or `sparse` (sparse terminates on the target's first unlock).
  - Each reset draws a world seed from the pool using gymnasium's `np_random`; `options={"world_seed": s}` forces one.
  - `info` adds `world_seed`, `episode_step`, `success`, `first_unlock` (collect_wood, place_table, make_wood_pickaxe, plus the target), and `crafter_reward`.
  - Death → `terminated`; the length limit → `truncated`.
  - Tests: `tests/test_crafter_task.py`, 15 stdlib-`unittest` tests (no pytest). Run with `.venv-sb3/bin/python -m unittest discover -s tests -t .`. **All pass, 37 s.**
  - What the tests cover:
    - same seed → identical first frame and 100-step trajectory;
    - a 500-step trajectory with other envs alive;
    - episode history doesn't matter;
    - identical across processes with PYTHONHASHSEED=0 and 12345;
    - 20 seeds → 20 distinct maps;
    - pool sampling is reproducible and stays in the pool; train and held-out pools are disjoint;
    - reward modes (unlock faked by patching Crafter's step), truncation, gymnasium `check_env`.
  - SB3 `PPO("CnnPolicy")` trains 256 steps on `make_vec_env(make_env)` without errors. SB3 transposes the obs to (3,64,64) itself.
- **Found and fixed (key result):**
  - **Crafter 1.8.3 is not deterministic given seed + actions.** `Env._balance_object` picks the creature to despawn by indexing a list built from a per-chunk `set` (`crafter/engine.py` `World._chunks`), and set order follows memory addresses.
  - Two envs with the same seed diverged within the same process at steps 50, 125, and 143 (seeds 0, 3, 7). Cross-process runs happened to match.
  - Fix: `_DeterministicCrafterEnv` sorts the chunk's objects by position before Crafter's unchanged random pick, so the distribution is the same. After the fix: 4 seeds × full episodes, no divergence. Recorded in `third_party/PATCHES.md` (no third-party files edited).
  - `info["player_pos"]` aliased Crafter's live array across steps (`check_env` caught it). It is now copied.
- **Measurements (rough; machine not idle):**
  - `reset()` takes **~0.52 s** (Python world generation).
  - Stepping: ~260 sps, 1 process, random actions.
  - Random-action episodes ended by death after ~150–260 steps in the 4 seeds checked. So for a random policy, reset is roughly a third of wall-clock. T04 should report sps both with and without resets.
  - Learning: n/a.
- **Follow-up (same day):**
  - Installed gymnasium 1.4.0 (+ farama-notifications 0.0.6) into `.venv-dreamer`; numpy stays 1.26.4. Updated `requirements-dreamer.txt` and the lock.
  - The 15 tests pass in `.venv-dreamer` (39 s).
  - The same 500-step trajectory fingerprints (seeds 0, 7, 10000) match exactly across `.venv-sb3` (numpy 2.4.6) and `.venv-dreamer` (numpy 1.26.4), so training in one venv and evaluating in the other sees the same worlds.
- **Issues / open items:**
  - Reset cost may matter for eval: 40 episodes × 0.5 s = 20 s per checkpoint, which is small next to the stepping cost.
- **Next:** T04 (random baseline + sps on an idle machine), T05, T06.

### 2026-11-04 — T01/T02 — MacBook Pro M2 Max
- **Completed:**
  - **T01** `.venv-sb3` from `/opt/homebrew/bin/python3.11`: torch 2.15.0, stable-baselines3 2.9.2, sb3-contrib 2.9.2, crafter 1.8.3, gymnasium 1.4.0, numpy 2.4.6. Files: `requirements-sb3.txt` (top-level pins) and `requirements-sb3.lock` (`pip freeze`, 20 packages).
  - **T02** `third_party/dreamerv3` submodule pinned at `e01491f` (2026-10-03). `.venv-dreamer` from python3.11: jax/jaxlib 0.4.33 (CPU), numpy 1.26.4, optax 0.2.5, chex 0.1.90, ninjax 3.6.3, elements 3.22.4, portal 3.8.3, granular 0.24.2, scope 0.8.1, crafter 1.8.3. Files: `requirements-dreamer.txt`, `requirements-dreamer.lock` (94 packages), and `third_party/PATCHES.md` (no patches yet).
  - Changes from upstream `requirements.txt`: `jax[cuda12]` → CPU `jax`+`jaxlib==0.4.33`. Dropped `nvidia-cuda-nvcc-cu12`, `ale_py`, `autorom`, `ipdb`. Added `crafter` and `cloudpickle`.
- **Smoke tests:**
  - `.venv-sb3`: all imports load. `torch.backends.mps.is_available()` = True and a conv runs on MPS. `crafter.Env(seed=0)` gives a (64,64,3) uint8 obs and 17 actions. PPO (1024 steps) and RecurrentPPO (256 steps) train on CartPole on CPU without errors.
  - `.venv-dreamer`: `jax.devices()` = `[CpuDevice(id=0)]`.
  - DreamerV3 debug run: `python dreamerv3/main.py --configs crafter debug --jax.platform cpu --run.steps 2000 --logdir runs/stage0/dreamer_debug`. Exit 0, no warnings, wrote `metrics.jsonl`, `scores.jsonl`, and a checkpoint. Console log copied to `logs/t02_dreamer_debug_crafter_cpu.log`.
- **Measurements (rough; machine not idle, so not for the cost table):**
  - Crafter raw env, 1 process, random actions, 2000 steps: **~300 sps**, measured while the Dreamer pip install was running. Proper benchmark is T04.
  - Dreamer debug run (11,735-param debug net, 4 envs, train_ratio 8): **62 s wall-clock** for ~2k env steps, including XLA compile. Peak RSS 2.4 GB. `fps/policy` 45–57, `fps/train` 330–455.
  - Learning: **not expected and not observed.** The debug net is a smoke test only. `loss/image` went 1684 → 1588; `loss/rew` (1.61) and `loss/con` (0.90) stayed flat.
- **Issues / open items:**
  - Crafter 1.8.3 uses the old gym API: `reset()` returns obs only and `step()` returns a 4-tuple. T03's `CrafterTask` must adapt it to gymnasium for SB3.
  - Random-action Crafter at ~300 sps is right at the Stage 0 "< 300 sps → 4 subprocess envs" line. Re-measure on an idle machine in T04.
  - `.venv-sb3` has no matplotlib or pandas yet. T09 (plots) needs them; add them to `requirements-sb3.txt` then.
  - `git submodule add` staged `.gitmodules` and `third_party/dreamerv3`. Nothing is committed.
- **Next:** T03 (`CrafterTask` + determinism test). T05 and T10 can run alongside it.

### 2026-11-04 — T00 — MacBook Pro M2 Max
- **Completed:** T00 machine check; filled the M2 Max table in §1.
- **Measurements:**
  - Hardware: Mac14,5, Apple M2 Max, 12 CPU cores (8P + 4E), **30-core GPU** (Metal 4), 32 GB RAM, 533 GB disk free, macOS 26.6.2 (25G83).
  - Raw output: `sw_vers` → 26.6.2 / 25G83; `sysctl` → `Apple M2 Max`, 12, 8, 4, 34359738368; `df -h ~` → 926Gi size, 364Gi used, 533Gi avail, 41%.
  - `/opt/homebrew/bin` has only `python3.14`. `which` → git `/usr/bin/git`; cmake, ffmpeg, uv not found. `python3.11` → command not found.
  - No env sps or training numbers yet.
- **Issues / open items:**
  - **No Python 3.11 on this machine at first.** Found: Anaconda base 3.13.5 (first on PATH as `python3`, with torch 2.9.1, MPS True), Homebrew 3.14.6, python.org 3.12.2, conda env `usc` 3.12.14, system 3.9.6. **Resolved:** ran `brew install python@3.11` → 3.11.18 at `/opt/homebrew/bin/python3.11` (matches CONTRIBUTING.md and the iMac). Default `python3` is unchanged (still Anaconda).
  - Anaconda shadows `python3`, so venvs must be created with an explicit interpreter path, never bare `python3`.
  - Background load at check time: load avg ≈ 2.3–2.5, 5.6 GB swap in use. Close other apps before T04/T05 timing runs (CONTRIBUTING.md rule 5).
- **Next:** T01 and T02 in parallel, both created from `/opt/homebrew/bin/python3.11`.

### 2026-11-04 — planning — iMac M1
- **Completed:**
  - Machine check of the iMac (§1).
  - Feasibility research from READMEs, source, and issues (§2).
  - Staged plan, `git init`, `.gitignore`, PLAN.md, CONTRIBUTING.md.
- **Measurements:** none yet. All §5 numbers are guesses.
- **Issues / open items:**
  - The M2 Max has not been inspected (T00).
  - Official DreamerV3 pins `jax[cuda12]`, so the CPU-jax install on macOS is unverified.
  - Crafter seed forcing touches private attributes (needs the determinism test).
  - No published Apple Silicon DreamerV3 speeds exist.
- **Sources:**
  - github.com/danijar/crafter, github.com/danijar/dreamerv3 (agent.py, rssm.py, configs.yaml, requirements.txt, embodied/envs/crafter.py)
  - github.com/Eclectic-Sheep/sheeprl (incl. issue #336), github.com/NM512/dreamerv3-torch, github.com/NM512/r2dreamer
  - github.com/mikelma/craftium (incl. issues #19, #32), minigrid.farama.org
  - github.com/vwxyzjn/cleanrl, stable-baselines3.readthedocs.io, sb3-contrib.readthedocs.io, github.com/DLR-RM/rl-baselines3-zoo
  - arXiv 2109.06780 (Crafter)
- **Next:** move the repo to the M2 Max and start Stage 0 (T00 → T01/T02 → T03).
