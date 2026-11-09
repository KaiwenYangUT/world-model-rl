# PLAN — DQN / PPO / PPO-LSTM / DreamerV3 on Crafter (Apple Silicon, CPU-first)

> **Source of truth for the project.** Anyone who finishes a work session must:
> (1) update task statuses in §Task list, (2) update **Current status**, (3) add a dated entry to §Progress log.
> Statuses: `todo` · `in progress` · `done` · `blocked`.

## Current status
- **Date:** 2026-11-09
- **Stage:** Stages 0–3 done in preliminary form on the M2 Max. All four methods ran through the pipeline, and **preliminary results exist for RQ1, RQ2, and RQ3** (see *Preliminary results* below). The week-1 gate (§6) passes for PPO and DreamerV3. Repo is on GitHub (`KaiwenYangUT/world-model-rl`).
- **Next action:**
  1. Read the preliminary results and decide the Stage 4 budget and settings.
  2. DreamerV3: ratio 128 fixed the world model's unlock prediction and the policy now learns, but it learns the easy achievements, not `place_table` (see results). Next candidates: the sparse-reward ablation at ratio 128 (rewards only the target), then a 2nd/3rd seed of ratio 128.
  3. T15 + T16 proper: held-out trajectories for RQ3.
  4. ~~Step-indexed DreamerV3 checkpoints~~ done 2026-11-09 (`ckpt_every_steps` in the config; first used by the sparse ablation run).
  5. Re-time PPO/DQN/PPO-LSTM on an idle machine.
- **Running:** DreamerV3 sparse-reward ablation at ratio 128, seed 0 (`configs/ablation_sparse_dreamer_size1m_r128_100k.yaml`), started 13:20 on 2026-11-09, ETA ~18:00 plus ~25 min for the harness eval of 11 step checkpoints. Log: `runs/logs/ablation_sparse_dreamer_r128_seed0.log`. Don't run other jobs on the M2 Max until it finishes (timing, rule 5).
- **Blockers:** none. **All timings tonight are provisional** (several jobs shared the machine; load avg up to ~100), per CONTRIBUTING.md rule 5.
- **Key dates:**
  - Week-1 kill/go gate: **Wed 2026-11-11**.
  - Results frozen: **Sun 2026-12-06**.
  - Presentation: 2nd week of December.
  - Final code and report: Dec 13–18.

---

## Preliminary results (2026-11-04, overnight; all provisional)
Setting: Crafter, target `place_table`, standard reward unless stated, 250k env steps (DreamerV3 100k).
- **Eval:** shared harness, 50 episodes per split per checkpoint, every 10k steps, fixed worlds and eval RNG.
- **"Final":** the last 2 checkpoints × all seeds, pooled; 95% Wilson CI.
- **Random policy:** 2.0% train / 4.5% held-out (n=200 each).
- **Sources:** `results/main/summary.md` and `figures/main_*.png`.

| Method | Seeds | Final success, train worlds | Final success, held-out worlds | Gap (pp) | Return (train) | §6 gate |
|---|---|---|---|---|---|---|
| PPO (CnnPolicy) | 3 | **62.0%** [56.4, 67.3] | **52.0%** [46.4, 57.6] | +10.0 | 4.42 | pass |
| PPO-LSTM | 3 | 45.0% [39.5, 50.7] | 40.3% [34.9, 46.0] | +4.7 | 3.75 | pass |
| DQN | 3 | 16.3% [12.6, 20.9] | 17.7% [13.8, 22.4] | −1.3 | 2.42 | pass (barely; seed-dependent) |
| DQN 3-step (try 2) | 3 | 47.0% [41.4, 52.7] | 37.0% [31.7, 42.6] | +10.0 | 3.88 | pass |
| DreamerV3 size1m, ratio 32, 100k | 3 | 4.7% [2.3, 9.3] (harness, latest ckpts ≈ 96k / 70k / 79k, n=150) | 2.7% [1.0, 6.7] | +2.0 | 1.37 | "completing": pass; learning: no |
| DreamerV3 size1m, **ratio 64**, 100k | 1 | 2.0% (harness, latest ckpt ≈ 85k, n=50) | 6.0% | – | 0.56 | "completing": pass; learning: no |
| DreamerV3 size1m, **ratio 128**, 100k | 1 | 0.0% [0.0, 7.1] (harness, latest ckpt ≈ 97k, n=50) | 0.0% [0.0, 7.1] | – | 1.84 | "completing": pass; learning: yes, but not the target |
| PPO, **sparse reward** (ablation) | 3 | **96.0%** [93.1, 97.7] | 82.3% [77.6, 86.2] | +13.7 | – | – |

- **RQ1 (learning efficiency):**
  - PPO learns fastest and best: `place_table` goes 4% → 62% in 250k. The Crafter paper reports 66% for PPO at 1M.
  - PPO-LSTM is slower and lower (45%).
  - DQN is slow and seed-dependent: one seed stays at random level, one reaches ~50% after 150k.
  - 3-step returns plus longer exploration (DQN try 2) raise DQN to 47% (3 seeds), close to PPO-LSTM.
  - DreamerV3 at size1m does not learn a policy in 100k steps at train ratio 32 (3 seeds) or 64 (1 seed). At ratio 128 (1 seed) it learns the easy achievements but not `place_table` (below).
- **RQ2 (generalization):**
  - Held-out success is lower than train for PPO (+10 pp gap, CIs barely overlap), DQN 3-step (+10, CIs barely overlap), PPO-LSTM (+4.7), and sparse-reward PPO (+13.7, CIs disjoint).
  - The gap opens after ~100k steps (`figures/main_learning_curves_train_heldout.png`).
  - DQN shows no gap at its low success level.
- **RQ3 (world-model reliability, preliminary):**
  - DreamerV3's world model clearly learns: image loss 1566 → 117, reward loss 5.5 → 0.26, continuation loss 0.72 → 0.05.
  - Measured on the ~96k checkpoint over replay buffers (`figures/main_wm_error.png`), overall reward MAE is small (~0.03) and flat over k = 0..16.
  - **But the reward head never predicts an achievement unlock** (|r̂| > 0.5): recall is 0 at every horizon, including the posterior, although 1.5% of replay steps are unlocks. Replicated on seed 1, whose in-training `report/wm/` logs show unlock recall 0 at 13k, 52k, and 92k steps while posterior reward MAE falls 0.13 → 0.04.
  - Reward MAE on reward steps is ~0.4–0.6, and terminal recall is 0.
  - Likely link to RQ1: the actor learns only from imagined rewards, which never contain the +1 unlocks.
- **DreamerV3 at train ratio 64** (1 seed, 2.35 h):
  - Slightly better world model: reward loss 0.20 vs 0.26; the posterior now catches 5% of unlocks (precision 0.85).
  - Open-loop unlock recall is still 0.
  - The policy is still at random level (2% / 6%).
  - So doubling the ratio is not enough at 100k steps. Candidates: a larger model (size12m+), ratio ≥ 128, a longer budget, or the sparse reward (see the PPO ablation).
- **DreamerV3 at train ratio 128** (1 seed, `configs/dreamer_size1m_r128_100k.yaml`, 4.61 h wall, the only job on the machine; load avg 2 at start):
  - **World model now predicts unlocks.** Unlock recall at the posterior is 0.30 (precision 0.85) on the training replay and 0.30 (precision 0.98) on the eval replay, vs 0.05 at ratio 64 and 0 at ratio 32. Open-loop recall stays at 0.1–0.6 over k = 1..16 instead of 0 (`figures/main_wm_error_dreamerv3_r128.png`). Final losses: reward 0.18 (r64 0.20), image 86 (r64 107).
  - **The policy learns, but not the target.** Harness eval of the ≈ 97k checkpoint (n=50 per split) vs the random policy (n=400): collect_sapling 97% vs 51%, collect_wood 41% vs 21%, collect_drink 23% vs 8%, eat_cow 8% vs 1%, defeat_zombie 7% vs 0%; return 1.84 vs 1.32. But `place_table` is 0/100, and in training episodes its rate falls from 7.6% (first 20k steps) to 0/111 (last 20k).
  - Reading (guess): with the standard reward, the cheapest +1s (sapling, wood, drink) dominate what the actor sees in imagination, and nothing pushes it to keep wood and place a table. This is the case for the sparse-reward DreamerV3 ablation, which PPO says is worth +34 pp.
  - Compute: policy fps 6.0 vs 11.9 at ratio 64 (train fps ≈ 770 replayed steps/s in both), so wall-clock scales about linearly with the ratio.
- **Sanity check against the literature: PPO at 1M steps** (1 seed, `configs/ppo_cnn_1m.yaml`, `results/budget1m/`; 18 min training).
  - Harness eval of the final checkpoints: place_table 94% train / 87% held-out, collect_wood 100%, make_wood_pickaxe 74–78%.
  - Paper-style metric (success over all 5,151 training episodes in the budget, as the Crafter paper reports it): collect_wood 86.4% / place_table 72.9% / make_wood_pickaxe 40.1%, vs the paper's PPO at 1M of 83.0 / 66.1 / 21.1%.
  - So our PPO pipeline is on par or slightly better.
  - Possible reasons (guess): SB3 Atari hyperparameters; our 50-world training pool vs fresh worlds every episode in the paper.
  - At 1M steps the generalization gap persists (+7 pp on place_table).
- **Reward ablation:** with the target-only sparse reward PPO reaches 96% train / 82% held-out, against 62% / 52% with the standard reward. Optimizing exactly the measured target helps a lot, but the generalization gap grows.
- **Caveats:**
  - DreamerV3's final numbers now come from the shared harness on its final checkpoint (T12, `scripts/eval_dreamer.py`; upstream keeps only the latest checkpoint). Its learning curve is still the in-loop eval (randomly sampled worlds, ~10 episodes per split per round).
  - RQ3 uses replay buffers (eval buffer mixes train and held-out worlds), not T15 trajectories.
  - DQN and the ablation follow rl-zoo Atari settings scaled to 250k, with no tuning.
  - Single machine, provisional timings.

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
- **Measured (T05, 2026-11-04, provisional load):** for PPO `CnnPolicy` with the Atari hyperparameters, **MPS wins**: 1,660 sps vs 347 sps on CPU, both with 8 `SubprocVecEnv` envs. The PPO update is ~19× faster on MPS. Single-observation rollout inference is somewhat slower on MPS, but the update dominates. SB3's `get_device` passes `"mps"` straight to `torch.device` (it only special-cases CUDA), and it works.
- Re-check the device for DQN (batch 32, an update every 4 steps) and RecurrentPPO; small batches may favour CPU.
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
- **Gate:** choose the target whose random success is non-zero and ≤ ~10% (expected `place_table`). If `place_table` is ≥ 20%, use `make_wood_pickaxe`. **Result (T04, 2026-11-04): `place_table`** (2.0% train, 4.5% held-out, n=200 each).
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
| PPO-CNN (SB3, CPU) | 150–600 sps | 2–6 min | 3–11 min | 7–28 min | T05 benchmark (random-ish early policy, no eval): **MPS + 8 subproc 1,660 sps** → 100k ≈ 1 min, 250k ≈ 2.5 min; CPU + 8 subproc 347 sps. Full runs (Stage 1): 250k in 162 s (1,553 sps, load ≈ 2) to 268 s (938 sps, load ≈ 20), + ~25 s cache warm-up and ~2.5 min eval (26 ckpts × 100 episodes, 8 workers) |
| PPO-LSTM | 0.3–0.5× PPO | 5–20 min | 10–40 min | 25–90 min | 250k: 9–10 min training (404–454 sps, mps + 8 subproc), +~3 min eval; **provisional (load avg 10–20)** |
| DQN (train_freq 4) | 100–300 sps | 3–8 min | 6–17 min | 14–42 min | 250k: 20–24 min (173–210 sps, mps, 1 env); **provisional** |
| DreamerV3 size1m, ratio 32 | t_update 0.3–1.5 s | 0.2–0.8 h | 0.3–1.5 h | 0.8–3.5 h | 100k: **93 min** wall incl. 7 eval rounds (fps/train 420–630 replay steps/s ≈ t_update 1.6–2.4 s; peak RSS 9.9 GB); **provisional (shared CPU)** |
| DreamerV3 size1m, ratio 64 | 2× updates | 0.4–1.5 h | 0.7–3 h | 1.6–7 h | 100k: 2.35 h wall (load avg ~10, other jobs at times); **provisional** |
| DreamerV3 size1m, ratio 128 | 4× updates | – | – | – | 100k: **4.61 h** wall, only job (≈ 28 min / 10k steps; fps/train ≈ 770 replay steps/s; peak memory 10.5 GB) |

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
| T04 | Random baseline + env sps benchmark | `results/stage0/random.csv`, target choice | T03 | 2 h | – | done (sps to re-time idle) | Kevin |
| T05 | CPU vs MPS PPO benchmark | device decision | T01 | 1 h | yes | done (mps + subproc) | Kevin |
| T06 | Eval harness + adapter protocol + CSV schema + Wilson CI | `rlproj/eval`, tests | T03 | 4–6 h | – | done | Kevin |
| T07 | `train_sb3.py` + PPO config + checkpointing | PPO run | T06 | 3–4 h | – | done | Kevin |
| T08 | PPO 100k run + eval → **first result** | `results/ppo_cnn`, plot | T07, T09 | 0.5 day | – | done (250k × 3 seeds) | Kevin |
| T09 | ★ Plotting scripts against the §7 schema (start with fake CSVs) | `plots/*.py` | schema only | 4–6 h | **yes** | done (first version) | Kevin |
| T10 | Dreamer launcher registering `CrafterTask`; size1m 2k-step smoke | `train_dreamer.py` | T02, T03 | 4–6 h | yes, with T06–T08 | done | Kevin |
| T11 | Dreamer 50–100k run + throughput report | `results/dreamer`, §5 measured | T10 | 0.5 day + overnight | – | done (ratio 32 × 3 seeds, ratio 64 × 1 seed, ratio 128 × 1 seed) | Kevin |
| T12 | Dreamer checkpoint adapter for harness | `adapters/dreamer.py` | T06, T10 | 4–6 h | – | done (every 10k-step checkpoint for runs with `ckpt_every_steps`, else final checkpoint only; eval reproducible byte for byte) | Kevin |
| T13 | ★ PPO-LSTM config + adapter state handling + test | LSTM run | T06, T07 | 0.5 day | **yes** | done (250k × 3 seeds; recurrent-state adapter test in `tests/test_sb3_adapter.py`) | Kevin |
| T14 | ★ DQN config (memory-safe) + 50k smoke | DQN run | T06, T07 | 0.5 day | **yes** | done (250k × 3 seeds + 3-step variant) | Kevin |
| T15 | ★ Held-out trajectory collector | `collect_heldout.py` + npz | T12 | 3–4 h | **yes** | todo | |
| T16 | ★ `wm_eval` patch + script + plot | `wm_error.csv`, figure | T10 (any checkpoint), T15 | 1 day | **yes** | in progress (preliminary: on replay buffers, not T15 trajectories) | Kevin |
| T17 | ★ Demo GIF tool (incl. Dreamer imagined-vs-real) | `make_demo.py` | T06 | 4 h | **yes** | todo | |
| T18 | ★ TA update note (§10) | email text | T04 | 1 h | **yes** | todo | |
| T19 | ★ README with exact reproduction commands (kept current) | `README.md` | ongoing | ongoing | **yes** | in progress (first version 2026-11-05) | Kevin |
| T20 | Stage 4 main runs (4 methods × 3 seeds) | `results/main` | T08, T11–T14, gate | 1–2 weeks of nights | split by machine | todo | |
| T21 | Ablation runs (reward sparsity) | `results/ablation` | T20 setup | compute | split by machine | in progress (PPO × 3 seeds done; DreamerV3 ratio 128 seed 0 running) | Kevin |
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

### 2026-11-09 — DreamerV3 ratio 128; repo published — MacBook Pro M2 Max
- **Completed:**
  - README for teammates (key terms, results, problems fixed, limitations, next steps) and CONTRIBUTING.md with the project rules.
  - Repo pushed to GitHub (`KaiwenYangUT/world-model-rl`).
  - DreamerV3 size1m at train ratio 128, 100k steps, seed 0 (`configs/dreamer_size1m_r128_100k.yaml`; checkpoint every 10 min instead of 30, so the final one is at ≈ 97k), then log conversion, `wm_eval`, and harness eval.
  - Summary table and figures refreshed (`results/main/summary.md`, `figures/main_*.png`, new `figures/main_wm_error_dreamerv3_r128.png`); learning-curve legend now 3 columns so long labels fit.
- **Measurements:**
  - Wall-clock 16,610 s (4.61 h) for 100k steps, ≈ 28 min per 10k steps; only job running (load avg 2.1 at start, ~9.8 during the run from DreamerV3's own threads). Peak memory 10.5 GB.
  - Harness, ≈ 97k ckpt: place_table 0% / 0% (n=50 each), collect_wood 46% / 36%, return 1.84 (random 1.32).
  - Learning observed: **yes, but not on the target.** Unlock rates vs random: collect_sapling 97% vs 51%, collect_wood 41% vs 21%, collect_drink 23% vs 8%, eat_cow 8% vs 1%, defeat_zombie 7% vs 0%. place_table in training episodes falls 7.6% → 0%.
  - RQ3: unlock recall at the posterior 0.30 (was 0.05 at r64, 0 at r32); open-loop recall 0.1–0.6 over k = 1..16 (was 0).
- **Issues / open items:**
  - One seed only.
  - The standard reward pulls DreamerV3 towards the easy achievements; place_table is not one it chases at this budget.
- README: problems section rewritten with what this run showed (standard reward pulls DreamerV3 to the easy achievements; compute grows with the ratio; DreamerV3 needs >8 GB so it cannot run on the iMac; one seed each for r64/r128; in-loop curves too noisy).
- **Next:** DreamerV3 sparse-reward ablation at ratio 128 (~4.6 h); if it learns the table, add seeds 1–2.
- **Later the same day:**
  - Step-indexed DreamerV3 checkpoints: `scripts/train_dreamer.py` wraps upstream's `make_agent` / `make_logger` (no change to `third_party/`) and saves the agent every `ckpt_every_steps` env steps, plus step 0 and the end, to `runs/<exp>/<method>/seed<k>/ckpt_steps/step_<n>/` (~8.5 MB each). The env-step count is upstream's `logger.step`, which the train driver increments once per env step (checked in `embodied/run/train_eval.py`).
  - `scripts/eval_dreamer.py` scores every step checkpoint with the shared harness and writes `eval.csv` / `eval_episodes.csv` / `eval_meta.json` like `scripts/eval.py`; for these runs `dreamer_results.py` writes the in-loop eval to `eval_inloop*.csv`. Plots and the summary label a curve "in-loop" only when `eval_meta.json` is missing.
  - Smoke test (2.5k steps, checkpoint every 1k): 4 checkpoints written; weights differ between checkpoints; harness episodes change once training starts; harness re-run byte-identical; 30 tests pass.
  - Started the sparse-reward ratio-128 run (seed 0) at 13:20.

### 2026-11-05 (early morning) — overnight runs, continued — MacBook Pro M2 Max
- **Completed after the entry below:**
  - DQN 3-step × 3 seeds.
  - T12 DreamerV3 harness adapter (`rlproj/adapters/dreamer.py`, `scripts/eval_dreamer.py`; byte-identical on re-run).
  - DreamerV3 ratio 32 seeds 1–2 and ratio 64 seed 0, each harness-scored.
  - T13 adapter tests (30 tests total).
  - README (T19).
  - PPO at 1M steps (budget/literature check).
- **Measurements:**
  - DQN 3-step 47.0% / 37.0% (3 seeds).
  - DreamerV3 r32 4.7% / 2.7% (3 seeds, harness); r64 2% / 6% (1 seed; 2.35 h wall).
  - PPO 1M: 94% / 87% final; paper-style 72.9% vs 66.1% (Crafter paper).
  - RQ3: unlock recall 0 at every horizon for all 3 r32 seeds; r64's posterior catches 5% of unlocks.
- **Issues / open items:**
  - Two DreamerV3 runs in parallel oversubscribe the CPU (load avg ~100), so run them one at a time.
  - Upstream saves only every 30 min and keeps one checkpoint, so DreamerV3 "final" checkpoints sit at 70–96k of 100k.
  - Shell gotcha: the Bash tool runs zsh, which does not word-split unquoted `$var`. A queued job silently waited on a wrong path for 30 min until fixed.
- **Next:** review; decide the Stage 4 plan (see Current status); re-time on an idle machine.

### 2026-11-04 (night) — overnight runs — MacBook Pro M2 Max
- **Completed:**
  - **T06** eval harness:
    - `rlproj/eval/harness.py` (fixed, policy-independent world seeds per split; `evaluate` / `summarize`);
    - `rlproj/adapters` protocol `reset(n, seed)` / `act(obs, first)` plus `RandomAdapter`;
    - `rlproj/eval/metrics.py` (Wilson CI, Crafter score with the formula from crafter `analysis/common.py`, AUC, steps-to-fraction);
    - `rlproj/utils/csvlog.py` (§7 schemas, schema-checked writer);
    - `tests/test_eval.py`. 26 tests pass in both venvs.
  - **T07** SB3 pipeline:
    - `rlproj/adapters/sb3.py`;
    - `scripts/train_sb3.py` (step-indexed checkpoints with training wall-clock, per-episode training log, meta.json);
    - `scripts/eval.py` (parallel per-checkpoint eval; **re-running eval gives a byte-identical eval.csv**).
  - **T08/T13/T14** runs (configs `ppo_cnn_250k`, `ppo_lstm_250k`, `dqn_250k`, `dqn_nstep3_250k`, all rl-zoo Atari blocks scaled to 250k; see each file's header):
    - PPO, PPO-LSTM, DQN: 250k × 3 seeds each;
    - DQN 3-step: 3 seeds.
  - **T21 (early):** PPO with the target-only sparse reward, 250k × 3 seeds (`configs/ablation_sparse_ppo_cnn_250k.yaml`).
  - **T10/T11** DreamerV3:
    - `rlproj/envs/embodied_crafter.py` + `scripts/train_dreamer.py`: official DreamerV3 on CrafterTask via a wrapped `train_eval`. Upstream is unmodified for training; eval env 0 uses the train pool, env 1 the held-out pool.
    - `scripts/dreamer_results.py` converts its logs.
    - size1m, ratio 32, 100k: 3 seeds; ratio 64: 1 seed. All harness-scored at their latest checkpoint (T12, added later that night).
  - **T16 (preliminary):**
    - Patched `Agent.report` to emit per-horizon open-loop reward/continuation errors (`third_party/patches/dreamerv3-report-wm-horizon.patch`, recorded in PATCHES.md).
    - `scripts/wm_eval.py` evaluates a checkpoint on its replay buffers.
    - `plots/plot_wm_error.py`.
  - **T09** plots: `plot_learning_curves.py`, `plot_generalization.py`, `plot_ablation.py`, `plot_wm_error.py`, and `summarize_results.py` (summary table and §6 gate). matplotlib added to `.venv-sb3`.
- **Measurements:**
  - See *Preliminary results* at the top for success, gap, and RQ3.
  - Training speed: PPO 938–1,553 sps (mps + 8 subproc); PPO-LSTM 404–454 sps (mps; cpu was 123); DQN 168–210 sps (mps, 1 env; cpu was 64); DreamerV3 size1m r32 100k = 93 min wall (peak RSS 9.9 GB).
  - Wall-clock per 10k steps: PPO 6.4–10.7 s, PPO-LSTM 22–25 s, DQN 48–60 s, DreamerV3 ~9.3 min including in-loop eval.
  - **All provisional:** many jobs ran at once; the load avg reached ~100 when two DreamerV3 runs overlapped.
  - Learning observed:
    - PPO yes (4% → 62% place_table);
    - PPO-LSTM yes (→ 45%);
    - DQN partly (1 of 3 seeds clearly; 3-step variant yes on seed 0);
    - DreamerV3 no (world-model losses fall, policy at random level).
- **Decisions to review with the team:**
  - Eval uses **50 episodes per split** (PLAN said 20), because CIs at n=20 are very wide.
  - The DQN 3-step variant counts as "DQN try 2" (§4 allows 2–3 tries).
  - I started the sparse-reward ablation early.
  - I stopped a DreamerV3 ratio-64 run that overlapped another DreamerV3 run (oversubscription) and requeued it sequentially.
- **Issues / open items:**
  - DreamerV3 evaluation is not through the shared harness yet (T12). Its "final" numbers rest on about 20 in-loop episodes.
  - RQ3 uses replay buffers. The eval buffer mixes train and held-out worlds and has no split label. Proper T15 trajectories are still needed.
  - DreamerV3's `train_eval` saves checkpoints only every 30 min, so the latest checkpoint can lag the end of training (the seed-0 one is at ~96k of 100k).
  - Upstream `report()` bug noted in PATCHES.md (`mets.update(mets)`); left unchanged.
  - T13's recurrent-adapter test is now done: `tests/test_sb3_adapter.py` checks state carry-over, per-slot reset on `first`, and seeded sampling; DQN ε rate is checked too. 30 tests in total.
  - No timing numbers are fit to quote (rule 5).
- **Next:**
  - Review results.
  - Decide whether DreamerV3 should get a higher train ratio or a larger model, based on the ratio-64 run.
  - T12 adapter; T15 + T16 proper.
  - Re-time on an idle machine.
  - Stage 4 budget: PPO is cheap enough for 1M steps.

### 2026-11-04 — T05 — MacBook Pro M2 Max
- **Completed:** T05.
  - `scripts/bench_device.py` and `configs/stage0_bench_device.yaml`. SB3 PPO `CnnPolicy` with the rl-zoo `atari` hyperparameters (checked in `hyperparams/ppo.yml`): 8 envs, n_steps 128, 4 epochs, batch 256, lr 2.5e-4, clip 0.1, vf 0.5, ent 0.01. For timing I used constant lr/clip instead of the linear schedules and no frame stack.
  - Grid: {cpu, mps} × {DummyVecEnv, SubprocVecEnv} × 2 repeats, interleaved. 20,480 steps (20 iterations) per run, with the world cache warmed first.
  - A callback splits each iteration into rollout and update time, with `torch.mps.synchronize()` at each timestamp (MPS is async).
  - Outputs: `results/stage0/bench_device/{bench_device.csv, meta.json}`. Commit `24ccf5b`, clean. Total 471 s.
- **Measurements (provisional: load avg 3.4 at start, up to 7 during runs):**

  | Device / vec env | sps (mean of 2) | Rollout s / run | Update s / run |
  |---|---|---|---|
  | cpu / dummy | 292 | 17.0 | 53.1 |
  | cpu / subproc | 347 | 5.8 | 53.2 |
  | mps / dummy | 848 | 21.3 | 2.9 |
  | **mps / subproc** | **1,660** | 9.6 | 2.8 |

  - Repeats agree within 1%. The first MPS iteration costs ~0.6–1.4 s (warm-up).
  - On CPU the PPO update dominates: 2.66 s per iteration, i.e. 16 gradient steps (1,024 samples / batch 256 × 4 epochs) at ~166 ms each, with 8 torch threads. On MPS it is 0.14 s (~9 ms per step), **~19× faster**.
  - Rollout is slower on MPS than CPU: per-step inference on 8 observations has GPU-sync overhead.
  - Losses are identical across repeats on the same device (deterministic given the seed); CPU and MPS differ slightly, as expected.
  - Learning: n/a (timing only).
- **Decision:** PPO-CNN runs on **`device="mps"` with `SubprocVecEnv`**. At this throughput a 100k-step PPO run is ~1 min of training (eval overhead not included), well under the §5 guesses.
- **Issues / open items:**
  - With MPS, rollout is now ~75% of PPO wall-clock. More envs (e.g. 16) might amortize inference; try it in T07 if it matters.
  - DQN and RecurrentPPO have different batch shapes; benchmark the device again in T13/T14.
  - Re-time on an idle machine before quoting any of this.
- **Next:** T06 (eval harness), T07 (`train_sb3.py` + PPO config). T10 can start in parallel.

### 2026-11-04 — world cache — MacBook Pro M2 Max
- **Completed:** per-process world cache in `CrafterTask` (`_DeterministicCrafterEnv.reset`, `cache_worlds=True` by default), as proposed in the T04 entry.
  - The first reset on a world seed generates the world and stores a deep copy of the world state and player. Later resets restore a copy.
  - Objects' `world` pointers are remapped to the env's own `World`, which the renderers hold. `world.random` is copied once, so all objects still share it.
  - Recorded in `third_party/PATCHES.md`.
- **Verification:**
  - 2 new tests (18 total, all pass in both venvs, 27 s):
    - 500-step fingerprints with cache off vs cache miss vs hit vs second hit vs hit from another instance (4 seeds) are identical;
    - after restore, every object points at the env's World, shares `world.random`, and zombies/skeletons point at the env's player.
  - Re-ran T04 with the cache. `random.csv` and `random_episodes.csv` are **byte-identical** to the uncached run.
- **Measurements (provisional: load avg 3.3–4.5 from other apps):**
  - Reset: **~490 ms → ~1.1 ms** on a cache hit. The first visit to each seed still costs ~0.5 s, so warming all 50 train seeds takes ~25 s per process, once.
  - Env sps, random actions, warm cache (uncached T04 values in parentheses):

    | Procs | sps |
    |---|---|
    | 1 | **1,617** (280) |
    | 4 | **3,805** (311) |
    | 8 | **6,181** (351) |

  - Lockstep vector envs now scale, since resets no longer stall them.
  - Memory: 150 cached worlds ≈ 37 MB RSS per process (33 KB/world pickled).
  - T04 wall-clock 233 s → 133 s. The episode phase barely changed (38 → 32 s) because most of its world visits are first visits per worker.
  - `env_sps.csv` gained `warmup_s` and `warm_cache` columns. Config: `sps.warm_cache: true`.
- **Implication:** the env is no longer a bottleneck for model-free training (PPO's compute will dominate). The Stage 0 "< 300 sps" fallback no longer applies.
- **Next:** T05 (CPU vs MPS), T06 (eval harness). Re-time env sps on an idle machine before quoting it.

### 2026-11-04 — T04 — MacBook Pro M2 Max
- **Completed:** T04.
  - New code:
    - `scripts/random_baseline.py` and `configs/stage0_random.yaml`;
    - `rlproj/utils/config.py` (YAML via ruamel.yaml, a crafter dependency in both venvs);
    - `rlproj/utils/meta.py` (meta.json: git hash + dirty flag, config, hardware, packages, load averages, wall-clock);
    - `rlproj/eval/metrics.py` (`wilson_ci`, to be reused by T06).
  - Episodes are deterministic and don't depend on the worker count. Episode i uses world seed `pool[i % len(pool)]` and action RNG `default_rng([seed, split, i])`.
  - Outputs in `results/stage0/`:
    - `random.csv` (§7 eval.csv schema, one row per split × achievement-as-target);
    - `random_episodes.csv` (400 episodes, first-unlock step for all 22 achievements);
    - `env_sps.csv`;
    - `meta.json`.
  - Run: `.venv-sb3/bin/python scripts/random_baseline.py --config configs/stage0_random.yaml`. 233 s total. Made from commit `c7f6ea5` + uncommitted changes (`dirty: true` in meta.json).
- **Random baseline (200 episodes per split, 95% Wilson CI):**

  | Achievement | Train (0–49) | Held-out (10000–10099) | Paper random (§2) |
  |---|---|---|---|
  | collect_wood | 21.0% [15.9, 27.2] | 21.5% [16.4, 27.7] | 24.4% |
  | place_table | **2.0% [0.8, 5.0]** | **4.5% [2.4, 8.3]** | 4.4% |
  | make_wood_pickaxe | 0.0% [0.0, 1.9] | 0.0% [0.0, 1.9] | 0.3% |
  | collect_stone | 0.0% [0.0, 1.9] | 0.0% [0.0, 1.9] | 0.0% |

  - Others (train / held-out):
    - wake_up 96% / 94%;
    - collect_sapling 50% / 52%;
    - place_plant 48% / 48%;
    - collect_drink 5.5% / 10.5%;
    - eat_cow 0.5% / 0.5%.
  - Mean return 1.32 / 1.40.
  - Episode length: median 164 / 172 (p25–p75 149–186 / 154–192, max 282 / 371). **Every episode ended in death; none reached the 10k length limit.**
  - Consistent with the paper. Train vs held-out CIs overlap, so for a random policy the "gap" is noise.
- **Gate decision: target = `place_table`** (non-zero and ≤ 10% on both splits).
  - Week-1 criterion (b), §6: PPO place_table success on train must be ≥ max(2×2.0%, 2.0% + 10 pp) = **12%**, with its Wilson lower bound above **5.0%**.
- **Env throughput** (random actions, train pool; **provisional: load avg ≈ 3 from other apps**):

  | Procs | sps (incl. resets) | Notes |
  |---|---|---|
  | 1 | 280 | step() alone = **1,652 sps**; 121 resets at **0.494 s each** = 60 s of the 71 s |
  | 4 (AsyncVectorEnv) | 311 | startup 0.72 s (excluded) |
  | 8 (AsyncVectorEnv) | 351 | startup 0.77 s (excluded) |

  - **Reset (Python world generation, `crafter/worldgen.py`) is ~84% of env time for a random policy.** One reset costs as much as ~800 steps.
  - Vectorized envs barely help: they step in lockstep, so any env's 0.5 s reset stalls all of them. The Stage 0 fallback "< 300 sps → 4 subprocess envs" therefore does not work here; 4 procs give +11%.
  - Trained agents should survive longer, so they reset less often.
  - Not checked: whether SB3 `SubprocVecEnv` behaves the same. It should, since it is also lockstep.
- **Fix during T04:** `CrafterTask` `first_unlock` now uses `-1` (`NOT_UNLOCKED`) instead of `None`. gymnasium's AsyncVectorEnv crashed stacking mixed None/int infos (SB3 VecEnvs would not have). Added a regression test, which fails on the old code; 16/16 tests pass.
- **Issues / open items:**
  - **Possible speed-up (not done, needs a decision):** world seeds come from fixed pools (150 seeds), so `CrafterTask` could cache each seed's freshly generated world (map, objects, player, `world.random` state) and restore a copy on reset instead of regenerating. That would turn a 0.5 s reset into ~ms. It must keep the determinism tests passing.
  - **Eval noise:** the plan's 20 eval episodes per split give wide CIs. At p≈5%, the Wilson 95% CI with n=20 is ~[0.9%, 24%]. Consider more eval episodes in T06, or rely on pooling over the last eval points and seeds.
  - Re-time env sps on an idle M2 Max before quoting it.
- **Next:** decide on world caching; then T05 (CPU vs MPS) and T06 (eval harness).

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
