# DQN vs PPO vs PPO-LSTM vs DreamerV3 on Crafter

ECE1508 reinforcement learning course project. We compare four reinforcement-learning
methods on **Crafter**, a 2D Minecraft-like game with sparse rewards and a partial view of
the world, under the same budget of game steps and the same evaluation code.

**Status (2026-11-09):** preliminary results exist for all three research questions.
Final experiments run Nov 16 – Dec 4; results freeze on Dec 6.
Full plan, decisions and run log: [PLAN.md](PLAN.md).

## Research questions
- **RQ1, learning efficiency:** how fast does each method learn (success rate and return vs. game steps)?
- **RQ2, generalization:** how much worse does each method do on game worlds it never trained on?
- **RQ3, world-model reliability:** how accurate are DreamerV3's predicted rewards and episode
  endings as it imagines further ahead?

## Key terms
- **Game world.** Crafter generates each 64×64 map (terrain, trees, water, caves, animals)
  from a number called the *world seed*. The same world seed always gives the same map.
  We train on **50 worlds** (seeds 0–49) and also test on **100 unseen worlds** (10000–10099).
- **Task.** The agent must **place a crafting table**: chop trees until it holds 2 wood, then
  use the place-table action facing an empty grass, sand or path tile. Success = it happens at
  least once in an episode. Random play succeeds 2–4.5% of the time.
- **Reward.** The standard Crafter reward: +1 the first time each of 22 achievements happens in
  an episode (placing a table is one of them), ±0.1 for health changes. An ablation rewards
  only the table.
- **Run seed.** A separate random seed for each training run. It sets the network's starting
  weights, the order of worlds and the random exploratory moves. "3 seeds" = three independent
  runs of the same method, to tell real results from luck.

## The four methods
| Method | Family | In one sentence | Implementation here |
|---|---|---|---|
| **DQN** | value-based | Learns a score for each action (expected future reward) from a memory of past moves, and picks the best one. | Stable-Baselines3; 3-step-return variant as second try |
| **PPO** | policy-based | Learns action probabilities directly, with small, clipped updates from fresh play. | Stable-Baselines3, 8 games in parallel, no memory |
| **PPO-LSTM** | policy-based + memory | PPO plus an LSTM memory of earlier frames (the agent sees only a small patch of the map). | sb3-contrib RecurrentPPO |
| **DreamerV3** | model-based | Learns a world model (a learned simulator), then trains its policy on imagined play. | Official authors' code, smallest size (~0.7M parameters), CPU |

## What has been done
1. **Setup and baselines.**
   - One shared environment wrapper, `rlproj/envs/crafter_task.py`, with the world pools, the target, the two reward modes and determinism tests.
   - A random-policy baseline: 400 episodes, which is how we picked the target.
   - A speed test: PyTorch trains PPO 4.8× faster on the M2 Max GPU (MPS) than on the CPU.
2. **Pipeline.** Every method writes a checkpoint every 10k steps. Checkpoints are scored by one shared evaluation harness:
   - fixed worlds, 50 episodes on training worlds and 50 on unseen worlds;
   - 95% confidence intervals;
   - byte-identical results on a re-run.

   Every run records its git commit, config and hardware.
3. **Experiments (preliminary).**
   - PPO, PPO-LSTM, DQN and DQN 3-step: 250k steps × 3 seeds each.
   - DreamerV3: 100k steps × 3 seeds, plus one run each with 2× and 4× training updates.
   - A reward-sparsity ablation for PPO.
   - PPO at 1M steps, to compare with the Crafter paper.
   - World-model error vs. imagination horizon for every DreamerV3 run.

## Preliminary results
Success = placing a table. Final = last two checkpoints of each of 3 seeds, pooled
(300 episodes per world set). Brackets = 95% confidence interval.

| Method | Steps | Training worlds | Unseen worlds | Gap |
|---|---|---|---|---|
| Random policy | – | 2.0% | 4.5% | – |
| **PPO** | 250k | **62.0%** [56–67] | **52.0%** [46–58] | +10 pts |
| PPO-LSTM | 250k | 45.0% [40–51] | 40.3% [35–46] | +5 pts |
| DQN | 250k | 16.3% [13–21] (learns on 1 of 3 seeds) | 17.7% [14–22] | ≈0 |
| DQN, 3-step returns | 250k | 47.0% [41–53] | 37.0% [32–43] | +10 pts |
| DreamerV3 (smallest) | 100k | 4.7% [2–9] | 2.7% [1–7] | – |
| DreamerV3, 4× updates (1 seed) | 100k | 0% [0–7] | 0% [0–7] | – |
| PPO, sparse reward (ablation) | 250k | 96.0% [93–98] | 82.3% [78–86] | +14 pts |

- **RQ1:** PPO learns fastest. Memory (PPO-LSTM) does not help at this budget. Plain DQN is
  unreliable across seeds; 3-step returns fix that. DreamerV3 at this size stays at random level
  with the default number of updates. With 4× updates it does learn (it collects saplings 97% of
  the time vs. 51% for random play, wood 41% vs. 21%) but goes for these easy achievements and
  never places a table.
- **RQ2:** agents that learn do 5–10 points worse on unseen worlds. Both curves match until
  ~100k steps; after that, training-world success rises faster. Unseen-world success still
  rises with more training (PPO at 1M: 94% training, 87% unseen).
- **RQ3:** DreamerV3's world model learns the images and the average reward, but never predicts
  the +1 reward for unlocking an achievement, even while watching real frames (3 of 3 seeds).
  Its policy learns only from those predictions, so it gets no signal for the task. With 4×
  updates the world model catches 30% of unlocks while watching and 10–60% while imagining up
  to 16 steps ahead ([figure](figures/main_wm_error_dreamerv3_r128.png)), and that is when its
  policy starts to learn.
- **Against published work:** scored the way the Crafter paper scores it (success over all
  training episodes), our PPO at 1M steps gets 86% / 73% / 40% on collect wood / place table /
  make wood pickaxe vs. the paper's 83% / 66% / 21%. Our random baseline also matches the paper's.

![Learning curves](figures/main_learning_curves.png)

More figures: [generalization](figures/main_generalization.png),
[reward ablation](figures/ablation_sparse_ppo_cnn.png),
[world-model error](figures/main_wm_error.png). Full table: [results/main/summary.md](results/main/summary.md).

## Problems found and fixed
- **Crafter was not reproducible:** the same world seed and actions could give different games.
  The cause was Python set ordering when despawning creatures. Our wrapper fixes it, and tests cover it.
- **World generation was the bottleneck:** 0.5 s per reset. Caching each world per seed cut it
  to ~1 ms (env speed 351 → 6,181 steps/s with 8 processes) with identical results.
- **Upstream DreamerV3 dropped its report metrics:** we added a documented patch that measures
  reward/continuation error per imagination step (`third_party/PATCHES.md`).
- **Evaluation noise:** 20 episodes per checkpoint gave intervals like 1–24%, so we use 50.

## Current problems and limitations
- **DreamerV3 is too small and too slow here.** On this laptop it runs only on the CPU, so we use
  the smallest model (~0.7M parameters) and few training updates per game step (train ratio 32,
  vs. 512 in the authors' Crafter setting with a much larger model). At ratio 32 (3 seeds) and
  ratio 64 (1 seed) its policy stays at random level.
- **With 4× updates, DreamerV3 learns the wrong things.** At ratio 128 (1 seed) the world model
  finally sees the unlock rewards and the policy clearly learns, but it goes for the easiest
  +1s: saplings (97% vs. 51% for random play), wood, water, cows. It never places a table
  (0 of 100 eval episodes), and its table rate during training falls from 7.6% in the first
  20k steps to 0% in the last 20k. With the standard reward every achievement is worth the same
  +1, and placing a table needs a longer chain (two wood, then the right tile), so the cheap
  ones win at this budget. This is our reading of the numbers, not yet tested; the sparse-reward
  run below tests it.
- **DreamerV3 compute grows with the update ratio.** Training time grows roughly in proportion
  to the ratio: 100k steps take ~1.5 h at ratio 32, 2.35 h at ratio 64 and 4.6 h at ratio 128
  (~6 game steps/s). Three seeds at ratio 128 are ~14 h, and each further DreamerV3 experiment
  (sparse reward, more seeds) costs the same again. The authors' ratio 512 would be roughly
  18 h per run at this size (an extrapolation, not measured).
- **Only one machine can run DreamerV3.** Its runs peak at 10–11 GB of memory (9.9 GB at
  ratio 32, 10.5 GB at ratio 128), more than the iMac's 8 GB, so all DreamerV3 runs share the
  M2 Max with the Stage 4 runs of the other methods, one job at a time.
- **Few DreamerV3 seeds.** Ratio 64 and ratio 128 have one seed each, so their differences from
  ratio 32 are not yet confirmed across seeds.
- **DreamerV3 evaluation gaps.** Upstream keeps only its latest checkpoint, so the shared harness
  scores only the end of each run: 70k–96k of 100k steps at ratio 32 (saved every 30 min), ~97k
  at ratio 128 (saved every 10 min). Its learning curves come from the in-loop evaluation, which
  plays only ~10 episodes per world set per round and is too noisy to say when learning starts.
- **RQ3 data:** measured on the agent's replay memory, not on a separate set of unseen-world games.
- **Timing numbers are mostly not reportable yet:** several jobs shared the laptop during the
  overnight runs. The DreamerV3 ratio-128 run is the exception (it ran alone).
- **Small study:** default (rl-zoo) settings, no tuning, 3 seeds, 250k steps.

## Next steps
1. **DreamerV3:**
   - Run its sparse-reward ablation at 4× updates, which rewards only the table; if it
     learns, add two more seeds.
   - Otherwise, time the next model size (12M parameters) before committing to it.
   - Look for an NVIDIA GPU (university cluster or cloud), which DreamerV3 is built for.
2. **Measurement:**
   - Save DreamerV3 checkpoints by step count, so it gets proper learning curves.
   - Collect separate unseen-world games for RQ3.
   - Re-time all methods on an idle machine.
3. **Final experiments (Nov 16 – Dec 4):**
   - PPO, PPO-LSTM and DQN at 1M steps × 3 seeds.
   - Decide whether DQN 3-step becomes our main DQN.
4. **Communication:** send the TA update, then build the presentation figures from frozen results.

## Repository layout
- `rlproj/`: shared code.
  - `envs/`: the Crafter wrapper.
  - `eval/`: the harness and metrics.
  - `adapters/`: one interface over all four methods.
  - `utils/`: config, metadata and CSV schemas.
- `scripts/`: training, evaluation and analysis entry points.
- `configs/`: one YAML file per experiment.
- `plots/`: figures and summary tables.
- `results/`: small CSV and JSON outputs (committed).
- `figures/`: plots.
- `runs/`: checkpoints and logs (not committed).
- `tests/`: 30 unit tests.
- `third_party/`: the pinned DreamerV3 submodule; every local change is listed in `PATCHES.md`.

## Setup (macOS arm64, Python 3.11)
```bash
brew install python@3.11
git submodule update --init
git -C third_party/dreamerv3 apply ../patches/dreamerv3-report-wm-horizon.patch
/opt/homebrew/bin/python3.11 -m venv .venv-sb3
.venv-sb3/bin/pip install -r requirements-sb3.lock
/opt/homebrew/bin/python3.11 -m venv .venv-dreamer
.venv-dreamer/bin/pip install -r requirements-dreamer.lock
```

## Reproduce
```bash
# Tests
.venv-sb3/bin/python -m unittest discover -s tests -t .
# Random baseline and device benchmark
.venv-sb3/bin/python scripts/random_baseline.py --config configs/stage0_random.yaml
.venv-sb3/bin/python scripts/bench_device.py --config configs/stage0_bench_device.yaml
# Model-free methods: train, then evaluate every checkpoint
# (also ppo_lstm_250k, dqn_250k, dqn_nstep3_250k, ablation_sparse_ppo_cnn_250k, ppo_cnn_1m)
for s in 0 1 2; do .venv-sb3/bin/python scripts/train_sb3.py --config configs/ppo_cnn_250k.yaml --seed $s; done
# DreamerV3: train, convert logs, world-model error, harness eval of the last checkpoint
C=configs/dreamer_size1m_r32_100k.yaml
.venv-dreamer/bin/python scripts/train_dreamer.py --config $C --seed 0
.venv-sb3/bin/python scripts/dreamer_results.py --config $C --seed 0
.venv-dreamer/bin/python scripts/wm_eval.py --config $C --seed 0
.venv-dreamer/bin/python scripts/eval_dreamer.py --config $C --seed 0
# Tables and figures
.venv-sb3/bin/python plots/summarize_results.py --exp main
.venv-sb3/bin/python plots/plot_learning_curves.py --exp main
.venv-sb3/bin/python plots/plot_generalization.py --exp main
.venv-sb3/bin/python plots/plot_wm_error.py --exp main
.venv-sb3/bin/python plots/plot_ablation.py --method ppo_cnn
```

## Acknowledgements
AI coding assistants were used to help with parts of the implementation.

## References
- Hafner, *Benchmarking the Spectrum of Agent Capabilities* (Crafter), arXiv:2109.06780.
- Mnih et al., *Human-level control through deep reinforcement learning* (DQN), Nature 2015.
- Schulman et al., *Proximal Policy Optimization Algorithms*, arXiv:1707.06347.
- Hafner et al., *Mastering Diverse Domains through World Models* (DreamerV3), arXiv:2301.04104.
- Raffin et al., *Stable-Baselines3*, JMLR 2021.
