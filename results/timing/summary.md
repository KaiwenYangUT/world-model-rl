Idle M2 Max, only job, on AC (runs/logs/timing_load.log). One run each; training time only (no eval). SB3: steady per-10k segments; DreamerV3: steady fps after prefill. 1M projections are extrapolations.

| Method | Device | Steps run | Training s / 10k steps | Steps / s | Projected h / 100k | Projected h / 1M | Peak memory (GB) | Detail |
|---|---|---|---|---|---|---|---|---|
| ppo_cnn | mps | 50176 | 6.2 | 1609 | 0.017 | 0.17 |  | 6.6 6.2 6.2 6.2 6.1 |
| ppo_lstm | mps | 50176 | 16.2 | 616 | 0.045 | 0.45 |  | 16.0 16.7 16.8 15.8 15.0 |
| dqn | mps | 50000 | 27.9 | 358 | 0.078 | 0.78 |  | 6.9 28.2 28.0 27.9 27.9 |
| dqn_nstep3 | mps | 50000 | 27.9 | 359 | 0.077 | 0.77 |  | 7.0 27.8 28.0 27.8 28.0 |
| dreamerv3_size12m_r128 | cpu (jax) | 1944 | 6544.4 | 1.53 | 18.2 | 181.8 | 6.5 | fps/train 196 (size1m: 764) |
