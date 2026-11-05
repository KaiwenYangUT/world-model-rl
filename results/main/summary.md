Final = last 2 eval checkpoints x all seeds, pooled. Target = place_table. Random policy (train, n=200): success 2.0% [0.8%, 5.0%], return 1.32.

| Method | Seeds | Steps | Train success [95% CI] | Held-out success [95% CI] | Gap (pp) | Train return | Crafter score (train) | AUC | Steps to 50% | Gate (a) / (b) |
|---|---|---|---|---|---|---|---|---|---|---|
| PPO | 3 | 250k | 62.0% [56.4%, 67.3%] (n=300) | 52.0% [46.4%, 57.6%] (n=300) | +10.0 | 4.42 [4.23, 4.61] | 5.40 | 38.5% | 80000 | pass / pass |
| PPO-LSTM | 3 | 250k | 45.0% [39.5%, 50.7%] (n=300) | 40.3% [34.9%, 46.0%] (n=300) | +4.7 | 3.75 [3.60, 3.90] | 3.98 | 30.3% | 50000 | pass / pass |
| DQN | 3 | 250k | 16.3% [12.6%, 20.9%] (n=300) | 17.7% [13.8%, 22.4%] (n=300) | -1.3 | 2.42 [2.23, 2.60] | 2.71 | 5.5% | 160000 | pass / pass |
| DreamerV3 | 1 | 96k | 0.0% [0.0%, 7.1%] (n=50) | 2.0% [0.4%, 10.5%] (n=50) | -2.0 | 1.34 [1.01, 1.67] | 1.23 | 3.4% | 278 | fail / fail |
| DQN 3-step | 3 | 250k | 47.0% [41.4%, 52.7%] (n=300) | 37.0% [31.7%, 42.6%] (n=300) | +10.0 | 3.88 [3.74, 4.02] | 3.79 | 16.5% | 190000 | pass / pass |
