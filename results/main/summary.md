Final = last 2 eval checkpoints x all seeds, pooled. Target = place_table. Random policy (train, n=200): success 2.0% [0.8%, 5.0%], return 1.32.

| Method | Seeds | Steps | Train success [95% CI] | Held-out success [95% CI] | Gap (pp) | Train return | Crafter score (train) | AUC | Steps to 50% | Gate (a) / (b) |
|---|---|---|---|---|---|---|---|---|---|---|
| PPO | 3 | 250k | 62.0% [56.4%, 67.3%] (n=300) | 52.0% [46.4%, 57.6%] (n=300) | +10.0 | 4.42 [4.23, 4.61] | 5.40 | 38.5% | 80000 | pass / pass |
