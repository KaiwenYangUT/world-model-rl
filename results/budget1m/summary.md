Final = last 2 eval checkpoints x all seeds, pooled. Target = place_table. Random policy (train, n=200): success 2.0% [0.8%, 5.0%], return 1.32.

| Method | Seeds | Steps | Train success [95% CI] | Held-out success [95% CI] | Gap (pp) | Train return | Crafter score (train) | AUC | Steps to 50% | Gate (a) / (b) |
|---|---|---|---|---|---|---|---|---|---|---|
| PPO | 1 | 1000k | 94.0% [87.5%, 97.2%] (n=100) | 87.0% [79.0%, 92.2%] (n=100) | +7.0 | 7.61 [7.21, 8.01] | 10.95 | 74.1% | 100000 | pass / pass |
