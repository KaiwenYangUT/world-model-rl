Final = last 2 eval checkpoints x all seeds, pooled. Target = place_table. Random policy (train, n=200): success 2.0% [0.8%, 5.0%], return 1.32.

| Method | Seeds | Steps | Train success [95% CI] | Held-out success [95% CI] | Gap (pp) | Train return | Crafter score (train) | AUC | Steps to 50% | Gate (a) / (b) |
|---|---|---|---|---|---|---|---|---|---|---|
| PPO | 3 | 250k | 96.0% [93.1%, 97.7%] (n=300) | 82.3% [77.6%, 86.2%] (n=300) | +13.7 | 0.96 [0.94, 0.98] | 0.98 | 56.4% | 100000 | fail / pass |
| DreamerV3 ratio 128 | 3 | 100k | 17.7% [13.8%, 22.4%] (n=300) | 17.0% [13.2%, 21.7%] (n=300) | +0.7 | 0.18 [0.13, 0.22] | 2.30 | 6.6% | 90000 | fail / pass |
