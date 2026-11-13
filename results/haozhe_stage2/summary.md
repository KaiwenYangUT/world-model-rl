Final = last 2 eval checkpoints x all seeds, pooled. Target = place_table. Random policy (train, n=200): success 2.0% [0.8%, 5.0%], return 1.32.

| Method | Seeds | Steps | Train success [95% CI] | Held-out success [95% CI] | Gap (pp) | Train return | Crafter score (train) | AUC | Steps to 50% | Gate (a) / (b) |
|---|---|---|---|---|---|---|---|---|---|---|
| dreamerv3_size12m_r128_crafter_500k | 3 | 500k | 93.3% [89.9%, 95.6%] (n=300) | 77.0% [71.9%, 81.4%] (n=300) | +16.3 | 6.47 [6.26, 6.68] | 7.69 | 80.9% | 250000 | pass / pass |
| dreamerv3_size1m_r128_crafter_500k | 3 | 500k | 1.7% [0.7%, 3.8%] (n=300) | 3.0% [1.6%, 5.6%] (n=300) | -1.3 | 3.04 [2.92, 3.16] | 2.41 | 3.9% | 0 | pass / fail |
| ppo_cnn_500k | 3 | 500k | 88.0% [83.8%, 91.2%] (n=300) | 75.0% [69.8%, 79.6%] (n=300) | +13.0 | 5.91 [5.69, 6.14] | 8.73 | 56.7% | 150000 | pass / pass |
