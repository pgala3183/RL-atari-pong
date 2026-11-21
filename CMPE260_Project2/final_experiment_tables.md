### E1: REINFORCE: first tuned config and variants (earlier)

| run                        | role      | change                                    | status     | frames to +15   | frames to solve   | Δ frames to solve   | wall-min to solve   |   best reward_100 |   final reward_100 | eval (stochastic)   |   learning-signal std | V(s) bias   | GPU                                |
|:---------------------------|:----------|:------------------------------------------|:-----------|:----------------|:------------------|:--------------------|:--------------------|------------------:|-------------------:|:--------------------|----------------------:|:------------|:-----------------------------------|
| reinforce_seed42           | reference | —                                         | solved     | 17,637,792      | 17,829,456        | —                   | 89.1                |             19    |              19    | +18.40 ± 5.14       |                 0.47  | —           | NVIDIA GeForce RTX 5080 Laptop GPU |
| reinforce_T256_seed42      | earlier   | REINFORCE T=256 (reference of R1-R4)      | solved     | 14,759,696      | 14,834,848        | -17%                | 77.3                |             19.04 |              19.04 | +20.23 ± 1.43       |                 0.441 | —           | NVIDIA GeForce RTX 5080 Laptop GPU |
| reinforce_lr1e-3_seed42    | earlier   | REINFORCE lr 1e-3 (T=512)                 | max_frames | —               | —                 | not solved          | —                   |            -20.02 |             -20.27 | -20.33 ± 0.87       |                 0.34  | —           | NVIDIA GeForce RTX 5080 Laptop GPU |
| reinforce_vbaseline_seed42 | earlier   | REINFORCE + learned V(s) baseline (T=512) | max_frames | —               | —                 | not solved          | —                   |              5.52 |               5.04 | +5.10 ± 3.21        |                 0.418 | -0.161      | NVIDIA GeForce RTX 5080 Laptop GPU |

### E2: A2C n=5 over three seeds (earlier)

| run        | role      | change           | status   |   frames to +15 |   frames to solve | Δ frames to solve   |   wall-min to solve |   best reward_100 |   final reward_100 | eval (stochastic)   |   learning-signal std |   V(s) bias | GPU                                |
|:-----------|:----------|:-----------------|:---------|----------------:|------------------:|:--------------------|--------------------:|------------------:|-------------------:|:--------------------|----------------------:|------------:|:-----------------------------------|
| a2c_seed42 | reference | —                | solved   |       2,500,672 |         3,894,752 | —                   |                33.3 |             19.02 |              19.02 | +19.70 ± 1.53       |                 0.115 |      -0.033 | NVIDIA GeForce RTX 5080 Laptop GPU |
| a2c_seed43 | earlier   | A2C n=5, seed 43 | solved   |       1,779,712 |         3,859,472 | -1%                 |                35.3 |             19    |              19    | +19.77 ± 1.23       |                 0.112 |       0.005 | NVIDIA GeForce RTX 5080 Laptop GPU |
| a2c_seed44 | earlier   | A2C n=5, seed 44 | solved   |       2,222,624 |         3,772,640 | -3%                 |                34.4 |             19    |              19    | +19.97 ± 1.33       |                 0.116 |      -0.051 | NVIDIA GeForce RTX 5080 Laptop GPU |

### E3: A2C variants of the n=5 config (earlier)

| run                | role      | change                             | status     | frames to +15   | frames to solve   | Δ frames to solve   | wall-min to solve   |   best reward_100 |   final reward_100 | eval (stochastic)   |   learning-signal std |   V(s) bias | GPU                                |
|:-------------------|:----------|:-----------------------------------|:-----------|:----------------|:------------------|:--------------------|:--------------------|------------------:|-------------------:|:--------------------|----------------------:|------------:|:-----------------------------------|
| a2c_seed42         | reference | —                                  | solved     | 2,500,672       | 3,894,752         | —                   | 33.3                |             19.02 |              19.02 | +19.70 ± 1.53       |                 0.115 |      -0.033 | NVIDIA GeForce RTX 5080 Laptop GPU |
| a2c_lin7e-4_seed42 | earlier   | A2C n=5, lr 7e-4 with linear decay | solved     | 2,884,208       | 3,372,448         | -13%                | 28.1                |             19    |              19    | +17.50 ± 2.17       |                 0.132 |       0.045 | NVIDIA GeForce RTX 5080 Laptop GPU |
| a2c_n20_seed42     | earlier   | A2C n=20 (reference of A1-A8)      | solved     | 3,140,192       | 3,397,472         | -13%                | 19.4                |             19    |              19    | +20.07 ± 1.98       |                 0.229 |      -0.066 | NVIDIA GeForce RTX 5080 Laptop GPU |
| a2c_32envs_seed42  | earlier   | A2C n=5, 32 envs                   | solved     | 3,959,232       | 4,108,544         | +5%                 | 22.0                |             19    |              19    | +17.90 ± 2.12       |                 0.134 |       0.166 | NVIDIA GeForce RTX 5080 Laptop GPU |
| a2c_noent_seed42   | earlier   | A2C n=5, no entropy bonus          | max_frames | —               | —                 | not solved          | —                   |              9.82 |             -21    | +9.30 ± 0.90        |                 0.077 |       0.052 | NVIDIA GeForce RTX 5080 Laptop GPU |

### R1: REINFORCE: seeds of the reference (T=256)

| run                   | role      | change   | status   | frames to +15   | frames to solve   | Δ frames to solve   | wall-min to solve   | best reward_100   | final reward_100   | eval (stochastic)   | learning-signal std   | V(s) bias   | GPU                                |
|:----------------------|:----------|:---------|:---------|:----------------|:------------------|:--------------------|:--------------------|:------------------|:-------------------|:--------------------|:----------------------|:------------|:-----------------------------------|
| reinforce_T256_seed42 | reference | —        | solved   | 14,759,696      | 14,834,848        | —                   | 77.3                | +19.04            | +19.04             | +20.23 ± 1.43       | 0.441                 | —           | NVIDIA GeForce RTX 5080 Laptop GPU |
| reinforce_T256_seed43 | new       | seed 43  | to run   | —               | —                 | —                   | —                   | —                 | —                  | —                   | —                     | —           | —                                  |
| reinforce_T256_seed44 | new       | seed 44  | to run   | —               | —                 | —                   | —                   | —                 | —                  | —                   | —                     | —           | —                                  |

### R2: REINFORCE: rollout length (T)

| run                   | role      | change                                        | status   | frames to +15   | frames to solve   | Δ frames to solve   | wall-min to solve   | best reward_100   | final reward_100   | eval (stochastic)   | learning-signal std   | V(s) bias   | GPU                                |
|:----------------------|:----------|:----------------------------------------------|:---------|:----------------|:------------------|:--------------------|:--------------------|:------------------|:-------------------|:--------------------|:----------------------|:------------|:-----------------------------------|
| reinforce_T256_seed42 | reference | —                                             | solved   | 14,759,696      | 14,834,848        | —                   | 77.3                | +19.04            | +19.04             | +20.23 ± 1.43       | 0.441                 | —           | NVIDIA GeForce RTX 5080 Laptop GPU |
| reinforce_T128_seed42 | new       | rollout 128                                   | to run   | —               | —                 | —                   | —                   | —                 | —                  | —                   | —                     | —           | —                                  |
| reinforce_seed42      | earlier   | REINFORCE T=512, lr 5e-4 (first tuned config) | solved   | 17,637,792      | 17,829,456        | +20%                | 89.1                | +19.00            | +19.00             | +18.40 ± 5.14       | 0.470                 | —           | NVIDIA GeForce RTX 5080 Laptop GPU |

### R3: REINFORCE: learning rate (T=256; lr 1e-3 was run at T=512)

| run                            | role      | change                    | status     | frames to +15   | frames to solve   | Δ frames to solve   | wall-min to solve   | best reward_100   | final reward_100   | eval (stochastic)   | learning-signal std   | V(s) bias   | GPU                                |
|:-------------------------------|:----------|:--------------------------|:-----------|:----------------|:------------------|:--------------------|:--------------------|:------------------|:-------------------|:--------------------|:----------------------|:------------|:-----------------------------------|
| reinforce_T256_seed42          | reference | —                         | solved     | 14,759,696      | 14,834,848        | —                   | 77.3                | +19.04            | +19.04             | +20.23 ± 1.43       | 0.441                 | —           | NVIDIA GeForce RTX 5080 Laptop GPU |
| reinforce_T256_lr2.5e-4_seed42 | new       | lr 2.5e-4                 | to run     | —               | —                 | —                   | —                   | —                 | —                  | —                   | —                     | —           | —                                  |
| reinforce_T256_lr7e-4_seed42   | new       | lr 7e-4                   | to run     | —               | —                 | —                   | —                   | —                 | —                  | —                   | —                     | —           | —                                  |
| reinforce_lr1e-3_seed42        | earlier   | REINFORCE lr 1e-3 (T=512) | max_frames | —               | —                 | not solved          | —                   | -20.02            | -20.27             | -20.33 ± 0.87       | 0.340                 | —           | NVIDIA GeForce RTX 5080 Laptop GPU |

### R4: REINFORCE ablation: return standardisation

| run                          | role      | change                    | status   | frames to +15   | frames to solve   | Δ frames to solve   | wall-min to solve   | best reward_100   | final reward_100   | eval (stochastic)   | learning-signal std   | V(s) bias   | GPU                                |
|:-----------------------------|:----------|:--------------------------|:---------|:----------------|:------------------|:--------------------|:--------------------|:------------------|:-------------------|:--------------------|:----------------------|:------------|:-----------------------------------|
| reinforce_T256_seed42        | reference | —                         | solved   | 14,759,696      | 14,834,848        | —                   | 77.3                | +19.04            | +19.04             | +20.23 ± 1.43       | 0.441                 | —           | NVIDIA GeForce RTX 5080 Laptop GPU |
| reinforce_T256_nonorm_seed42 | new       | no return standardisation | to run   | —               | —                 | —                   | —                   | —                 | —                  | —                   | —                     | —           | —                                  |

### A1: A2C: seeds of the reference (n=20)

| run            | role      | change   | status   | frames to +15   | frames to solve   | Δ frames to solve   | wall-min to solve   | best reward_100   | final reward_100   | eval (stochastic)   | learning-signal std   | V(s) bias   | GPU                                |
|:---------------|:----------|:---------|:---------|:----------------|:------------------|:--------------------|:--------------------|:------------------|:-------------------|:--------------------|:----------------------|:------------|:-----------------------------------|
| a2c_n20_seed42 | reference | —        | solved   | 3,140,192       | 3,397,472         | —                   | 19.4                | +19.00            | +19.00             | +20.07 ± 1.98       | 0.229                 | -0.066      | NVIDIA GeForce RTX 5080 Laptop GPU |
| a2c_n20_seed43 | new       | seed 43  | to run   | —               | —                 | —                   | —                   | —                 | —                  | —                   | —                     | —           | —                                  |
| a2c_n20_seed44 | new       | seed 44  | to run   | —               | —                 | —                   | —                   | —                 | —                  | —                   | —                     | —           | —                                  |

### A2: A2C: n-step length

| run            | role      | change                                | status   | frames to +15   | frames to solve   | Δ frames to solve   | wall-min to solve   | best reward_100   | final reward_100   | eval (stochastic)   | learning-signal std   | V(s) bias   | GPU                                |
|:---------------|:----------|:--------------------------------------|:---------|:----------------|:------------------|:--------------------|:--------------------|:------------------|:-------------------|:--------------------|:----------------------|:------------|:-----------------------------------|
| a2c_n20_seed42 | reference | —                                     | solved   | 3,140,192       | 3,397,472         | —                   | 19.4                | +19.00            | +19.00             | +20.07 ± 1.98       | 0.229                 | -0.066      | NVIDIA GeForce RTX 5080 Laptop GPU |
| a2c_seed42     | earlier   | A2C n=5 (first tuned config), seed 42 | solved   | 2,500,672       | 3,894,752         | +15%                | 33.3                | +19.02            | +19.02             | +19.70 ± 1.53       | 0.115                 | -0.033      | NVIDIA GeForce RTX 5080 Laptop GPU |
| a2c_n10_seed42 | new       | n=10                                  | to run   | —               | —                 | —                   | —                   | —                 | —                  | —                   | —                     | —           | —                                  |
| a2c_n40_seed42 | new       | n=40                                  | to run   | —               | —                 | —                   | —                   | —                 | —                  | —                   | —                     | —           | —                                  |
| a2c_n80_seed42 | new       | n=80                                  | to run   | —               | —                 | —                   | —                   | —                 | —                  | —                   | —                     | —           | —                                  |

### A3: A2C: GAE λ (n=20)

| run                    | role      | change     | status   | frames to +15   | frames to solve   | Δ frames to solve   | wall-min to solve   | best reward_100   | final reward_100   | eval (stochastic)   | learning-signal std   | V(s) bias   | GPU                                |
|:-----------------------|:----------|:-----------|:---------|:----------------|:------------------|:--------------------|:--------------------|:------------------|:-------------------|:--------------------|:----------------------|:------------|:-----------------------------------|
| a2c_n20_seed42         | reference | —          | solved   | 3,140,192       | 3,397,472         | —                   | 19.4                | +19.00            | +19.00             | +20.07 ± 1.98       | 0.229                 | -0.066      | NVIDIA GeForce RTX 5080 Laptop GPU |
| a2c_n20_gae0.9_seed42  | new       | GAE λ=0.9  | to run   | —               | —                 | —                   | —                   | —                 | —                  | —                   | —                     | —           | —                                  |
| a2c_n20_gae0.95_seed42 | new       | GAE λ=0.95 | to run   | —               | —                 | —                   | —                   | —                 | —                  | —                   | —                     | —           | —                                  |
| a2c_n20_gae0.98_seed42 | new       | GAE λ=0.98 | to run   | —               | —                 | —                   | —                   | —                 | —                  | —                   | —                     | —           | —                                  |

### A4: A2C: entropy coefficient (n=20; 'no entropy' was run at n=5)

| run                     | role      | change                    | status     | frames to +15   | frames to solve   | Δ frames to solve   | wall-min to solve   | best reward_100   | final reward_100   | eval (stochastic)   | learning-signal std   | V(s) bias   | GPU                                |
|:------------------------|:----------|:--------------------------|:-----------|:----------------|:------------------|:--------------------|:--------------------|:------------------|:-------------------|:--------------------|:----------------------|:------------|:-----------------------------------|
| a2c_n20_seed42          | reference | —                         | solved     | 3,140,192       | 3,397,472         | —                   | 19.4                | +19.00            | +19.00             | +20.07 ± 1.98       | 0.229                 | -0.066      | NVIDIA GeForce RTX 5080 Laptop GPU |
| a2c_n20_ent0.005_seed42 | new       | entropy 0.005             | to run     | —               | —                 | —                   | —                   | —                 | —                  | —                   | —                     | —           | —                                  |
| a2c_n20_ent0.02_seed42  | new       | entropy 0.02              | to run     | —               | —                 | —                   | —                   | —                 | —                  | —                   | —                     | —           | —                                  |
| a2c_n20_ent0.05_seed42  | new       | entropy 0.05              | to run     | —               | —                 | —                   | —                   | —                 | —                  | —                   | —                     | —           | —                                  |
| a2c_noent_seed42        | earlier   | A2C n=5, no entropy bonus | max_frames | —               | —                 | not solved          | —                   | +9.82             | -21.00             | +9.30 ± 0.90        | 0.077                 | +0.052      | NVIDIA GeForce RTX 5080 Laptop GPU |

### A5: A2C: learning rate (n=20)

| run                     | role      | change    | status   | frames to +15   | frames to solve   | Δ frames to solve   | wall-min to solve   | best reward_100   | final reward_100   | eval (stochastic)   | learning-signal std   | V(s) bias   | GPU                                |
|:------------------------|:----------|:----------|:---------|:----------------|:------------------|:--------------------|:--------------------|:------------------|:-------------------|:--------------------|:----------------------|:------------|:-----------------------------------|
| a2c_n20_seed42          | reference | —         | solved   | 3,140,192       | 3,397,472         | —                   | 19.4                | +19.00            | +19.00             | +20.07 ± 1.98       | 0.229                 | -0.066      | NVIDIA GeForce RTX 5080 Laptop GPU |
| a2c_n20_lr2.5e-4_seed42 | new       | lr 2.5e-4 | to run   | —               | —                 | —                   | —                   | —                 | —                  | —                   | —                     | —           | —                                  |
| a2c_n20_lr5e-4_seed42   | new       | lr 5e-4   | to run   | —               | —                 | —                   | —                   | —                 | —                  | —                   | —                     | —           | —                                  |
| a2c_n20_lr1e-3_seed42   | new       | lr 1e-3   | to run   | —               | —                 | —                   | —                   | —                 | —                  | —                   | —                     | —           | —                                  |

### A6: A2C: optimiser (n=20)

| run                 | role      | change                  | status   | frames to +15   | frames to solve   | Δ frames to solve   | wall-min to solve   | best reward_100   | final reward_100   | eval (stochastic)   | learning-signal std   | V(s) bias   | GPU                                |
|:--------------------|:----------|:------------------------|:---------|:----------------|:------------------|:--------------------|:--------------------|:------------------|:-------------------|:--------------------|:----------------------|:------------|:-----------------------------------|
| a2c_n20_seed42      | reference | —                       | solved   | 3,140,192       | 3,397,472         | —                   | 19.4                | +19.00            | +19.00             | +20.07 ± 1.98       | 0.229                 | -0.066      | NVIDIA GeForce RTX 5080 Laptop GPU |
| a2c_n20_adam_seed42 | new       | Adam instead of RMSprop | to run   | —               | —                 | —                   | —                   | —                 | —                  | —                   | —                     | —           | —                                  |

### A7: A2C: value-loss coefficient (n=20)

| run                      | role      | change          | status   | frames to +15   | frames to solve   | Δ frames to solve   | wall-min to solve   | best reward_100   | final reward_100   | eval (stochastic)   | learning-signal std   | V(s) bias   | GPU                                |
|:-------------------------|:----------|:----------------|:---------|:----------------|:------------------|:--------------------|:--------------------|:------------------|:-------------------|:--------------------|:----------------------|:------------|:-----------------------------------|
| a2c_n20_seed42           | reference | —               | solved   | 3,140,192       | 3,397,472         | —                   | 19.4                | +19.00            | +19.00             | +20.07 ± 1.98       | 0.229                 | -0.066      | NVIDIA GeForce RTX 5080 Laptop GPU |
| a2c_n20_vcoef0.25_seed42 | new       | value coef 0.25 | to run   | —               | —                 | —                   | —                   | —                 | —                  | —                   | —                     | —           | —                                  |
| a2c_n20_vcoef1_seed42    | new       | value coef 1.0  | to run   | —               | —                 | —                   | —                   | —                 | —                  | —                   | —                     | —           | —                                  |

### A8: A2C: shared vs separate actor-critic network (n=20)

| run                     | role      | change                  | status   | frames to +15   | frames to solve   | Δ frames to solve   | wall-min to solve   | best reward_100   | final reward_100   | eval (stochastic)   | learning-signal std   | V(s) bias   | GPU                                |
|:------------------------|:----------|:------------------------|:---------|:----------------|:------------------|:--------------------|:--------------------|:------------------|:-------------------|:--------------------|:----------------------|:------------|:-----------------------------------|
| a2c_n20_seed42          | reference | —                       | solved   | 3,140,192       | 3,397,472         | —                   | 19.4                | +19.00            | +19.00             | +20.07 ± 1.98       | 0.229                 | -0.066      | NVIDIA GeForce RTX 5080 Laptop GPU |
| a2c_n20_separate_seed42 | new       | separate critic network | to run   | —               | —                 | —                   | —                   | —                 | —                  | —                   | —                     | —           | —                                  |
