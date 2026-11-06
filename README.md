# RL on Atari Pong: REINFORCE and A2C vs DQN (CMPE 260, Project 2)

Policy-gradient agents for `ALE/Pong-v5`, compared against the DQN agents from Project 1.
The environment, preprocessing, network torso, seed (42), γ = 0.99 and "solved" rule
(mean reward of the last 100 episodes ≥ 19) are identical to Project 1. "Frames" means agent steps;
each agent step is 4 emulator frames.

## Contents

| Path | What it is |
|---|---|
| `CMPE260_Project2_PG_Pong.ipynb` | The notebook: REINFORCE, A2C, hyper-parameter sweeps, evaluation, comparison |
| `CMPE260_Project2_PG_Pong_executed.ipynb` | The same notebook with all outputs from the training run |
| `pong_envs.py` | Atari wrappers (written by the notebook; needed by Windows worker processes) |
| `CMPE260_Project2/` | Results of the run: logs, tables, plots, final models |

Inside `CMPE260_Project2/`:

* `reinforce_seed42/`, `a2c_seed42/`: the final runs. Each has `results.json`, `episodes.npz`
  (every episode's frame, reward, length and wall-clock time), TensorBoard logs and `best.pt`
  (the trained model).
* `sweep_*`: the hyper-parameter trials (logs only), with summaries in `sweep_reinforce.csv` and `sweep_a2c.csv`.
* `eval_*.json`: 30-episode evaluations, including the value-estimate bias measurement.
* `comparison_table.md` / `.csv`, `learning_curves.png`, `value_bias.png`: the comparison.

Resume checkpoints (`latest.pt`) and the sweep-trial models are not included because of their size.

## Results (single seed)

Training hardware: NVIDIA RTX 5080 Laptop GPU, 24 CPU cores, 16 parallel environments.

| | REINFORCE | A2C |
|---|---|---|
| Tuned settings | Adam, lr 5e-4, 16 envs × 512 steps per update | RMSprop, lr 3.5e-4, 16 envs × 5 steps per update |
| Status | not solved in 15M frames | **solved** |
| Frames to +15 | — | 2.50M |
| Frames to solve (≥ 19) | — | 3.89M |
| Best / final mean-100 reward | −8.73 / −10.15 | +19.02 / +19.02 |
| Wall-clock | 74.9 min | 33.3 min |
| Gradient updates | 1,832 | 48,684 |
| Evaluation (30 games, sampling from the policy) | −14.93 ± 2.42 | +19.70 ± 1.53 |
| Value-estimate bias, V(s) − actual return | n/a (no critic) | −0.033 |

Project 1 DQN, for reference: the ε-greedy baseline reached +15 at 0.76M frames (best mean-100 reward
18.98 in 2M frames), and NoisyNet + PER solved at 0.38M frames. DQN needs the fewest frames because it
replays each transition about 32 times. A2C reaches the same final score with more frames but in less
wall-clock time thanks to parallel environments. REINFORCE, which has no critic, learns slowest
because of its high-variance Monte-Carlo gradient.

![Learning curves](CMPE260_Project2/learning_curves.png)

Caveats: one seed per method. The REINFORCE sweep trials (3M frames each) all ended before
REINFORCE began improving, so they could not separate the learning rates. Wall-clock times are not
directly comparable with Project 1, which ran on Colab GPUs with one environment.

## Running it

Requirements: Python 3.10–3.12, an NVIDIA GPU, and the CUDA build of PyTorch:

```
pip install torch --index-url https://download.pytorch.org/whl/cu128
```

Open the notebook from the repository folder and run all cells; the install cell adds the remaining
packages. Runs checkpoint regularly and resume when the cell is re-run.
