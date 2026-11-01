# Post 1: what are latent state models for training dynamics useful for?

The post asks whether a Gaussian HMM's transition structure is needed to predict convergence
time, or whether a static Gaussian mixture over the same weight statistics does as well.

This repository expects data produced from [Hu et al.](https://arxiv.org/abs/2308.09543v3)

## Before you start

```bash
python3 -m pip install -r requirements.txt
```

You need Python 3.10 or newer. The pinned versions are the ones used to produce the post's
results.

| | |
| --- | --- |
| `parity_models.py` | fits the HMM and GMM, and turns each into per-run occupancy features |
| `convergence_regression.py` | runs the five-fold convergence-time regression |
| `make_state_figures.py` | draws the state rasters, training maps, and L1 mechanism figure |

## 1. Train the cohorts

```bash
git clone https://github.com/michahu/modeling-training.git
```

Train the runs below with Hu et al.'s code. This project's training code and metric logging came
from there. `convergence_regression.py` reimplements their regression from the paper's description.

The paper is Hu, Chen, Saphra and Cho, *Latent State Models of Training Dynamics*, TMLR 2023
([arXiv](https://arxiv.org/abs/2308.09543v3)). Section 2.1 covers the metric trajectories and
per-run standardization, and section 2.3 covers the state-count regression.

Train one cohort for each task:

**Sparse parity, 60 runs.** Seeds 0–59, 300 epochs each, all 300 checkpoints logged. 40 input
bits, parity of the first 3, 1,000 training and 100 test examples, a one-hidden-layer MLP of
width 1,000, SGD at learning rate 0.1 with weight decay 0.01, batch size 32. Save and restore the
Python, NumPy and Torch RNG state around evaluation and metric logging, so that logging does not
change the training path.

**Modular addition, 40 runs.** AdamW at learning rate 1e-3, unit initialization scaling, 500
checkpoints logged every tenth epoch.

**File format.** Give each cohort its own directory, with one file per run named `seed<N>.csv` or
`seed<N>.csv.gz`, numbered from 0. Each file needs the 14 weight statistics per checkpoint (`l1`,
`l2`, `trace`, `spectral`, `code_sparsity`, `computational_sparsity`, `mean_lambda`,
`variance_lambda`, `mean_w`, `median_w`, `var_w`, `mean_b`, `median_b`, `var_b`), plus `step` and
`eval_accuracy`. `step` must be strictly increasing. The models take the weight statistics as
inputs, and `eval_accuracy` gives the convergence label.

## 2. Run the regression

```bash
python3 convergence_regression.py <parity runs>  --runs 60 --horizon 300
python3 convergence_regression.py <modular runs> --runs 40 --horizon 500
```

This is the regression behind the post's numbers. `--runs` is the number of runs in the cohort, and
`--horizon` is the number of checkpoints kept per run. It prints the five-fold held-out R² for the
HMM, the GMM, the averaged metrics and the averaged L1 norm alone.

What it does:

- Uses K = 6 latent states.
- Runs two restarts per fit, with 60 EM iterations each, and picks the restart using training
  likelihood only.
- The regression features are HMM occupancy, GMM occupancy, the 14 weight statistics averaged over
  each run, and the averaged L1 norm alone. The averages are computed before per-run centering, so
  they keep absolute scale.
- The convergence label is the first logged evaluation epoch that reaches 90% accuracy, read from
  the whole run even when `--horizon` is shorter. Every run must reach it, or the script stops with
  an error.
- Each training fold refits the latent model and the regression (48/12 runs for parity, 32/8 for
  modular).
- R² is computed over the held-out predictions for each model and fit seed, then averaged across
  fit seeds 0, 1 and 2.

Each run is standardized using all of its retained checkpoints, including for held-out runs. That
means this analyzes finished runs and can't forecast from a partial one.

## 3. Draw the figures

```bash
python3 make_state_figures.py --data <dir>        # holding parity/ and modular/
```

`make_state_figures.py` draws the latent states. It fits the HMM and GMM on all of your runs with
the same fitter as the regression, using fit seed 0, then produces:

- `state_raster`: the decoded state sequence of every run over training, for both tasks, with a
  zoomed modular panel.
- `training_map`: the HMM transition graph and state occupancies for each task.
- `l1_mechanism`: shows why a single averaged weight statistic predicts convergence on sparse
  parity. It plots L1 trajectories colored by convergence epoch, their spread at a fixed epoch
  against their spread aligned on convergence, and each run's average L1 against its convergence
  epoch, labeled with the correlation and the five-fold R².

State labels are arbitrary, so the raster and the map relabel states by mean epoch before drawing.
The script writes to `assets/`, creating it if needed.
