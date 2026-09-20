# Post 1: what are latent state models for training dynamics useful for?

The post asks whether a Gaussian HMM's transition structure is needed to predict convergence
time, or whether a static Gaussian mixture over the same weight statistics does as well.

This repository carries the analysis, not the data. Training runs are not distributed here: you
produce them with the original authors' code and point the analysis at them. Three steps, all of
which run on a cohort of your own.

## Before you start

```bash
python3 -m pip install -r requirements.txt
```

Python 3.10 or newer. The pinned versions are the ones the post's results were produced with.

| | |
| --- | --- |
| `parity_models.py` | fits the latent state models — HMM, GMM, LDS, ARHMM — and turns each into per-run features |
| `convergence_regression.py` | the convergence-time regression and its two evaluation protocols |
| `make_state_figures.py` | state rasters, training maps, and the L1 mechanism figure |
| `make_figure.py` | the occupancy comparison figure and tables |

## 1. Train the cohorts

```bash
git clone https://github.com/michahu/modeling-training.git
```

That repository is where this project's training code and metric instrumentation came from, and
it is where to start. Step 2 is not a rerun of released code: the convergence regression in
`convergence_regression.py` is a reimplementation of the method as written, which is why it
ships here. The paper is Hu, Chen, Saphra and Cho, *Latent State Models of Training Dynamics*,
TMLR 2023 ([arXiv](https://arxiv.org/abs/2308.09543v3)); §2.1 covers the metric trajectories and
per-run standardization, §2.3 the state-count regression.

Two cohorts, trained separately and never pooled.

**Sparse parity, 60 runs.** Seeds 0–59, 300 epochs each, all 300 checkpoints logged. 40 input
bits, parity of the first 3, 1,000 training and 100 test examples, a one-hidden-layer MLP of
width 1,000, SGD at learning rate 0.1 with weight decay 0.01, batch size 32. Preserve Python,
NumPy and Torch RNG state across evaluation and metric probes, so the logging cadence does not
change the training path.

**Modular addition, 40 runs.** AdamW at learning rate 1e-3, unit initialization scaling, 500
checkpoints logged every tenth epoch.

**What each cohort must look like.** One directory per cohort, holding one file per run named
`seed<N>.csv` or `seed<N>.csv.gz`, numbered from 0. Each file needs the 14 weight statistics per
checkpoint — `l1`, `l2`, `trace`, `spectral`, `code_sparsity`, `computational_sparsity`,
`mean_lambda`, `variance_lambda`, `mean_w`, `median_w`, `var_w`, `mean_b`, `median_b`, `var_b` —
plus `step` and `eval_accuracy`, with `step` strictly increasing. The weight statistics are the
model inputs; accuracy supplies the convergence label.

A cohort trained this way will not match the post checkpoint for checkpoint. Compare the
pattern, not the digits.

## 2. Run the regression

```bash
python3 convergence_regression.py <parity runs>  --runs 60 --horizon 300
python3 convergence_regression.py <modular runs> --runs 40 --horizon 500
```

This is the step that produces the post's numbers. It prints held-out R² per readout under both
protocols, as a mean and sample SD across fit seeds.

| flag | |
| --- | --- |
| `--runs` | number of runs in the cohort (required) |
| `--horizon` | checkpoints retained per run (required) |
| `--stride` | keep every *n*th checkpoint within the horizon; default 1 |
| `--k` | latent states, and the LDS dimension; default 6 |
| `--models` | families to fit; default `hmm gmm`. Raw-mean baselines are always included |
| `--subset` | which columns to model: `weight`, `functional`, or `weight+functional` |
| `--fit-seeds` | fit/split seeds to average over; default `0 1 2` |
| `--threshold` | convergence accuracy; default 0.9 |
| `--label` | cohort name recorded in the output; defaults to the directory name |
| `--out` | write the summary scores to a CSV — the input to `make_figure.py` |

What it does:

- Latent state models at K = 6 states, LDS latent dimension M = 6 to match.
- Two restarts per fit, 60 EM iterations each, restarts selected on **training** likelihood only.
- Readouts: HMM, GMM and ARHMM occupancy; LDS mean; LDS mean plus temporal SD; quantized LDS
  occupancy; and raw-mean and raw-mean-plus-SD baselines with no latent model at all. Raw
  summaries are taken before per-run centering, so they keep absolute scale.
- Convergence label: the first logged evaluation epoch reaching the threshold, read from the
  complete record. Runs that never cross are reported and excluded, not silently dropped.
- Two evaluation protocols on the same folds: fully inductive five-fold, which refits both
  representation and readout on each training fold (48/12 runs for parity, 32/8 for modular),
  and shared-representation leave-one-out, which fits the representation on all runs and holds
  out only the regression label.
- R² over the held-out predictions of each protocol/model/seed group, then mean and sample SD
  across fit seeds. A group that did not predict every run is left unscored rather than scored
  on the subset that happened to succeed.

Per-run standardization uses each run's complete analyzed trajectory, including for held-out
runs. Both protocols are therefore retrospective analyses of completed runs, not forecasts.

## 3. Draw the figures

Both scripts run on the same cohorts as step 2. Nothing is loaded from a previous study: the
figures are drawn from your runs and from scores you just computed.

```bash
python3 convergence_regression.py <parity runs>  --runs 60 --horizon 300 \
    --label "Sparse parity"    --out parity.csv
python3 convergence_regression.py <modular runs> --runs 40 --horizon 500 \
    --label "Modular addition" --out modular.csv
python3 make_figure.py parity.csv modular.csv
python3 make_state_figures.py --data <dir>        # holding parity/ and modular/
```

`make_figure.py` draws the occupancy comparison — one panel per cohort and protocol — plus a
comparison table and, per cohort, a table of every readout with the conditional density of the
fit behind it. It fits nothing; every number comes from the scores you pass it.

`make_state_figures.py` draws the latent states rather than the regression scores. It refits the
HMM and GMM with the same fitter and seed the regression uses, then produces:

- `state_raster` — the decoded state sequence of every run over training, for both tasks, with a
  zoomed modular panel.
- `training_map` — the HMM transition graph and state occupancies, per task.
- `l1_mechanism` — why a single averaged weight statistic predicts convergence on sparse parity:
  L1 trajectories coloured by convergence epoch, their spread at a fixed epoch against their
  spread aligned on convergence, and the correlation between each run's average L1 and its
  convergence epoch.

State labels are arbitrary in any fit, so the raster and the map relabel states by mean epoch
before drawing. `--root <research repository>` reads that repository's sweep outputs instead of
a trajectory directory. Both scripts write into `assets/`, which they create.

## Scope

Cohort size and cadence differ between the tasks: parity has 60 runs logged every epoch over a
300-checkpoint window, modular 40 runs logged every tenth epoch over 500. Matching the analysis
protocol does not make these matched experimental conditions, and absolute density scores are not
comparable across the two. The fit/split-seed SD is a spread over fit seeds, not a confidence
interval over training runs.
