# Latent-state models of training dynamics — blog post code

Code behind a blog post series on describing neural network training trajectories with
latent-state models (HMMs, GMMs and linear dynamical systems fitted to per-checkpoint training
metrics).

One directory per post, carrying the analysis each post reports: the model fitting, the
regression and the figures. Training runs are not distributed here — you produce them from the
original authors' code and point the analysis at them. The posts themselves live with the blog.

| Post | Directory | Subject |
| --- | --- | --- |
| 1 | [post_1_useful_states/](post_1_useful_states/) | Whether discrete latent states carry information about training beyond simple summary statistics, on sparse parity and modular addition |


## Running it

```bash
cd post_1_useful_states
python3 -m pip install -r requirements.txt
python3 convergence_regression.py <your runs> --runs 60 --horizon 300
```

Python 3.10 or newer. Figures are written into `assets/`, which is generated rather than
committed.

Train the cohorts first, with the original authors'
[modeling-training](https://github.com/michahu/modeling-training) and the configurations in each
post's README, which also gives the CSV layout the analysis expects.