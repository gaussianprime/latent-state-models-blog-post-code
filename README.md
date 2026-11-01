# Latent-state models of training dynamics — blog post code

Code behind a blog post series on describing neural network training trajectories with latent-state models.

There is one directory per post. This repository expects data produced from
[Hu et al.](https://arxiv.org/abs/2308.09543v3)

| Post | Directory | Subject |
| --- | --- | --- |
| 1 | [post_1_useful_states/](post_1_useful_states/) | Whether an HMM's states predict convergence time better than a GMM's states or simple averages, on sparse parity and modular addition |


## Running it

Train the runs first with [modeling-training](https://github.com/michahu/modeling-training). Each
post's README lists the settings and the CSV format.

```bash
cd post_1_useful_states
python3 -m pip install -r requirements.txt
python3 convergence_regression.py <your runs> --runs 60 --horizon 300
```

Python 3.10 or newer. Figures are written to `assets/`.