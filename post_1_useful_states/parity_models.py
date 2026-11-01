"""Fit the HMM and the GMM and turn each into per-run occupancy features.

Restarts are selected using training likelihood only.
"""

import numpy as np
from sklearn.mixture import GaussianMixture
from hmmlearn.hmm import GaussianHMM


WEIGHT_COLUMNS = ["l1", "l2", "trace", "spectral", "code_sparsity",
                  "computational_sparsity", "mean_lambda", "variance_lambda",
                  "mean_w", "median_w", "var_w", "mean_b", "median_b", "var_b"]


def zscore_run(x):
    sd = x.std(axis=0)
    return (x - x.mean(axis=0)) / np.where(sd > 0, sd, 1.0)


def occupancy(model, runs):
    """The fraction of each run's checkpoints in each state."""
    return np.stack([np.bincount(model.predict(x), minlength=model.n_components) / len(x)
                     for x in runs])


def _fit_once(kind, runs, k, seed, n_iter):
    if kind == "gmm":
        return GaussianMixture(k, covariance_type="diag", n_init=1, max_iter=n_iter,
                               reg_covar=1e-6, random_state=seed).fit(np.vstack(runs))
    return GaussianHMM(k, covariance_type="diag", n_iter=n_iter, min_covar=1e-6,
                       random_state=seed).fit(np.vstack(runs), [len(x) for x in runs])


def _loglik(kind, model, x):
    # GaussianMixture.score averages over samples; GaussianHMM.score sums them.
    return model.score_samples(x).sum() if kind == "gmm" else model.score(x)


def fit_model(kind, training_runs, k, seed, n_iter=60, n_starts=2):
    """Only training data is accepted: test scores cannot select a restart."""
    best, best_ll = None, -np.inf
    for restart in range(n_starts):
        model = _fit_once(kind, training_runs, k, seed + 10000 * restart, n_iter)
        ll = sum(_loglik(kind, model, x) for x in training_runs)
        if ll > best_ll:
            best, best_ll = model, ll
    return best
