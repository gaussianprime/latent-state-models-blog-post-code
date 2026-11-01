"""Reproduce the post's convergence-time regression from per-run trajectories.

Given a directory of per-run CSVs, this fits the HMM and the GMM at K = 6 states,
regresses convergence time on their occupancy features, on the run-averaged weight
statistics, and on the run-averaged L1 norm alone, and reports five-fold held-out R².

This is a reimplementation of the method described in §2.3 of Hu et al. (2023):

    python3 convergence_regression.py <parity runs>  --runs 60 --horizon 300
    python3 convergence_regression.py <modular runs> --runs 40 --horizon 500

Each CSV must be named seed<N>.csv or seed<N>.csv.gz and carry the weight-statistic
columns plus `step` and `eval_accuracy`.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from parity_models import WEIGHT_COLUMNS, fit_model, occupancy, zscore_run

K = 6
THRESHOLD = 0.9
FIT_SEEDS = (0, 1, 2)
READOUTS = ("hmm", "gmm", "raw_mean", "l1_mean")


def read_run(csvdir, seed):
    path = csvdir / f"seed{seed}.csv.gz"
    return pd.read_csv(path if path.exists() else csvdir / f"seed{seed}.csv")


def load_runs(csvdir, n, horizon):
    """Per-run weight statistics and convergence labels.

    The label is the first checkpoint reaching the accuracy threshold, read from
    the complete record; the weight statistics are the retained window.
    """
    raw, labels = [], []
    for seed in range(n):
        df = read_run(csvdir, seed)
        labels.append(float(df.loc[df.eval_accuracy >= THRESHOLD, "step"].iloc[0]))
        raw.append(df.iloc[:horizon][WEIGHT_COLUMNS].to_numpy(float))
    return raw, np.asarray(labels)


def regression_predictions(xtrain, xtest, ytrain):
    reg = make_pipeline(StandardScaler(), LinearRegression())
    return reg.fit(xtrain, ytrain).predict(xtest)


def averages_r2(averages, y, fit_seed):
    """Held-out R² of features averaged over each run, which need no model fit."""
    predicted = np.empty(len(y))
    for tr, te in KFold(5, shuffle=True, random_state=fit_seed).split(y):
        predicted[te] = regression_predictions(averages[tr], averages[te], y[tr])
    return r2_score(y, predicted)


def five_fold_r2(raw, y, fit_seed):
    """Held-out R² per readout, refitting the models and regression on each training fold."""
    runs = [zscore_run(x) for x in raw]
    predicted = {kind: np.empty(len(y)) for kind in ("hmm", "gmm")}
    for fold, (tr, te) in enumerate(KFold(5, shuffle=True, random_state=fit_seed).split(y)):
        train_runs, test_runs = [runs[i] for i in tr], [runs[i] for i in te]
        for kind in predicted:
            model = fit_model(kind, train_runs, k=K, seed=100 * fit_seed + fold)
            predicted[kind][te] = regression_predictions(occupancy(model, train_runs),
                                                         occupancy(model, test_runs), y[tr])
    # Averaged before per-run standardization, which would remove the mean.
    averages = np.stack([x.mean(0) for x in raw])
    l1 = averages[:, [WEIGHT_COLUMNS.index("l1")]]
    return {"hmm": r2_score(y, predicted["hmm"]), "gmm": r2_score(y, predicted["gmm"]),
            "raw_mean": averages_r2(averages, y, fit_seed), "l1_mean": averages_r2(l1, y, fit_seed)}


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("data", type=Path, help="directory of seed<N>.csv[.gz] trajectories")
    parser.add_argument("--runs", type=int, required=True, help="number of runs in the cohort")
    parser.add_argument("--horizon", type=int, required=True, help="checkpoints retained per run")
    args = parser.parse_args()

    raw, y = load_runs(args.data, args.runs, args.horizon)
    scores = pd.DataFrame([five_fold_r2(raw, y, fit_seed) for fit_seed in FIT_SEEDS])
    print(f"Five-fold held-out R² over {len(y)} runs, K={K}, horizon {args.horizon}, "
          f"mean across {len(FIT_SEEDS)} fit seeds\n")
    for readout in READOUTS:
        print(f"{readout:<10} {scores[readout].mean():.4f}")


if __name__ == "__main__":
    main()
