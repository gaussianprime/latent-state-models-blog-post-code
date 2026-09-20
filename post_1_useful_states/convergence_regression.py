"""Reproduce the post's convergence-time regression from per-run trajectories.

Given a directory of per-run CSVs, this fits latent state models at K states, turns
them into per-run features, regresses convergence time on those features, and
reports held-out R² under both evaluation protocols.

This is the reimplementation of the method described in §2.3 of Hu et al. (2023);
it is the procedure the post reports, lifted from the research repository's study
driver so that it runs against any cohort:

    python3 convergence_regression.py <parity runs>  --runs 60 --horizon 300
    python3 convergence_regression.py <modular runs> --runs 40 --horizon 500

Each CSV must be named seed<N>.csv or seed<N>.csv.gz and carry the weight-statistic
columns plus `step` and `eval_accuracy`. Runs that never reach the accuracy
threshold are reported and excluded from the regression, not silently dropped.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, LeaveOneOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from parity_models import SUBSETS, WEIGHT_COLUMNS, fit_model, raw_features, zscore_run

PROTOCOLS = ("shared_representation_loo", "inductive_5fold")


def read_run(csvdir, seed):
    path = csvdir / f"seed{seed}.csv.gz"
    return pd.read_csv(path if path.exists() else csvdir / f"seed{seed}.csv")


def load_runs(csvdir, n, horizon, stride, subset, threshold):
    """Per-run model inputs, raw inputs and convergence labels.

    The label is the first checkpoint reaching the accuracy threshold, read from
    the complete record; the model input is the retained window, standardized
    within the run.
    """
    raw, normalized, seeds, labels = [], [], [], []
    for seed in range(n):
        df = read_run(csvdir, seed)
        if df.step.duplicated().any() or not df.step.is_monotonic_increasing:
            raise ValueError(f"Invalid checkpoint sequence for seed {seed}")
        hit = df.loc[df.eval_accuracy >= threshold, "step"].to_list()
        labels.append(float(hit[0]) if hit else np.nan)
        x = df.iloc[:horizon:stride][SUBSETS[subset]].to_numpy(float)
        raw.append(x)
        normalized.append(zscore_run(x))
        seeds.append(seed)
    return raw, normalized, np.asarray(seeds), np.asarray(labels)


def regression_predictions(xtrain, xtest, ytrain):
    reg = make_pipeline(StandardScaler(), LinearRegression())
    return reg.fit(xtrain, ytrain).predict(xtest)


def evaluate(raw, runs, seeds, y, k, fit_seed, kinds):
    """Held-out predictions for every readout, under both protocols.

    The inductive protocol also scores each fitted model's conditional likelihood
    on the very same held-out fold, so density and regression describe one fit.
    """
    predictions, density, failures = [], [], []
    summary = raw_features(raw)
    for protocol in PROTOCOLS:
        shared = {}
        if protocol == "shared_representation_loo":
            for kind in kinds:
                try:
                    m = fit_model(kind, runs, k=k, seed=fit_seed)
                    shared[kind] = m.features(runs, training_runs=runs)
                except Exception as exc:
                    failures.append(dict(protocol=protocol, fold="all", model=kind, error=str(exc)))
            splits = LeaveOneOut().split(seeds)
        else:
            splits = KFold(5, shuffle=True, random_state=fit_seed).split(seeds)
        for fold, (tr, te) in enumerate(splits):
            train_features = {"raw_mean": summary[tr, :len(WEIGHT_COLUMNS)], "raw_mean_sd": summary[tr]}
            test_features = {"raw_mean": summary[te, :len(WEIGHT_COLUMNS)], "raw_mean_sd": summary[te]}
            if protocol == "shared_representation_loo":
                for features in shared.values():
                    train_features.update({key: value[tr] for key, value in features.items()})
                    test_features.update({key: value[te] for key, value in features.items()})
            else:
                train_runs, test_runs = [runs[i] for i in tr], [runs[i] for i in te]
                for kind in kinds:
                    try:
                        m = fit_model(kind, train_runs, k=k, seed=100 * fit_seed + fold)
                        train_features.update(m.features(train_runs, training_runs=train_runs))
                        test_features.update(m.features(test_runs))
                        density.extend(dict(model=kind, fold=fold, seed=int(seeds[i]),
                                            per_observation=float(m.conditional_loglik(x) / (len(x) - 1)))
                                       for i, x in zip(te, test_runs))
                    except Exception as exc:
                        failures.append(dict(protocol=protocol, fold=fold, model=kind, error=str(exc)))
            for readout in train_features:
                try:
                    pred = regression_predictions(train_features[readout], test_features[readout], y[tr])
                    if not np.isfinite(pred).all():
                        raise FloatingPointError("Non-finite regression prediction")
                    for i, value in zip(te, pred):
                        predictions.append(dict(protocol=protocol, model=readout, fold=fold,
                                                seed=int(seeds[i]), actual=float(y[i]),
                                                predicted=float(value)))
                except Exception as exc:
                    failures.append(dict(protocol=protocol, fold=fold, model=readout, error=str(exc)))
    return predictions, density, failures


def scores(predictions, expected):
    """R² over the held-out predictions of each protocol/model/seed group.

    A group that did not predict every run is left unscored rather than scored on
    the subset that happened to succeed.
    """
    df = pd.DataFrame(predictions)
    rows = []
    for (fit_seed, protocol, model), g in df.groupby(["fit_seed", "protocol", "model"]):
        complete = len(g) == expected and g.seed.nunique() == expected
        denom = ((g.actual - g.actual.mean()) ** 2).sum()
        r2 = 1 - ((g.actual - g.predicted) ** 2).sum() / denom if complete and denom > 0 else np.nan
        rows.append(dict(fit_seed=fit_seed, protocol=protocol, model=model, r2=r2, complete=complete))
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("data", type=Path, help="directory of seed<N>.csv[.gz] trajectories")
    parser.add_argument("--runs", type=int, required=True, help="number of runs in the cohort")
    parser.add_argument("--horizon", type=int, required=True, help="checkpoints retained per run")
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--k", type=int, default=6, help="latent states, and LDS dimension")
    parser.add_argument("--subset", default="weight", choices=sorted(SUBSETS))
    parser.add_argument("--models", nargs="+", default=["hmm", "gmm"],
                        help="model families to fit; raw-mean baselines are always included")
    parser.add_argument("--fit-seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--threshold", type=float, default=0.9, help="convergence accuracy")
    parser.add_argument("--label", help="cohort name recorded in the output; defaults to the directory name")
    parser.add_argument("--out", type=Path, help="write the summary scores to this CSV")
    args = parser.parse_args()

    raw, runs, seeds, y = load_runs(args.data.resolve(), args.runs, args.horizon,
                                    args.stride, args.subset, args.threshold)
    valid = np.flatnonzero(np.isfinite(y))
    if len(valid) < len(y):
        print(f"excluded {len(y) - len(valid)} run(s) that never reached "
              f"accuracy {args.threshold}: {seeds[~np.isfinite(y)].tolist()}", flush=True)
    raw, runs = [raw[i] for i in valid], [runs[i] for i in valid]
    seeds, y = seeds[valid], y[valid]
    if len(runs) < 10:
        raise SystemExit("Fewer than ten converged runs; the regression is not meaningful")

    predictions, density = [], []
    for fit_seed in args.fit_seeds:
        rows, scored, failures = evaluate(raw, runs, seeds, y, args.k, fit_seed, args.models)
        for row in rows:
            row["fit_seed"] = fit_seed
        predictions.extend(rows)
        density.extend(scored)
        for failure in failures:
            print(f"  fit failure: {failure}", flush=True)
        print(f"fit seed {fit_seed}: {len(rows)} held-out predictions", flush=True)

    per_seed = scores(predictions, expected=len(runs))
    summary = (per_seed.groupby(["protocol", "model"])["r2"]
               .agg(["mean", "std", "count"]).reset_index()
               .sort_values(["protocol", "mean"], ascending=[True, False]))
    print(f"\nR² over {len(runs)} held-out runs, K={args.k}, horizon {args.horizon}, "
          f"mean and sample SD across {len(args.fit_seeds)} fit seeds\n")
    print(f"{'protocol':<30} {'readout':<14} {'R²':>8} {'SD':>8}")
    for row in summary.itertuples():
        print(f"{row.protocol:<30} {row.model:<14} {row.mean:>8.4f} {row.std:>8.4f}")
    if args.out:
        nats = (pd.DataFrame(density).groupby("model")["per_observation"].mean().to_dict()
                if density else {})
        summary.insert(0, "cohort", args.label or args.data.resolve().name)
        summary["k"], summary["n_runs"], summary["horizon"] = args.k, len(runs), args.horizon
        # Density is measured on the inductive folds only; leave it blank elsewhere.
        summary["conditional_nats_per_observation"] = summary.model.map(nats).where(
            summary.protocol == "inductive_5fold")
        summary = summary.rename(columns={"mean": "r2", "std": "r2_sd"}).drop(columns=["count"])
        summary.to_csv(args.out, index=False)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
