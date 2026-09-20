"""Draw the occupancy comparison figure and tables from convergence_regression.py scores.

Give it one summary CSV per cohort, as written by `convergence_regression.py --out`:

    python3 convergence_regression.py <parity runs>  --runs 60 --horizon 300 \\
        --label "Sparse parity" --out parity.csv
    python3 convergence_regression.py <modular runs> --runs 40 --horizon 500 \\
        --label "Modular addition" --out modular.csv
    python3 make_figure.py parity.csv modular.csv

It writes occupancy_comparison.{svg,png} and one Markdown table per cohort into assets/.
It fits nothing; every number it draws comes from the scores it is given.
"""
from __future__ import annotations

import argparse
import csv
import math
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MODELS = ("hmm", "gmm")
EVALUATIONS = (("inductive_5fold", "Five-fold representation refit"),
               ("shared_representation_loo", "Shared representation; regression LOO"))
READOUTS = {
    "gmm": "GMM occupancy", "hmm": "HMM occupancy", "lds": "LDS mean",
    "lds_ms": "LDS mean + SD", "lds_occ": "Quantized LDS occupancy",
    "arhmm_diag": "ARHMM occupancy", "raw_mean": "Raw means",
    "raw_mean_sd": "Raw means + SD",
}


def read_rows(path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def load(paths):
    """Score rows keyed by cohort, protocol and readout, keeping the given cohort order."""
    cohorts, rows = [], {}
    for path in paths:
        for row in read_rows(path):
            required = {"cohort", "protocol", "model", "r2", "n_runs", "horizon"}
            if not required <= row.keys():
                raise ValueError(f"{path} is not a convergence_regression.py summary")
            score = float(row["r2"])
            if not math.isfinite(score) or score > 1:
                raise ValueError(f"Invalid R² in {path}: {row}")
            rows[row["cohort"], row["protocol"], row["model"]] = row
            if row["cohort"] not in cohorts:
                cohorts.append(row["cohort"])
    if not cohorts:
        raise ValueError("No scores to draw")
    return cohorts, rows


def panels(cohorts, rows):
    found = [(cohort, protocol, label) for cohort in cohorts for protocol, label in EVALUATIONS
             if all((cohort, protocol, model) in rows for model in MODELS)]
    if not found:
        raise ValueError("Need HMM and GMM scores for at least one cohort and protocol")
    return found


def result_table(selected, rows):
    lines = ["| Cohort and evaluation | Runs | Checkpoints | HMM R² | GMM R² |",
             "|---|---:|---:|---:|---:|"]
    for cohort, protocol, label in selected:
        h, g = (rows[cohort, protocol, model] for model in MODELS)
        lines.append(f"| {cohort} · {label} | {h['n_runs']} | {h['horizon']} | "
                     f"{float(h['r2']):.4f} | {float(g['r2']):.4f} |")
    return "\n".join(lines) + "\n"


def main_table(cohort, rows):
    """Every readout scored for one cohort, with the density of the fit behind it."""
    lines = ["| Model/readout | Conditional nats/observation | Five-fold refit R² | Shared-representation LOO R² |",
             "|---|---:|---:|---:|"]
    for model, label in READOUTS.items():
        inductive = rows.get((cohort, "inductive_5fold", model))
        shared = rows.get((cohort, "shared_representation_loo", model))
        if not inductive or not shared:
            continue
        nats = inductive.get("conditional_nats_per_observation") or ""
        density = f"{float(nats):.4f}" if nats else "—"
        if model in ("lds_ms", "lds_occ"):
            density = "same LDS fit"
        lines.append(f"| {label} | {density} | {float(inductive['r2']):.4f} | "
                     f"{float(shared['r2']):.4f} |")
    return "\n".join(lines) + "\n"


def make_figure(selected, rows, assets):
    os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache/matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "svg.hashsalt": "useful-states-fresh-parity",
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.spines.left": False, "axes.edgecolor": "#b8c1cc",
                         "text.color": "#202735", "axes.labelcolor": "#202735",
                         "xtick.color": "#586373", "ytick.color": "#202735"})
    two_tasks = len(selected) > 2
    fig, axes = plt.subplots(2 if two_tasks else 1, 2, figsize=(11.6, 7.6 if two_tasks else 4.5))
    fig.subplots_adjust(left=.10, right=.98, bottom=.19 if two_tasks else .29,
                        top=.76 if two_tasks else .65, wspace=.35, hspace=1.1)
    fig.suptitle("State occupancy and convergence time", x=.045, y=.97,
                 ha="left", fontsize=19, weight="bold")
    k = rows[selected[0][0], selected[0][1], "hmm"].get("k", "6")
    fig.text(.045, .90 if two_tasks else .86,
             f"K = {k} · Same fitting and evaluation protocol · Complete analyzed trajectories",
             fontsize=11, color="#586373")
    for ax, (cohort, protocol, evaluation) in zip(axes.flat, selected):
        scores = [float(rows[cohort, protocol, model]["r2"]) for model in MODELS]
        row = rows[cohort, protocol, "hmm"]
        ax.set_title(f"{cohort} · {row['n_runs']} runs · {row['horizon']} checkpoints\n{evaluation}",
                     loc="left", fontsize=11, pad=14)
        ax.barh([1, 0], scores, height=.43, color=("#3865a4", "#c8732c"))
        ax.set_yticks([1, 0], ["HMM", "GMM"])
        ax.tick_params(axis="y", length=0, pad=9)
        ax.set_xlim(min(0, min(scores) - .05), 1.17)
        ax.set_ylim(-.6, 1.6)
        ax.set_xticks([0, .25, .5, .75, 1], ["0", "0.25", "0.50", "0.75", "1.00"])
        ax.set_xlabel("Retrospective regression R²", labelpad=10)
        ax.set_axisbelow(True)
        ax.grid(axis="x", color="#e4e8ed", linewidth=.8)
        for y, score in zip([1, 0], scores):
            ax.text(score + .018, y, f"{score:.4f}", va="center", fontsize=12)
    for ax in axes.flat[len(selected):]:
        ax.set_axis_off()
    fig.text(.045, .10 if two_tasks else .11,
             "Means over the fit/split seeds scored. Both evaluations are retrospective, "
             "using observations after convergence.", fontsize=10, color="#586373")
    fig.text(.045, .045,
             "Close scores do not establish statistical equivalence; fit/split variability "
             "is not a run-level confidence interval.", fontsize=10, color="#586373")
    fig.savefig(assets / "occupancy_comparison.svg", metadata={"Date": None}, facecolor="white")
    fig.savefig(assets / "occupancy_comparison.png", dpi=180, facecolor="white")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("scores", type=Path, nargs="+",
                        help="summary CSVs from convergence_regression.py --out, one per cohort")
    args = parser.parse_args()
    try:
        cohorts, rows = load(args.scores)
        selected = panels(cohorts, rows)
        assets = ROOT / "assets"
        assets.mkdir(exist_ok=True)
        make_figure(selected, rows, assets)
        (assets / "occupancy_table.md").write_text(result_table(selected, rows), encoding="utf-8")
        written = ["occupancy_table.md"]
        for cohort in cohorts:
            table = main_table(cohort, rows)
            if table.count("\n") > 2:
                name = f"{cohort.lower().replace(' ', '_')}_main_table.md"
                (assets / name).write_text(table, encoding="utf-8")
                written.append(name)
    except (ValueError, KeyError, OSError) as exc:
        parser.exit(1, f"Error: {exc}\n")
    print(f"Drew {len(selected)} panels from {len(cohorts)} cohort(s).")
    print("Wrote occupancy_comparison.{svg,png} and " + ", ".join(written))


if __name__ == "__main__":
    main()
