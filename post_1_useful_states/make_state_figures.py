"""Draw what the latent states are, for sparse parity and modular addition.

Reads per-run trajectories, fits the HMM and the GMM with the same fitter the
regression uses (shared representation, K=6, fit seed 0), and writes state_raster,
training_map and l1_mechanism into assets/.

Needs only a trajectory directory and parity_models.py beside it, and writes
nothing outside assets/. Run it from this folder:

    python3 make_state_figures.py --data <dir>          # <dir>/parity and <dir>/modular
    python3 make_state_figures.py --root <research repo>  # read that repo's sweeps
"""
import argparse
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm, LinearSegmentedColormap
from matplotlib.patches import FancyArrowPatch, Circle
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent
_parser = argparse.ArgumentParser(description=__doc__)
_parser.add_argument("--data", type=Path, default=HERE / "data",
                     help="directory holding parity/ and modular/ trajectory folders")
_parser.add_argument("--root", type=Path,
                     help="read the research repository's sweep outputs instead")
_args = _parser.parse_args()
DATA, ROOT = _args.data, _args.root.resolve() if _args.root else None
from parity_models import WEIGHT_COLUMNS, zscore_run, fit_model

SWEEPS = {"parity": "output/parity_rerun_20260912T202037Z",
          "modular": "output/modular_rerun_20260914T083633Z"}


def inputs(key):
    """A trajectory directory: under --data by default, or the sweep's csv/ under --root."""
    return DATA / key if ROOT is None else ROOT / SWEEPS[key] / "csv"


def read_csv(csvdir, seed):
    path = csvdir / f"seed{seed}.csv.gz"
    return pd.read_csv(path if path.exists() else csvdir / f"seed{seed}.csv")

OUT = HERE / "assets"
OUT.mkdir(exist_ok=True)
# Documented blue ramp, steps 100->700: monotone lightness, adjacent dL >= 0.06.
PHASE = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#1c5cab", "#0d366b"]
ACCENT = "#eb6834"      # categorical slot 2, used only for the convergence annotation
SURFACE, INK, INK2, MUTED, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#c3c2b7"

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"],
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE, "text.color": INK,
    "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.edgecolor": AXIS, "axes.linewidth": 0.8,
    "font.size": 9, "axes.titlesize": 10, "xtick.labelsize": 8, "ytick.labelsize": 8,
})


def load(csvdir, n, horizon):
    runs, steps, y = [], [], []
    for s in range(n):
        df = read_csv(csvdir, s)
        y.append(float(df.loc[df.eval_accuracy >= 0.9, "step"].iloc[0]))
        sel = df.iloc[:horizon]
        runs.append(zscore_run(sel[WEIGHT_COLUMNS].to_numpy(float)))
        steps.append(sel.step.to_numpy(float))
    return runs, steps, np.array(y)


def ordered(seqs, steps, K=6):
    """Relabel states by mean epoch so colour reads early -> late."""
    mean_epoch = [np.mean(np.concatenate([e[s == k] for s, e in zip(seqs, steps) if (s == k).any()]))
                  for k in range(K)]
    order = np.argsort(mean_epoch)
    lookup = np.empty(K, int)
    lookup[order] = np.arange(K)
    return [lookup[s] for s in seqs], np.sort(mean_epoch)


def panels():
    data = {}
    for task, key, n, horizon in (("Sparse parity", "parity", 60, 300),
                                  ("Modular addition", "modular", 40, 500)):
        runs, steps, y = load(inputs(key), n, horizon)
        hmm_model = fit_model("hmm", runs, k=6, seed=0)
        hmm = [hmm_model.states(x) for x in runs]
        gmm = [fit_model("gmm", runs, k=6, seed=0).states(x) for x in runs]
        order = np.argsort(y)
        out = {}
        for name, seqs in (("HMM", hmm), ("GMM", gmm)):
            lab, mean_epoch = ordered(seqs, steps)
            out[name] = dict(raster=np.stack([lab[i] for i in order]), mean_epoch=mean_epoch,
                             seqs=[lab[i] for i in order], steps=[steps[i] for i in order])
        data[task] = dict(models=out, y=y[order], n=n, epochs=steps[0], span=steps[0][-1] + (steps[0][1] - steps[0][0]))
        print(f"loaded {task}: {n} runs, window to epoch {data[task]['span']:.0f}", flush=True)
    return data


def raster_figure(data):
    ZOOM = 700
    tasks = list(data.items())
    rows = [(tasks[0][0], tasks[0][1], None),
            (tasks[1][0], tasks[1][1], None),
            (tasks[1][0], tasks[1][1], ZOOM)]
    fig, axes = plt.subplots(3, 2, figsize=(9.4, 9.2), constrained_layout=True)
    cmap, norm = ListedColormap(PHASE), BoundaryNorm(np.arange(-0.5, 6.5), 6)
    for row, (task, d, zoom) in enumerate(rows):
        for col, model in enumerate(("HMM", "GMM")):
            ax, m = axes[row, col], d["models"][model]
            ax.imshow(m["raster"], aspect="auto", interpolation="nearest", cmap=cmap, norm=norm,
                      extent=[0, d["span"], d["n"], 0])
            ax.plot(d["y"], np.arange(d["n"]) + 0.5, color=ACCENT, lw=2.0, solid_capstyle="round")
            suffix = f" · first {zoom} epochs" if zoom else ""
            ax.set_title(f"{task} · {model}{suffix}", loc="left", color=INK, pad=6)
            if zoom:
                ax.set_xlim(0, zoom)
            ax.set_xlabel("epoch")
            if col == 0:
                ax.set_ylabel(f"{d['n']} runs, sorted by convergence")
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
    handles = [Line2D([], [], marker="s", ls="", ms=9, mfc=c, mec="none", label=f"phase {i+1}")
               for i, c in enumerate(PHASE)]
    handles.append(Line2D([], [], color=ACCENT, lw=2.0, label="convergence epoch"))
    fig.legend(handles=handles, loc="outside lower center", ncol=7, frameon=False,
               handletextpad=0.5, columnspacing=1.4, labelcolor=INK2)
    fig.suptitle("Decoded states over training, K = 6 (phases numbered by mean epoch)",
                 x=0.01, ha="left", color=INK, fontsize=12)
    for ext in ("png", "svg"):
        fig.savefig(OUT / f"state_raster.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote state_raster.png / .svg", flush=True)


def map_figure(data):
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.6), constrained_layout=True)
    for ax, (task, d) in zip(axes, data.items()):
        m = d["models"]["HMM"]
        seqs = m["seqs"]
        K = 6
        counts = np.zeros((K, K))
        occ = np.zeros(K)
        for s in seqs:
            occ += np.bincount(s, minlength=K) / len(s) / len(seqs)
            for a, b in zip(s[:-1], s[1:]):
                if a != b:
                    counts[a, b] += 1
        probs = counts / np.maximum(counts.sum(1, keepdims=True), 1)
        x = m["mean_epoch"] / d["span"]
        yv = np.array([0.32 if k % 2 else -0.32 for k in range(K)])
        for i in range(K):
            for j in range(K):
                if probs[i, j] >= 0.02 and i != j:
                    ax.add_patch(FancyArrowPatch((x[i], yv[i]), (x[j], yv[j]),
                        connectionstyle="arc3,rad=0.25", arrowstyle="-|>", mutation_scale=9,
                        lw=0.6 + 3.4 * probs[i, j], color="#2a78d6",
                        alpha=0.25 + 0.6 * probs[i, j], shrinkA=13, shrinkB=13, zorder=1))
        for k in range(K):
            ax.add_patch(Circle((x[k], yv[k]), 0.055 + 0.16 * occ[k], color=PHASE[k],
                                ec=AXIS, lw=0.8, zorder=2, transform=ax.transData))
            ax.text(x[k], yv[k], str(k + 1), ha="center", va="center", zorder=3,
                    color=INK if k <= 2 else "#ffffff", fontsize=9)
            ax.text(x[k], yv[k] - 0.26 if yv[k] < 0 else yv[k] + 0.26, f"{m['mean_epoch'][k]:.0f}",
                    ha="center", va="center", color=MUTED, fontsize=7.5, zorder=3)
        n_edges = int(((probs >= 0.02) & ~np.eye(K, dtype=bool)).sum())
        ax.set_title(f"{task} · {n_edges} transitions above 2%", loc="left", color=INK, pad=6)
        ax.set_xlim(-0.12, 1.12); ax.set_ylim(-0.75, 0.75)
        ax.set_xlabel("mean epoch of the state, as a fraction of the window")
        ax.set_yticks([]); ax.set_xticks([0, 0.5, 1]); ax.set_xticklabels(["start", "half", "end"])
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
    fig.suptitle("Training maps: states placed by mean epoch, sized by occupancy;\n"
                 "arrows are transitions between different states, self-transitions omitted",
                 x=0.01, ha="left", color=INK, fontsize=12)
    for ext in ("png", "svg"):
        fig.savefig(OUT / f"training_map.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote training_map.png / .svg", flush=True)


def l1_figure():
    """Why one averaged metric predicts convergence on sparse parity."""
    csvdir = inputs("parity")
    curves, y = [], []
    for s in range(60):
        df = read_csv(csvdir, s)
        y.append(int(df.loc[df.eval_accuracy >= 0.9, "step"].iloc[0]))
        curves.append(df.iloc[:300]["l1"].to_numpy(float))
    epochs = read_csv(csvdir, 0).iloc[:300].step.to_numpy(float)
    y, L = np.array(y), np.stack(curves)
    means, at_conv = L.mean(1), np.array([L[i, t] for i, t in enumerate(y)])
    cv_fixed = np.median([L[:, t].std() / L[:, t].mean() for t in range(60, 240)])
    off = np.arange(-60, 60)
    aligned = np.stack([[L[i, t + o] for o in off] for i, t in enumerate(y)])
    cv_aligned = np.median(aligned.std(0) / aligned.mean(0))
    r = np.corrcoef(means, y)[0, 1]
    print(f"L1 norm at convergence {at_conv.mean():.0f} +/- {at_conv.std():.0f}; spread across runs "
          f"{cv_fixed*100:.0f}% at a fixed epoch, {cv_aligned*100:.1f}% aligned; "
          f"corr(run average, convergence epoch) = {r:+.3f}", flush=True)

    cmap, norm = LinearSegmentedColormap.from_list("convergence", PHASE), plt.Normalize(y.min(), y.max())
    fig, (axA, axB, axC) = plt.subplots(1, 3, figsize=(12.4, 3.9), constrained_layout=True)
    order = np.argsort(y)

    for i in order:
        axA.plot(epochs, L[i], color=cmap(norm(y[i])), lw=0.9, alpha=0.85, zorder=1)
    axA.scatter(y, at_conv, s=16, color=ACCENT, zorder=3, linewidths=0, label="convergence epoch")
    for i in (order[0], order[-1]):
        axA.plot(epochs, L[i], color=cmap(norm(y[i])), lw=2.2, zorder=2)
    axA.set_xlim(0, 300)
    axA.set_xlabel("epoch"); axA.set_ylabel("L1 norm of the weights")
    axA.set_title("Runs follow the same decay,\nat different times", loc="left", color=INK, pad=6)
    axA.legend(frameon=False, loc="upper right", labelcolor=INK2)

    for i in order:
        axB.plot(epochs - y[i], L[i], color=cmap(norm(y[i])), lw=0.9, alpha=0.85, zorder=1)
    axB.scatter(np.zeros_like(at_conv), at_conv, s=16, color=ACCENT, zorder=3, linewidths=0)
    axB.set_xlim(-90, 110)
    axB.set_xlabel("epochs since that run's convergence"); axB.set_ylabel("L1 norm of the weights")
    axB.set_title("Aligned on convergence,\nthey nearly collapse", loc="left", color=INK, pad=6)

    axC.scatter(means, y, s=42, c=y, cmap=cmap, norm=norm, edgecolor=SURFACE, linewidths=0.8, zorder=2)
    xs = np.linspace(means.min(), means.max(), 2)
    axC.plot(xs, np.polyval(np.polyfit(means, y, 1), xs), color=MUTED, lw=1.5, zorder=1)
    axC.set_xlabel("run average of the L1 norm"); axC.set_ylabel("convergence epoch")
    axC.set_title("so the run average recovers\nthe timing", loc="left", color=INK, pad=6)
    axC.annotate(f"r = {r:.2f}\nfive-fold R² = 0.94\nfrom this one feature",
                 (0.04, 0.96), xycoords="axes fraction", va="top", color=INK2, fontsize=8.5)

    for ax in (axA, axB, axC):
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    bar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=[axA, axB, axC],
                       location="right", fraction=0.025, pad=0.02)
    bar.set_label("convergence epoch", color=INK2)
    bar.outline.set_visible(False)
    fig.suptitle("Sparse parity: why averaging one weight statistic recovers convergence time",
                 x=0.01, ha="left", color=INK, fontsize=12)
    for ext in ("png", "svg"):
        fig.savefig(OUT / f"l1_mechanism.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote l1_mechanism.png / .svg", flush=True)


d = panels()
raster_figure(d)
map_figure(d)
l1_figure()
