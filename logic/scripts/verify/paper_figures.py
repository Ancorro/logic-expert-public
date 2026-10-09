"""Draw the paper's ProofWriter accuracy and routing-heatmap figures from the 3-seed runs.

Usage: python paper_figures.py RESULTS_DIR OUT_DIR
RESULTS_DIR is logic/docs/verification/results/raw. OUT_DIR is usually logic/LaTeX_dir/build/plots.
aggregate.py keeps its own figures (with the earlier single-run values) for the verification record.
"""
import os
import sys

import numpy as np

from aggregate import LABEL, load, ms

ROWS = ["base", "nogate", "intra8", "inter8", "a01", "learn", "inter16", "noxattn_inter8"]


def main(results_dir, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(out_dir, exist_ok=True)
    runs = load(results_dir)

    # Dot plot, not bars: every mean sits in 0.925-0.939, so the axis is zoomed and bars would mislead.
    fig, ax = plt.subplots(figsize=(9, 4.2))
    x = np.arange(len(ROWS))
    for i, k in enumerate(ROWS):
        accs = [r["acc"] for r in runs["fix_" + k].values()]
        m, sd, _ = ms(accs)
        ax.plot(np.full(len(accs), x[i]) + 0.12, accs, "o", color="#7a828c", ms=4, zorder=2)
        ax.errorbar(x[i] - 0.12, m, yerr=sd, fmt="o", color="#2f6db5", ms=8, capsize=4, lw=2, zorder=3)
    ax.errorbar([], [], yerr=[], fmt="o", color="#2f6db5", ms=8, capsize=4, lw=2, label="mean ± SD (3 seeds)")
    ax.plot([], [], "o", color="#7a828c", ms=4, label="individual seeds")
    ax.set_xticks(x, [LABEL[k].replace("Routed ", "R. ") for k in ROWS], rotation=25, ha="right", fontsize=8)
    ax.set_xlim(-0.6, len(ROWS) - 0.4)
    ax.set_ylim(0.90, 0.96)
    ax.set_ylabel("ProofWriter val accuracy (final epoch)")
    ax.grid(axis="y", color="#e3e3e3", lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(fontsize=8, loc="lower right", framealpha=1)
    ax.set_title("ProofWriter validation accuracy, rulebase in the input")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "proofwriter_accuracy.png"), dpi=200)

    arr = np.load(os.path.join(results_dir, "fix_inter8_s42_routing.npy"))
    fig, ax = plt.subplots(figsize=(8, 4.5))
    im = ax.imshow(arr, aspect="auto", cmap="viridis")
    ax.set_xlabel("routed channel (AND | OR | NOT groups)")
    ax.set_ylabel("layer")
    fig.colorbar(im, ax=ax, label="mean routing weight")
    ax.set_title("Routing weights, Routed inter G=8 (seed 42, one val batch)")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "routing_heatmap.png"), dpi=200)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
