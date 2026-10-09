"""How far the routed AND/OR gates sit from their logical (nonlinear) regime, as a function of routing sharpness.

Usage: python gate_regime.py OUT_PNG

LogicLayer feeds each gate u = sigmoid(W_g x) * softmax(W_r x) over 3G channels, so u <= routing weight.
For a given mean normalized routing entropy, the largest routing weight is reached when one channel is
peaked and the rest share the remainder; AND over a window of 2 is then at most that weight squared.
The "typical" curve draws Gaussian router logits and standard-normal gate logits (illustration only).
The reference line is the DLGN-style alternative, where the softmax mixes operator outputs and the
operators see unscaled sigmoid inputs. No model is run.
"""
import sys

import numpy as np
from scipy.optimize import brentq

N = 24  # 3G channels for G = 8
OBSERVED = (0.972, 0.976)  # normalized routing entropy range in the paper's Table 1


def norm_entropy_one_peak(p):
    q = (1 - p) / (N - 1)
    return (-p * np.log(p) - (N - 1) * q * np.log(q)) / np.log(N)


def main(out_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(0)
    h_grid = np.linspace(0.30, 0.995, 60)
    bound = [brentq(lambda p: norm_entropy_one_peak(p) - h, 1 / N + 1e-9, 1 - 1e-9) ** 2 for h in h_grid]

    typ_h, typ_and = [], []
    for sd in np.linspace(0.05, 4.0, 40):
        z = rng.normal(0, sd, (20000, N))
        p = np.exp(z - z.max(1, keepdims=True))
        p /= p.sum(1, keepdims=True)
        s = 1 / (1 + np.exp(-rng.normal(0, 1, (20000, 2, N))))
        u_now, u_prev = s[:, 0] * p, s[:, 1] * p  # same routing at both tokens: favours a large AND
        typ_h.append((-(p * np.log(p)).sum(1) / np.log(N)).mean())
        typ_and.append((u_now * u_prev).mean())
    dlgn_and = (1 / (1 + np.exp(-rng.normal(0, 1, 200000)))) * (1 / (1 + np.exp(-rng.normal(0, 1, 200000))))

    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.axvspan(*OBSERVED, color="#e9b949", alpha=0.35, lw=0, label="paper's routed runs")
    ax.plot(h_grid, bound, color="#2f6db5", lw=2, label="largest possible AND output")
    ax.plot(typ_h, typ_and, color="#2f6db5", lw=2, ls="--", label="typical AND output (simulated)")
    ax.axhline(dlgn_and.mean(), color="#7a828c", lw=1.5, ls=":",
               label=f"typical AND if softmax mixed operator outputs ({dlgn_and.mean():.2f})")
    ax.set_yscale("log")
    ax.set_xlim(0.3, 1.0)
    ax.set_ylim(1e-4, 1)
    ax.set_xlabel("normalized routing entropy (1 = uniform over 24 channels)")
    ax.set_ylabel("AND output, window of 2 (log scale)")
    ax.set_title("Routed AND gates are near zero unless routing is sharp")
    ax.grid(axis="y", color="#e3e3e3", lw=0.8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(fontsize=7.5, loc="lower left", framealpha=1)
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    i = np.argmin(np.abs(h_grid - 0.972))
    print(f"at 0.972: bound {bound[i]:.4f}; DLGN-style typical {dlgn_and.mean():.3f}")


if __name__ == "__main__":
    main(sys.argv[1])
