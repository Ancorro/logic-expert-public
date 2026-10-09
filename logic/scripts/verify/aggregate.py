"""Aggregate verification reruns into tables, pre-registered checks, and figures.

Usage: python aggregate.py RESULTS_DIR OUT_DIR
RESULTS_DIR holds <key>_s<seed>.json (+ optional _routing.npy) harvested from the HPC.
Multi-task per-task accuracies come from the final history row (val_{proofwriter,mnli}_acc_final).
Only MNLI rows are compared with published values: the published ProofWriter numbers were
measured on question-only inputs (see VERIFICATION.md 2b), so the corrected track has no
published counterpart and is reported next to them instead.
"""
import glob
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ORIG = json.load(open(os.path.join(HERE, "../../docs/verification/orig_configs.json")))
ORIG_RUNS = {r["id"]: r for r in json.load(open(os.path.join(HERE, "../../docs/verification/wandb_original_runs.json")))}
ENTITY = "ancorro-oregon-state-university"
SEEDS = (42, 777, 123)

# Values printed in the paper (acc, loss) for the verification check.
PAPER = {
    "base": (0.417, 1.063), "nogate": (0.525, 0.820), "intra8": (0.542, 0.818),
    "a01": (0.546, 0.819), "inter8": (0.562, 0.816), "learn": (0.546, 0.819),
    "inter16": (0.501, 0.837), "mnli_nogate": (0.888, 0.34), "mnli_intra": (0.880, 0.36),
    "mnli_inter": (0.878, 0.39),
}
PAPER_MULTI = {"multi_nogate": (0.544, 0.879), "multi_logic": (0.536, 0.881)}
LABEL = {
    "base": "Baseline", "nogate": "No-Gate Control", "intra8": "Routed intra G=8",
    "inter8": "Routed inter G=8 (a=0.01)", "a01": "Routed inter G=8 (a=0.1)",
    "learn": "Routed inter G=8 (learned a)", "inter16": "Routed inter G=16",
    "noxattn_inter8": "Routed inter G=8, no cross-attn",
    "mnli_nogate": "MNLI No-Gate Control", "mnli_intra": "MNLI Routed intra G=8",
    "mnli_inter": "MNLI Routed inter G=8",
    "multi_nogate": "Multi-task No-Gate Control", "multi_logic": "Multi-task Routed inter G=8",
}


def final_row(res):
    rows = [r for r in res["history"] if r.get("epoch", 0) > 0]
    return rows[-1]


def load(results_dir):
    runs = {}
    for f in glob.glob(os.path.join(results_dir, "*_s*.json")):
        key, seed = os.path.basename(f)[:-5].rsplit("_s", 1)
        res = json.load(open(f))
        fr = final_row(res)
        runs.setdefault(key, {})[int(seed)] = dict(
            acc=fr.get("val_acc"), loss=fr.get("val_loss"), alpha=fr.get("fusion_alpha"),
            entropy=fr.get("routing_entropy"), grad=fr.get("grad_norm"),
            peak_mem=fr.get("peak_gpu_mem_gb"), epoch_sec=fr.get("epoch_sec"),
            pw_acc=fr.get("val_proofwriter_acc_final"), mnli_acc=fr.get("val_mnli_acc_final"),
            wandb_id=res.get("wandb_run_id"), n_val=res["overrides"].get("val_max_samples"),
            history=[(r["epoch"], r.get("val_acc")) for r in res["history"]])
    return runs


def ms(vals):
    v = [x for x in vals if x is not None and not (isinstance(x, float) and math.isnan(x))]
    if not v:
        return None, None, 0
    return float(np.mean(v)), (float(np.std(v, ddof=1)) if len(v) > 1 else 0.0), len(v)


def fmt(m, s, d=3):
    return "--" if m is None else f"{m:.{d}f} ± {s:.{d}f}"


def paired(runs, a, b, metric="acc"):
    common = sorted(set(runs.get(a, {})) & set(runs.get(b, {})))
    d = [runs[a][s][metric] - runs[b][s][metric] for s in common]
    if len(d) < 2:
        return None
    m, sd = float(np.mean(d)), float(np.std(d, ddof=1))
    se = sd / math.sqrt(len(d))
    return dict(a=a, b=b, n=len(d), diffs=d, mean=m, se=se, supported=bool(m > 0 and m > 2 * se))


def main(results_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    runs = load(results_dir)
    lines, summary = [], {}
    lines.append("| Row | Track | n | Val acc | Val loss | alpha | Routing H | Grad norm | Peak GB | s/epoch |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    order = ["base", "nogate", "intra8", "inter8", "a01", "learn", "inter16", "noxattn_inter8",
             "multi_nogate", "multi_logic", "mnli_nogate", "mnli_intra", "mnli_inter"]
    for track, prefix in (("corrected", "fix_"), ("as-published", "")):
        for base in order:
            k = prefix + base
            if k not in runs:
                continue
            R = runs[k].values()
            row = {m: ms([r.get(m) for r in R]) for m in ("acc", "loss", "alpha", "entropy", "grad", "peak_mem", "epoch_sec", "pw_acc", "mnli_acc")}
            summary[k] = {m: {"mean": v[0], "sd": v[1], "n": v[2]} for m, v in row.items()}
            summary[k]["per_seed_acc"] = {s: runs[k][s]["acc"] for s in runs[k]}
            acc = row["acc"] if base not in ("multi_nogate", "multi_logic") else row["pw_acc"]
            lines.append(f"| {LABEL[base]} | {track} | {row['acc'][2]} | {fmt(*acc[:2])} | {fmt(*row['loss'][:2])} | "
                         f"{fmt(*row['alpha'][:2], 4)} | {fmt(*row['entropy'][:2], 4)} | {fmt(*row['grad'][:2], 2)} | "
                         f"{fmt(*row['peak_mem'][:2], 1)} | {fmt(*row['epoch_sec'][:2], 0)} |")
            if base in ("multi_nogate", "multi_logic"):
                lines[-1] += f" MNLI {fmt(*row['mnli_acc'][:2])}"

    # Verification of published numbers against Track A (pre-registered rule).
    ver = ["", "| Row | Paper acc | Rerun acc | Seed-42 rerun | Original W&B | Tolerance | Verified |", "|---|---|---|---|---|---|---|"]
    for base, (pacc, _) in PAPER.items():
        if base not in runs:
            continue
        metric = "acc"
        m, sd, n = ms([r.get(metric) for r in runs[base].values()])
        if m is None:
            continue
        nval = next(iter(runs[base].values()))["n_val"]
        se_bin = math.sqrt(max(m * (1 - m), 1e-9) / nval)
        tol = 2 * max(sd, se_bin)
        s42 = runs[base].get(42, {}).get(metric)
        rid = [v["id"] for kk, v in ORIG.items() if kk == base]
        o = ORIG_RUNS.get(rid[0], {}).get("acc") if rid else None
        ok = abs(pacc - m) <= tol
        summary.setdefault("verification", {})[base] = dict(paper=pacc, rerun_mean=m, rerun_sd=sd, tol=tol, verified=ok, seed42=s42, original=o)
        ver.append(f"| {LABEL[base]} | {pacc:.3f} | {m:.3f} ± {sd:.3f} | {s42 if s42 is None else round(s42, 4)} | "
                   f"{o if o is None else round(o, 4)} | ±{tol:.3f} | {'yes' if ok else '**no**'} |")

    tests = {
        "H1 routed inter8 > No-Gate (corrected)": paired(runs, "fix_inter8", "fix_nogate"),
        "H2 inter8 vs no-cross-attn (corrected)": paired(runs, "fix_inter8", "fix_noxattn_inter8"),
        "routed intra8 > No-Gate (corrected)": paired(runs, "fix_intra8", "fix_nogate"),
        "post hoc: routed inter8 no-cross-attn > No-Gate, used params matched (corrected)": paired(
            runs, "fix_noxattn_inter8", "fix_nogate"),
        "No-Gate > Baseline (corrected)": paired(runs, "fix_nogate", "fix_base"),
        "Routed inter16 > No-Gate (corrected)": paired(runs, "fix_inter16", "fix_nogate"),
        "multi-task Routed > No-Gate, ProofWriter (corrected)": paired(runs, "fix_multi_logic", "fix_multi_nogate", "pw_acc"),
        "multi-task Routed > No-Gate, MNLI (corrected)": paired(runs, "fix_multi_logic", "fix_multi_nogate", "mnli_acc"),
        "MNLI Routed inter > No-Gate": paired(runs, "mnli_inter", "mnli_nogate"),
        "MNLI Routed intra > No-Gate": paired(runs, "mnli_intra", "mnli_nogate"),
    }
    summary["tests"] = tests
    tl = ["", "| Comparison (seed-paired acc diff) | n | mean diff | SE | supported (mean>0 and >2SE) |", "|---|---|---|---|---|"]
    for name, t in tests.items():
        if t:
            tl.append(f"| {name} | {t['n']} | {t['mean']:+.4f} | {t['se']:.4f} | {'yes' if t['supported'] else 'no'} |")

    md = "\n".join(["## Rerun summary (mean ± SD over seeds)", *lines, "", "## Verification of published numbers", *ver,
                    "", "## Pre-registered comparisons", *tl])
    open(os.path.join(out_dir, "rerun_summary.md"), "w", encoding="utf-8").write(md + "\n")
    json.dump(summary, open(os.path.join(out_dir, "rerun_summary.json"), "w"), indent=1, default=float)
    print(md)
    figures(runs, results_dir, out_dir)


def figures(runs, results_dir, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = ["base", "nogate", "intra8", "inter8", "a01", "learn", "inter16", "noxattn_inter8"]
    fig, ax = plt.subplots(figsize=(9, 4.2))
    x = np.arange(len(rows))
    ax.bar(x - 0.18, [PAPER.get(k, (0,))[0] for k in rows], 0.34, color="#9aa3ad",
           label="published (question-only inputs, 1 run)")
    means = [ms([r["acc"] for r in runs.get("fix_" + k, {}).values()])[0] for k in rows]
    ax.bar(x + 0.18, [m if m is not None else 0 for m in means], 0.34, color="#2f6db5",
           label="corrected rerun (theory + question, mean of 3 seeds)")
    for i, k in enumerate(rows):
        for r in runs.get("fix_" + k, {}).values():
            ax.plot(x[i] + 0.18, r["acc"], "o", color="black", ms=3)
    ax.axhline(0.4646, ls="--", color="#c0392b", lw=1, label="majority class (val, seed 42)")
    ax.set_xticks(x, [LABEL[k].replace("Routed ", "R. ") for k in rows], rotation=25, ha="right", fontsize=8)
    ax.set_ylabel("ProofWriter val accuracy (final epoch)")
    ax.set_ylim(0.2, 1.1)
    ax.legend(fontsize=7.5, loc="upper center", ncol=3, framealpha=1)
    ax.set_title("ProofWriter: published vs corrected rerun (dots = seeds)")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "proofwriter_reruns.png"), dpi=200)

    npy = os.path.join(results_dir, "fix_inter8_s42_routing.npy")
    if os.path.exists(npy):
        arr = np.load(npy)
        fig, ax = plt.subplots(figsize=(8, 4.5))
        im = ax.imshow(arr, aspect="auto", cmap="viridis")
        ax.set_xlabel("routed channel (AND | OR | NOT groups)")
        ax.set_ylabel("layer")
        fig.colorbar(im, ax=ax, label="mean routing weight")
        ax.set_title("Routing weights, corrected Routed inter G=8 (seed 42, one val batch)")
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "routing_heatmap_corrected.png"), dpi=200)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
