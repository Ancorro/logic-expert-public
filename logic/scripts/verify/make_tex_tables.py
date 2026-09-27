"""Emit the paper's LaTeX table bodies from results/rerun_summary.json (no hand transcription).

Usage: python make_tex_tables.py RESULTS_DIR   -> writes RESULTS_DIR/paper_tables.tex
"""
import json
import math
import os
import sys

from aggregate import PAPER, PAPER_MULTI

GATES = {"fix_intra8": 8, "fix_inter8": 8, "fix_a01": 8, "fix_learn": 8, "fix_inter16": 16,
         "fix_noxattn_inter8": 8, "fix_multi_logic": 8, "mnli_inter": 8, "mnli_intra": 8}


def pm(d, digits=3):
    return f"{d['mean']:.{digits}f} $\\pm$ {d['sd']:.{digits}f}"


def norm_entropy(key, d):
    g = GATES.get(key)
    if g is None or not d["mean"]:
        return "--"
    return f"{d['mean'] / math.log(3 * g):.3f}"


def main(results_dir):
    s = json.load(open(os.path.join(results_dir, "rerun_summary.json")))
    raw = os.path.join(results_dir, "raw")
    params = {k: json.load(open(os.path.join(raw, f"{k}_s42.json")))["param_count"]
              for k in ("fix_base", "fix_nogate", "fix_inter8", "fix_inter16", "fix_noxattn_inter8")}
    out = []

    out.append("% Table 1: corrected ProofWriter (mean +- SD, 3 seeds)")
    rows = [("Baseline", "fix_base", "base", "--"),
            ("No-Gate Control", "fix_nogate", "nogate", "0.01 (fixed)"),
            ("Routed (intra, G=8)", "fix_intra8", "intra8", "0.01 (fixed)"),
            ("Routed (inter, G=8)", "fix_inter8", "inter8", "0.01 (fixed)"),
            ("Routed (inter, G=8)", "fix_a01", "a01", "0.1 (fixed)"),
            ("Routed (inter, G=8)", "fix_learn", "learn", None),
            ("Routed (inter, G=16)", "fix_inter16", "inter16", "0.01 (fixed)"),
            ("Routed (inter, G=8), no cross-attn", "fix_noxattn_inter8", None, "0.01 (fixed)")]
    for name, k, pk, alpha in rows:
        d = s[k]
        if alpha is None:
            alpha = f"{d['alpha']['mean']:.4f} (learned from 0.01)"
        pub = f"{PAPER[pk][0]:.3f}" if pk else "--"
        out.append(f"{name} & {pm(d['acc'])} & {pm(d['loss'])} & {alpha} & {norm_entropy(k, d['entropy'])} "
                   f"& {d['grad']['mean']:.1f} & {pub} \\\\")

    out.append("\n% Table 2: trainable parameters (from the rerun JSONs)")
    for name, k in (("Baseline", "fix_base"), ("No-Gate Control", "fix_nogate"), ("Routed (G=8)", "fix_inter8"),
                    ("Routed (G=16)", "fix_inter16"), ("Routed (G=8), no cross-attn", "fix_noxattn_inter8")):
        out.append(f"{name} & {params[k]:,} \\\\")

    out.append("\n% Table 3: MNLI (mean +- SD, 3 seeds; val n=500)")
    for name, k in (("No-Gate Control", "mnli_nogate"), ("Routed (intra, G=8)", "mnli_intra"),
                    ("Routed (inter, G=8)", "mnli_inter")):
        v = s["verification"][k]
        out.append(f"{name} & {pm(s[k]['acc'])} & {pm(s[k]['loss'])} & {PAPER[k][0]:.3f} & "
                   f"{'yes' if v['verified'] else 'no'} \\\\")

    out.append("\n% Table 4: multi-task (mean +- SD, 3 seeds; corrected ProofWriter)")
    for name, k, pk in (("No-Gate Control", "fix_multi_nogate", "multi_nogate"),
                        ("Routed (inter, G=8)", "fix_multi_logic", "multi_logic")):
        d = s[k]
        mean = (d["pw_acc"]["mean"] + d["mnli_acc"]["mean"]) / 2
        out.append(f"{name} & {pm(d['pw_acc'])} & {pm(d['mnli_acc'])} & {mean:.3f} & "
                   f"{PAPER_MULTI[pk][0]:.3f} / {PAPER_MULTI[pk][1]:.3f} \\\\")

    out.append("\n% Pre-registered and secondary comparisons (seed-paired acc diff, pp)")
    for name, t in s["tests"].items():
        if t:
            out.append(f"% {name}: {100 * t['mean']:+.1f} pp, SE {100 * t['se']:.1f}, n={t['n']}, "
                       f"supported={t['supported']}")
    path = os.path.join(results_dir, "paper_tables.tex")
    open(path, "w", encoding="utf-8", newline="\n").write("\n".join(out) + "\n")
    print("\n".join(out))


if __name__ == "__main__":
    main(sys.argv[1])
