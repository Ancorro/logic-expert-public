"""Build the rerun manifest from the original W&B run configs.

Each paper row maps to one original W&B run (see docs/verification/VERIFICATION.md) and to
the notebook commit that was current when that run started. Only fields logged in W&B are
overridden; everything else keeps the pinned notebook's defaults, as in the original run.

Usage: python make_manifest.py orig_configs.json OUT_DIR
"""
import json
import os
import sys

SEEDS = (42, 777, 123)

# paper row key -> (pinned notebook, original W&B run id)
ROWS = {
    "base":         ("V5_Eval@f6714ce", "wy9c4qdh"),
    "nogate":       ("V5_Eval@f6714ce", "p77m4hx7"),
    "intra8":       ("V5_Eval@f6714ce", "45o0qen1"),
    "inter8":       ("V5_Eval@f6714ce", "fa3al9os"),
    "inter16":      ("V5_Eval@f6714ce", "w8fnquk8"),
    "learn":        ("V5_Eval@758bca2", "fkv5gltf"),
    "a01":          ("V5_Eval@758bca2", "q4zj8fhp"),
    "mnli_inter":   ("V5_Eval@27918e7", "t66n8g2u"),
    "mnli_nogate":  ("V5_Eval@27918e7", "gucn27u2"),
    "mnli_intra":   ("V5_Eval@ba16102", "aapet4cg"),
    "multi_logic":  ("V6_multi_eval@7172b79", "ckb9q7ut"),
    "multi_nogate": ("V6_multi_eval@7172b79", "xa5kmu55"),
}
# New ablation (replaces the old-architecture appendix table): inter8 without cross-attention.
ABLATIONS = {"noxattn_inter8": ("inter8", {"use_cross_attn": False})}

# Corrected track: ProofWriter with the rulebase (`theory`) restored, run in the fixed code
# tree. One driver (V6) for every variant so rows differ only in the listed fields; paper
# hyperparameters are kept, gradient checkpointing is on (inputs are ~14x longer), and
# "fixed" alpha is actually frozen (learn_fusion_alpha=False).
FIX_NB = "V6_multi_eval@7172b79"
FIX_SINGLE = {"enable_multitask_training": False, "enable_gradient_checkpointing": True,
              "gate_mode": "soft", "dataset_name": "proofwriter"}
FIXED = {  # key -> (source row for base config, extra overrides)
    "fix_base":           ("base",    {"learn_fusion_alpha": False}),  # no alpha; V6 guard
    "fix_nogate":         ("nogate",  {"learn_fusion_alpha": False}),
    "fix_intra8":         ("intra8",  {"learn_fusion_alpha": False}),
    "fix_inter8":         ("inter8",  {"learn_fusion_alpha": False}),
    "fix_inter16":        ("inter16", {"learn_fusion_alpha": False}),
    "fix_learn":          ("inter8",  {"learn_fusion_alpha": True}),
    "fix_a01":            ("inter8",  {"learn_fusion_alpha": False, "alpha_init": 0.1}),
    "fix_noxattn_inter8": ("inter8",  {"learn_fusion_alpha": False, "use_cross_attn": False}),
}
FIXED_MULTI = {"fix_multi_logic": "multi_logic", "fix_multi_nogate": "multi_nogate"}


def overrides_from_wandb(cfg: dict) -> dict:
    m, t, d = cfg["model"], cfg["train"], cfg["data"]
    o = {
        "run_model": m["variant"],
        "train_max_samples": d["train_max_samples"],
        "val_max_samples": d["val_max_samples"],
        "proofwriter_depth": d["proofwriter_depth"],
        "proofwriter_config": d["proofwriter_config"],
        "k_epochs": t["epochs"],
        "train_batch_size": t["batch_size"],
        "eval_batch_size": t["eval_batch_size"],
        "learning_rate": float(t["learning_rate"]),
        "weight_decay": t["weight_decay"],
        "max_length": t["max_length"],
        "use_bf16_if_available": t["use_bf16_if_available"],
        "logic_dim": m["logic_dim"],
        "num_gates": m["num_gates"],
        "cross_attn_heads": m["cross_attn_heads"],
        "alpha_init": m["alpha_init"],
        "gate_window_axis": m["gate_window_axis"],
        "use_param_matched_baseline": bool(m["use_param_matched_baseline"]),
    }
    # MNLI runs logged proofwriter_config='mnli'; later notebooks default to ProofWriter.
    o["dataset_name"] = "nyu-mll/multi_nli" if d["proofwriter_config"] == "mnli" else "proofwriter"
    if m.get("param_matched_no_gate_update_type"):
        o["param_matched_no_gate_update_type"] = m["param_matched_no_gate_update_type"]
    if "learn_fusion_alpha" in m:
        o["learn_fusion_alpha"] = bool(m["learn_fusion_alpha"])
    if "multitask_enabled" in t:
        o["enable_multitask_training"] = bool(t["multitask_enabled"])
        o["multitask_pw_ratio"] = t["multitask_pw_ratio"]
    return o


def main(orig_path: str, out_dir: str) -> None:
    orig = {v["id"]: v for v in json.load(open(orig_path)).values()}
    jobs = []
    specs = {k: (nb, orig[rid]["config"], {}, "orig") for k, (nb, rid) in ROWS.items()}
    for k, (src, extra) in ABLATIONS.items():
        nb, cfg, _, _ = specs[src]
        specs[k] = (nb, cfg, extra, "orig")
    for k, (src, extra) in FIXED.items():
        specs[k] = (FIX_NB, orig[ROWS[src][1]]["config"], {**FIX_SINGLE, **extra}, "fixed")
    for k, src in FIXED_MULTI.items():
        specs[k] = (FIX_NB, orig[ROWS[src][1]]["config"], {"enable_gradient_checkpointing": True}, "fixed")
    os.makedirs(out_dir, exist_ok=True)
    for key, (nb, cfg, extra, tree) in specs.items():
        for seed in SEEDS:
            o = overrides_from_wandb(cfg)
            o.update(extra)
            o.update({"seed": seed, "run_seeds": [seed],
                      "wandb_project": "logic-expert-verify", "wandb_group": f"verify-{key}"})
            name = f"{key}_s{seed}"
            # LF endings: these files are read by bash on the cluster.
            json.dump(o, open(os.path.join(out_dir, f"{name}.json"), "w", newline="\n"), indent=1)
            jobs.append(f"{name} {nb} {tree}")
    open(os.path.join(out_dir, "jobs.txt"), "w", newline="\n").write("\n".join(jobs) + "\n")
    print(len(jobs), "jobs")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
