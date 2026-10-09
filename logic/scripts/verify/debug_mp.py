"""Debug: compare single-GPU vs 2-GPU (model-parallel) eval forward on real batches.

Run on a 2-GPU node from repo root:  python logic/scripts/verify/debug_mp.py {proofwriter|mnli}
"""
import sys

import torch
import torch.nn.functional as F

from logic.core.data_utils import get_tokenizer, load_proofwriter
from logic.core.logic_llama_model import LogicLlamaModel

TASK = sys.argv[1] if len(sys.argv) > 1 else "proofwriter"
NAME = "meta-llama/Llama-3.1-8B"
tok = get_tokenizer(NAME)
print("tokenizer padding_side", tok.padding_side, "pad", tok.pad_token, tok.pad_token_id, "vocab", len(tok))

if TASK == "proofwriter":
    ds, collate = load_proofwriter(model_name=NAME, max_length=256, depth="all", split="validation",
                                   config_name="default", max_samples=5000, seed=42)
    batches = [collate([ds[i] for i in range(b * 30, b * 30 + 30)]) for b in range(4)]
else:
    from datasets import load_dataset
    v = load_dataset("nyu-mll/multi_nli")["validation_matched"].shuffle(seed=42).select(range(40))
    m = {0: 1, 1: 2, 2: 0}
    batches = []
    for b in range(4):
        ex = [v[i] for i in range(b * 10, b * 10 + 10)]
        bt = tok([e["premise"] for e in ex], [e["hypothesis"] for e in ex], padding=True, truncation=True, return_tensors="pt")
        bt["labels"] = torch.tensor([m[e["label"]] for e in ex])
        batches.append(dict(bt))

for i, b in enumerate(batches):
    am = b["attention_mask"]
    print(f"batch {i}: shape {tuple(b['input_ids'].shape)} max_id {int(b['input_ids'].max())} "
          f"min_len {int(am.sum(1).min())} left_pad {bool((am[:, 0] == 0).any())}")


def build():
    torch.manual_seed(0)
    return LogicLlamaModel(model_name=NAME, logic_dim=256, num_gates=8, cross_attn_heads=4,
                           gate_window_axis="inter_token", use_no_gate_stream=True, no_gate_update_type="mlp",
                           no_gate_match_logic_params=True, routing_top_k=None, routing_train_mode="dense",
                           fusion_mode="mlp", alpha_init=0.01, learn_fusion_alpha=False, num_labels=3).eval()


def run(model, dev0):
    outs = []
    for b in batches:
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            o = model(input_ids=b["input_ids"].to(dev0), attention_mask=b["attention_mask"].to(dev0))
        lg = o.logits.float().cpu()
        outs.append(lg)
        loss = F.cross_entropy(lg, b["labels"])
        print("   nan_logits_rows", int(torch.isnan(lg).any(1).sum()), "loss", float(loss),
              "fused_nan", bool(torch.isnan(o.fused_hidden).any()))
    return outs


m = build()
print("dtype", next(m.backbone.parameters()).dtype)
m.to("cuda:0")
torch.cuda.synchronize()
print("== single GPU")
single = run(m, "cuda:0")
print("== single GPU again (determinism)")
single2 = run(m, "cuda:0")
for i, (a, b) in enumerate(zip(single, single2)):
    print(f"batch {i} max|single-single2| {float((a - b).abs().max()):.4g}")
m.enable_model_parallel(["cuda:0", "cuda:1"])  # same weights, now split
print("== model parallel")
mp = run(m, "cuda:0")
for i, (a, b) in enumerate(zip(single, mp)):
    print(f"batch {i} max|single-mp| {float((a - b).abs().nan_to_num(1e9).max()):.4g}")
