"""Localize single-vs-split divergence: per-layer hidden states of the bare backbone."""
import torch
from transformers import AutoModel
from logic.core.data_utils import load_proofwriter
from logic.core.model_parallel import dispatch_backbone

NAME = "meta-llama/Llama-3.1-8B"
ds, collate = load_proofwriter(model_name=NAME, max_length=256, depth="all", split="validation",
                               config_name="default", max_samples=5000, seed=42)
b = collate([ds[i] for i in range(8)])
bb = AutoModel.from_pretrained(NAME).to("cuda:0").eval()
print("attn impl", bb.config._attn_implementation, "dtype", bb.dtype)
store = {}
def rec(tag):
    def h(i):
        def f(_m, _a, o):
            store.setdefault(tag, {})[i] = (o[0] if isinstance(o, tuple) else o).float().cpu()
        return f
    return h
def run(tag):
    hs = [l.register_forward_hook(rec(tag)(i)) for i, l in enumerate(bb.layers)]
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        out = bb(input_ids=b["input_ids"].to("cuda:0"), attention_mask=b["attention_mask"].to("cuda:0"))
    for h in hs: h.remove()
    return out.last_hidden_state.float().cpu()
a = run("single")
dispatch_backbone(bb, ["cuda:0", "cuda:1"])
c = run("mp")
m = b["attention_mask"].bool()
for i in sorted(store["single"]):
    d = (store["single"][i] - store["mp"][i]).abs()
    print(f"layer {i:2d} max|diff| real-tokens {float(d[m].max()):.4g}  pad-tokens {float(d[~m].max()) if (~m).any() else 0:.4g}")
print("last_hidden max|diff| real", float((a - c).abs()[m].max()))
