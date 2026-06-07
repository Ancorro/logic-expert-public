# Parallel Logic Expert with Operator Routing

**🌐 Interactive project page: [ancorro.github.io/logic-expert-public](https://ancorro.github.io/logic-expert-public/)** — overview, fuzzy-logic calculator, routing simulator, and results.

A logic-augmented transformer pathway that routes token-level representations into fixed
fuzzy logical operators (AND / OR / NOT) — or a straight-through estimator — so the
network learns compositional structure as an architectural prior, rather than as an
ensemble of expert networks.

At each backbone layer, hidden states are projected to a lower-dimensional Q / K space
that a running logic state queries via cross-attention. Routed attention selects logical
operators along either the **intra-token** (feature) axis or the **inter-token**
(sequence) axis, and the final logic state is fused back into the backbone before
classification — imposing a structured, interpretable composition over token features.

> Author: **Steven Cleasby-Mayeda** — Oregon State University
> Paper (CVPR-style writeup): [`logic/LaTeX_dir/latex/Logic_Expert.pdf`](logic/LaTeX_dir/latex/Logic_Expert.pdf)

---

## Highlights

- **Non-invasive logic stream** — runs in parallel to a frozen / fine-tuned HuggingFace
  backbone (Llama 3.1 family supported); no changes to attention internals.
- **Differentiable fuzzy AND / OR / NOT gates** plus a straight-through-estimator (STE)
  variant for binary-style gating, controlled by a single config flag.
- **Operator routing** with dense softmax routing in training and top-k sparse routing
  at inference, plus a configurable temperature and a cutoff ablation mode.
- **Two composition axes** — intra-token (within a token's feature vector) vs.
  inter-token (across neighboring tokens) — each ablated under matched parameter budgets.
- **Mechanistic diagnostics** logged to W&B: routing entropy, fusion alpha, gate
  utilization, per-layer logic-state norms.
- **Matched-budget baseline** (`BaselineModel`) so observed gains are attributable to the
  logic pathway rather than added capacity.

## Repo layout

```
logic/
  __init__.py                 # public re-exports (FusionMLP, LogicLayer, LogicStream, ...)
  core/
    logic_llama_model.py      # full Llama-backed logic-augmented model
    logic_stream.py           # per-layer logic-state recurrence
    logic_layer.py            # routing + gate aggregation + state update
    logic_projection.py       # [B,S,H] -> [B,S,L] projection per layer
    logic_cross_attn.py       # layerwise cross-attention into logic space
    logic_gates.py            # fuzzy AND / OR / NOT + STE variants
    routing.py                # dense / top-k / cutoff routing module
    fusion.py                 # residual fusion w/ learnable alpha
    baseline_model.py         # matched-budget baseline
    no_gate_layer.py          # ablation: same path, no gates
    data_utils.py             # ProofWriter / CoLA loaders
  docs/
    ARCHITECTURE.md           # full design write-up
    shape_flow.md             # tensor shapes through the pipeline
    results.md                # ablation notes (gate vs. no-gate, LoRA, routing)
    paper/run_configs/        # exact YAML configs used for paper runs
  notebooks/
    V0_Eval.ipynb, V5_Eval.ipynb, V6_multi_eval.ipynb
                              # eval drivers used for the paper
    logic_hf.ipynb            # interactive walkthrough on a HF backbone
  LaTeX_dir/latex/
    Logic_Expert.pdf          # paper
    plots/                    # paper figures (architecture, routing heatmap, ...)
  requirements-logic.txt
```

## Install

Requires Python 3.10+ and a CUDA-capable GPU for the Llama 3.1 8B path.

```bash
# PyTorch matched to your CUDA version, e.g. CUDA 12.6:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126

# Project deps:
pip install -r logic/requirements-logic.txt
```

For Llama 3.1 access you'll need a HuggingFace account with the gated-model agreement
accepted, and a token exposed via `HF_TOKEN` / `huggingface_hub.login(...)`.

## Quickstart

The cleanest entry point is [`logic/notebooks/logic_hf.ipynb`](logic/notebooks/logic_hf.ipynb),
which loads a backbone, wraps it with `LogicLlamaModel`, and exercises the gates,
routing, and fusion in isolation.

Programmatic use:

```python
from logic.core.logic_llama_model import LogicLlamaModel

model = LogicLlamaModel.from_pretrained(
    backbone_name="meta-llama/Llama-3.1-8B-Instruct",
    num_labels=2,
    logic_dim=64,
    gate_mode="soft",          # or "ste"
    routing_train_mode="dense",  # or "cutoff"
    routing_temperature=1.0,
)

logits = model(input_ids, attention_mask=attention_mask).logits
```

## Reproducing the paper runs

The exact configs used to produce the paper's intra-token and inter-token comparisons
live under [`logic/docs/paper/run_configs/`](logic/docs/paper/run_configs/). Each
directory contains matched `base.yaml` / `logic.yaml` pairs so the only delta between a
baseline and a logic-augmented run is the logic pathway itself.

## Citation

If you reference this work, please cite the paper:

```bibtex
@misc{cleasbymayeda2026logicexpert,
  author = {Cleasby-Mayeda, Steven},
  title  = {Parallel Logic Expert with Operator Routing},
  year   = {2026},
  note   = {Course project, Oregon State University AI 535}
}
```

## License

MIT — see [LICENSE](LICENSE).
