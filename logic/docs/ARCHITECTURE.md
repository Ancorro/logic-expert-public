# Logic-Stream Architecture

## Overview
This project augments a HuggingFace transformer backbone with a parallel logic stream.

The objective is to add a structured reasoning pathway without sacrificing the transfer and scaling properties of pretrained transformers. Instead of editing attention blocks, logic computation is implemented as a parallel module that reads intermediate hidden states and writes back a compact sequence-level signal. This preserves compatibility with off-the-shelf HuggingFace backbones and enables controlled comparison against standard fine-tuning.

Design rationale and hypothesis: if a transformer is augmented with a compact, layerwise-accumulated logic state and this state is fused into the final representation, then classification performance should improve on tasks requiring compositional structure relative to a matched baseline, with minimal degradation in pretrained feature reuse. This hypothesis is testable via architecture-controlled ablations that vary only logic-path components while holding backbone, data, and optimization fixed.

Two model paths are supported:
- Logic-augmented: backbone + logic stream + fusion + task head
- Baseline: backbone + task head only

The logic stream is non-invasive with respect to transformer internals: it consumes per-layer hidden states, iteratively updates a sequence-level logic state, and fuses that state into the final hidden representation before prediction.

This decomposition yields a clean ablation axis. The baseline and logic-augmented variants share tokenizer, backbone family, data pipeline, and optimizer settings, so observed deltas are attributable to the added logic pathway rather than architectural confounds.

## Component Map
- Backbone (AutoModel): produces final hidden state; per-layer hidden states are either streamed via hooks (default) or returned with output_hidden_states=True (fallback)
- LogicProjection: maps [B,S,H] -> [B,S,L] per layer
- LogicStream: stacks one LogicLayer per transformer layer
- LogicLayer: routing, gate aggregation/composition, logic state update
- RoutingModule: dense softmax routing in training, top-k sparse routing in inference
- FusionMLP: residual fusion with learnable scale alpha
- Task head: linear classifier from pooled fused hidden state

The backbone supplies high-capacity token representations, while the logic path enforces a low-dimensional compositional bottleneck. LogicProjection maps [B,S,H] to [B,S,L], reducing the cost of expert routing and gate composition. Each LogicLayer performs a recurrent-style update: routed token evidence is aggregated and merged into a persistent logic_state.

RoutingModule provides adaptive expert selection. Dense softmax routing is used during training for stable gradients; top-k sparse routing is used at inference to reduce compute. FusionMLP injects the final logic_state through residual fusion with learnable scale alpha, allowing the model to tune reliance on logic features. The task head remains linear to isolate representational gains to logic and fusion modules.

## Data Flow
```text
Input tokens + attention mask
  -> HF backbone
  -> logic stream updates per layer during backbone forward (default memory-optimized path)
     or consumes returned hidden_states (fallback path)
  -> per-layer logic projection [B,S,H] -> [B,S,L]
  -> logic stream across N layers (persistent logic_state [B,L])
  -> broadcast final logic_state to [B,S,L]
  -> residual fusion with final backbone hidden [B,S,H]
  -> pooling
     - encoder models: token at position 0
     - decoder/causal models: last non-padding token
  -> task head -> logits [B,num_labels]
```

The forward pass is defined by explicit tensor interfaces. Hidden states from successive transformer layers are projected into logic space and consumed in depth order, allowing cumulative evidence integration as features become more abstract. The resulting logic_state is sequence-level and acts as a compact latent summary of rule-relevant information.

Before classification, logic_state is broadcast across token positions and fused with the final backbone hidden state. Pooling follows standard backbone conventions (position 0 for encoder models; last non-padding token for decoder/causal models), preserving compatibility with conventional classification heads.

## Tensor Shape Mini-Table
| Stage | Shape |
|---|---|
| Backbone layer hidden | [B,S,H] |
| Projected token logic | [B,S,L] | #DEPRECATED
| Routing weights | [B,S,G] |       
| Gate inputs (einsum) | [B,G,L] |
| Logic state (per/final) | [B,L] |
| Broadcast logic state | [B,S,L] |
| Fused hidden | [B,S,H] |
| Logits | [B,num_labels] |

The shape progression highlights alternating computation between token space and compact logic space. This alternation preserves contextual detail while imposing a global bottleneck for structured aggregation. Explicit dimensional contracts also improve implementation reliability by making routing, einsum, and broadcast assumptions auditable.

## Training Integration
- Model selection is config-driven (use_logic_stream): logic model vs baseline model.
- Stage 1 logic training freezes the backbone and trains logic/fusion/head modules.
- Optimization uses AdamW, warmup scheduling, and gradient clipping.
- Typical metrics include validation loss/accuracy, plus routing entropy and fusion alpha for logic runs.
- Runs save config.json and checkpoint.pt under runs/run_<pid>/.

Training is organized for attribution and reproducibility. Stage 1 freezes the backbone and optimizes only logic, fusion, and head parameters, directly testing whether the added pathway contributes predictive signal. This setup can be extended to later joint fine-tuning once isolated gains are established.

Routing entropy and fusion alpha serve as mechanistic diagnostics. High entropy can indicate weak expert specialization; near-zero alpha can indicate that fused logic features are being ignored. Saving per-run config and checkpoint artifacts supports exact reruns and post-hoc analysis.

## Configuration Option Reference

This section lists the current configurable options used by the training and evaluation pipeline.

### model block

| Key | Type | Values / Notes |
|---|---|---|
| `model_name` | string | HuggingFace backbone id (for example `meta-llama/Meta-Llama-3.1-8B`) |
| `use_logic_stream` | bool | `true` = logic-augmented path, `false` = baseline path |
| `num_labels` | int | Number of classifier output labels |
| `freeze_backbone` | bool | `true` freezes backbone parameters in stage-1 style runs |
| `logic_dim` | int | Logic bottleneck dimension `L` (logic model only) |
| `num_gates` | int | Number of logic gates / experts `G` (logic model only) |
| `stream_logic_during_backbone` | bool | `true` (default) runs logic updates during backbone layer forward for lower memory; `false` falls back to `output_hidden_states` collection |
| `routing_top_k` | int | Inference-time sparse routing top-k (must be `>=1`) |
| `routing_train_mode` | string | `dense` or `cutoff` (training-time routing behavior) |
| `routing_train_cutoff` | float | Cutoff threshold in `[0,1)`; used in `routing_train_mode=cutoff` |
| `routing_temperature` | float | Softmax temperature (`>0`); lower = sharper, higher = smoother |
| `gate_mode` | string | `soft` or `ste_binary` |
| `fusion_hidden_dim` | int | Hidden width of fusion MLP |
| `alpha_init` | float | Initial fusion residual scale alpha |

Optional nested block:

| Key | Type | Values / Notes |
|---|---|---|
| `lora.enabled` | bool | Enables PEFT LoRA on backbone |
| `lora.r` | int | LoRA rank |
| `lora.lora_alpha` | int | LoRA alpha |
| `lora.dropout` | float | LoRA dropout |
| `lora.target_modules` | list[string] | Optional override for adapted module names |

### data block

| Key | Type | Values / Notes |
|---|---|---|
| `dataset` | string | Dataset name passed to data loading pipeline |
| `config` | string | Dataset config/subset |
| `text_col` | string | Input text column name |
| `label_col` | string | Label column name |
| `max_length` | int | Tokenization truncation length |
| `max_samples` | int or null | Optional sample cap for quick runs |

### train block

| Key | Type | Values / Notes |
|---|---|---|
| `batch_size` | int | Mini-batch size |
| `epochs` | int | Number of epochs |
| `learning_rate` | float | AdamW learning rate |
| `weight_decay` | float | AdamW weight decay |
| `warmup_ratio` | float | LR warmup fraction of total steps |
| `max_grad_norm` | float | Gradient clipping norm |
| `seed` | int | Global random seed |

### integrity block

| Key | Type | Values / Notes |
|---|---|---|
| `fail_closed` | bool | Required and must be `true` |
| `strict_splits` | bool | Template integrity option |
| `require_checkpoint_for_eval` | bool | Template integrity option |
| `disallow_dataset_fallbacks` | bool | Template integrity option |

### wandb block

| Key | Type | Values / Notes |
|---|---|---|
| `enabled` | bool | Enable/disable W&B logging |
| `project` | string | W&B project name when enabled |
| `entity` | string or null | Optional W&B entity |
| `run_name` | string or null | Optional run name |
| `log_model` | bool | Optional artifact logging toggle |

## Backpropagation Specification
Training uses standard supervised backpropagation from cross-entropy loss on logits.

For one batch with logits `z` and labels `y`:
- `L = CE(z, y)`
- Backward pass computes `dL/dtheta` for all trainable parameters.

Expected gradient flow in the logic-augmented path:
- `task_head` and `pre_head_norm` receive gradients directly from `L`.
- `FusionMLP` (including learnable `alpha`) receives gradients through `fused_hidden`.
- `LogicStream` and per-layer `RoutingModule` receive gradients through fusion into `logic_state`.
- `LogicProjection` receives gradients from each logic layer input.
- Backbone hidden states receive gradients through two routes when unfrozen:
  - classification route: `backbone -> pooled -> task_head`
  - logic route: `backbone hidden states -> logic_projection -> logic_stream -> fusion -> task_head`

Backpropagation behavior should be stage-controlled:
- Stage 1 (attribution): freeze backbone (`requires_grad=False`), train logic/fusion/head only.
- Stage 2 (optional adaptation): unfreeze some or all backbone layers and continue joint optimization.

Routing-specific note:
- During training, routing now supports two modes:
  - `dense`: use full softmax weights for all gates.
  - `cutoff`: zero out gates with weight below `routing_train_cutoff`, then renormalize.
- Top-k sparse routing remains inference-only efficiency (`inference=True`).

### Routing Training Modes (Ablation)
- `routing_train_mode: dense`:
  - No training-time cutoff.
  - Fully dense softmax routing weights.
  - Smoothest gradient behavior.
- `routing_train_mode: cutoff`:
  - Applies a value threshold `routing_train_cutoff` in `[0,1)` after softmax.
  - Gates below cutoff become zero, then remaining weights are renormalized.
  - If all gates are removed for a token, routing falls back to that token's argmax gate.

### Routing Temperature (Ablation)
- `routing_temperature > 0` scales router logits before softmax:
  - `weights = softmax(scores / routing_temperature)`
- Lower values (`< 1.0`) sharpen routing distributions (more peaked).
- Higher values (`> 1.0`) smooth routing distributions (more diffuse).
- `routing_temperature = 1.0` reproduces the prior behavior.

### Gate Training Modes (Ablation)
To test whether harder logical decisions improve compositional behavior, gate computation now supports two modes:
- `soft` (current baseline): gates are continuous `sigmoid` values in `[0,1]`.
- `ste_binary` (new ablation): forward pass uses hard thresholded binary-like gates, backward pass uses straight-through estimation (STE) so gradients flow as if the pre-threshold soft gate were used.

STE formulation used for gate values `g`:
- `g_soft = sigmoid(.)`
- `g_hard = 1[g_soft >= 0.5]`
- `g_ste = stopgrad(g_hard - g_soft) + g_soft`

This gives hard gate behavior in forward while preserving usable gradient signals in backward.

### What STE Skips (and What It Does Not)
STE does not skip the full logic stream. It only bypasses the non-differentiable derivative of the hard threshold used to binarize gate values.

- Forward behavior: uses hard binary-like gates (`g_hard`).
- Backward behavior: gradients flow as if soft gates (`g_soft`) were used.

So in `ste_binary` mode, the threshold operation is the only step with a surrogate gradient. Other logic-path operations still participate in normal backpropagation, including:
- routing softmax (training path),
- token-to-gate aggregation (`einsum`),
- gate scalar projection,
- logic update projection,
- fusion and task head.

In short: STE replaces the gradient of binarization, not the gradient of the whole logic pathway.

## Current Implementation Differences (Important)
The current code mostly matches the above, with the following concrete differences/constraints:

1. Stage control is config-dependent, not hard-coded.
- `train/train.py` requires `model.freeze_backbone`, but whether backbone grads are disabled depends entirely on that config value.
- This means stage-1 behavior is supported, but not automatically enforced unless config sets it true.

2. No explicit stage-2 schedule exists in one run.
- There is no built-in two-phase loop that freezes then unfreezes within a single training invocation.
- Stage 2 currently requires a separate run/config change.

3. LR schedule is warmup-then-constant.
- Scheduler linearly warms up, then stays flat (`LambdaLR` returns `1.0` after warmup).
- If decay after warmup is desired, current implementation differs and would need scheduler changes.

4. Routing mode is now config-driven in training.
- `routing_train_mode: dense` keeps prior behavior (dense softmax).
- `routing_train_mode: cutoff` adds post-softmax thresholding and renormalization.
- Top-k sparse routing is still only active when `model.eval()` sets `inference=True` in `LogicLlamaModel.forward`.

5. Gate-mode ablation is now implemented and config-driven.
- `model.gate_mode: soft | ste_binary` controls logic-gate behavior.
- `soft` reproduces prior behavior; `ste_binary` enables hard-forward / STE-backward testing.

6. Baseline pooling is not causal-aware today.
- `LogicLlamaModel` pools decoder models at the last non-padding token and encoder models at token 0.
- `BaselineModel` currently always pools token 0.
- So for causal backbones, baseline and logic variants do not yet share identical pooling semantics.

7. Apparent gains can be confounded by trainable parameter count.
- Logic variants add extra learnable modules (logic projection/stream, fusion, and sometimes additional head-side parameters), so improvements may partially reflect higher adaptation capacity rather than logic structure alone.
- To reduce this confound, report trainable-parameter counts per variant and include at least one parameter-matched baseline control (for example, a wider/deeper non-logic head with comparable trainable parameter budget).

8. Stronger attribution control: parallel stream without gates.
- Add a no-gate control that keeps the same parallel-stream scaffold (projection, per-layer accumulation path, and fusion) but removes logical gate operators and routing.
- Example replacement: each layer applies a learned linear or MLP update to token-projected features, pools to sequence state, and accumulates into the stream state without AND/OR/NOT composition.
- This isolates whether gains come from logical computation versus simply adding a second trainable pathway.
- Recommended comparison set: baseline, no-gate parallel stream, full logic stream (all with matched training budget and reported trainable-parameter counts).

## Suggested STE vs Soft Experiment
Run two matched configs and compare validation accuracy/loss plus routing entropy/fusion alpha:
- Soft gates: `configs/augmented_v1.yaml` (`gate_mode: soft`)
- STE gates: `configs/augmented_ste_v1.yaml` (`gate_mode: ste_binary`)

Keep backbone, dataset split, seed, optimizer, and training budget fixed for fair attribution.

## Suggested Routing Cutoff vs Dense Experiment
Run two matched configs and compare validation accuracy/loss plus routing entropy/fusion alpha:
- Dense routing: `configs/augmented_v1.yaml` (`routing_train_mode: dense`)
- Cutoff routing: `configs/augmented_cutoff_v1.yaml` (`routing_train_mode: cutoff`)

Keep backbone, dataset split, seed, optimizer, gate mode, and training budget fixed for fair attribution.

## Suggested No-Gate Parallel Stream Control
To test whether improvements are specifically due to gates/routing (instead of extra pathway capacity), run a no-gate variant with the same high-level topology as the logic model:
- Keep: backbone hidden-state taps, projection to logic-space, per-layer stream recurrence, fusion module, and task head.
- Remove: gate scalar projection, AND/OR/NOT composition, and routing-based token-to-gate aggregation.
- Replace with: a dense per-layer update block (for example, MLP or linear + activation) that maps projected token features to a sequence-level update and accumulates it into stream state.

Report all three variants under identical settings:
- Baseline (backbone + head)
- No-gate parallel stream control
- Full logic stream

Primary readout:
- If no-gate is close to full logic, gains are likely from added trainable pathway/capacity.
- If full logic clearly outperforms no-gate at similar parameter budget, that supports gate/routing-specific benefit.

## Suggested Routing Temperature Sweep
Run matched configs while varying only `routing_temperature` (example: `0.7`, `1.0`, `1.5`) and compare validation accuracy/loss and routing entropy.

## Potential Train-Eval Mismatch Risk

There is an important evaluation caveat in the current design:

1. During training, routing uses either:
- `routing_train_mode=dense` (full softmax), or
- `routing_train_mode=cutoff` (thresholded softmax + renormalization).

2. During evaluation/inference, routing switches to top-k sparse routing (`routing_top_k`).

This means the routing operator at test time is not identical to the training operator. That can introduce a distribution shift in gate selection and reduce accuracy, especially when:

- `routing_temperature` is low (very sharp routing),
- `routing_top_k` is small (aggressive sparsification), or
- cutoff thresholding and top-k ranking disagree on which gates survive.

Practical interpretation:

- `dense` train -> `top-k` eval is usually the largest operator gap.
- `cutoff` train -> `top-k` eval may be closer, but still not identical.

Suggested evaluation practice:

1. Report a matched-routing metric (eval with train-like routing behavior) to measure representation quality.
2. Report top-k inference metric to measure deployment behavior.
3. Treat the delta between these two as sparsification mismatch cost.

## Baseline Contrast
The baseline path removes logic projection, routing, logic layers, and fusion. It uses backbone output + task head only, providing a direct comparison point for the logic-augmented architecture.

This contrast is required for credible gain attribution. Because both paths share preprocessing, backbone choice, optimization, and evaluation, performance differences are interpretable as effects of logic-specific modules. The baseline also remains a low-complexity deployment option under strict latency or memory constraints.

## Key Files
- logic/logic_llama_model.py
- logic/logic_stream.py
- logic/logic_layer.py
- logic/routing.py
- logic/fusion.py
- logic/baseline_model.py
- train/train.py
- logic/docs/shape_flow.md
- logic/docs/ste_mode_flow_ascii.txt
- logic/docs/ste_mode_flow_mermaid.md
