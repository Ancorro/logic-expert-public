# V3_Eval Current Run Spec: Logic Model Implementation and Metrics

## Abstract

This document specifies the exact implementation and metric behavior used by the current `logic/notebooks/V3_Eval.ipynb` run protocol.

The core model extends a transformer backbone with a layer-wise logic stream. During backbone traversal, each layer hidden state is projected into logic space and used to update token-level logic state through cross-attention and routed gate composition. After the final backbone layer, the final logic state is fused once into final backbone token features through a residual linear fusion bridge. The implementation is designed to support controlled baseline comparisons, deterministic data subsampling, and reproducible experiment tracking with Weights & Biases (W&B).

The intended use is controlled comparison between:
- Baseline transformer classifier (no logic stream)
- Logic-augmented classifier (logic stream + fusion)

with matched data budget, seed policy, and training schedule.

---

## Outline

1. System Goals and Scope
2. High-Level Architecture
3. Detailed Implementation
4. Forward Pass Walkthrough
5. Training and Experiment Procedure
6. Metrics Under Test
7. W&B Logging Schema
8. Reproducibility and Fairness Rules
9. Known Limitations and Interpretability Notes

---

## 1. System Goals and Scope

### Goals
- Add a logic-computation path to a transformer classifier without changing the baseline backbone.
- Measure whether logic-path computation improves validation metrics under controlled budgets.
- Track mechanism-level diagnostics (routing entropy, fusion alpha) in addition to task metrics.

### In Scope
- Baseline vs logic-augmented training/evaluation
- Gate and routing behavior diagnostics
- W&B logging for quality, stability, throughput, and resource cost

### Out of Scope
- New gate families beyond current AND/OR/NOT composition
- Full profiler-based exact FLOPs accounting
- Large architecture rewrites of the transformer backbone

---

## 2. High-Level Architecture

### Baseline Path
The baseline variant is intentionally minimal so that it remains a strict control condition. It runs a standard backbone from `transformers.AutoModel`, applies the same tokenizer and collation pipeline used by the logic variant, and maps pooled hidden features to logits with a linear task head. The pooling path is matched to model family behavior (for example, causal-style last-token pooling versus encoder-style token-0 pooling), so any measurable differences between runs can be attributed to the logic path rather than to data or head mismatches.

### Logic-Augmented Path
The logic-augmented variant keeps the same backbone, but adds a parallel logic-state stream that is updated in sync with backbone depth. The implementation is streaming: backbone layers execute normally, while per-layer hidden states are tapped and used to update token-level logic state through cross-attention and routed gate composition. In other words, logic updates are computed during backbone traversal, but backbone internals are not rewritten in-place at each layer.

At each layer, hidden states are projected from backbone width `H` to logic width `L`, mixed with current logic state through cross-attention, and then passed through the routed logic update block. This produces a new logic state that is carried to the next layer. After the final backbone layer, the final logic state is fused once into the final backbone hidden tensor with the residual bridge formula `h_fused = h_llm + alpha * W(h_logic)` in linear-bridge mode. The learnable scalar `alpha` controls how strongly logic updates influence the final representation and is logged as an interpretability signal.

---

## 3. Detailed Implementation

### 3.1 Core Modules

#### `logic_projection.py`
This module handles dimensional alignment between backbone token features and logic features. In the current run setup, it projects tensors from [B, S, H] to [B, S, L], where B is batch size, S is sequence length, H is backbone hidden width, and L is logic width.

#### `routing.py`
The routing module converts each token's logic feature into gate assignment weights of shape [B, S, 3G], where G is the base gate count per operator bank and the factor 3 corresponds to AND/OR/NOT channel banks. In the current notebook configuration, routing is dense and soft in both training and evaluation (no top-k pruning), so each token keeps a full probability distribution over all gate channels.

#### `logic_gates.py`
This module implements differentiable gate composition primitives that approximate logical operations while remaining trainable with backpropagation. Supported compositions include AND, OR, and NOT-style combinations, along with capacity-oriented composition behavior used to modulate how gate outputs combine across token positions. The goal is to expose structured computation while preserving stable optimization dynamics.

#### `logic_layer.py`
`logic_layer.py` is the central per-layer update block. It takes token-aligned logic features and current logic state (both `[B, S, L]`) and applies a strict sequence: pre-update normalization, routing weight computation, gate-scalar projection, gate composition, logic-delta projection, then residual update plus post-residual normalization. In the notebook path, `pre_routing_mlp` is disabled, `gate_mode='soft'`, `gate_operator='all'`, and `gate_capacity=2`. It returns the updated logic state and routing weights, enabling both forward progress and mechanism diagnostics.

#### `logic_stream.py`
The logic stream constructs one logic update stage per backbone layer and manages state threading across depth. In the streaming-forward path used by the notebook, per-layer logic updates are invoked directly from `logic_stream.layers[layer_idx]` during backbone hooks, and initial logic state is seeded from the first projected backbone layer hidden state in that hook path. This depth-coupled recurrence allows later logic updates to depend on earlier ones and enables per-layer diagnostics such as routing-history traces.

#### `fusion.py`
fusion.py contains the canonical linear fusion bridge used in this phase. It maps logic-space features back into backbone feature space and injects them through a residual path: fused = llm_hidden + alpha * bridge(logic_hidden). Because alpha is learnable and logged during training, this module supports both controlled integration and interpretability; alpha near zero often indicates that the model is choosing to rely less on logic contributions.

#### `logic_llama_model.py`
This wrapper orchestrates the complete forward path: it runs the backbone, captures layerwise hidden states, updates logic state in lockstep with depth, fuses final logic state with final backbone hidden states, pools, normalizes for head input, and emits logits. It also records intermediate artifacts such as routing history and cross-attention history for debugging and analysis.

#### `baseline_model.py`
The baseline model intentionally mirrors data flow and output formatting without introducing logic modules. It runs a standard backbone forward pass, pools token representations, and applies a linear classification head. This parity in interface and training setup makes baseline-versus-logic comparisons easier to audit and interpret.

### 3.2 Operation-Level Flow For One Logic Layer

This section explains one logic-layer update in plain language, with each phrase defined before it is used.

What the symbols mean:
1. `B`: how many examples are in the batch.
2. `S`: how many token positions each example has after padding.
3. `L`: logic feature size per token.
4. `G`: number of base gates in each gate family.
5. `C`: gate capacity (`C=2` in this notebook path).
6. `token_logic` `[B, S, L]`: per-token logic features coming into this layer.
7. `logic_state` `[B, S, L]`: running logic memory from the previous layer.
8. `token_mask` `[B, S]`: token validity (`True` = real token, `False` = padding).
9. `mask` `[B, S]`: internal boolean copy of `token_mask` on the right device.

Step 1: Shape and mask checks. The layer confirms `token_logic` and `logic_state` both have shape `[B, S, L]`, then prepares `mask` as a boolean tensor. This prevents shape bugs and ensures padding is handled consistently.

Step 2: Normalize input before computing updates. `pre_update_norm` (RMSNorm) is applied to `token_logic`, producing `update_in` `[B, S, L]`. Plain meaning: the layer rescales features first so later computations are numerically stable.

Step 3: Zero out padded positions before routing. The layer applies `update_in = update_in * mask.unsqueeze(-1)`. `unsqueeze(-1)` turns `[B, S]` into `[B, S, 1]` so the mask can be broadcast across the feature dimension `L`.

Step 4: Choose routing input. In this notebook path, `pre_routing_mlp` is off, so `routing_input = update_in` with no extra transform.

Step 5: Compute routing distribution. The routing module maps `routing_input` `[B, S, L]` to `routing_logits` `[B, S, 3G]`, then softmax converts them into `routing_weights` `[B, S, 3G]`. Meaning of `3G`: there are three gate families (AND, OR, NOT), each with `G` channels. "Dense routing" here means `routing_top_k=None`, so no channels are dropped.

Step 6: Compute gate strengths, then weight by routing. `gate_scalar(routing_input)` with input `[B, S, L]` produces `gate_logits` `[B, S, 3G]`. Sigmoid converts that to `gate_values` `[B, S, 3G]` with element values in `[0, 1]`. The layer then multiplies by routing weights:
`routed_gate_values = gate_values * routing_weights`, so `routed_gate_values` is `[B, S, 3G]`.
Plain meaning: `gate_values` says how strong each gate channel is, and `routing_weights` says how much to trust each channel for each token.

Step 7: Split into gate families. `routed_gate_values` is split on the last dimension into:
1. `and_values` `[B, S, G]`
2. `or_values` `[B, S, G]`
3. `not_values` `[B, S, G]`
Each tensor now holds one gate family only.

Step 8: Compose gates with capacity (simple view). With `gate_operator='all'` and `gate_capacity=2`, each channel index looks at a tiny local window of 2 tokens along the sequence axis `S` (not along gate axis `G`) and builds three summaries:
1. AND branch (`and_values` `[B, S, G]` -> `and_out` `[B, S, G]`): high only when both values in the 2-token window are high.
2. OR branch (`or_values` `[B, S, G]` -> `or_out` `[B, S, G]`): high when at least one value in the 2-token window is high.
3. NOT branch (`not_values` `[B, S, G]` -> `not_out` `[B, S, G*C]`): takes complement-style values (`1 - x`) for each position in the 2-token window, so it expands by `C`.

Implementation detail: for each branch, causal token windows are formed by shifting along the sequence dimension of `[B, S, G]` without wrap-around, creating an intermediate window tensor of shape `[B, S, G, C]` before reduction/reshape. A token-validity mask is applied in this window space so padded or invalid tokens do not contribute to the reductions.

Plain meaning: the layer processes three different routed channel banks (`and_values`, `or_values`, `not_values`) with three different operators:
1. AND bank -> strict co-activation summary,
2. OR bank -> permissive activation summary,
3. NOT bank -> inverse/complement summary.

So these are not the same signal; they come from different channel slices of `routed_gate_values`, then each slice is transformed by a different rule over nearby token positions.

These are concatenated into `composed_gates` with shape `[B, S, G*(C+2)]`; in this run (`C=2`) that is `[B, S, 4G]`. This combined tensor is the input to `logic_update`.

Step 9: Project to logic update. `logic_update(composed_gates)` maps `composed_gates` `[B, S, 4G]` to `logic_delta` `[B, S, L]`. Plain meaning: this is the proposed change to logic memory at this layer. The mask is applied again so padded tokens cannot receive fake updates, yielding masked `logic_delta` `[B, S, L]`.

Step 10: Add update to current memory. The layer computes:
`residual_state = logic_state + logic_delta`.
`residual_state` is `[B, S, L]`. This is the standard residual add in logic space.

Step 11: Normalize the new residual state. `post_residual_norm` (RMSNorm) is applied to `residual_state`, giving `new_logic_state` `[B, S, L]`.

Step 12: Preserve old values for padding tokens. The layer applies:
`new_logic_state = torch.where(mask.unsqueeze(-1), new_logic_state, logic_state)`.
Here `mask.unsqueeze(-1)` is `[B, S, 1]`, and both `new_logic_state` and `logic_state` are `[B, S, L]`. Plain meaning: real tokens use the new value, padded tokens keep the previous value.

Step 13: Return outputs for the next layer and diagnostics:
1. `new_logic_state` `[B, S, L]` (input memory for the next depth).
2. `routing_weights` `[B, S, 3G]` (used for routing-entropy and routing-trace analysis).

In the current run protocol, this output pair is consumed immediately by the streaming path in `logic_llama_model.py`, and the same process repeats at the next backbone layer.

### 3.3 Normalization Inventory (Every Norm)

There are four normalization locations in the logic-enabled path used by the notebook.

In `logic_layer.py`, `pre_update_norm` is applied before routing and gate operations. Norm type: RMSNorm (`torch.nn.RMSNorm`) in the current run. Implementation fallback: LayerNorm if RMSNorm is unavailable. This is a pre-op norm that stabilizes the input to routing and gate scalar projections.

In `logic_layer.py`, `post_residual_norm` is applied after residual addition of `logic_delta` to logic state. Norm type: RMSNorm (`torch.nn.RMSNorm`) in the current run. Implementation fallback: LayerNorm if RMSNorm is unavailable. This is a post-residual norm that stabilizes the state passed to the next depth.

In `logic_cross_attn.py`, `pre_attn_norm` is applied to logic-state queries before cross-attention projections. Norm type: RMSNorm (`torch.nn.RMSNorm`) in the current run. Implementation fallback: LayerNorm if RMSNorm is unavailable. This is a pre-attention normalization. The cross-attention residual add itself does not apply an additional post-attention norm inside that module.

In `logic_llama_model.py`, `pre_head_norm` is applied to pooled features before the classifier head. Norm type: RMSNorm (`torch.nn.RMSNorm`) in the current run. Implementation fallback: LayerNorm if RMSNorm is unavailable. This ensures head-input scale consistency across training.

All normalization points listed above are active in the notebook execution path.

### 3.4 Mathematical Formulation (Methods Style)

This subsection provides compact equations for the main operations described above. Let batch size be $B$, sequence length be $S$, backbone width be $H$, logic width be $L$, and gate count be $G$.

Routing logits and dense routing weights are computed as:

$$
R = \frac{W_r X + b_r}{\tau}, \qquad
P = \operatorname{softmax}(R, \text{dim}=-1), \qquad
P \in \mathbb{R}^{B \times S \times 3G}
$$

where $X \in \mathbb{R}^{B \times S \times L}$ is the routing input and $\tau$ is routing temperature.

Gate scalars are formed by projecting routing input and modulating by routing weights:

$$
U = \sigma(W_g X + b_g), \qquad
U = [U_{\text{and}}, U_{\text{or}}, U_{\text{not}}], \qquad
U^{\mathrm{w}}_{*} = U_{*} \odot P
$$

where each split gate channel has shape $\mathbb{R}^{B \times S \times G}$ before composition (with NOT expanding by capacity in composition output).

Capacity-based fuzzy composition over token windows can be summarized as:

$$
\operatorname{AND}(v) = \prod_{c=0}^{C-1} \operatorname{shift}_{S}(v, c),
\qquad
\operatorname{OR}(v) = 1 - \prod_{c=0}^{C-1} \left(1 - \operatorname{shift}_{S}(v, c)\right),
\qquad
\operatorname{NOT}(v) = 1 - \operatorname{shift}_{S}(v, c)
$$

where $\operatorname{shift}_{S}(v, c)$ means shift by $c$ previous tokens along sequence axis $S$ with zero-fill (no circular wrap). Clipping to $[0,1]$ is applied before fuzzy operations, and mask-aware neutral values are used so padded tokens do not affect reductions.

Logic delta projection and residual update follow:

$$
\Delta L_t = W_\Delta \; \operatorname{Compose}(U^{\mathrm{w}}_{\text{and}}, U^{\mathrm{w}}_{\text{or}}, U^{\mathrm{w}}_{\text{not}}) + b_\Delta
$$

$$
L_{t+1} = \operatorname{Norm}_{\text{post}}\left(L_t + \Delta L_t\right)
$$

where $\operatorname{Norm}_{\text{post}}$ is RMSNorm.

Layerwise logic-query cross-attention update is:

$$
Q = W_Q\,\operatorname{Norm}_{\text{attn}}(L_t), \qquad
K = W_K\,H_t, \qquad
V = W_V\,H_t
$$

$$
A = \operatorname{softmax}\left(\frac{QK^\top}{\sqrt{d_h}}\right), \qquad
\hat{L}_t = L_t + A V
$$

where $H_t \in \mathbb{R}^{B \times S \times H}$ is current backbone hidden state and $\operatorname{Norm}_{\text{attn}}$ is pre-attention normalization.

Final fusion at the end of backbone traversal is:

$$
H_{\text{fused}} = H_{\text{last}} + \alpha\,W_f L_{\text{last}}
$$

and pooled features are normalized before classification:

$$
z = \operatorname{Norm}_{\text{head}}\left(\operatorname{Pool}(H_{\text{fused}})\right), \qquad
\ell = W_c z + b_c
$$

Mask semantics apply during logic updates so padded tokens do not contribute spurious routing or delta updates.

---

## 4. Forward Pass Walkthrough

Given tokenized inputs (`input_ids`, `attention_mask`):

The forward pass begins with standard backbone execution under the provided attention mask. As each backbone layer finishes, layer hidden states are captured for logic-stream processing. On the first layer, projected hidden states initialize the logic-state container for this forward pass.

For each depth, the current logic state is mixed with current backbone hidden states through layerwise logic-query cross-attention (with pre-attention normalization on logic queries). The attended logic representation is then passed into the routed logic layer, where routing, gate-scalar formation, composition, residual update, and post-residual normalization are executed in fixed order.

The resulting logic state is carried to the next depth and the process repeats until the final backbone layer is processed. After backbone traversal completes, the final logic state is fused once with final backbone hidden states through the configured fusion bridge (linear-bridge mode in this phase).

After fusion, pooled features are extracted according to model-family pooling policy, then normalized by pre-head normalization and projected by the task head to logits. Diagnostic artifacts such as routing history and cross-attention history are returned alongside logits.

Key tensor symbols:
- `B`: batch size
- `S`: sequence length
- `H`: backbone hidden size
- `L`: logic dimension
- `G`: number of gates

---

## 5. Training and Experiment Procedure

### Data Policy
Data loading supports deterministic subsampling through max_samples and seed values, which is critical for controlled comparisons across model variants. The training loader shuffles each epoch to maintain robust optimization behavior, while the full validation loader remains stable and unshuffled so that epoch-level validation metrics are directly comparable over time.

### Fair Comparison Policy
To preserve fairness, baseline and logic runs should share the same seed list, sample caps, epoch count, and optimizer settings. In this notebook, one model variant is selected per execution (`run_model`) and run sequentially across `run_seeds`; model-to-model comparison is performed by running the notebook once per variant with the same settings.

### Sequential Multi-Seed Execution
Runs are executed sequentially by seed for the selected model variant to reduce peak memory pressure and keep run bookkeeping simple. After each seed run, the workflow deletes the model object, triggers Python garbage collection, and clears CUDA cache when available.

### Optimization Schedule Used
The notebook uses AdamW with a fixed learning rate (`learning_rate`) and does not apply a learning-rate scheduler. The logged `val/lr` metric therefore tracks a constant value across epochs.

### Evaluation Cadence Used
Each epoch runs two validation pathways: repeated mini-evaluations and one full validation pass. Mini-evaluations are controlled by `eval_checks_per_epoch` and `mini_eval_batches_per_check` and are used to compute spread statistics (`std`, `min`, `max`, and mini means). The full validation pass provides the primary epoch-level `val/loss` and `val/acc`.

### Notebook Configuration In Use
Current notebook defaults for large runs are:
- `run_model='baseline_llama'` (switch to `'logic_llama'` for logic runs)
- `run_seeds=(42, 123, 777)`
- `train_max_samples=100000`, `val_max_samples=20000`
- `k_epochs=6`
- `learning_rate=1e-5`, `weight_decay=0.01`
- `eval_checks_per_epoch=10`, `mini_eval_batches_per_check=8`
- `fusion_mode='linear_bridge'`
- `routing_train_mode='dense'`, `routing_top_k=None`
- `gate_mode='soft'`, `gate_operator='all'`, `gate_capacity=2`

### Logic-State Lifecycle During Training
Within each forward call, logic state is initialized fresh, updated layer-by-layer, and discarded after logits are produced. There is no persistent logic memory shared across batches. This is important for interpreting diagnostics: routing entropy and fusion alpha trends represent per-batch/per-epoch behavior under repeated fresh state initialization, not long-lived recurrent memory across dataset steps.

Mask semantics are applied throughout logic updates so padded positions do not inject spurious routing or gate activity. In practice, this means masked positions are zeroed or preserved appropriately at the relevant update points, preventing artificial entropy or update artifacts from padding tokens.

---

## 6. Metrics Under Test

### 6.1 Task Quality
Task quality metrics track whether optimization is progressing and whether the model generalizes to validation data. train/loss captures fit to the training objective, while val/loss and val/acc provide the primary quality indicators used for model comparison and final curve reporting.

### 6.2 Optimization and Stability
Optimization and stability metrics are used to detect failure modes early. train/grad_norm is monitored for exploding or vanishing updates, val/lr records the effective learning rate over time, and reliability/non_finite_loss_count surfaces numerical breakdown events such as inf or nan losses.

### 6.3 Performance and Cost
Performance and cost metrics quantify practical runtime behavior. Throughput is tracked with perf/samples_per_sec and perf/tokens_per_sec, while time/epoch_sec and time/total_train_sec summarize temporal cost at epoch and run scales. system/peak_gpu_mem_gb captures memory pressure and helps explain throughput differences and run feasibility.

### 6.4 Logic-Mechanism Diagnostics
Logic-mechanism diagnostics measure whether the added logic path is actually being used. train/routing_entropy provides a coarse signal about routing diversity and specialization, and train/fusion_alpha tracks the learned strength of logic-to-backbone fusion. Together these help interpret whether quality gains align with active mechanism behavior.

### 6.5 FLOPs Estimates (Approximate)
FLOPs-related metrics are logged as explicit estimates to enable quality-versus-cost analysis without expensive profiling overhead in every run. The current approximation assumes flops_per_token_est ~= 6 * trainable_params, then scales by observed tokens and optimizer steps to derive per-step, per-epoch, and total estimates. These values are useful for relative comparison within the same setup but should not be interpreted as kernel-accurate compute measurements.

---

## 7. W&B Logging Schema

### Run Grouping
All sequential model runs should share the same W&B group so baseline and logic curves can be compared directly in aligned dashboards.

### Recommended Config Columns
At minimum, log configuration fields that determine architecture behavior, optimization schedule, and data budget. This includes grouping metadata (`config.wandb.group`, `config.wandb.run_name`), logic controls actually used (`config.model.logic_dim`, `config.model.num_gates`, `config.model.cross_attn_heads`, `config.model.alpha_init`), and core training controls (`config.train.epochs`, `config.train.batch_size`, `config.train.eval_batch_size`, `config.train.learning_rate`, `config.train.weight_decay`, `config.train.seed`). Keeping these fields complete is necessary for reliable post-hoc filtering and fair curve interpretation.

### Summary Fields
Summary fields should include final validation quality and run-level cost indicators so cross-run ranking can be done without scanning all step logs. Recommended entries are summary.val/acc, time/total_train_sec, perf/flops_total_est, and perf/trainable_params.

---

## 8. Reproducibility and Fairness Rules

Reproducibility depends on controlling the experimental degrees of freedom. Keep sample caps and seed policy fixed when comparing variants, and hold the validation split constant across all runs. Record complete configuration metadata for each run so any discrepancy can be audited after the fact. Final conclusions should rely on the multi-seed results, not single-seed outcomes.

---

## 9. Known Limitations and Interpretability Notes

Several caveats should be considered when interpreting results. FLOPs values are approximate and can differ from true kernel-level compute. Throughput and wall-time measurements are sensitive to hardware class, precision mode, and shared runtime contention, so they are most meaningful under matched environments. Routing entropy should only be compared across compatible routing configurations, and fusion alpha values near zero may indicate a weak or bypassed logic contribution path.

---
