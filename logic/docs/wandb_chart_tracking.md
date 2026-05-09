# W&B Chart Tracking Guide

This guide defines what to track in Weights & Biases across all experiment nodes,
and what to additionally track per node in logic_hf.ipynb.

## Scope

- Global charts: should appear for every training run.
- Node-specific charts: should be emphasized for the node's experiment goal.
- Naming convention: use consistent prefixes so panels auto-group cleanly.

## Global Charts (All Nodes)

Track these for every run where training occurs (Node 3, Node 4, Node 5).

### 1) Training Quality

- train/loss vs trainer_step
- val/loss vs epoch
- val/acc vs epoch

Why: gives optimization health and generalization trend in one view.

### 2) Optimization and Stability

- train/grad_norm vs trainer_step
- val/lr vs epoch
- reliability/non_finite_loss_count vs epoch

Why: catches exploding gradients, scheduler issues, and numerical failures early.

### 3) Throughput and Cost

- perf/samples_per_sec vs epoch
- perf/tokens_per_sec vs epoch
- time/epoch_sec vs epoch
- time/total_train_sec (summary)

Why: compares training speed and hardware utilization across variants.

### 4) Resource Usage

- system/peak_gpu_mem_gb vs epoch

Why: identifies variants that are too memory-heavy for deployment or scaling.

### 5) Run Identity and Config Columns (Runs Table)

Add these columns in the W&B runs table:

- config.wandb.group
- config.wandb.run_name
- config.model.gate_mode
- config.model.routing_top_k
- config.model.use_no_gate_stream
- config.model.lora.enabled
- config.train.epochs
- config.train.batch_size
- config.train.learning_rate
- summary.val/acc

Why: makes comparison possible without opening each run page.

## Node-Specific Tracking

## Node 2: First Benchmark (Single Benchmark)

Node 2 is benchmark-first, not training-first.

Track and save:

- Dataset-level project_model accuracy
- Dataset-level llama baseline accuracy
- Delta: project_model accuracy minus llama accuracy

Recommended artifact/report files:

- runs/logic_vs_llama31_report.json

Recommended charts (manual or report dashboard):

- Bar chart: project vs llama accuracy by dataset
- Delta chart: accuracy delta by dataset

## Node 3: Four-Way Variants

Variants: baseline_a, baseline_b, logic_a, logic_b.

Primary comparison charts:

- val/acc by variant (overlay)
- val/loss by variant (overlay)
- train/loss by variant (overlay)

Secondary diagnostics:

- perf/samples_per_sec by variant
- system/peak_gpu_mem_gb by variant

Interpretation focus:

- Baseline A vs Logic A: logic stream value without LoRA.
- Baseline B vs Logic B: logic stream value with LoRA.
- A vs B within baseline/logic: LoRA contribution.

## Node 4: No-Gate Attribution Control

Variants: nogate_a, nogate_b (parallel stream without logic gates/routing).

Primary comparison charts:

- val/acc for Node 4 variants vs Node 3 logic variants
- val/loss for Node 4 variants vs Node 3 logic variants

Attribution diagnostics:

- system/peak_gpu_mem_gb (cost parity checks)
- perf/samples_per_sec (throughput parity checks)

Interpretation focus:

- If no-gate ~= logic: gains may be mostly extra capacity.
- If logic > no-gate at similar cost: logic computation likely adds value.

## Node 5: LoRA No-Gate vs LoRA Soft vs LoRA STE

Variants: lora_nogate, lora_soft, lora_ste.

Primary comparison charts:

- val/acc by variant
- val/loss by variant
- train/loss by variant

Gating-specific diagnostics:

- train/routing_entropy vs trainer_step
- train/fusion_alpha vs trainer_step

Interpretation focus:

- lora_soft vs lora_ste: soft vs binary routing behavior tradeoff.
- lora_nogate vs lora_soft/lora_ste: routing logic contribution with LoRA held constant.

## Node 6: Inference Top-k Sweep vs Soft Test

Variants: soft_test, topk_1, topk_2, topk_4, topk_8 (bounded by num_gates).

Node 6 is evaluation-first from a fixed checkpoint.

Track and save:

- Project accuracy by dataset for each top-k variant
- Relative change vs soft_test baseline

Recommended artifact/report files:

- runs/node6_topk_vs_soft_summary.json
- runs/logic_vs_llama31_node6_<variant>.json

Recommended charts:

- Line chart: accuracy vs top-k for each dataset
- Bar chart: soft_test delta for each top-k

Interpretation focus:

- Find smallest top-k with minimal accuracy drop.
- Quantify speed/quality tradeoff at inference routing sparsity.

## Suggested W&B Dashboard Layout

Place panels in this order:

1. Run table with config columns
2. val/acc overlay
3. val/loss overlay
4. train/loss overlay
5. grad_norm + non_finite_loss_count
6. throughput and epoch time
7. peak GPU memory
8. node-specific panels (routing entropy, fusion alpha, top-k sweep charts)

## Panel Naming Convention

Use short, stable titles:

- Global/Val Accuracy
- Global/Val Loss
- Global/Train Loss
- Global/Grad Norm
- Global/Throughput Samples
- Global/GPU Peak Memory
- Node3/Four-Way Accuracy
- Node4/No-Gate Attribution
- Node5/Gate Mode Comparison
- Node6/TopK Sweep

## Quick Start Checklist

For each new experiment batch:

1. Ensure group and run_name are set in config generation.
2. Confirm required global metrics appear in first epoch.
3. Add/pin key config columns in runs table.
4. Save a dashboard view per node and one global view.
5. Export summary JSON for benchmark-only nodes.
