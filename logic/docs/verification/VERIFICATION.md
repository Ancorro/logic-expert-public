# Results verification (branch `verify-results`, started 2026-09-26)

Scope: verify every number in the paper (`ancorro.github.io/logic-expert/paper.tex`,
compiled to `logic/LaTeX_dir/latex/Logic_Expert.pdf`) by (1) tracing it to the W&B run
that produced it and (2) re-running every table row with 3 seeds on the OSU HPC.

## 1. Code provenance

- Paper runs were launched from Colab notebooks (`V5_Eval.ipynb`, `V6_multi_eval.ipynb`)
  cloned from `Ancorro/AI535project` branch `V5`. W&B recorded no git commit, so code is
  pinned by run start time: the ProofWriter sweep (group `baser`, 2026-03-20 10:48 PDT)
  maps to AI535project `f6714ce`; MNLI / multi-task runs map to `e981d1b`.
- `logic/core` at `f6714ce`, `e981d1b`, V5 HEAD and this public repo are identical
  (only a blank-line difference in `__init__.py`). Reruns therefore use this repo's code.
- The appendix "earlier version" runs (group `v3-eval-sequential`, 2026-03-17) map to
  AI535project `41e9737`..`fdcd4b7` (no `logic/core` change within that window).
  That version **already had layerwise cross-attention**; the main difference is that gate
  windows were built with `torch.roll` over the gate-channel axis (circular wrap-around).

Snapshot of all original runs: `wandb_original_runs.json` (script: `pull_wandb.py`).

## 2. Provenance audit (paper value vs. logged W&B value)

Main table (ProofWriter, 10k train / 5k val, seed 42, 3 epochs, final-epoch `val/acc`):

| Paper row | Paper acc / loss / alpha / H / grad | W&B run | Logged acc / loss / alpha / H / grad | Status |
|---|---|---|---|---|
| Baseline | 0.417 / 1.063 / - / - / 29.3765 | `baseline_llama` (wy9c4qdh) | **0.4646** / 1.063 / - / - / 29.3765 | acc wrong |
| No-Gate Control | 0.525 / 0.820 / 1e-5 fixed / - / 11.1387 | `augmented_mlp_llama` (p77m4hx7) | **0.5416** / 0.820 / **0.01** / 0 / 11.1387 | acc, alpha wrong |
| Routed intra G=8 | 0.542 / 0.818 / 1e-5 fixed / 3.0675 / 10.5972 | `logic_intra_gates8` (45o0qen1) | 0.5418 / 0.818 / **0.01** / 3.0675 / 10.5972 | alpha wrong |
| Routed inter G=8 a=0.1 | 0.546 / 0.819 / 0.1 / 3.0668 / 10.4154 | `logic_fusion_pt1` (q4zj8fhp) | 0.5456 / 0.819 / 0.1 / 3.0668 / **10.4054** | grad swapped |
| Routed inter G=8 a=0.01 (best) | **0.562** / 0.816 / 0.01 / 3.0666 / 10.4054 | `logic_inter_gates8` (fa3al9os) | **0.5474** / 0.816 / 0.01 / 3.0666 / **10.4154** | acc wrong (headline), grad swapped |
| Routed inter G=8 learned | 0.546 / 0.819 / 0.006 / 3.0666 / 10.5527 | `logic_learn_fusion` (fkv5gltf) | 0.5460 / 0.819 / 0.0062 / 3.0666 / 10.5527 | OK |
| Routed inter G=16 | 0.501 / 0.837 / 1e-5 fixed / 3.7483 / 10.9610 | `logic_inter_gates16` (w8fnquk8) | **0.5034** / 0.837 / **0.01** / 3.7483 / 10.961 | acc rounding, alpha wrong |

- 0.562 and 0.417 appear in no logged run, summary or history. 0.417 matches `val/acc_min`
  (worst mini-eval) of an unrelated V3 baseline run; 0.525 matches no run.
- The Baseline's full-val accuracy is 0.4646 at **every** epoch in every Baseline run
  (V3 and V5): the classifier collapsed to a constant prediction.
- Using the logged values, the best routed variant beats No-Gate Control by 0.6 pp
  (0.5474 vs 0.5416), about 1 binomial SE at n=5000 (SE ~ 0.70 pp), not the 3.7 pp claimed.
- The paper says alpha was 1e-5 for No-Gate, intra and G=16; every logged run used 0.01.
- Parameter-count table: all four values match `perf/trainable_params`. Training is a full
  fine-tune (backbone unfrozen).

MNLI (5k train / **500** val, 2 epochs, seed 42): No-Gate 0.888 / 0.337 (`baseline_mnli`),
intra 0.880 / **0.400** (paper 0.36, `logic_intra`), inter 0.878 / 0.392 (`logic_mnli`).
The table rounds accuracy to 2 decimals (0.89 / 0.88 / 0.88). The routed MNLI runs used
alpha = 0.1. The 500-example val set (SE ~ 1.5 pp) is not stated in the paper.

Multi-task (10k train, 1k val per task, 6 epochs, seed 42): No-Gate PW 0.544 / MNLI 0.879
(`big_baseline`), routed PW 0.536 / MNLI 0.881 (`big_logic`). OK. The paper says "matched to
single-task runs"; it actually ran 6 epochs vs 3.

Appendix "earlier version without cross-attention, mean of 3 seeds":
- Routed-Gate-Only 0.5348 / 0.832 is seed 42 acc plus seed 777 loss, not a mean.
  The 3-seed mean is 0.5330 +/- 0.0036 (0.5348, 0.5288, 0.5354).
- The "No-Gate Control" row (0.4646 / 1.065) is actually `baseline_llama` (no parallel
  branch, collapsed), not a No-Gate Control.
- The architecture had cross-attention (see section 1), so the caption and the conclusion
  drawn from it are unsupported. Decision (user, 2026-09-26): replace the table with a real
  routed-without-cross-attention ablation.

## 2b. Critical data-pipeline finding: the rulebase was never shown to the model

`tasksource/proofwriter` (the first source tried by `logic/core/data_utils.load_proofwriter`)
has columns `id, maxD, NFact, NRule, theory, question, answer, QDep, QLen, allProofs, config`.
The loader's text builder checks for `input`, then `context`, then any of `facts/rules/question`.
Only `question` matches, so every example is rendered as `"Question: <question>"` and the
`theory` (facts + rules) is dropped. Decoded model inputs from the repo loader
(validation, seed 42):

    'Question: Fred is not big.'                   label 2
    'Question: The rabbit does not need the lion.' label 0
    'Question: The cat does not chase the mouse.'  label 1

Consequences:
- Every ProofWriter number in the paper (main table, multi-task ProofWriter column, old
  appendix table) measures **question-only** classification: surface priors of the
  question text with no premises. The task is unsolvable above chance-plus-artifacts, and
  it is not a test of multi-step logical inference.
- Validation class prior (seed 42, n=5000): Unknown 0.4646, False 0.2706, True 0.2648.
  The Baseline's constant 0.4646 is exactly the majority class ("always Unknown"). Its
  epoch-0 accuracy of 0.2648 is exactly "always True".
- 5000 validation examples contain only 1191 distinct question strings.
- MNLI is unaffected: its collate uses premise and hypothesis pairs directly.

## 3. Pre-registered verification protocol (written before any rerun finished)

- Rerun every row of the main, MNLI and multi-task tables, plus the new no-cross-attention
  ablation, with seeds {42, 777, 123} using the exact W&B config of the original run.
  Seed 42 is a determinism check against the original run.
- Hardware differs (H100 80GB on HPC vs RTX PRO 6000 Blackwell on Colab), so bit-exact
  reproduction is not expected.
- Metric: final-epoch full-validation `val/acc` (no best-epoch selection).
- **An original value counts as verified** if it lies within rerun mean +/- 2 x max(SD
  across seeds, binomial SE of the val set). Otherwise the paper reports the rerun
  mean +/- SD and the change is logged below.
- **Comparative claims** (for example, routed > No-Gate Control) are kept only if the
  3-seed mean difference exceeds 2 x its standard error (seed-paired). Otherwise they are
  reworded as "comparable".
- **Baseline collapse check:** compute the validation class prior. If Baseline accuracy
  equals the majority-class rate, the paper must say so.

### 3a. Second bug: the Baseline is input-blind by construction

`BaselineModel.forward` pooled `last_hidden_state[:, 0, :]`. For a causal decoder
(Llama), position 0 is the BOS token, and under causal attention it cannot attend to any
later token. The pooled vector is therefore identical for every input, and the head can
only learn the class prior. That is why every Baseline run sits at exactly 0.4646 (the
majority class) from epoch 1 onward. `LogicLlamaModel` correctly pools the last
non-padding token, so the paper's Baseline vs. logic comparison was never a comparison of
architectures. Fix (Track B, `repo-fixed`): `BaselineModel` now pools the last non-padding
token for causal backbones, the same rule as `LogicLlamaModel`. Track A keeps the
published pooling.

### 3a'. Hardware for reruns

`dgxh` (H100 80GB) had a ~900-job backlog with an estimated start more than 24 h out, so an
opt-in 2-GPU path was added (`VERIFY_MP=1`, `logic/core/model_parallel.py`): backbone layers
split across 2x A40 48GB in one process, logic modules and head on GPU 0, no sharding.
A tiny-model fp64 check on a CPU+CUDA split gave |Δlogits| < 2e-7.

**Abandoned (2026-09-27).** On the `ampere` A40 nodes, direct GPU-to-GPU (P2P) copies
silently corrupted tensors: a `.to("cuda:1")` round trip mismatched nearly every element,
while the same copy staged through CPU was exact. Split-model forwards diverged from
single-GPU from the first layer on GPU 1 (layer 16) and produced NaN losses. This is a
node/driver fault, not a code bug; it has not been reported to HPC staff yet. No result in
this report used the model-parallel path.

All reported reruns ran on one GPU on `dgxh` (H100 80GB on dgxh-2/3, H200 141GB on dgxh-4),
using `slurm/rerun_array.sbatch` with atomic claims (`claims/<job>`); the node of every run is
in `claims/<job>.where`. MNLI runs have no gradient checkpointing and dynamic padding, so
peak memory depends on the seed's batches: `mnli_nogate_s777` ran out of memory on an H100
(79 GB in use) and was rerun on the H200; the remaining MNLI jobs were pinned to `h200`. This
changes memory headroom only, not the computation.

**Scope reduction (user decision, 2026-09-27).** Only runs that appear in the paper are
rerun: the corrected ProofWriter track (Track B below, 30 runs) and the MNLI rows (9 runs).
The question-only Track A ProofWriter reruns were dropped, so the published ProofWriter
numbers are not re-measured; they are superseded by Track B, and the question-only finding
rests on the loader audit in section 2b.

### 3b. When the bug entered

- 2026-03-12 to 03-13: the loader used `longface/ProofWriter` (columns `facts, rules, question, ...`).
  The facts/rules/question branch was written for that schema and worked.
- 2026-03-14 (AI535project `c5397e6`): `tasksource/proofwriter` became the first source. It
  stores the rulebase in `theory`, so the same branch silently matched only `question`.
- Every paper run (2026-03-17 to 03-21) is after this change. The early node5/node7
  exploratory runs (Mar 13 to 15) are a mix.

### 3c. Two rerun tracks (user decision, 2026-09-26)

- **Track A, as-published** (`repo-orig`, loader bug intact): verifies the published numbers
  as a *question-only probe*, the analogue of hypothesis-only baselines in NLI
  (Poliak et al., 2018; Gururangan et al., 2018). MNLI rows are rerun here too; that
  loader is unaffected.
- **Track B, corrected** (`repo-fixed`): `data_utils.load_proofwriter` renders
  `"{theory}\nQuestion: {question}"` and truncates from the left (max 250 tokens, so the
  256 cap never truncates). Corrected inputs average 120 tokens, versus 8.5 before.
  - All Track B rows use one driver (`V6_multi_eval@7172b79`) with the paper's
    hyperparameters (lr 1e-5, wd 0.01, bs 30, 3 epochs, 10k/5k, logic_dim 256).
  - Gradient checkpointing is on. It is numerically equivalent and only slower.
  - "Fixed" alpha rows really freeze alpha (`learn_fusion_alpha=False`). The learned-alpha
    row uses V6's 20x alpha learning rate.
  - Rows: Baseline, No-Gate Control, routed intra G=8, inter G=8, inter G=16,
    learned alpha, alpha=0.1, inter G=8 without cross-attention, and multi-task
    (No-Gate vs routed).

### 3d. Pre-registered claims for Track B (written before any Track B run)

- **H1 (paper's main claim):** routed inter G=8 > No-Gate Control on corrected ProofWriter.
  Supported only if the seed-paired mean difference is > 0 and > 2 SE.
  Otherwise the paper states that routing gives no measurable gain over a
  parameter-matched control.
- **H2 (cross-attention):** routed inter G=8 vs the same model without cross-attention,
  using the same test.
- **H3 (baseline health):** if the corrected Baseline still sits at the majority rate
  (+/- 1 pp) on all seeds, the paper reports it as a failed (collapsed) baseline under these
  hyperparameters, not as evidence for the parallel branch.
- **Kill criterion:** if routed <= No-Gate Control on corrected data, the abstract and
  conclusion are rewritten to a negative or neutral result. No retuning to rescue the
  claim is done inside this verification.

### 3e. Third finding: "learned" alpha cannot move from 0.1 (bf16 rounding)

`FusionMLP` creates alpha in fp32 (`fusion.py:35`), but `LogicLlamaModel` casts the fusion
module to the backbone dtype (`logic_llama_model.py:263`), which is bf16. Every driver gives
alpha AdamW lr = 20 x base = 2e-4, and an Adam step is at most about lr in size. The bf16
spacing is 2^-11 ≈ 4.9e-4 around 0.1 and 2^-14 ≈ 6.1e-5 around 0.01. So:

- alpha_init = 0.1 (the alpha=0.1 row, all MNLI rows, both multi-task rows): a 2e-4 step is
  below half the spacing (2.4e-4) and rounds away. Alpha stays at bf16(0.1) = 0.100098 for
  the whole run even with `learn_fusion_alpha=True`. The reruns confirm this: `delta = 0`
  every epoch with a non-zero `alpha_grad_abs`. These rows are fixed-alpha runs in practice.
- alpha_init = 0.01 (the learned-alpha row): a 2e-4 step exceeds half the spacing (3.05e-5),
  so alpha can move, in steps of at least one bf16 ulp. Check `fix_learn` in section 4.

More generally, the backbone and logic modules train in pure bf16 without fp32 master
weights, so small updates to large weights are also rounded away (cf. Zamirai et al., 2020,
"Revisiting BFloat16 Training"). This is recorded as a limitation. It is not corrected here,
because the reruns keep the paper's setup.

## 4. Rerun results (all 39 runs complete, 2026-09-27)

All numbers come from `results/raw/*.json`. Two scripts generate everything from those files:
`scripts/verify/aggregate.py results/raw results` writes `results/rerun_summary.{md,json}`,
`results/proofwriter_reruns.png` and `results/routing_heatmap_corrected.png`, and
`scripts/verify/make_tex_tables.py results` writes `results/paper_tables.tex`. The node
behind each run is in `results/raw/<run>.where`.

**Seeds and data.** The seed picks the train and validation subsets (`load_proofwriter` and
MNLI `shuffle(seed)`) as well as the initialisation. Runs with the same seed see identical
data, so the comparisons below are seed-paired and each SD includes variation from the data
subset. ProofWriter validation n = 5000 (1000 for multi-task); MNLI validation n = 500.

### 4.1 Corrected ProofWriter (Track B, rulebase + question, mean ± SD over 3 seeds)

| Row | Val acc | Val loss | alpha (final) | Routing H / ln(3G) | Params | Published (question-only) |
|---|---|---|---|---|---|---|
| Baseline | 0.925 ± 0.005 | 0.198 ± 0.003 | -- | -- | 7,504,936,963 | 0.417 |
| No-Gate Control | **0.939 ± 0.004** | **0.167 ± 0.014** | 0.01 fixed | -- | 7,582,192,483 | 0.525 |
| Routed intra G=8 | 0.936 ± 0.004 | 0.194 ± 0.023 | 0.01 fixed | 0.972 | 7,579,068,163 | 0.542 |
| Routed inter G=8, alpha 0.01 | 0.927 ± 0.012 | 0.203 ± 0.021 | 0.01 fixed | 0.972 | 7,579,068,163 | 0.562 |
| Routed inter G=8, alpha 0.1 | 0.936 ± 0.005 | 0.179 ± 0.021 | 0.1 fixed | 0.972 | 7,579,068,163 | 0.546 |
| Routed inter G=8, learned alpha | 0.938 ± 0.011 | 0.172 ± 0.007 | 0.0077 (from 0.01) | 0.973 | 7,579,068,163 | 0.546 |
| Routed inter G=16 | 0.934 ± 0.002 | 0.198 ± 0.022 | 0.01 fixed | 0.976 | 7,579,725,059 | 0.501 |
| Routed inter G=8, no cross-attn | 0.937 ± 0.010 | 0.178 ± 0.031 | 0.01 fixed | 0.974 | 7,507,724,035 | -- |

With the rulebase restored, every variant lands between 0.925 and 0.939, which is 36 to 51 pp
above the published question-only numbers. The published ProofWriter numbers therefore do
not describe the architecture on the actual task. They are replaced, not "verified".

### 4.2 Pre-registered claims (section 3d)

| Claim | Seed-paired diff | SE | Verdict |
|---|---|---|---|
| **H1** Routed inter G=8 > No-Gate Control | **-1.3 pp** (-2.5, +0.3, -1.7) | 0.8 | **not supported, so the kill criterion applies** |
| **H2** inter G=8 vs same without cross-attention | -1.1 pp | 1.1 | no difference; the no-cross-attn model is at least as good with 71M fewer parameters |
| **H3** Baseline collapsed to majority (0.465)? | Baseline = 0.925 | -- | not triggered; the Baseline learns once pooling is fixed |

Secondary comparisons, all with the same test (mean > 0 and > 2 SE):
- Routed intra G=8 vs No-Gate: -0.4 pp (SE 0.2).
- Routed inter G=16 vs No-Gate: -0.5 pp (SE 0.3).
- No-Gate vs Baseline: **+1.4 pp (SE 0.2), supported.** The No-Gate model differs from the
  Baseline by more than the parallel branch: it also has the fusion MLP and an RMSNorm before
  the head, while the Baseline puts a linear head directly on the last token. The gain belongs
  to that whole head path, not to parallel capacity alone.

**Kill criterion outcome:** on the task it was designed for, routing gives no measurable gain
over a parameter-matched control, and neither does cross-attention. The abstract and conclusion are
rewritten as a null result. No retuning was done.

### 4.3 MNLI (verification of published values; rerun on the published loader, which is unaffected)

| Row | Published | Rerun (3 seeds) | Seed 42 | Tolerance | Verified |
|---|---|---|---|---|---|
| No-Gate Control | 0.888 | 0.871 ± 0.006 | 0.876 | ±0.030 | yes |
| Routed intra G=8 | 0.880 | 0.860 ± 0.033 | 0.880 (exact) | ±0.066 | yes |
| Routed inter G=8 | 0.878 | 0.851 ± 0.048 | 0.870 | ±0.096 | yes |

All three pass the pre-registered rule, but the routed rows pass mainly because the seed
spread is large: a ±0.096 tolerance says little. Every rerun mean is 1.7 to 2.7 pp below the
published single run, which is consistent with the published values being favourable draws.
Routed vs No-Gate on MNLI is -2.0 pp (inter) and -1.1 pp (intra), neither significant. The
paper now reports the 3-seed means.

### 4.4 Multi-task (50/50 ProofWriter + MNLI, corrected ProofWriter)

| Row | ProofWriter | MNLI | Mean | Published PW / MNLI |
|---|---|---|---|---|
| No-Gate Control | 0.918 ± 0.017 | **0.892 ± 0.012** | 0.905 | 0.544 / 0.879 |
| Routed inter G=8 | **0.937 ± 0.009** | 0.882 ± 0.016 | 0.909 | 0.536 / 0.881 |

Routed minus No-Gate is +1.8 pp on ProofWriter (SE 0.6) and -1.0 pp on MNLI (SE 0.8).
**The ProofWriter gain is exploratory, not confirmatory.** This comparison was added after the
pre-registration, it is one of 9 comparisons, and with n = 3 (2 degrees of freedom) the 2 SE rule
is lenient: the 95% t critical value is about 4.3, and t here is about 3.2. It is worth a
pre-registered follow-up, but it does not override H1.

### 4.5 Mechanism diagnostics

- **Routing is close to uniform.** The normalised routing entropy H / ln(3G) is 0.97 to 0.98
  in every routed run, for both G = 8 and G = 16. In the corrected heatmap
  (`results/routing_heatmap_corrected.png`, seed 42, one validation batch, token-averaged) most
  cells sit near the uniform value 1/24 = 0.042, with isolated cells up to 0.17. The paper's
  "sparse, few dominant operators" reading does not hold. Raw entropy cannot be compared across
  different G, so the published "higher entropy at G = 16 goes with worse accuracy" claim is withdrawn.
- **Learned alpha moves down.** Starting at 0.01, alpha ends at 0.0077 ± 0.0014, and the seed-42
  trace is 0.0100, 0.0074, 0.0063, 0.0062. The model turns the logic branch down, which fits the
  null accuracy result. Alpha stays frozen at 0.1 wherever it starts there (section 3e).
- **Gradient norms** (mean over the final epoch): Baseline 28.5, No-Gate 20.3, routed 19.6 to 22.9.
  The published "routed variants have markedly lower gradient norms" pattern shrinks to
  "anything with the fusion head has lower norms than the Baseline".
- **Training length.** Corrected accuracy still rose from epoch 2 to 3 in 17 of 24 single-task
  runs and fell by less than 1 pp in the other 7 (largest drop 0.9 pp). The paper's
  claim that "longer training only decreased accuracy due to overfitting" came from the
  question-only setting and is removed.

### 4.6 Paper corrections beyond the tables

- **MNLI appendix.** The paper says 4,000 train, 200 validation, batch 30/100. The logged config
  (and the reruns) used 5,000 / 500, batch 20/10, and 2 epochs.
- **Table 1 alpha labels.** The paper lists 0.00001 (fixed) for No-Gate, intra G=8 and G=16.
  Their logged `alpha_init` is 0.01, and the original runs did not set `learn_fusion_alpha`.
- **Appendix Table A1 (old architecture without cross-attention).** Its No-Gate value, 0.4646,
  is exactly the ProofWriter majority rate, so it was also an input-blind run. It is replaced by
  the corrected no-cross-attention ablation (section 4.1).
- **Hardware.** The original runs used an RTX PRO 6000 on Colab. The reruns used one H100 80GB
  or H200 141GB per run on OSU HPC (`dgxh`, `preempt`).
