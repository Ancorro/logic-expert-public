## Rerun summary (mean ± SD over seeds)
| Row | Track | n | Val acc | Val loss | alpha | Routing H | Grad norm | Peak GB | s/epoch |
|---|---|---|---|---|---|---|---|---|---|
| Baseline | corrected | 3 | 0.925 ± 0.005 | 0.198 ± 0.003 | -- | -- | 28.52 ± 2.11 | 70.0 ± 0.0 | 429 ± 33 |
| No-Gate Control | corrected | 3 | 0.939 ± 0.004 | 0.167 ± 0.014 | 0.0100 ± 0.0000 | 0.0000 ± 0.0000 | 20.27 ± 4.88 | 70.2 ± 0.0 | 404 ± 95 |
| Routed intra G=8 | corrected | 3 | 0.936 ± 0.004 | 0.194 ± 0.023 | 0.0100 ± 0.0000 | 3.0904 ± 0.0021 | 19.63 ± 2.44 | 71.7 ± 0.0 | 427 ± 42 |
| Routed inter G=8 (a=0.01) | corrected | 3 | 0.927 ± 0.012 | 0.203 ± 0.021 | 0.0100 ± 0.0000 | 3.0904 ± 0.0017 | 22.93 ± 4.50 | 71.7 ± 0.0 | 486 ± 124 |
| Routed inter G=8 (a=0.1) | corrected | 3 | 0.936 ± 0.005 | 0.179 ± 0.021 | 0.1001 ± 0.0000 | 3.0904 ± 0.0016 | 22.21 ± 1.54 | 71.7 ± 0.0 | 590 ± 197 |
| Routed inter G=8 (learned a) | corrected | 3 | 0.938 ± 0.011 | 0.172 ± 0.007 | 0.0077 ± 0.0014 | 3.0907 ± 0.0019 | 20.91 ± 5.60 | 71.7 ± 0.0 | 518 ± 70 |
| Routed inter G=16 | corrected | 3 | 0.934 ± 0.002 | 0.198 ± 0.022 | 0.0100 ± 0.0000 | 3.7802 ± 0.0034 | 20.32 ± 3.20 | 71.7 ± 0.0 | 562 ± 53 |
| Routed inter G=8, no cross-attn | corrected | 3 | 0.937 ± 0.010 | 0.178 ± 0.031 | 0.0100 ± 0.0000 | 3.0948 ± 0.0036 | 21.70 ± 1.15 | 70.1 ± 0.0 | 602 ± 23 |
| Multi-task No-Gate Control | corrected | 3 | 0.918 ± 0.017 | 0.209 ± 0.028 | 0.1001 ± 0.0000 | 0.0000 ± 0.0000 | 16.54 ± 3.26 | 70.2 ± 0.0 | 316 ± 45 | MNLI 0.892 ± 0.012
| Multi-task Routed inter G=8 | corrected | 3 | 0.937 ± 0.009 | 0.187 ± 0.029 | 0.1001 ± 0.0000 | 3.0980 ± 0.0049 | 16.15 ± 0.24 | 71.7 ± 0.6 | 435 ± 39 | MNLI 0.882 ± 0.016
| MNLI No-Gate Control | as-published | 3 | 0.871 ± 0.006 | 0.393 ± 0.058 | 0.1001 ± 0.0000 | 0.0000 ± 0.0000 | 25.19 ± 2.11 | 76.9 ± 6.2 | 169 ± 68 |
| MNLI Routed intra G=8 | as-published | 3 | 0.860 ± 0.033 | 0.429 ± 0.114 | 0.1001 ± 0.0000 | 3.0904 ± 0.0015 | 22.07 ± 1.78 | 78.7 ± 7.0 | 185 ± 28 |
| MNLI Routed inter G=8 | as-published | 3 | 0.851 ± 0.048 | 0.451 ± 0.132 | 0.1001 ± 0.0000 | 3.0888 ± 0.0008 | 23.10 ± 2.96 | 78.7 ± 7.0 | 148 ± 10 |

## Verification of published numbers

| Row | Paper acc | Rerun acc | Seed-42 rerun | Original W&B | Tolerance | Verified |
|---|---|---|---|---|---|---|
| MNLI No-Gate Control | 0.888 | 0.871 ± 0.006 | 0.876 | 0.888 | ±0.030 | yes |
| MNLI Routed intra G=8 | 0.880 | 0.860 ± 0.033 | 0.88 | 0.88 | ±0.066 | yes |
| MNLI Routed inter G=8 | 0.878 | 0.851 ± 0.048 | 0.87 | 0.878 | ±0.096 | yes |

## Pre-registered comparisons

| Comparison (seed-paired acc diff) | n | mean diff | SE | supported (mean>0 and >2SE) |
|---|---|---|---|---|
| H1 routed inter8 > No-Gate (corrected) | 3 | -0.0127 | 0.0083 | no |
| H2 inter8 vs no-cross-attn (corrected) | 3 | -0.0105 | 0.0114 | no |
| routed intra8 > No-Gate (corrected) | 3 | -0.0036 | 0.0019 | no |
| No-Gate > Baseline (corrected) | 3 | +0.0139 | 0.0021 | yes |
| Routed inter16 > No-Gate (corrected) | 3 | -0.0050 | 0.0029 | no |
| multi-task Routed > No-Gate, ProofWriter (corrected) | 3 | +0.0183 | 0.0058 | yes |
| multi-task Routed > No-Gate, MNLI (corrected) | 3 | -0.0100 | 0.0083 | no |
| MNLI Routed inter > No-Gate | 3 | -0.0200 | 0.0247 | no |
| MNLI Routed intra > No-Gate | 3 | -0.0107 | 0.0157 | no |
