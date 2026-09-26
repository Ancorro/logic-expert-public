#!/bin/bash
# Download Llama-3.1-8B, its tokenizer and both datasets into HF_HOME once, before the array.
#SBATCH -J lv-prefetch
#SBATCH -A eecs
#SBATCH -p share
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH -t 04:00:00
#SBATCH -o /nfs/hpc/share/cleasbys/logic-verify/logs/%x_%j.out

set -eo pipefail
source /nfs/hpc/share/cleasbys/logic-verify/repo-fixed/logic/scripts/verify/slurm/_env.sh
python - <<'EOF'
from huggingface_hub import snapshot_download
from datasets import load_dataset
p = snapshot_download("meta-llama/Llama-3.1-8B", allow_patterns=["*.json", "*.safetensors", "tokenizer*"])
print("llama at", p)
load_dataset("nyu-mll/multi_nli")
print("mnli ok")
EOF
python - <<'EOF2'
from datasets import load_dataset
load_dataset("tasksource/proofwriter", "default")   # primary source in logic/core/data_utils.load_proofwriter
print("proofwriter ok")
EOF2
