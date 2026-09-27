#!/bin/bash
# 2-GPU (A40) variant of rerun_array.sbatch. One verification rerun per array index; index N = line N+1 of manifest/jobs.txt.
# Submit:  sbatch --array=0-68%8 $ROOT/repo-fixed/logic/scripts/verify/slurm/rerun_array.sbatch
# Column 3 of jobs.txt picks the code tree: repo-orig (published loader, rulebase dropped)
# or repo-fixed (theory+question).
# Resumable: a job whose result JSON already exists exits immediately.
#SBATCH -J lv-rerun-mp
#SBATCH -A eecs
#SBATCH -p ampere
#SBATCH --gres=gpu:2
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 08:00:00
#SBATCH -o /nfs/hpc/share/cleasbys/logic-verify/logs/%x_%A_%a.out

set -eo pipefail
export VERIFY_MP=1   # 2x A40 48GB: split backbone layers across both GPUs
source /nfs/hpc/share/cleasbys/logic-verify/repo-fixed/logic/scripts/verify/slurm/_env.sh
read -r NAME NB TREE < <(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$ROOT/repo-fixed/logic/scripts/verify/manifest/jobs.txt")
V=$ROOT/repo-$TREE/logic/scripts/verify
export VERIFY_OVERRIDES="$V/manifest/$NAME.json"
export VERIFY_OUT="$ROOT/out/$NAME.json"
if [[ -s "$VERIFY_OUT" ]]; then echo "done already: $NAME"; exit 0; fi
# Atomic claim so the same manifest can be submitted to several partitions without duplicates.
mkdir -p "$ROOT/claims"
if ! mkdir "$ROOT/claims/$NAME" 2>/dev/null; then echo "claimed elsewhere: $NAME"; exit 0; fi
trap 'rc=$?; [[ $rc -ne 0 ]] && rmdir "$ROOT/claims/$NAME"; exit $rc' EXIT
echo "${SLURM_JOB_PARTITION:-?} $(hostname) ${SLURM_ARRAY_JOB_ID:-}_${SLURM_ARRAY_TASK_ID:-}" > "$ROOT/claims/$NAME.where"

echo "job=$NAME notebook=$NB tree=$TREE host=$(hostname) start=$(date -Is)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
cd "$ROOT/repo-$TREE"   # notebooks locate the repo root from the working directory
python -u "$V/gen/$NB.py"
echo "end=$(date -Is)"
