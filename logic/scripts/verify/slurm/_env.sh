# Shared environment for verification reruns on the OSU HPC.
ROOT=/nfs/hpc/share/cleasbys/logic-verify
source /nfs/hpc/share/cleasbys/envs/logic-verify/bin/activate
export HF_HOME=/nfs/hpc/share/cleasbys/hf_cache
# Gated Llama access: token from `hf auth login` (default location, independent of HF_HOME).
if [[ -z "${HF_TOKEN:-}" && -s ~/.cache/huggingface/token ]]; then
    export HF_TOKEN="$(cat ~/.cache/huggingface/token)"
fi
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:256
export TOKENIZERS_PARALLELISM=false
# The pinned notebooks require WANDB_API_KEY in the environment (read from ~/.netrc).
export WANDB_API_KEY="$(awk '/api.wandb.ai/{f=1} f&&/password/{print $2; exit}' ~/.netrc)"
export WANDB_ENTITY=ancorro-oregon-state-university
