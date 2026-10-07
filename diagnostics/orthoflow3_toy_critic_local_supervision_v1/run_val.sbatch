#!/bin/bash
#SBATCH --job-name=ortho3-local-val
#SBATCH --partition=gpu
#SBATCH --array=0-5%6
#SBATCH --gres=shard:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=6G
#SBATCH --time=02:00:00
#SBATCH --output=/home/zhihan/research/Basin_C1/diagnostics/orthoflow3_toy_critic_local_supervision_v1/val/logs/%A_%a.out
#SBATCH --error=/home/zhihan/research/Basin_C1/diagnostics/orthoflow3_toy_critic_local_supervision_v1/val/logs/%A_%a.err
set -euo pipefail
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=.10
cd /home/zhihan/research/Basin_C1
/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python \
  diagnostics/orthoflow3_toy_critic_local_supervision_v1/evaluate.py \
  --phase val --shard "${SLURM_ARRAY_TASK_ID}" --shards 6 \
  --cache-preflight diagnostics/orthoflow3_toy_critic_local_supervision_v1/val/cache_preflight.json
