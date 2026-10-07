#!/bin/bash
#SBATCH --job-name=c1-gen-state-swap
#SBATCH --partition=cpu
#SBATCH --array=0-14%15
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=08:00:00
#SBATCH --output=diagnostics/orthoflow3_core_hypothesis_audit_v1/swap_%A_%a.out
#SBATCH --error=diagnostics/orthoflow3_core_hypothesis_audit_v1/swap_%A_%a.err
set -euo pipefail
cd /home/zhihan/research/Basin_C1
export PYTHONPATH=/home/zhihan/research/Basin_C1
export JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 XLA_PYTHON_CLIENT_PREALLOCATE=false
/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python -m diagnostics.orthoflow3_core_hypothesis_audit_v1.run_generator_swap \
  --shard "${SLURM_ARRAY_TASK_ID}" --shards 15 \
  --cache-preflight diagnostics/orthoflow3_core_hypothesis_audit_v1/cache_preflight_generator_swap.json
