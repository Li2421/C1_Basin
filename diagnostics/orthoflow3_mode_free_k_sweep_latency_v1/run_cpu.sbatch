#!/bin/bash
#SBATCH --job-name=ortho3-modefree-K16
#SBATCH --partition=cpu
#SBATCH --array=0-11%12
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=08:00:00
#SBATCH --output=/home/zhihan/research/Basin_C1/diagnostics/orthoflow3_mode_free_k_sweep_latency_v1/logs/%A_%a.out
#SBATCH --error=/home/zhihan/research/Basin_C1/diagnostics/orthoflow3_mode_free_k_sweep_latency_v1/logs/%A_%a.err
set -euo pipefail
export JAX_PLATFORMS=cpu
export CUDA_VISIBLE_DEVICES=''
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
cd /home/zhihan/research/Basin_C1
/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python \
  diagnostics/orthoflow3_mode_free_k_sweep_latency_v1/run_extended_rollouts.py \
  --shard "${SLURM_ARRAY_TASK_ID}" --shards 12 \
  --cache-preflight diagnostics/orthoflow3_mode_free_k_sweep_latency_v1/cache_preflight_resume.json
