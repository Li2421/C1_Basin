#!/bin/bash
#SBATCH --job-name=ortho3-hygiene-tail
#SBATCH --partition=cpu
#SBATCH --array=17,35,53,71
#SBATCH --cpus-per-task=1
#SBATCH --mem=7G
#SBATCH --time=04:00:00
#SBATCH --output=diagnostics/orthoflow3_data_hygiene_v2/logs/tail_%A_%a.out
#SBATCH --error=diagnostics/orthoflow3_data_hygiene_v2/logs/tail_%A_%a.err

set -euo pipefail
cd /home/zhihan/research/Basin_C1
export PYTHONPATH=/home/zhihan/research/Basin_C1
export JAX_PLATFORMS=cpu
export JAX_PLATFORM_NAME=cpu
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
exec /home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python \
  diagnostics/orthoflow3_data_hygiene_v2/backfill_ring.py \
  --shard "${SLURM_ARRAY_TASK_ID}" --shards 72
