#!/bin/bash
#SBATCH --job-name=ring-k16-diag
#SBATCH --partition=cpu
#SBATCH --array=0-17
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=7-00:00:00
#SBATCH --output=diagnostics/orthoflow3_ring_k16_diagnostic_v1/logs/%A_%a.out
#SBATCH --error=diagnostics/orthoflow3_ring_k16_diagnostic_v1/logs/%A_%a.err

set -euo pipefail
cd /home/zhihan/research/Basin_C1
mkdir -p diagnostics/orthoflow3_ring_k16_diagnostic_v1/logs
export PYTHONPATH=/home/zhihan/research/Basin_C1
export JAX_PLATFORMS=cpu
export JAX_PLATFORM_NAME=cpu
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
exec /home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python \
  diagnostics/orthoflow3_ring_k16_diagnostic_v1/run_k16.py \
  run --shard "${SLURM_ARRAY_TASK_ID}" --shards 18
