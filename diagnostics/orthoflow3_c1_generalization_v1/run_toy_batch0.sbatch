#!/bin/bash
#SBATCH --job-name=c1-toy-confirm0
#SBATCH --partition=cpu
#SBATCH --array=0-5%6
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=08:00:00
#SBATCH --output=diagnostics/orthoflow3_c1_generalization_v1/toy_replication/batch0_%A_%a.out
#SBATCH --error=diagnostics/orthoflow3_c1_generalization_v1/toy_replication/batch0_%A_%a.err
set -euo pipefail
cd /home/zhihan/research/Basin_C1
export PYTHONPATH=/home/zhihan/research/Basin_C1
export JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES=''
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python \
  diagnostics/orthoflow3_c1_generalization_v1/run_toy_replication.py \
  --shard "${SLURM_ARRAY_TASK_ID}" --shards 6 \
  --cache-preflight diagnostics/orthoflow3_c1_generalization_v1/toy_replication/preflight_proposal_batch0.json
