#!/bin/bash
#SBATCH --job-name=of3_qv2_follow
#SBATCH --partition=debug
#SBATCH --array=0-1
#SBATCH --gres=shard:1
#SBATCH --cpus-per-task=3
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=/home/zhihan/research/Basin_C1/diagnostics/orthoflow3_q_learnability_v2/logs/follow_%A_%a.out
#SBATCH --error=/home/zhihan/research/Basin_C1/diagnostics/orthoflow3_q_learnability_v2/logs/follow_%A_%a.err

set -euo pipefail
: "${PLAN:?PLAN must name a frozen follow-up JSON}"
export OMP_NUM_THREADS=3
export OPENBLAS_NUM_THREADS=3
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.35
cd /home/zhihan/research/Basin_C1
exec /home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python \
  diagnostics/orthoflow3_q_learnability_v2/run_q_rollouts.py \
  --plan "${PLAN}" \
  --shard "${SLURM_ARRAY_TASK_ID}" \
  --shards 2 \
  --batch 32
