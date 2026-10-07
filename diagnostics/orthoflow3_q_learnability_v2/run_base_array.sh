#!/bin/bash
#SBATCH --job-name=of3_qv2_base
#SBATCH --partition=debug
#SBATCH --array=0-1
#SBATCH --gres=shard:1
#SBATCH --cpus-per-task=3
#SBATCH --mem=18G
#SBATCH --time=01:00:00
#SBATCH --output=/home/zhihan/research/Basin_C1/diagnostics/orthoflow3_q_learnability_v2/logs/base_%A_%a.out
#SBATCH --error=/home/zhihan/research/Basin_C1/diagnostics/orthoflow3_q_learnability_v2/logs/base_%A_%a.err

set -euo pipefail
export OMP_NUM_THREADS=3
export OPENBLAS_NUM_THREADS=3
cd /home/zhihan/research/Basin_C1
exec /home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python \
  diagnostics/orthoflow3_q_learnability_v2/run_q_rollouts.py \
  --plan base_rollout_plan.json \
  --shard "${SLURM_ARRAY_TASK_ID}" \
  --shards 2 \
  --batch 32
