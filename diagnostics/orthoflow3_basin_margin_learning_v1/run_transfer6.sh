#!/bin/bash
#SBATCH --partition=debug
#SBATCH --gres=shard:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
set -euo pipefail
export JAX_PLATFORM_NAME=gpu
export JAX_ENABLE_X64=True
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export OMP_NUM_THREADS=2
ROOT=/home/zhihan/research/Basin_C1/diagnostics/orthoflow3_basin_margin_learning_v1
PY=/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python3.11
PLAN_NAME="$1"
exec "$PY" "$ROOT/new_transfer.py" run-plan --plan "$ROOT/new_anchor_transfers/plans/$PLAN_NAME/shard${SLURM_ARRAY_TASK_ID}.jsonl" --work "${PLAN_NAME}_shard${SLURM_ARRAY_TASK_ID}"
