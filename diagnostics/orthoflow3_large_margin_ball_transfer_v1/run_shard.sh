#!/bin/bash
#SBATCH --partition=debug
#SBATCH --gres=shard:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
set -euo pipefail
export JAX_PLATFORM_NAME=gpu
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export OMP_NUM_THREADS=2
PLAN_DIR="$1"
WORK_PREFIX="$2"
PY=/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python3.11
SCRIPT=/home/zhihan/research/Basin_C1/diagnostics/orthoflow3_large_margin_ball_transfer_v1/transfer_audit.py
exec "$PY" "$SCRIPT" run-plan --plan "$PLAN_DIR/shard${SLURM_ARRAY_TASK_ID}.jsonl" --work "${WORK_PREFIX}_shard${SLURM_ARRAY_TASK_ID}"
