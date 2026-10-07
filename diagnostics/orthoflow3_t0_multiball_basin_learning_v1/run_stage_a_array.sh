#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=shard:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=06:00:00
set -euo pipefail
export JAX_PLATFORM_NAME=gpu
export JAX_ENABLE_X64=True
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export OMP_NUM_THREADS=2
PY=/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python3.11
ROOT=/home/zhihan/research/Basin_C1/diagnostics/orthoflow3_t0_multiball_basin_learning_v1
exec "$PY" "$ROOT/run_stage_a_state.py" --index "$SLURM_ARRAY_TASK_ID"
