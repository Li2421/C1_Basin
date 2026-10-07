#!/bin/bash
#SBATCH --partition=debug
#SBATCH --gres=shard:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:45:00
set -euo pipefail
export JAX_PLATFORM_NAME=gpu
export JAX_ENABLE_X64=True
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export OMP_NUM_THREADS=2
PY=/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python3.11
RUN=/home/zhihan/research/Basin_C1/diagnostics/orthoflow3_t0_basin_structure_v1/run_t0_anchor.py
INDICES=(4 6 7)
exec "$PY" "$RUN" --index "${INDICES[$SLURM_ARRAY_TASK_ID]}" --cap-steps 3000000
