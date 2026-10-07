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
IDS=(N_r049_s159 N_r126_s461 N_r002_m120)
SID="${IDS[$SLURM_ARRAY_TASK_ID]}"
ROOT=/home/zhihan/research/Basin_C1/diagnostics/orthoflow3_basin_margin_learning_v1
PY=/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python3.11
exec "$PY" "$ROOT/build_one_anchor.py" --state-id "$SID" --out "$ROOT/anchor_builds/$SID"
