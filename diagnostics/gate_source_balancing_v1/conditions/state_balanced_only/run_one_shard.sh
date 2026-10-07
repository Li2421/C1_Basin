#!/usr/bin/env bash
#SBATCH --job-name=gate-balance-state
#SBATCH --partition=gpu
#SBATCH --gres=shard:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:35:00
#SBATCH --output=/home/zhihan/research/Basin_C1/diagnostics/gate_source_balancing_v1/conditions/state_balanced_only/state-balanced-%j.out
#SBATCH --error=/home/zhihan/research/Basin_C1/diagnostics/gate_source_balancing_v1/conditions/state_balanced_only/state-balanced-%j.err

set -euo pipefail
TASK_ROOT=/home/zhihan/research/Basin_C1
TASK_PY=/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python
export PYTHONPATH="$TASK_ROOT"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.20
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
cd "$TASK_ROOT"
"$TASK_PY" diagnostics/gate_source_balancing_v1/conditions/state_balanced_only/run_state_balanced_only.py
