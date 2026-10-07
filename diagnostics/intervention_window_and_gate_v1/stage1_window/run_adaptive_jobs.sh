#!/usr/bin/env bash
#SBATCH --job-name=iwg-stage1-adaptive
#SBATCH --partition=gpu
#SBATCH --gres=shard:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G
#SBATCH --time=04:00:00
#SBATCH --output=/home/zhihan/research/Basin_C1/diagnostics/intervention_window_and_gate_v1/stage1_window/slurm-adaptive-%j.out
#SBATCH --error=/home/zhihan/research/Basin_C1/diagnostics/intervention_window_and_gate_v1/stage1_window/slurm-adaptive-%j.err

set -euo pipefail
AUDIT_ROOT=/home/zhihan/research/Basin_C1
AUDIT_PY=/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python
STAGE_DIR="$AUDIT_ROOT/diagnostics/intervention_window_and_gate_v1/stage1_window"

export PYTHONPATH="$AUDIT_ROOT"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.12
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4

cd "$AUDIT_ROOT"
exec "$AUDIT_PY" "$STAGE_DIR/adaptive_job_orchestrator.py" execute

