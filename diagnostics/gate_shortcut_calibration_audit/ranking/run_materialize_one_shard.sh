#!/usr/bin/env bash
#SBATCH --job-name=gate-audit-folds
#SBATCH --partition=gpu
#SBATCH --gres=shard:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH --output=/home/zhihan/research/Basin_C1/diagnostics/gate_shortcut_calibration_audit/ranking/materialize-%j.out
#SBATCH --error=/home/zhihan/research/Basin_C1/diagnostics/gate_shortcut_calibration_audit/ranking/materialize-%j.err

set -euo pipefail
AUDIT_ROOT=/home/zhihan/research/Basin_C1
AUDIT_PY=/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python
export PYTHONPATH="$AUDIT_ROOT"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.20
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
cd "$AUDIT_ROOT"
nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu --format=csv,noheader
"$AUDIT_PY" diagnostics/gate_shortcut_calibration_audit/ranking/materialize_shared_checkpoints.py
