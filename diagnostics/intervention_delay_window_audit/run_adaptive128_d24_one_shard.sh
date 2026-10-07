#!/usr/bin/env bash
#SBATCH --job-name=delay128-d24
#SBATCH --partition=gpu
#SBATCH --gres=shard:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=01:00:00
#SBATCH --output=/home/zhihan/research/Basin_C1/diagnostics/intervention_delay_window_audit/slurm-adaptive128-d24-%j.out
#SBATCH --error=/home/zhihan/research/Basin_C1/diagnostics/intervention_delay_window_audit/slurm-adaptive128-d24-%j.err

set -euo pipefail
AUDIT_ROOT=/home/zhihan/research/Basin_C1
AUDIT_PY=/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python
AUDIT_DIR="$AUDIT_ROOT/diagnostics/intervention_delay_window_audit"
export PYTHONPATH="$AUDIT_ROOT"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.12
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
cd "$AUDIT_ROOT"
nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu --format=csv,noheader
exec "$AUDIT_PY" "$AUDIT_DIR/run_delay_rollouts.py" \
  --delays 2,4 --shard 0 --shards 1 --device gpu --batch 16 \
  --y-long 1 --selection-json "$AUDIT_DIR/adaptive_to128_plan.json" \
  --stage adaptive_to128_d24_nonzero
