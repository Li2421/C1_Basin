#!/usr/bin/env bash
#SBATCH --job-name=iwg-stage1-short
#SBATCH --partition=gpu
#SBATCH --gres=shard:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G
#SBATCH --time=04:00:00
#SBATCH --output=/home/zhihan/research/Basin_C1/diagnostics/intervention_window_and_gate_v1/stage1_window/slurm-short-%j.out
#SBATCH --error=/home/zhihan/research/Basin_C1/diagnostics/intervention_window_and_gate_v1/stage1_window/slurm-short-%j.err
set -euo pipefail
AUDIT_ROOT=/home/zhihan/research/Basin_C1
AUDIT_PY=/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python
STAGE_DIR="$AUDIT_ROOT/diagnostics/intervention_window_and_gate_v1/stage1_window"
export PYTHONPATH="$AUDIT_ROOT"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.12
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
cd "$AUDIT_ROOT"
"$AUDIT_PY" "$STAGE_DIR/run_delay_rollouts.py" \
  --delays 0,4,8,16 --stage coarse_short --shard 0 --shards 1 --device gpu --batch 16
"$AUDIT_PY" "$STAGE_DIR/check_completed_stage.py" \
  --stage coarse_short --shard 0 --expected-delays 0,4,8,16 --expected-new 20928
