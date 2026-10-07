#!/usr/bin/env bash
#SBATCH --job-name=delay-a256-d01
#SBATCH --partition=gpu
#SBATCH --gres=shard:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=00:45:00
#SBATCH --output=/home/zhihan/research/Basin_C1/diagnostics/intervention_delay_window_audit/slurm-a256-d01-%j.out
#SBATCH --error=/home/zhihan/research/Basin_C1/diagnostics/intervention_delay_window_audit/slurm-a256-d01-%j.err

set -euo pipefail
AUDIT_ROOT=/home/zhihan/research/Basin_C1
AUDIT_PY=/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python
export PYTHONPATH="$AUDIT_ROOT"
export XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.18
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
cd "$AUDIT_ROOT"
exec "$AUDIT_PY" diagnostics/intervention_delay_window_audit/run_delay_rollouts.py \
  --delays 0,1 --y-long all --shard 0 --shards 1 --device gpu --batch 16 \
  --stage adaptive256_d01 --seed-start 0 --seed-stop 128 \
  --selection-json diagnostics/intervention_delay_window_audit/adaptive_to256_plan.json
