#!/usr/bin/env bash
#SBATCH --job-name=one-step-ext
#SBATCH --partition=gpu
#SBATCH --gres=shard:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=01:00:00
#SBATCH --output=/home/zhihan/research/Basin_C1/diagnostics/gate_conceptual_validity_audit/temporal/slurm-%j.out
#SBATCH --error=/home/zhihan/research/Basin_C1/diagnostics/gate_conceptual_validity_audit/temporal/slurm-%j.err

set -euo pipefail
ROOT=/home/zhihan/research/Basin_C1
PY=/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python
export PYTHONPATH="$ROOT"
export XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.12
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
cd "$ROOT"
exec "$PY" diagnostics/gate_conceptual_validity_audit/temporal/one_step_rollout.py \
  --shard "${SHARD_INDEX:?}" --shards 3 --device gpu --batch 16 \
  --seed-stop "${SEED_STOP:-64}" --stage "${STAGE:-extension_to128}" \
  --selection-json "${SELECTION_JSON:-diagnostics/gate_conceptual_validity_audit/temporal/adaptive_extension_plan.json}"
