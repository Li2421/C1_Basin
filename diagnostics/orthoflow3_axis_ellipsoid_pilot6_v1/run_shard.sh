#!/bin/bash
#SBATCH --job-name=of3-axis-ell6
#SBATCH --partition=debug
#SBATCH --gres=shard:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=/home/zhihan/research/Basin_C1/diagnostics/orthoflow3_axis_ellipsoid_pilot6_v1/logs/%x_%A_%a.out
#SBATCH --error=/home/zhihan/research/Basin_C1/diagnostics/orthoflow3_axis_ellipsoid_pilot6_v1/logs/%x_%A_%a.err
set -euo pipefail
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=.12
OUT=/home/zhihan/research/Basin_C1/diagnostics/orthoflow3_axis_ellipsoid_pilot6_v1
STAGE=${ELLIPSOID_STAGE:?}
PLAN_DIR=${ELLIPSOID_PLAN_DIR:?}
cd "$OUT"
exec /home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python axis_ellipsoid.py run-plan \
  --plan "$PLAN_DIR/shard${SLURM_ARRAY_TASK_ID}.jsonl" \
  --work "${STAGE}_shard${SLURM_ARRAY_TASK_ID}" --phase "$STAGE"
