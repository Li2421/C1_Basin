#!/usr/bin/env bash
# One resumable compute shard; the batch launcher owns the mandatory video gate.
set -euo pipefail
cd /home/zhihan/research/Basin_C1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 JAX_PLATFORMS=cpu
export XLA_PYTHON_CLIENT_PREALLOCATE=false
exec /home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python \
  diagnostics/state_nuisance_rootcause_v1/real_state_expansion.py worker \
  --stage "$1" --worker "${SLURM_ARRAY_TASK_ID}" --workers "$2" --chunk-size 192
