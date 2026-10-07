#!/usr/bin/env bash
set -euo pipefail

here=/home/zhihan/research/Basin_C1/diagnostics/gphi_fixed_cadence_ablation_v1
py=/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python
export PYTHONPATH=/home/zhihan/research/Basin_C1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

"$py" "$here/run_evaluation.py" \
  --cadence 1 --namespace smoke_h1_parity --limit 1 --device cpu
"$py" "$here/run_evaluation.py" \
  --cadence 4 --namespace smoke_h4_semantics --limit 1 --device cpu
"$py" "$here/verify_smoke.py"
