#!/usr/bin/env bash
set -euo pipefail
export JAX_PLATFORMS=cuda OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 XLA_PYTHON_CLIENT_PREALLOCATE=false
cd /home/zhihan/research/Basin_C1
BASE=diagnostics/controller_loco_real_flow_v1
PY=/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python
read -r fold arm < <("$PY" -c "import json; print(*json.load(open('$BASE/stage_c/PREDICTION_TASKS.json'))[int(__import__('os').environ['SLURM_ARRAY_TASK_ID'])])")
"$PY" "$BASE/stage_c_predict.py" "$fold" "$arm"
