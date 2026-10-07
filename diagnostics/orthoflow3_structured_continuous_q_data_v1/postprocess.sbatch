#!/bin/bash
#SBATCH --job-name=structured_q_post
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=/home/zhihan/research/Basin_C1/diagnostics/orthoflow3_structured_continuous_q_data_v1/logs/post_%j.out
#SBATCH --error=/home/zhihan/research/Basin_C1/diagnostics/orthoflow3_structured_continuous_q_data_v1/logs/post_%j.err
set -euo pipefail
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 XLA_PYTHON_CLIENT_PREALLOCATE=false
ROOT=/home/zhihan/research/Basin_C1
H=${ROOT}/diagnostics/orthoflow3_structured_continuous_q_data_v1
PY=/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python3.11
cd "${ROOT}"
EXP=$(${PY} - <<'PY'
from pathlib import Path
from shared_rollout_db.src.rollout_db import uid
p=Path('/home/zhihan/research/Basin_C1/diagnostics/orthoflow3_structured_continuous_q_data_v1').resolve()
print(uid('exp',{'path':str(p)}))
PY
)
${PY} -m shared_rollout_db.src.merger --experiment "${EXP}"
${PY} "${H}/register_experiment.py"
${PY} -m shared_rollout_db.plan --manifest "${H}/planned_rollouts.json" --output "${H}/cache_postflight.json" > "${H}/cache_postflight.stdout"
${PY} "${H}/build_datasets.py"
${PY} "${H}/train_evaluate.py"
${PY} "${H}/finalize.py"
