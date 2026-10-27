#!/bin/bash
#SBATCH --job-name=baseline2-flow
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=3G
#SBATCH --time=6-00:00:00
#SBATCH --output=/home/zhihan/research/Basin_C1/diagnostics/baseline2_literature_scenes_v1/logs/flow_%A_%a.log
set -euo pipefail
cd /home/zhihan/research/Basin_C1
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export JAX_PLATFORMS=cpu
unset XLA_FLAGS
exec /home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python -B diagnostics/baseline2_literature_scenes_v1/worker_flow_memo.py "$BASELINE2_STAGE" "$SLURM_ARRAY_TASK_ID" "$BASELINE2_SHARDS"
