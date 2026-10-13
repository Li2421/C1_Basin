#!/bin/bash
#SBATCH --job-name=c1-q-final-seeds
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=5G
#SBATCH --time=02:00:00
#SBATCH --output=diagnostics/transferable_state_conditioning_v1/final/logs/%x_%A_%a.log
set -euo pipefail
cd /home/zhihan/research/Basin_C1
export PYTHONPATH=/home/zhihan/research/Basin_C1
export JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
: "${FINAL_BANK:?proposal or anchor must be supplied with --export}"
/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python \
  -m diagnostics.transferable_state_conditioning_v1.final_worker run \
  --bank "$FINAL_BANK" --index "$SLURM_ARRAY_TASK_ID"
