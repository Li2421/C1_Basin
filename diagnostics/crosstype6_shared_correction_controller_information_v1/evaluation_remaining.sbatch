#!/bin/bash
#SBATCH --job-name=crosstype6_evaluation_remaining
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=24:00:00
#SBATCH --output=/home/zhihan/research/Basin_C1/diagnostics/crosstype6_shared_correction_controller_information_v1/evaluation_remaining_%A_%a.log
#SBATCH --array=0-17%18
set -euo pipefail
cd /home/zhihan/research/Basin_C1
export PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cpu
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export CUDA_CACHE_PATH=/home/zhihan/research/Basin_C1/diagnostics/crosstype6_shared_correction_controller_information_v1/cuda_cache
export CROSSTYPE_TEMPORARY_RESOURCE=0
if (( ${SLURM_ARRAY_TASK_ID:-0} >= 18 )); then export CROSSTYPE_TEMPORARY_RESOURCE=1; fi
worker_ids=(0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17)
unset XLA_FLAGS
/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python -B /home/zhihan/research/Basin_C1/diagnostics/crosstype6_shared_correction_controller_information_v1/worker_evaluation_c6.py "${worker_ids[${SLURM_ARRAY_TASK_ID:-0}]}" 18 remaining
