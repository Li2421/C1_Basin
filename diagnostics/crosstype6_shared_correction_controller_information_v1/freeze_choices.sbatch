#!/bin/bash
#SBATCH --job-name=crosstype6_freeze_choices
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=20G
#SBATCH --time=24:00:00
#SBATCH --output=/home/zhihan/research/Basin_C1/diagnostics/crosstype6_shared_correction_controller_information_v1/freeze_choices_%A_%a.log
#SBATCH --gres=shard:1
set -euo pipefail
cd /home/zhihan/research/Basin_C1
export PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cuda
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export CUDA_CACHE_PATH=/home/zhihan/research/Basin_C1/diagnostics/crosstype6_shared_correction_controller_information_v1/cuda_cache
export CROSSTYPE_TEMPORARY_RESOURCE=0
if (( ${SLURM_ARRAY_TASK_ID:-0} >= 2 )); then export CROSSTYPE_TEMPORARY_RESOURCE=1; fi
/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python -B /home/zhihan/research/Basin_C1/diagnostics/crosstype6_shared_correction_controller_information_v1/resume_steps_c6.py freeze_choices
