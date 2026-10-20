#!/bin/bash
#SBATCH --job-name=crosstype1c_new_evaluation_base
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=24:00:00
#SBATCH --output=/home/zhihan/research/Basin_C1/diagnostics/crosstype1c_two_eta_schedule_v1/new_evaluation_base_%A_%a.log
#SBATCH --array=0-17%18
set -euo pipefail
cd /home/zhihan/research/Basin_C1
export PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cpu
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export CROSSTYPE_TEMPORARY=0
/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python -B /home/zhihan/research/Basin_C1/diagnostics/crosstype1c_two_eta_schedule_v1/worker.py new_evaluation "${SLURM_ARRAY_TASK_ID:-0}"
