#!/bin/bash
#SBATCH --job-name=crosstype2_new_evaluation_run
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=24:00:00
#SBATCH --array=0-23%24
#SBATCH --output=/home/zhihan/research/Basin_C1/diagnostics/crosstype2_mixed_zoo_selector_v1/new_evaluation_run_%A_%a.log
set -euo pipefail
cd /home/zhihan/research/Basin_C1
export PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cpu
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python -B /home/zhihan/research/Basin_C1/diagnostics/crosstype2_mixed_zoo_selector_v1/worker.py new_evaluation "$((SLURM_ARRAY_TASK_ID + 0))"
