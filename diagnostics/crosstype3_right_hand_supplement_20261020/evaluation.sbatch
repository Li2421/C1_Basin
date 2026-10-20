#!/bin/bash
#SBATCH --job-name=rhr3_evaluation
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=03:00:00
#SBATCH --output=/home/zhihan/research/Basin_C1/diagnostics/crosstype3_right_hand_supplement_20261020/evaluation_%A_%a.log
#SBATCH --array=0-17%18
set -euo pipefail
cd /home/zhihan/research/Basin_C1
export PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cpu
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
unset XLA_FLAGS
/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python -B /home/zhihan/research/Basin_C1/diagnostics/crosstype3_right_hand_supplement_20261020/supplement.py worker "${SLURM_ARRAY_TASK_ID}"
