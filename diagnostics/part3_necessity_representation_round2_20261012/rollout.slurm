#!/bin/bash
#SBATCH --job-name=part3r2_q16
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=01:30:00
#SBATCH --array=0-3%4
#SBATCH --output=diagnostics/part3_necessity_representation_round2_20261012/worker_%a.log
set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 JAX_PLATFORMS=cpu
cd /home/zhihan/research/Basin_C1
exec /home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python diagnostics/part3_necessity_representation_round2_20261012/run.py worker --index "$SLURM_ARRAY_TASK_ID" --workers 4
