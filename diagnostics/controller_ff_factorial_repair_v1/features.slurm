#!/bin/bash
#SBATCH --job-name=q_factorial_static_features
#SBATCH --cpus-per-task=1
#SBATCH --mem=3G
#SBATCH --time=02:00:00
#SBATCH --array=0-31%2
#SBATCH --output=diagnostics/controller_ff_factorial_repair_v1/features_%a.log
set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 JAX_PLATFORMS=cpu
cd /home/zhihan/research/Basin_C1
exec /home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python -m diagnostics.controller_ff_factorial_repair_v1.features features --index "$SLURM_ARRAY_TASK_ID" --workers 32
