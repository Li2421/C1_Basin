#!/bin/bash
#SBATCH --job-name=csg_prec
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=3G
#SBATCH --time=01:00:00
#SBATCH --array=0-15
#SBATCH --output=diagnostics/controller_state_geometry_transfer_v1/logs/%x_%A_%a.log
set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 JAX_PLATFORMS=cpu
cd /home/zhihan/research/Basin_C1
exec /home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python -m diagnostics.controller_state_geometry_transfer_v1.supplement worker --index "$SLURM_ARRAY_TASK_ID" --workers 16
