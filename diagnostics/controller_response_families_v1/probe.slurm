#!/bin/bash
#SBATCH --job-name=q_root_response
#SBATCH --cpus-per-task=1
#SBATCH --mem=2500M
#SBATCH --time=04:00:00
#SBATCH --array=0-47%24
#SBATCH --output=diagnostics/controller_response_families_v1/worker_%a.log
set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 JAX_PLATFORMS=cpu
cd /home/zhihan/research/Basin_C1
exec /home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python -m diagnostics.controller_response_families_v1.study worker --index "$SLURM_ARRAY_TASK_ID" --workers 48
