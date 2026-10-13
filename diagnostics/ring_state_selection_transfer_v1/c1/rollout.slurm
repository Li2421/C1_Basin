#!/bin/bash
#SBATCH --job-name=ring_c1
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=3G
#SBATCH --time=02:00:00
#SBATCH --array=0-87%22
#SBATCH --output=diagnostics/ring_state_selection_transfer_v1/c1/rollout_%a.log
set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 JAX_PLATFORMS=cpu
cd /home/zhihan/research/Basin_C1
exec /home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python -m diagnostics.ring_state_selection_transfer_v1.c1.run worker --index "$SLURM_ARRAY_TASK_ID" --workers 88
