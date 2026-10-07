#!/bin/bash
#SBATCH --job-name=c1-db-source-fit
#SBATCH --partition=debug
#SBATCH --cpus-per-task=1
#SBATCH --mem=5G
#SBATCH --time=01:00:00
#SBATCH --array=0-14%5
#SBATCH --output=diagnostics/orthoflow3_critic_rootcause_resolution_v1/db_fit_%A_%a.out
#SBATCH --error=diagnostics/orthoflow3_critic_rootcause_resolution_v1/db_fit_%A_%a.err
set -euo pipefail
cd /home/zhihan/research/Basin_C1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 JAX_PLATFORMS=cpu XLA_PYTHON_CLIENT_PREALLOCATE=false
/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python -m diagnostics.orthoflow3_critic_rootcause_resolution_v1.db_transfer_train train --index "$SLURM_ARRAY_TASK_ID"
