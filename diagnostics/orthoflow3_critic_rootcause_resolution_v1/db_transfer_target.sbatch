#!/bin/bash
#SBATCH --job-name=c1-db-frozen-inputs
#SBATCH --partition=debug
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#SBATCH --array=0-4%5
#SBATCH --output=diagnostics/orthoflow3_critic_rootcause_resolution_v1/db_target_%A_%a.out
#SBATCH --error=diagnostics/orthoflow3_critic_rootcause_resolution_v1/db_target_%A_%a.err
set -euo pipefail
cd /home/zhihan/research/Basin_C1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 JAX_PLATFORMS=cpu XLA_PYTHON_CLIENT_PREALLOCATE=false
/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python -m diagnostics.orthoflow3_critic_rootcause_resolution_v1.db_transfer_evaluate inputs --worker "$SLURM_ARRAY_TASK_ID"
