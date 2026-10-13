#!/bin/bash
#SBATCH --job-name=c1-db-states
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=3G
#SBATCH --time=01:30:00
#SBATCH --output=diagnostics/physical_state_critic_formal_v1/logs/db_states_%A_%a.log
set -euo pipefail
cd /home/zhihan/research/Basin_C1
export PYTHONPATH=/home/zhihan/research/Basin_C1
export JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
splits=(train validation test)
split=${splits[$((SLURM_ARRAY_TASK_ID / 3))]}
regime=$((SLURM_ARRAY_TASK_ID % 3))
/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python -m diagnostics.physical_state_critic_formal_v1.prepare_db_states worker --split "$split" --regime-index "$regime"
