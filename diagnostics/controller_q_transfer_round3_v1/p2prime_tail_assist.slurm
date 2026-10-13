#!/bin/bash
#SBATCH --job-name=controller_q_round3_tailassist
#SBATCH --partition=cpu
#SBATCH --array=29,35
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=03:00:00
#SBATCH --output=diagnostics/controller_q_transfer_round3_v1/P2PRIME_TAIL_ASSIST_%a.log
set -euo pipefail
export JAX_PLATFORMS=cpu OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 XLA_PYTHON_CLIENT_PREALLOCATE=false
cd /home/zhihan/research/Basin_C1
/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python diagnostics/controller_q_transfer_round3_v1/p2prime_run.py worker --index "$SLURM_ARRAY_TASK_ID" --workers 132
