#!/bin/bash
#SBATCH --job-name=controller_q_round3_p1
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=02:00:00
#SBATCH --array=0-23%24
#SBATCH --output=diagnostics/controller_q_transfer_round3_v1/P1_ROLLOUT_%a.log
set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 JAX_PLATFORMS=cpu
cd /home/zhihan/research/Basin_C1
exec /home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python diagnostics/controller_q_transfer_round3_v1/p1_run.py worker --index "$SLURM_ARRAY_TASK_ID" --workers 24
