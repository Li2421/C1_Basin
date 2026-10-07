#!/bin/bash
#SBATCH --job-name=db_mode_transfer
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=5G
#SBATCH --time=12:00:00
#SBATCH --output=/home/zhihan/research/Basin_C1/diagnostics/orthoflow3_db_shared_mode_transfer_v1/logs/%A_%a.out
#SBATCH --error=/home/zhihan/research/Basin_C1/diagnostics/orthoflow3_db_shared_mode_transfer_v1/logs/%A_%a.err
set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
exec /home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python3.11 /home/zhihan/research/Basin_C1/diagnostics/orthoflow3_db_shared_mode_transfer_v1/run_db.py --plan "${TRANSFER_PLAN}" --shard "${SLURM_ARRAY_TASK_ID}"
