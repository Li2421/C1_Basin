#!/bin/bash
#SBATCH --job-name=shared_core
#SBATCH --partition=gpu
#SBATCH --gres=shard:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=/home/zhihan/research/Basin_C1/diagnostics/orthoflow3_shared_conservative_family_v1/logs/%A_%a.out
#SBATCH --error=/home/zhihan/research/Basin_C1/diagnostics/orthoflow3_shared_conservative_family_v1/logs/%A_%a.err
set -euo pipefail
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 XLA_PYTHON_CLIENT_PREALLOCATE=false
exec /home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python3.11 /home/zhihan/research/Basin_C1/diagnostics/orthoflow3_shared_conservative_family_v1/run_toy_shard.py --plan "${SHARED_PLAN}" --shard "${SLURM_ARRAY_TASK_ID}"
