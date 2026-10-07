#!/bin/bash
#SBATCH --job-name=sparse_fresh
#SBATCH --partition=gpu
#SBATCH --gres=shard:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=10G
#SBATCH --time=03:00:00
#SBATCH --output=/home/zhihan/research/Basin_C1/diagnostics/orthoflow3_sparse_mode_relearning_v1/logs/fresh_%A_%a.out
#SBATCH --error=/home/zhihan/research/Basin_C1/diagnostics/orthoflow3_sparse_mode_relearning_v1/logs/fresh_%A_%a.err
set -euo pipefail
export XLA_PYTHON_CLIENT_PREALLOCATE=false OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
exec /home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python3.11 /home/zhihan/research/Basin_C1/diagnostics/orthoflow3_sparse_mode_relearning_v1/run_fresh.py --phase "$PHASE" --shard "$SLURM_ARRAY_TASK_ID"
