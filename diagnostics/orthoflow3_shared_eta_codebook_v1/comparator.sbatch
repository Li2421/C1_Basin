#!/bin/bash
#SBATCH --job-name=eta_codebook_cmp
#SBATCH --partition=gpu
#SBATCH --gres=shard:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=10G
#SBATCH --time=02:00:00
#SBATCH --output=/home/zhihan/research/Basin_C1/diagnostics/orthoflow3_shared_eta_codebook_v1/logs/comparator_%A_%a.out
#SBATCH --error=/home/zhihan/research/Basin_C1/diagnostics/orthoflow3_shared_eta_codebook_v1/logs/comparator_%A_%a.err
set -euo pipefail
export XLA_PYTHON_CLIENT_PREALLOCATE=false OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
exec /home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python3.11 /home/zhihan/research/Basin_C1/diagnostics/orthoflow3_shared_eta_codebook_v1/comparator_run.py --shard "${SLURM_ARRAY_TASK_ID}"
