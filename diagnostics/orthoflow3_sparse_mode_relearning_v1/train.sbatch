#!/bin/bash
#SBATCH --job-name=sparse_train
#SBATCH --partition=gpu
#SBATCH --gres=shard:1
#SBATCH --cpus-per-task=3
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=/home/zhihan/research/Basin_C1/diagnostics/orthoflow3_sparse_mode_relearning_v1/logs/train_%A_%a.out
#SBATCH --error=/home/zhihan/research/Basin_C1/diagnostics/orthoflow3_sparse_mode_relearning_v1/logs/train_%A_%a.err
set -euo pipefail
export XLA_PYTHON_CLIENT_PREALLOCATE=false OMP_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3
if [[ "$SLURM_ARRAY_TASK_ID" == 0 ]]; then
  exec /home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python3.11 /home/zhihan/research/Basin_C1/diagnostics/orthoflow3_sparse_mode_relearning_v1/train.py train --kind linear --seed 0
fi
SEEDS=(17 23 41)
exec /home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python3.11 /home/zhihan/research/Basin_C1/diagnostics/orthoflow3_sparse_mode_relearning_v1/train.py train --kind mlp --seed "${SEEDS[$((SLURM_ARRAY_TASK_ID-1))]}"
