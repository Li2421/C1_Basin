#!/bin/bash
#SBATCH --job-name=eta_codebook_train
#SBATCH --partition=gpu
#SBATCH --gres=shard:1
#SBATCH --cpus-per-task=3
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=/home/zhihan/research/Basin_C1/diagnostics/orthoflow3_shared_eta_codebook_v1/logs/train_%A_%a.out
#SBATCH --error=/home/zhihan/research/Basin_C1/diagnostics/orthoflow3_shared_eta_codebook_v1/logs/train_%A_%a.err
set -euo pipefail
export XLA_PYTHON_CLIENT_PREALLOCATE=false OMP_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3
PY=/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python3.11
case "${SLURM_ARRAY_TASK_ID}" in
  0) exec "$PY" /home/zhihan/research/Basin_C1/diagnostics/orthoflow3_shared_eta_codebook_v1/train_models.py train --kind linear --seed 17;;
  1) exec "$PY" /home/zhihan/research/Basin_C1/diagnostics/orthoflow3_shared_eta_codebook_v1/train_models.py train --kind mlp --seed 17;;
  2) exec "$PY" /home/zhihan/research/Basin_C1/diagnostics/orthoflow3_shared_eta_codebook_v1/train_models.py train --kind mlp --seed 23;;
  3) exec "$PY" /home/zhihan/research/Basin_C1/diagnostics/orthoflow3_shared_eta_codebook_v1/train_models.py train --kind mlp --seed 41;;
esac
