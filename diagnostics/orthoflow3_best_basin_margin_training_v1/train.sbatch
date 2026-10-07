#!/bin/bash
#SBATCH --job-name=bestmargintrain
#SBATCH --partition=gpu
#SBATCH --gres=shard:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=/home/zhihan/research/Basin_C1/diagnostics/orthoflow3_best_basin_margin_training_v1/logs/train_%A_%a.out
#SBATCH --error=/home/zhihan/research/Basin_C1/diagnostics/orthoflow3_best_basin_margin_training_v1/logs/train_%A_%a.err
set -euo pipefail
export XLA_PYTHON_CLIENT_PREALLOCATE=false OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
case "${SLURM_ARRAY_TASK_ID}" in 0) seed=17;; 1) seed=23;; 2) seed=41;; *) exit 2;; esac
exec /home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python3.11 /home/zhihan/research/Basin_C1/diagnostics/orthoflow3_best_basin_margin_training_v1/train_margin.py train --seed "$seed"
