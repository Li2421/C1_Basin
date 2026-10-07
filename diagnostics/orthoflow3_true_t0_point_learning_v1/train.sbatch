#!/bin/bash
#SBATCH --job-name=t0train
#SBATCH --partition=gpu
#SBATCH --gres=shard:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=/home/zhihan/research/Basin_C1/diagnostics/orthoflow3_true_t0_point_learning_v1/logs/train_%A_%a.out
#SBATCH --error=/home/zhihan/research/Basin_C1/diagnostics/orthoflow3_true_t0_point_learning_v1/logs/train_%A_%a.err
set -euo pipefail
seeds=(17 23 41)
export XLA_PYTHON_CLIENT_PREALLOCATE=false
exec /home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python3.11 /home/zhihan/research/Basin_C1/diagnostics/orthoflow3_true_t0_point_learning_v1/build_and_train.py train --seed "${seeds[$SLURM_ARRAY_TASK_ID]}"
