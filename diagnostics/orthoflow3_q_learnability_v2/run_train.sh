#!/bin/bash
#SBATCH --job-name=of3_qv2_train
#SBATCH --partition=debug
#SBATCH --gres=shard:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=/home/zhihan/research/Basin_C1/diagnostics/orthoflow3_q_learnability_v2/logs/train_%j.out
#SBATCH --error=/home/zhihan/research/Basin_C1/diagnostics/orthoflow3_q_learnability_v2/logs/train_%j.err

set -euo pipefail
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.25
cd /home/zhihan/research/Basin_C1
exec /home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python \
  diagnostics/orthoflow3_q_learnability_v2/train_q_and_plan_followups.py
