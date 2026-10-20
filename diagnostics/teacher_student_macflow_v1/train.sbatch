#!/bin/bash
#SBATCH --job-name=r30_train
#SBATCH --partition=gpu
#SBATCH --gres=shard:2
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=08:00:00
#SBATCH --output=/home/zhihan/research/Basin_C1/diagnostics/teacher_student_macflow_v1/train_%j.log
set -euo pipefail
cd /home/zhihan/research/Basin_C1
export PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cuda
exec /home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python -B diagnostics/teacher_student_macflow_v1/train.py
