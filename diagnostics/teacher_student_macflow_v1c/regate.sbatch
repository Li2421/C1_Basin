#!/bin/bash
#SBATCH --job-name=r30c_regate
#SBATCH --partition=gpu
#SBATCH --gres=shard:2
#SBATCH --cpus-per-task=1
#SBATCH --mem=20G
#SBATCH --time=02:00:00
#SBATCH --output=/home/zhihan/research/Basin_C1/diagnostics/teacher_student_macflow_v1c/regate_%j.log
set -euo pipefail
cd /home/zhihan/research/Basin_C1
export PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cuda
exec /home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python -B diagnostics/teacher_student_macflow_v1c/regate.py
