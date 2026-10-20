#!/bin/bash
#SBATCH --job-name=r30b_numeric
#SBATCH --partition=gpu
#SBATCH --gres=shard:2
#SBATCH --cpus-per-task=1
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --output=/home/zhihan/research/Basin_C1/diagnostics/teacher_student_macflow_v1b/numeric_%j.log
set -euo pipefail
cd /home/zhihan/research/Basin_C1
export PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cuda
PY=/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python
$PY -B diagnostics/teacher_student_macflow_v1b/equivariance.py
$PY -B diagnostics/teacher_student_macflow_v1b/train.py
$PY -B diagnostics/teacher_student_macflow_v1b/offline.py
