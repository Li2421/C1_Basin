#!/bin/bash
#SBATCH --job-name=r30f_cpu
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=24:00:00
#SBATCH --output=/home/zhihan/research/Basin_C1/diagnostics/teacher_student_macflow_v1f/cpu_%A_%a.log
set -euo pipefail
cd /home/zhihan/research/Basin_C1
export PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cpu
/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python -B diagnostics/teacher_student_macflow_v1f/quota.py check
exec /home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python -B "diagnostics/teacher_student_macflow_v1f/$1" "${@:2}"
