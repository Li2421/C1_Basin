#!/bin/bash
#SBATCH --job-name=r30_numeric
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=/home/zhihan/research/Basin_C1/diagnostics/teacher_student_macflow_v1/numeric_%j.log
set -euo pipefail
cd /home/zhihan/research/Basin_C1
export PYTHONDONTWRITEBYTECODE=1
PY=/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python
"$PY" -B diagnostics/teacher_student_macflow_v1/postflight.py teacher
"$PY" -B diagnostics/teacher_student_macflow_v1/postflight.py main
"$PY" -B diagnostics/teacher_student_macflow_v1/analysis.py
/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1-plots/bin/python -B diagnostics/teacher_student_macflow_v1/plots.py
