#!/bin/bash
#SBATCH --job-name=r30_plots
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:20:00
#SBATCH --output=/home/zhihan/research/Basin_C1/diagnostics/teacher_student_macflow_v1/plots_%j.log
set -euo pipefail
cd /home/zhihan/research/Basin_C1
export PYTHONDONTWRITEBYTECODE=1
/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1-plots/bin/python -B diagnostics/teacher_student_macflow_v1/plots.py
/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python -B diagnostics/teacher_student_macflow_v1/wrapper_tests.py
